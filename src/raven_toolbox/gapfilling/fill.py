"""Connectivity gap-filling: add the fewest template reactions so reactions that are
*blocked* in a draft can carry flux.

For the other gap-filling flavour (add the fewest template reactions until the model's
own objective becomes feasible) use ``cobra.flux_analysis.gapfill`` — just align the
template's metabolite ids to the draft first, since cobra matches by id.

It solves an MILP: pick the minimum-penalty subset of template reactions such that the
blocked (irreversible) draft reactions can carry flux at steady state. Template
metabolites are matched to the draft by ``name[compartment]`` (via
:func:`add_reactions_from_model`), so templates in a different identifier namespace
than the model still work. Per-reaction ``scores`` (higher = prefer to include) map to
RAVEN's ``rxnScores``; the MILP minimises the penalty ``-score`` (default penalty
``1.0``, i.e. minimise the number of reactions added).
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import cobra
from cobra.flux_analysis import find_blocked_reactions, flux_variability_analysis
from optlang.interface import Variable  # untyped (resolves to Any); used for container hints

from raven_toolbox.manipulation.transfer import add_reactions_from_model


@dataclass
class GapFillResult:
    """Outcome of a connectivity gap-fill.

    ``added_reactions`` are the template reaction ids added to ``model``;
    ``newly_connected`` are draft reactions that were blocked but can now carry flux;
    ``cannot_connect`` are blocked reactions left unconnectable.
    """

    added_reactions: list[str]
    newly_connected: list[str]
    cannot_connect: list[str]
    model: cobra.Model


def _as_models(templates: cobra.Model | Iterable[cobra.Model]) -> list[cobra.Model]:
    return [templates] if isinstance(templates, cobra.Model) else list(templates)


def _merge_templates(model: cobra.Model, templates: list[cobra.Model]) -> tuple[cobra.Model, list[str]]:
    """Copy every template reaction (new ones only) into a working copy of ``model``.

    Returns the working model and the ids of the reactions that came from templates
    (the gap-fill candidates). Metabolites are matched by ``name[compartment]``.
    """
    working = model.copy()
    template_ids: list[str] = []
    for template in templates:
        new = [r.id for r in template.reactions if r.id not in working.reactions]
        if new:
            added = add_reactions_from_model(working, template, new, genes=False, note=None)
            template_ids += [r.id for r in added]
    return working, template_ids


def _solve_min_templates(
    working: cobra.Model,
    template_ids: list[str],
    *,
    scores: dict[str, float] | None,
    penalty: float,
    allow_net_production: bool,
) -> set[str] | None:
    """MILP: minimum-penalty template reactions making ``working`` feasible.

    The requirement (here, forced flux through the blocked reactions) must already be
    imposed on ``working``. Returns the template reaction ids to keep, or ``None`` if
    the problem is infeasible.
    """
    prob = working.problem
    indicators: dict[str, Variable] = {}
    extra = []
    # Big-M for the indicator constraints. A template reaction with an infinite
    # bound would otherwise put an infinite coefficient into the MILP; clamp it to
    # the largest finite bound magnitude in the model (>= any flux it can carry).
    finite_bounds = [
        abs(b)
        for r in working.reactions
        for b in (r.lower_bound, r.upper_bound)
        if math.isfinite(b)
    ]
    big_m = max(finite_bounds) if finite_bounds else 1000.0
    for rid in template_ids:
        rxn = working.reactions.get_by_id(rid)
        y = prob.Variable(f"_gf_keep_{rid}", type="binary")
        indicators[rid] = y
        # Flux is confined to [lb*y, ub*y]: zero unless the reaction is kept (y=1).
        ub = rxn.upper_bound if math.isfinite(rxn.upper_bound) else big_m
        lb = rxn.lower_bound if math.isfinite(rxn.lower_bound) else -big_m
        extra.append(prob.Constraint(rxn.flux_expression - ub * y, ub=0, name=f"_gf_ub_{rid}"))
        extra.append(prob.Constraint(rxn.flux_expression - lb * y, lb=0, name=f"_gf_lb_{rid}"))
    working.add_cons_vars(list(indicators.values()) + extra)

    if allow_net_production:  # relax steady state to Sv >= 0 (mets may accumulate)
        for met in working.metabolites:
            working.constraints[met.id].ub = None

    def pen(rid: str) -> float:
        return -scores[rid] if scores and rid in scores else penalty

    working.objective = prob.Objective(
        sum(pen(rid) * indicators[rid] for rid in template_ids), direction="min"
    )
    working.slim_optimize()
    if working.solver.status != "optimal":
        return None
    return {rid for rid, y in indicators.items() if (y.primal or 0) > 0.5}


def _build_filled(model: cobra.Model, templates: list[cobra.Model], chosen: set[str]) -> cobra.Model:
    filled = model.copy()
    remaining = set(chosen)
    for template in templates:
        ids = [r for r in remaining if r in template.reactions]
        if ids:
            add_reactions_from_model(filled, template, ids, genes=False, note="Added by connect_blocked_reactions")
            remaining -= set(ids)
    return filled


def connect_blocked_reactions(
    model: cobra.Model,
    templates: cobra.Model | Iterable[cobra.Model],
    *,
    scores: dict[str, float] | None = None,
    penalty: float = 1.0,
    allow_net_production: bool = False,
    eps: float = 1.0,
) -> GapFillResult:
    """Add template reactions so blocked draft reactions can carry flux.

    Finds reactions that
    cannot carry flux in ``model`` under its current bounds, then adds the minimum-penalty set of template
    reactions that lets the (irreversible) ones carry flux, and returns the filled
    model. Like RAVEN, only irreversible blocked reactions are forced — reversible
    ones can carry flux trivially in the split formulation, so forcing them is
    uninformative.

    For the *other* gap-filling flavour — adding reactions to make the model's
    objective feasible — use ``cobra.flux_analysis.gapfill`` after aligning the
    template's metabolite ids to the draft.

    The draft is expected to have exchange reactions for its nutrients (otherwise most
    reactions are trivially blocked).

    ``eps`` is the minimum flux a blocked reaction must be able to carry to count as
    rescuable, and the flux forced through it while solving. ``allow_net_production``
    relaxes the steady-state balance from ``S @ v == 0`` to ``S @ v >= 0`` (metabolites
    may accumulate), which can rescue reactions whose real fix is a missing
    consumption route rather than a genuine connectivity gap.
    """
    if penalty <= 0:
        raise ValueError(f"penalty must be > 0 (a non-positive cost malforms the MILP); got {penalty}.")
    templates = _as_models(templates)
    blocked = set(find_blocked_reactions(model, processes=1))
    candidates = [r for r in blocked if model.reactions.get_by_id(r).lower_bound >= 0]

    working, template_ids = _merge_templates(model, templates)

    target: list[str] = []
    if candidates:
        fva = flux_variability_analysis(
            working, reaction_list=candidates, fraction_of_optimum=0.0, processes=1
        )
        # A reaction can be missing from the FVA frame if the solver dropped it
        # (e.g. the reaction was eliminated upstream); treat that as "unreachable"
        # rather than letting the KeyError propagate.
        target = [
            r for r in candidates
            if r in fva.index and fva.at[r, "maximum"] > eps
        ]

    cannot = sorted(blocked - set(target))
    if not target:
        return GapFillResult([], [], cannot, model.copy())

    for rid in target:
        working.reactions.get_by_id(rid).lower_bound = eps
    chosen = _solve_min_templates(
        working, template_ids, scores=scores, penalty=penalty,
        allow_net_production=allow_net_production,
    )
    if chosen is None:
        raise RuntimeError(
            "Gap-filling is infeasible: the blocked reactions cannot all carry flux "
            "even with every template reaction added."
        )
    return GapFillResult(sorted(chosen), sorted(target), cannot, _build_filled(model, templates, chosen))
