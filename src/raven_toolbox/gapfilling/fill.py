"""Connectivity gap-filling: add the fewest template reactions so reactions that are
*blocked* in a draft can carry flux.

For the other gap-filling flavour (add the fewest template reactions until the model's
own objective becomes feasible) use ``cobra.flux_analysis.gapfill``, after aligning the
template's metabolite ids to the draft first, since cobra matches by id.

It solves an MILP: pick the minimum-penalty subset of template reactions such that the
blocked (irreversible) draft reactions can carry flux at steady state. Template
metabolites are matched to the draft by ``name[compartment]`` (via
:func:`add_reactions_from_model`), so templates in a different identifier namespace
than the model still work. Per-reaction ``scores`` (higher = prefer to include) map to
RAVEN's ``rxnScores``; the MILP minimises the penalty ``-score`` (default penalty
``1.0``, i.e. minimise the number of reactions added). As in RAVEN, a score that is not
negative is replaced by the least negative score among the template reactions (or by 0
when none is negative), so a reaction is never rewarded for being added.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import cobra
from cobra.flux_analysis import flux_variability_analysis
from optlang.interface import Variable  # untyped (resolves to Any); used for container hints

from raven_toolbox.gapfilling.templates import unique_templates
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


def _carrying_flux(
    model: cobra.Model, reaction_ids: list[str], *, threshold: float, net_production: bool
) -> set[str]:
    """Ids among ``reaction_ids`` that can carry a flux larger than ``threshold`` in either direction.

    Infinite bounds count as +-10000, and with ``net_production`` the steady state is
    relaxed to ``S @ v >= 0``, as in RAVEN's ``haveFlux`` on a model whose ``b`` upper
    bound is infinite.
    """
    if not reaction_ids:
        return set()
    # The rollback of ``with model`` covers reaction bounds but not constraint bounds, so
    # the steady-state bounds are restored explicitly.
    steady_state_ub = {met.id: model.constraints[met.id].ub for met in model.metabolites}
    try:
        with model as m:
            for rxn in m.reactions:
                if rxn.lower_bound == -math.inf:
                    rxn.lower_bound = -10000.0
                if rxn.upper_bound == math.inf:
                    rxn.upper_bound = 10000.0
            if net_production:
                for met in m.metabolites:
                    m.constraints[met.id].ub = None
            fva = flux_variability_analysis(
                m, reaction_list=reaction_ids, fraction_of_optimum=0.0, processes=1
            )
    finally:
        for mid, ub in steady_state_ub.items():
            model.constraints[mid].ub = ub
    return {
        rid for rid in reaction_ids
        if rid in fva.index
        and (fva.at[rid, "maximum"] > threshold or fva.at[rid, "minimum"] < -threshold)
    }


def _penalties(
    template_ids: list[str], scores: dict[str, float] | None, penalty: float
) -> dict[str, float]:
    """Penalty per template reaction: ``-score`` for negative scores.

    A score that is not negative takes the penalty of the least negative score, or 0 when
    no score is negative (RAVEN's ``getMinNrFluxes``).
    """
    raw = {rid: scores[rid] if scores and rid in scores else -penalty for rid in template_ids}
    negatives = [v for v in raw.values() if v < 0]
    fallback = -max(negatives) if negatives else 0.0
    return {rid: -v if v < 0 else fallback for rid, v in raw.items()}


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

    penalties = _penalties(template_ids, scores, penalty)
    working.objective = prob.Objective(
        sum(penalties[rid] * indicators[rid] for rid in template_ids), direction="min"
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
    threshold: float = 1.0,
    min_flux: float = 0.1,
) -> GapFillResult:
    """Add template reactions so blocked draft reactions can carry flux.

    Finds reactions that cannot carry a flux larger than ``threshold`` in ``model`` under its current bounds,
    then adds the minimum-penalty set of template reactions that lets the (irreversible)
    ones carry flux, and returns the filled model. Like RAVEN, only irreversible blocked
    reactions are forced; reversible ones can carry flux trivially in the split
    formulation, so forcing them is uninformative.

    For the *other* gap-filling flavour (adding reactions to make the model's
    objective feasible), use ``cobra.flux_analysis.gapfill`` after aligning the
    template's metabolite ids to the draft.

    The draft is expected to have exchange reactions for its nutrients (otherwise most
    reactions are trivially blocked).

    ``threshold`` is the flux a reaction must be able to exceed to count as unblocked, in
    the draft and, for a blocked reaction, with the templates added. ``min_flux`` is the
    flux forced through each rescuable reaction while solving. ``allow_net_production``
    relaxes the steady-state balance from ``S @ v == 0`` to ``S @ v >= 0`` (metabolites
    may accumulate) in both checks and in the MILP, which can rescue reactions whose real
    fix is a missing consumption route rather than a genuine connectivity gap.
    """
    if penalty <= 0:
        raise ValueError(f"penalty must be > 0 (a non-positive cost malforms the MILP); got {penalty}.")
    renames: dict[str, str] = {}
    templates = unique_templates(model, _as_models(templates), renames)
    working, template_ids = _merge_templates(model, templates)
    if scores:
        # Scores are keyed by the id the reaction has in the template.
        # A score under the renamed id takes precedence over the one under the original id.
        scores = {rid: scores[rid] if rid in scores else scores[renames.get(rid, rid)]
                  for rid in template_ids if rid in scores or renames.get(rid, rid) in scores}

    draft_ids = [r.id for r in model.reactions]
    unblocked = _carrying_flux(
        model, draft_ids, threshold=threshold, net_production=allow_net_production
    )
    blocked = {rid for rid in draft_ids if rid not in unblocked}
    candidates = [r for r in draft_ids if r in blocked and model.reactions.get_by_id(r).lower_bound >= 0]
    rescuable = _carrying_flux(
        working, candidates, threshold=threshold, net_production=allow_net_production
    )
    target = [r for r in candidates if r in rescuable]

    cannot = sorted(blocked - set(target))
    if not target:
        return GapFillResult([], [], cannot, model.copy())

    for rid in target:
        working.reactions.get_by_id(rid).lower_bound = min_flux
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
