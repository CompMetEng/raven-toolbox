"""LP-based gap-filling: fastGapFill and swiftGapFill.

For each blocked draft reaction, solves a single LP that identifies the
minimum-L1-flux (fastLP) or maximum-template-flux (swiftLP) extension of
the merged draft and template model that activates the blocked reaction. The
union of all template reactions activated across all LP solves is returned.

* **fastLP** minimises the sum of |template fluxes| while forcing the
  blocked reaction to carry >= epsilon flux (FASTCORE, Vlassis 2014).
* **swiftLP** maximises the sum of the fluxes of the irreversible template
  reactions in a single LP solve per blocked reaction (SWIFTCORE, Tefagh & Boyd 2020).
  The maximisation objective is degenerate, so results can differ between runs
  or solvers.

Unlike :func:`~raven_toolbox.gapfilling.fill.connect_blocked_reactions`,
this function solves LPs only and no MILP. The returned template set is a union
over individual LP solutions and can contain more reactions than a minimal set.

Template metabolites are matched to the draft by ``name[compartment]`` (via
:func:`~raven_toolbox.manipulation.transfer.add_reactions_from_model`), so templates with a
different identifier namespace than the model are supported.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import cobra
from cobra.flux_analysis import find_blocked_reactions, flux_variability_analysis
from cobra.util import ProcessPool

from raven_toolbox.gapfilling.templates import unique_templates
from raven_toolbox.manipulation.transfer import add_reactions_from_model


@dataclass
class FastLPResult:
    """Result of an LP-based gap-fill.

    Parameters
    ----------
    added_reactions:
        Template reaction IDs selected by the LP solver (union across all
        per-blocked-reaction LP solves).
    newly_connected:
        Draft reaction IDs that were blocked in the draft and can carry a flux above
        ``epsilon`` when all template reactions are added. The reactions in
        ``added_reactions`` alone are not checked for this flux.
    cannot_connect:
        Blocked draft reaction IDs that remain blocked even with the full
        template database added.
    candidates_per_reaction:
        Per blocked-reaction template reaction candidates (as returned by
        each individual LP solve).
    model:
        Draft model with ``added_reactions`` incorporated.
    """

    added_reactions: list[str] = field(default_factory=list)
    newly_connected: list[str] = field(default_factory=list)
    cannot_connect: list[str] = field(default_factory=list)
    candidates_per_reaction: dict[str, list[str]] = field(default_factory=dict)
    model: cobra.Model = field(default_factory=cobra.Model)


def _as_models(templates: cobra.Model | Iterable[cobra.Model]) -> list[cobra.Model]:
    return [templates] if isinstance(templates, cobra.Model) else list(templates)


def _merge_templates(
    model: cobra.Model, templates: list[cobra.Model]
) -> tuple[cobra.Model, list[str]]:
    """Merge all template reactions (new ones only) into a working copy.

    Returns the merged model and the ids of reactions from templates.
    """
    working = model.copy()
    template_ids: list[str] = []
    for t in templates:
        new = [r.id for r in t.reactions if r.id not in working.reactions]
        if new:
            added = add_reactions_from_model(working, t, new, genes=False, note=None)
            template_ids += [r.id for r in added]
    return working, template_ids


def _solve_fastlp(
    working: cobra.Model,
    blocked_rid: str,
    template_ids: list[str],
    *,
    epsilon: float,
) -> list[str]:
    """Run one FASTCORE LP for a blocked reaction.

    Minimises the L1 norm (sum of |flux|) over template reactions while
    forcing ``blocked_rid`` to carry >= ``epsilon`` flux. For irreversible
    template reactions the L1 norm is the flux value. For reversible
    template reactions, auxiliary variables linearise |v|.

    Returns the list of template reaction ids with |flux| > epsilon/2.
    """
    prob = working.problem
    aux_vars: list = []
    aux_cons: list = []
    obj_terms: list = []

    with working as m:
        blocked_rxn = m.reactions.get_by_id(blocked_rid)
        blocked_rxn.lower_bound = epsilon

        for rid in template_ids:
            rxn = m.reactions.get_by_id(rid)
            if rxn.lower_bound >= 0:
                # Irreversible — L1 = flux (already non-negative)
                obj_terms.append(rxn.flux_expression)
            else:
                # Reversible — |v| = t, with t >= v and t >= -v
                t = prob.Variable(f"_l1_{rid}", lb=0)
                c1 = prob.Constraint(
                    t - rxn.flux_expression, lb=0, name=f"_l1c1_{rid}"
                )
                c2 = prob.Constraint(
                    t + rxn.flux_expression, lb=0, name=f"_l1c2_{rid}"
                )
                aux_vars.append(t)
                aux_cons.extend([c1, c2])
                obj_terms.append(t)

        if aux_vars or aux_cons:
            m.add_cons_vars(aux_vars + aux_cons)

        if obj_terms:
            m.objective = prob.Objective(sum(obj_terms), direction="min")
        sol = m.optimize()

        if sol.status != "optimal":
            return []

        return [
            rid
            for rid in template_ids
            if abs(sol.fluxes.get(rid, 0.0)) > epsilon / 2
        ]


def _solve_swiftlp(
    working: cobra.Model,
    blocked_rid: str,
    template_ids: list[str],
    *,
    epsilon: float,
) -> list[str]:
    """Run one SWIFTCORE LP for a blocked reaction.

    Maximises the sum of template reaction fluxes (for irreversible
    templates) while forcing ``blocked_rid`` to carry >= ``epsilon`` flux.
    The LP is degenerate (many optimal solutions), so results can differ
    between solvers and runs.

    Returns the list of template reaction ids with flux > epsilon/2.
    """
    prob = working.problem

    with working as m:
        m.reactions.get_by_id(blocked_rid).lower_bound = epsilon

        # Maximise sum of (irreversible) template fluxes
        obj_terms = [
            m.reactions.get_by_id(rid).flux_expression
            for rid in template_ids
            if m.reactions.get_by_id(rid).lower_bound >= 0
        ]
        if obj_terms:
            m.objective = prob.Objective(sum(obj_terms), direction="max")
        sol = m.optimize()

        if sol.status != "optimal":
            return []

        return [
            rid
            for rid in template_ids
            if sol.fluxes.get(rid, 0.0) > epsilon / 2
        ]


# --------------------------------------------------------------------------- #
# Worker globals for the parallel path (n_proc > 1): populated once per
# worker process by _init_worker and reused for every blocked reaction it
# solves.
# --------------------------------------------------------------------------- #

_WORKER_MODEL: cobra.Model | None = None
_WORKER_TEMPLATE_IDS: list[str] | None = None
_WORKER_EPSILON: float | None = None
_WORKER_VARIANT: Literal["fast", "swift"] | None = None


def _init_worker(
    working: cobra.Model,
    template_ids: list[str],
    epsilon: float,
    variant: Literal["fast", "swift"],
) -> None:
    """Pool initialiser: store the model copy of this worker (deserialised by
    ``ProcessPool``) and everything else needed to solve one
    blocked reaction's LP."""
    global _WORKER_MODEL, _WORKER_TEMPLATE_IDS, _WORKER_EPSILON, _WORKER_VARIANT
    _WORKER_MODEL = working
    _WORKER_TEMPLATE_IDS = template_ids
    _WORKER_EPSILON = epsilon
    _WORKER_VARIANT = variant


def _solve_worker(blocked_rid: str) -> tuple[str, list[str]]:
    assert _WORKER_MODEL is not None, "_solve_worker called before _init_worker"
    assert _WORKER_TEMPLATE_IDS is not None
    assert _WORKER_EPSILON is not None
    assert _WORKER_VARIANT is not None
    solve_fn = _solve_fastlp if _WORKER_VARIANT == "fast" else _solve_swiftlp
    active = solve_fn(_WORKER_MODEL, blocked_rid, _WORKER_TEMPLATE_IDS, epsilon=_WORKER_EPSILON)
    return blocked_rid, active


def fill_gaps_fast_lp(
    model: cobra.Model,
    templates: cobra.Model | Iterable[cobra.Model],
    *,
    epsilon: float = 1e-4,
    variant: Literal["fast", "swift"] = "fast",
    verbose: bool = True,
    n_proc: int | None = None,
) -> FastLPResult:
    """LP-based gap-filling (fastGapFill / swiftGapFill).

    For each blocked draft reaction, runs a single LP that activates that
    reaction using a minimal (``"fast"``) or maximal (``"swift"``) set of template
    reactions. The union of all activated template reactions is returned.

    Parameters
    ----------
    model:
        Draft model to gap-fill.
    templates:
        Universal reaction database model(s).
    epsilon:
        Minimum flux threshold for a reaction to be considered active.
    variant:
        ``"fast"``: FASTCORE L1-norm LP (minimises the L1 norm of template fluxes).
        ``"swift"``: SWIFTCORE single LP (maximises the sum of the fluxes of the
        irreversible template reactions; results can differ between runs).
    verbose:
        Print progress messages.
    n_proc:
        Worker processes for solving each blocked reaction's LP in parallel.
        Defaults to ``cobra.Configuration().processes``; 1 solves
        serially in this process. Each blocked reaction's LP depends only on
        the merged draft and template model (reverted after each solve via
        ``with working as m:``) and never on the result of another blocked reaction.
        The LPs run in ``cobra.util.ProcessPool``, which cobra's
        ``flux_variability_analysis`` also uses.

    Returns
    -------
    FastLPResult
    """
    templates = unique_templates(model, _as_models(templates))

    # ---- Find blocked reactions in draft ----
    blocked = set(find_blocked_reactions(model, processes=1))
    if not blocked:
        if verbose:
            print("fill_gaps_fast_lp: no blocked reactions found.")
        return FastLPResult(model=model.copy())

    if verbose:
        print(f"fill_gaps_fast_lp: {len(blocked)} blocked reaction(s) found.")

    # ---- Merge draft + templates ----
    working, template_ids = _merge_templates(model, templates)
    if not template_ids:
        if verbose:
            print("fill_gaps_fast_lp: no new template reactions to add.")
        return FastLPResult(cannot_connect=sorted(blocked), model=model.copy())

    # ---- Determine which blocked reactions are rescuable in merged model ----
    blocked_in_model = [r for r in blocked if r in working.reactions]
    if blocked_in_model:
        fva = flux_variability_analysis(
            working, reaction_list=blocked_in_model, fraction_of_optimum=0.0, processes=1
        )
        rescuable = [
            r
            for r in blocked_in_model
            if r in fva.index and fva.at[r, "maximum"] > epsilon
        ]
    else:
        rescuable = []

    cannot_connect = sorted(blocked - set(rescuable))

    if not rescuable:
        if verbose:
            print("fill_gaps_fast_lp: no blocked reactions can be rescued.")
        return FastLPResult(cannot_connect=cannot_connect, model=model.copy())

    if verbose:
        print(
            f"fill_gaps_fast_lp: {len(rescuable)}/{len(blocked)} rescuable; "
            f"running {variant}LP..."
        )

    # ---- Run LP for each rescuable blocked reaction ----
    candidates_per_rxn: dict[str, list[str]] = {}
    all_added: set[str] = set()
    n_skipped = 0

    if n_proc is None:
        n_proc = cobra.Configuration().processes
    n_proc = max(1, int(n_proc))

    if n_proc == 1:
        solve_fn = _solve_fastlp if variant == "fast" else _solve_swiftlp
        results = [
            (blocked_rid, solve_fn(working, blocked_rid, template_ids, epsilon=epsilon))
            for blocked_rid in rescuable
        ]
    else:
        # ProcessPool (cobra.util.process_pool) handles serialising the
        # model to each worker -- including a Windows-specific performance
        # workaround -- so `working` is passed as-is, not pre-pickled.
        chunk = max(1, len(rescuable) // (n_proc * 4))
        with ProcessPool(
            n_proc, initializer=_init_worker,
            initargs=(working, template_ids, epsilon, variant),
        ) as pool:
            results = pool.map(_solve_worker, rescuable, chunksize=chunk)

    for blocked_rid, active in results:
        if active:
            candidates_per_rxn[blocked_rid] = active
            all_added.update(active)
        else:
            n_skipped += 1
            candidates_per_rxn[blocked_rid] = []

    if verbose and n_skipped:
        print(f"fill_gaps_fast_lp: {n_skipped} LP(s) returned no solution.")

    if verbose:
        print(f"fill_gaps_fast_lp: adding {len(all_added)} template reaction(s).")

    # ---- Build output model ----
    filled = model.copy()
    remaining = set(all_added)
    for t in templates:
        ids_from_t = [r.id for r in t.reactions if r.id in remaining]
        if ids_from_t:
            add_reactions_from_model(filled, t, ids_from_t, genes=False, note=None)
            remaining -= set(ids_from_t)

    return FastLPResult(
        added_reactions=sorted(all_added),
        newly_connected=sorted(rescuable),
        cannot_connect=cannot_connect,
        candidates_per_reaction=candidates_per_rxn,
        model=filled,
    )
