"""Task gap-filling for ftINIT.

After ftINIT extracts a context-specific model, some metabolic tasks may no longer be
feasible (the scoring removed reactions a task needs). :func:`fill_tasks` restores
feasibility by adding back the **minimum-cost** set of reactions from the reference
(template) model — cost = ``−score``, so high-scoring reactions are preferred — one
task at a time, only for tasks that are actually infeasible (a cheap LP check gates
the expensive MILP), accumulating additions across tasks.

This is a different MILP from ftINIT's main extraction: it *adds* reactions to satisfy
the task's ranged metabolite bounds (RAVEN's two-column ``b``), rather than selecting
which to keep by expression score. Exchange reactions are not used to fill gaps (task
inputs/outputs come from the task's ``b``), so they are excluded as candidates.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import cobra
from cobra.exceptions import OptimizationError
from optlang.symbolics import Real, add, mul

from raven_toolbox.manipulation.boundary import close_model_in_place
from raven_toolbox.tasks import Task
from raven_toolbox.tasks.check import (
    _metabolite_bounds,
    _set_constraint_bounds,
    apply_task_constraints,
    task_name_maps,
)
from raven_toolbox.utils.gpr import set_gene_rule

_FILL_TIME_LIMIT = 300.0  # RAVEN ftINITFillGapsMILP TimeLimit (per task)
_FILL_SEED = 26           # RAVEN ftINITFillGapsMILP Seed

_DEFAULT_SCORE = -1.0   # RAVEN: missing scores default to -1 (cost 1)
_MAX_SCORE = -0.1       # RAVEN min(score, -0.1): every added reaction costs ≥ 0.1

# A resolve_ties tie-break phase is adopted whether it proves or times out (see
# _resolve_ties_fill), so it is given a fraction of the primary fill's own budget rather
# than the full amount again: a phase that would not converge within the full budget has
# been observed (on genome-scale Human-GEM) to still not converge, only slower; a phase
# that does converge does so far inside either budget (a pure integer count/id-rank
# objective with a 0.4 absolute gap). Not independently tuned — see the ftINIT
# reproducibility study on raven-docs for the experiment that would calibrate this
# properly instead of halving on general principle.
_TIE_BREAK_TIME_FRACTION = 0.5


@dataclass
class TaskFillResult:
    """Result of :func:`fill_tasks`: the gap-filled model and what was added."""

    model: cobra.Model
    added_reactions: list[str]
    failed_tasks: list[str]


def _feasible(model: cobra.Model, task: Task, name_to_id, comp_to_ids) -> bool:
    """Is ``task`` feasible in ``model`` (boundaries closed)? Tested in place, then reverted.

    Avoids copying the (genome-scale) model for each of the task list's feasibility checks
    — the copy dominated gap-fill runtime. ``with model:`` reverts the closed boundaries and
    everything ``apply_task_constraints`` does through cobra's API; the untracked direct
    metabolite mass-balance bound edits are snapshotted and restored (as in check_tasks).
    """
    bounds, missing = _metabolite_bounds(task, name_to_id, comp_to_ids)
    if missing:
        return False
    saved = {mid: (model.constraints[mid].lb, model.constraints[mid].ub) for mid in bounds}
    try:
        with model:
            for rxn in model.boundary:
                rxn.bounds = (0.0, 0.0)
            _, error = apply_task_constraints(model, task, name_to_id, comp_to_ids)
            if error is not None:
                return False
            model.slim_optimize()
            return model.solver.status == "optimal"
    finally:
        for mid, (lb, ub) in saved.items():
            _set_constraint_bounds(model.constraints[mid], lb, ub)


def _add_reference_reactions(
    target: cobra.Model, reference_model: cobra.Model, rxn_ids: Iterable[str]
) -> None:
    """Add ``rxn_ids`` from ``reference_model`` to ``target`` by reconstruction.

    ``cobra.Reaction.copy()`` deep-copies through the reaction↔metabolite graph and hits the
    recursion limit on large (and, on newer Python, even small) models, so the reactions are
    rebuilt from their id/bounds/stoichiometry/GPR instead — matching RAVEN's ``addRxns``
    (which adds reactions from equations, not by copying objects). Metabolites the reactions
    need that are missing from ``target`` are created first.
    """
    have = {m.id for m in target.metabolites}
    new_mets: dict[str, cobra.Metabolite] = {}
    for rid in rxn_ids:
        for met in reference_model.reactions.get_by_id(rid).metabolites:
            if met.id not in have and met.id not in new_mets:
                new_mets[met.id] = cobra.Metabolite(
                    met.id, name=met.name, compartment=met.compartment,
                    formula=met.formula, charge=met.charge,
                )
    if new_mets:
        target.add_metabolites(list(new_mets.values()))
    built = []
    for rid in rxn_ids:
        src = reference_model.reactions.get_by_id(rid)
        new = cobra.Reaction(src.id, name=src.name,
                             lower_bound=src.lower_bound, upper_bound=src.upper_bound)
        built.append((new, src))
    target.add_reactions([new for new, _ in built])
    for new, src in built:
        new.add_metabolites({target.metabolites.get_by_id(m.id): c for m, c in src.metabolites.items()})
        set_gene_rule(new, src.gpr)


def _set_fill_solver(model: cobra.Model, time_limit: float | None, seed: int) -> None:
    """Match RAVEN ``ftINITFillGapsMILP``: single-threaded, fixed seed, tight integrality.

    The seed matters: the MILP is degenerate (many equal-cost fills), and a fixed seed
    makes the chosen reactions reproducible run to run.
    """
    try:  # Gurobi-specific; harmless on other backends
        params = model.solver.problem.Params
        params.Threads = 1
        params.Seed = seed
        params.IntFeasTol = 1e-9
        if time_limit is not None:
            params.TimeLimit = float(time_limit)
    except Exception:  # noqa: BLE001
        if time_limit is not None:
            model.solver.configuration.timeout = int(time_limit)


def _snapshot_solver_params(model: cobra.Model):
    """Save whatever :func:`_set_fill_solver` is about to overwrite.

    ``model.solver.problem.Params`` (Gurobi) and ``model.solver.configuration.timeout``
    (the generic optlang fallback) are raw solver state, not cobra/optlang model
    attributes — ``with model:`` does not know about them and will not revert them on
    exit, unlike everything else :func:`_gap_fill_task` touches on a shared reference
    model. Pair with :func:`_restore_solver_params`.
    """
    try:  # Gurobi-specific; harmless on other backends
        params = model.solver.problem.Params
        return "gurobi", (params.Threads, params.Seed, params.IntFeasTol, params.TimeLimit)
    except Exception:  # noqa: BLE001
        return "generic", model.solver.configuration.timeout


def _restore_solver_params(model: cobra.Model, saved) -> None:
    """Undo :func:`_set_fill_solver`, from a snapshot taken by :func:`_snapshot_solver_params`."""
    kind, value = saved
    if kind == "gurobi":
        try:  # noqa: BLE001 - if this now fails, the params were never changed either
            params = model.solver.problem.Params
            params.Threads, params.Seed, params.IntFeasTol, params.TimeLimit = value
        except Exception:  # noqa: BLE001
            pass
    else:
        model.solver.configuration.timeout = value


def _resolve_ties_fill(work, prob, candidates, cost_expr, time_limit) -> list[str] | None:
    """Pin the degenerate min-cost gap-fill to a single, reproducible set (on ``work``).

    Like the extraction MILP, the fill has many equal-cost solutions and the solver returns
    an arbitrary one (seed/version dependent). Hold the cost at its optimum, then lexicographically
    minimise the number of added reactions and then their summed id rank — both integer
    objectives, provable with a cheap absolute gap below 1. Returns the chosen reaction ids,
    or ``None`` if a phase did not converge (the caller then keeps the arbitrary min-cost fill).
    """
    yvars = {cid: work.variables[f"_fill_{cid}"] for cid in candidates}
    primary_cost = work.objective.value or 0.0
    tol = max(abs(primary_cost) * 1e-7, 1e-7)
    work.add_cons_vars([prob.Constraint(cost_expr, ub=primary_cost + tol, name="_fill_cost_floor")])
    count = add([mul([Real(1.0), y]) for y in yvars.values()])
    unproven: list[str] = []

    def _phase(objective, label: str) -> bool:
        work.objective = objective
        if time_limit is not None:
            # A fraction of the primary fill's own budget, not the full amount again —
            # see _TIE_BREAK_TIME_FRACTION. Overrides the TimeLimit _set_fill_solver
            # already put on `work.solver.problem.Params`, so this only ever shortens
            # the budget for this phase; _gap_fill_task restores the original value
            # afterward regardless of what a phase sets here.
            phase_limit = max(1.0, time_limit * _TIE_BREAK_TIME_FRACTION)
            try:
                work.solver.problem.Params.TimeLimit = phase_limit
            except Exception:  # noqa: BLE001 - harmless on other backends
                work.solver.configuration.timeout = int(phase_limit)
        try:  # integer objective → an absolute gap < 1 proves the optimum cheaply.
            work.solver.problem.Params.MIPGap = 0.0
            work.solver.problem.Params.MIPGapAbs = 0.4
        except Exception:  # noqa: BLE001 - harmless on other backends
            pass
        work.slim_optimize()
        # Mirrors _resolve_ties: a phase ending at the time limit still holds an
        # incumbent, and adopting it means the tie-break is itself an arbitrary
        # within-gap pick. Still adopted (it measurably reduces the spread) but
        # recorded, so resolve_ties=True cannot silently mean "tried, failed".
        if work.solver.status == "time_limit":
            unproven.append(label)
        if work.solver.status not in ("optimal", "feasible", "suboptimal", "time_limit"):
            return False
        try:
            return work.solver.problem.SolCount > 0
        except Exception:  # noqa: BLE001 - non-Gurobi backend: status is authoritative
            return True

    # fewest added reactions (parsimony) ...
    if not _phase(prob.Objective(count, direction="min"), "phase2a-parsimony"):
        return None
    # A timed-out phase can report a non-finite objective. ``inf + 0.5`` is still ``inf``,
    # which would make the cap below vacuous and silently disable the parsimony pin (the
    # same failure mode fixed for the main extraction's _resolve_ties), so fall back to
    # the incumbent's own achieved count instead.
    kmin = work.objective.value
    if kmin is None or not math.isfinite(kmin):
        kmin = float(sum(1 for y in yvars.values() if (y.primal or 0.0) > 0.5))
        unproven.append("phase2a-objective-not-finite")
    # ... then, among the sparsest, the unique lowest-id set.
    work.add_cons_vars([prob.Constraint(count, ub=kmin + 0.5, name="_fill_count_cap")])
    ranks = {cid: i for i, cid in enumerate(sorted(candidates))}
    idsum = add([mul([Real(float(1 + ranks[cid])), yvars[cid]]) for cid in candidates])
    ok = _phase(prob.Objective(idsum, direction="min"), "phase2b-idrank")
    if ok and unproven:
        warnings.warn(
            f"fill_tasks tie resolution did not converge ({', '.join(unproven)}): the "
            "selection among equal-cost fills is itself an unproven incumbent, so "
            "resolve_ties=True has reduced but not removed the run-to-run spread. "
            "Raise time_limit for a proven tie-break.",
            stacklevel=3,
        )
    return [cid for cid in candidates if (yvars[cid].primal or 0.0) > 0.5] if ok else None


def _gap_fill_task(
    reference_model: cobra.Model, present_ids: set[str], task: Task,
    costs: dict[str, float], *, time_limit: float | None, seed: int,
    resolve_ties: bool = False,
) -> list[str]:
    """Min-cost reference reactions that make ``task`` feasible (RAVEN ``ftINITFillGaps``).

    The MILP runs directly on ``reference_model``, closed and holding every candidate
    reaction already, rather than copying candidates into the target model one at a
    time (mirrors RAVEN's ``fullModel = tRefModel``) — copying the (thousands of)
    candidate reactions into a genome-scale model per task dominates runtime and can
    overflow the recursion limit, so a per-task copy risks never returning an
    incumbent within the time limit.

    Everything done to ``reference_model`` here is reverted before returning, same
    as if it were never touched: closing boundaries, the added binary variables and
    constraints, and the objective are all cobra/optlang context-tracked and undone
    by ``with reference_model:`` on the way out. Two things are not tracked and are
    snapshotted/restored by hand instead — the metabolite constraint bounds
    ``apply_task_constraints`` edits directly (same as :func:`_feasible`), and the
    raw Gurobi parameters :func:`_set_fill_solver` sets (``model.solver.problem.Params``
    is not a cobra/optlang attribute cobra's context system knows about). This matters
    because ``reference_model`` is ``prep.ref_model`` — shared and reused across every
    sample built from the same :class:`~raven_toolbox.init.PrepData`, so a change that
    leaked out of one task's gap-fill (single-threaded, 300 s time limit) would still
    be sitting there for the next sample's extraction.

    A prior version copied ``reference_model`` per call instead of mutating it in
    place, since that copy is what RAVEN's ``fullModel = tRefModel`` does too and it
    is cheap there — MATLAB arrays copy in a memcpy. ``cobra.Model.copy()`` on a
    Gurobi-backed model is not that: it deep-copies through pickle's
    ``__getstate__``/``__setstate__``, which rebuilds every variable and constraint
    from its sympy expression one at a time, and on a genome-scale reference model
    that rebuild dominated gap-fill runtime far more than the MILP solve itself (the
    same lesson :func:`_feasible` already applied to the per-task feasibility check,
    just not yet extended to this, the more expensive of the two copies it replaced).

    Every reference reaction not already ``present`` in the target model is gated by a binary
    (off ⇒ no flux) and its cost minimised subject to the task's ranged metabolite bounds.
    Returns the ids of the reactions the MILP turns on.
    """
    saved_solver_params = _snapshot_solver_params(reference_model)
    try:
        with reference_model:
            close_model_in_place(reference_model)  # task I/O via the task's b; holds all candidates
            candidates = [r.id for r in reference_model.reactions
                          if r.id not in present_ids and not r.boundary]
            if not candidates:  # nothing to add → task cannot be made feasible
                raise OptimizationError(f"gap-filling found no candidates for task {task.id!r}.")
            name_to_id, comp_to_ids = task_name_maps(reference_model)
            # apply_task_constraints edits these constraint bounds directly, bypassing
            # cobra's context tracking (same as _feasible) — snapshot before, restore after.
            met_bounds, _ = _metabolite_bounds(task, name_to_id, comp_to_ids)
            saved_met_bounds = {
                mid: (reference_model.constraints[mid].lb, reference_model.constraints[mid].ub)
                for mid in met_bounds
            }
            try:
                _, error = apply_task_constraints(reference_model, task, name_to_id, comp_to_ids)
                if error is not None:
                    raise OptimizationError(
                        f"task {task.id!r} could not be applied to the reference: {error}"
                    )

                work = reference_model
                prob = work.problem
                extras = []
                objective_terms = []
                for cid in candidates:
                    rxn = work.reactions.get_by_id(cid)
                    y = prob.Variable(f"_fill_{cid}", type="binary")
                    # off ⇒ no flux; on ⇒ the reaction's own bounds apply.
                    extras += [
                        y,
                        prob.Constraint(rxn.flux_expression - rxn.upper_bound * y, ub=0.0, name=f"_fillub_{cid}"),
                        prob.Constraint(rxn.flux_expression - rxn.lower_bound * y, lb=0.0, name=f"_filllb_{cid}"),
                    ]
                    objective_terms.append(mul([Real(costs[cid]), y]))
                work.add_cons_vars(extras)
                # add() over a flat list, not Python sum() — the latter is O(n²) in sympy and with
                # thousands of candidates dominates gap-fill runtime (see ftINIT/tINIT, same fix).
                cost_expr = add(objective_terms)
                work.objective = prob.Objective(cost_expr, direction="min")
                _set_fill_solver(work, time_limit, seed)
                work.slim_optimize()
                # Accept a near-optimal incumbent (time_limit); only a truly infeasible fill (no
                # incumbent) means the task cannot be satisfied from the reference.
                if work.solver.status not in ("optimal", "feasible", "suboptimal", "time_limit") or \
                        work.variables[f"_fill_{candidates[0]}"].primal is None:
                    raise OptimizationError(f"gap-filling found no way to make task {task.id!r} feasible.")

                chosen = [cid for cid in candidates
                          if (work.variables[f"_fill_{cid}"].primal or 0.0) > 0.5]
                # best-effort: pin the degenerate min-cost fill to the fewest, lowest-id
                # reactions so the added set does not depend on the solver seed/version.
                if resolve_ties:
                    canon = _resolve_ties_fill(work, prob, candidates, cost_expr, time_limit)
                    if canon is not None:
                        chosen = canon
                return chosen
            finally:
                for mid, (lb, ub) in saved_met_bounds.items():
                    _set_constraint_bounds(reference_model.constraints[mid], lb, ub)
    finally:
        _restore_solver_params(reference_model, saved_solver_params)


def fill_tasks(
    model: cobra.Model,
    reference_model: cobra.Model,
    tasks: Iterable[Task],
    *,
    rxn_scores: Mapping[str, float] | None = None,
    time_limit: float | None = _FILL_TIME_LIMIT,
    seed: int = _FILL_SEED,
    resolve_ties: bool = False,
    mutate_in_place: bool = False,
    verbose: bool = False,
) -> TaskFillResult:
    """Add minimum-cost reference reactions so every task is feasible in ``model``.

    Port of RAVEN ``fitTasks`` with ``gapFillMode`` ``'preMerged'`` (ftINIT's mode): task by
    task, if a task is infeasible in the (growing) model, :func:`_gap_fill_task` finds the
    minimum-cost set of reference
    reactions that restores it and they are added, carrying forward so later tasks see the
    earlier additions. ``reference_model`` supplies the candidates (its reactions not yet in
    the model, excluding exchange/boundary reactions); ``rxn_scores`` (original reaction id →
    score) sets each candidate's cost as ``−min(score, −0.1)`` (missing → cost 1).
    ``should_fail`` tasks are ignored. Each gap-fill MILP is single-threaded with a fixed
    ``seed`` and bounded by ``time_limit`` (RAVEN's 300 s). ``resolve_ties`` (opt-in) pins the
    degenerate min-cost fill to the fewest, lowest-id reactions so the added set does not
    depend on the solver seed/version — see :func:`_resolve_ties_fill`. At genome scale its
    own phases can themselves exhaust ``time_limit``; when that happens the (still-adopted)
    incumbent is an unproven tie-break and a warning is raised, same as the main extraction.

    Boundary reactions are closed while testing/solving each task, so task inputs and outputs
    come solely from the task's ranged metabolite bounds (RAVEN gap-fills the exchange-free
    model). The returned model keeps its boundary reactions. Tasks that could not be filled
    are returned in ``failed_tasks`` **and** raised as a warning — a non-empty list means the
    context model cannot perform those tasks, which callers should not ignore silently.

    ``mutate_in_place`` (default ``False``) skips the initial ``model.copy()`` and gap-fills
    ``model`` directly, returning it as ``TaskFillResult.model``. Copying a genome-scale,
    Gurobi-backed model is itself expensive (see :func:`_gap_fill_task`'s docstring for the
    same lesson applied to ``reference_model``), so a caller that already holds a disposable
    copy it will not use again — :func:`raven_toolbox.init.ftinit`, whose own ``out`` is
    exactly that — should pass ``mutate_in_place=True`` to avoid paying for a second one.
    Leave it ``False`` (the default) whenever the caller still needs ``model`` afterward.

    ``verbose`` prints one line per task (added count and running total, matching RAVEN
    ``fitTasks``'s per-task report), silent by default.
    """
    scores = dict(rxn_scores or {})
    tasks = list(tasks)
    n_tasks = len(tasks)

    out = model if mutate_in_place else model.copy()
    added: list[str] = []
    failed: list[str] = []

    # Cost is per reference reaction and independent of which tasks have already run, so
    # it is computed once rather than re-derived (a fresh sweep of reference_model, with
    # the score lookup and arithmetic repeated) for every infeasible task. It is safe to
    # hand _gap_fill_task the full dict, unfiltered by `present`: that function only ever
    # looks up a cost for a reaction it has already excluded from `present` itself.
    all_costs = {r.id: -min(scores.get(r.id, _DEFAULT_SCORE), _MAX_SCORE)
                 for r in reference_model.reactions if not r.boundary}

    present = {r.id for r in out.reactions}
    name_to_id, comp_to_ids = task_name_maps(out)
    n_mets = len(out.metabolites)

    for i, task in enumerate(tasks):
        if task.should_fail:
            continue
        if len(out.metabolites) != n_mets:
            # Only a successful gap-fill changes out's permanent metabolite set (task
            # application is reverted by _feasible's own `with model:`), so this is a
            # cheap, self-verifying way to skip re-sweeping every metabolite in `out` on
            # the (common) iterations where nothing changed since the last one.
            name_to_id, comp_to_ids = task_name_maps(out)
            n_mets = len(out.metabolites)
        if _feasible(out, task, name_to_id, comp_to_ids):
            if verbose:
                print(f"[{i + 1}/{n_tasks}] {task.id}: already feasible, "
                      f"0 reaction(s) added, {len(added)} total", flush=True)
            continue
        try:
            chosen = _gap_fill_task(reference_model, present, task, all_costs,
                                    time_limit=time_limit, seed=seed,
                                    resolve_ties=resolve_ties)
        except OptimizationError:
            failed.append(task.id)
            if verbose:
                print(f"[{i + 1}/{n_tasks}] {task.id}: FAILED to gap-fill, "
                      f"{len(added)} total", flush=True)
            continue
        if chosen:
            _add_reference_reactions(out, reference_model, chosen)
            added.extend(chosen)
            present.update(chosen)
        if verbose:
            print(f"[{i + 1}/{n_tasks}] {task.id}: added {len(chosen)} reaction(s), "
                  f"{len(added)} total", flush=True)
    if failed:
        warnings.warn(
            f"fill_tasks: {len(failed)} task(s) could not be gap-filled and remain "
            f"infeasible: {sorted(set(failed))}",
            stacklevel=2,
        )
    return TaskFillResult(out, added, failed)
