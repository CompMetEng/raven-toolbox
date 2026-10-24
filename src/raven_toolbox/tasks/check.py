"""Check whether a model performs a set of metabolic tasks.

For each task the model is constrained by the task's allowed inputs/outputs (and any
extra reactions / bound changes), then tested for feasibility: a task *passes* if a
steady-state flux exists, unless it is marked ``should_fail`` (then it passes iff
infeasible).

Inputs/outputs are encoded as ranges on the per-metabolite mass-balance constraint
(``model.constraints[met.id]``): an input allows net consumption (``Sv ∈ [-UB, -LB]``)
and an output allows / requires net production (``Sv ≤ UB``, and ``≥ LB`` if
``LB > 0``). By default, existing boundary reactions are left exactly as they are: a
task's inputs/outputs *add to* whatever the model's own open exchanges already allow,
rather than replacing them. That is what RAVEN's ``checkTasks`` does on a model without
boundary metabolites, such as one read from YAML. Pass ``close_boundaries=True`` for the
stricter reading, where a task's declared inputs/outputs are the complete boundary of the
system for that check; it corresponds to RAVEN's ``checkTasks(closeModel(model), ...)``,
which is how ``prepINITModel`` runs it.
"""
from __future__ import annotations

import concurrent.futures as cf
import pickle
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import cobra
from cobra.exceptions import OptimizationError
from cobra.flux_analysis import pfba
from optlang.symbolics import Zero, add

from raven_toolbox.manipulation.add import add_reactions_from_equations
from raven_toolbox.manipulation.boundary import close_model_in_place
from raven_toolbox.tasks.tasklist import Task, parse_task_list

_ALLMETS = "ALLMETS"
_ALLMETSIN = "ALLMETSIN"


@dataclass
class TaskResult:
    """Result of one task: ``passed`` is the verdict (accounts for ``should_fail``)."""

    id: str
    description: str
    passed: bool
    feasible: bool
    error: str | None = None


def _set_constraint_bounds(constraint, lb: float, ub: float) -> None:
    """Set an optlang constraint's bounds without a transient lb > ub."""
    if lb > constraint.ub:
        constraint.ub = ub
        constraint.lb = lb
    else:
        constraint.lb = lb
        constraint.ub = ub


def _classify(token: str) -> tuple[str, str]:
    """Return ``("all", "")``, ``("comp", COMP)``, or ``("met", token_upper)``.

    The arg is empty only for ``"all"``, which never reads it.
    """
    upper = token.upper()
    if upper == _ALLMETS:
        return "all", ""
    if upper.startswith(_ALLMETSIN + "[") and upper.endswith("]"):
        return "comp", upper[len(_ALLMETSIN) + 1: -1]
    return "met", upper  # incl. malformed ALLMETSIN[... → treated as a (missing) metabolite


def _metabolite_bounds(
    task: Task, name_to_ids: dict[str, list[str]], comp_to_ids: dict[str, list[str]]
) -> tuple[dict[str, list[float]], list[str]]:
    """Compute ``{met_id: [lb, ub]}`` from a task's inputs/outputs (RAVEN ``b``).

    Bulk tokens (ALLMETS / ALLMETSIN) are applied before specific metabolites, as
    RAVEN does. Returns the bounds and a list of unresolved tokens (→ task error).

    Raises
    ------
    ValueError
        If a metabolite is listed more than once among the inputs or among the
        outputs, or if it is both an input and an output with a nonzero lower bound
        on both sides. The messages are those of RAVEN's ``checkTasks``.
    """
    bounds: dict[str, list[float]] = {}
    missing: list[str] = []
    label = f'"[{task.id}] {task.description}"'

    def touch(mid: str) -> list[float]:
        return bounds.setdefault(mid, [0.0, 0.0])

    for entries, is_input in ((task.inputs, True), (task.outputs, False)):
        bulk = [(t, lb, ub) for (t, lb, ub) in entries if _classify(t)[0] != "met"]
        specific = [(t, lb, ub) for (t, lb, ub) in entries if _classify(t)[0] == "met"]
        # As in RAVEN, only the tokens that match a model metabolite are compared.
        listed = [_classify(t)[1] for t, _, _ in specific]
        listed = [arg for arg in listed if arg in name_to_ids]
        if len(listed) != len(set(listed)):
            side = "input(s)" if is_input else "output(s)"
            raise ValueError(
                f"The constraints on some {side} in {label} are defined more than one time"
            )
        for token, lb, ub in bulk + specific:
            kind, arg = _classify(token)
            if kind == "all":
                ids = [mid for group in comp_to_ids.values() for mid in group]
            elif kind == "comp":
                ids = comp_to_ids.get(arg, [])
            else:
                ids = name_to_ids.get(arg, [])
                if not ids:
                    missing.append(token)
                    continue
                if len(ids) > 1:
                    warnings.warn(
                        f"task {task.id!r}: {token!r} matches {len(ids)} model metabolites "
                        f"({', '.join(ids)}); all of them are constrained (RAVEN constrains "
                        "only the first).",
                        stacklevel=2,
                    )
            for mid in ids:
                b = touch(mid)
                if is_input:
                    b[0] = -ub  # allow net consumption up to UB (RAVEN b1 = -UBin)
                    if kind == "met":
                        b[1] = -lb
                else:
                    if kind == "met" and lb > 0 and b[1] < 0:
                        raise ValueError(
                            f"The IN LB and OUT LB in {label} cannot be nonzero for the "
                            "same metabolite"
                        )
                    b[1] = ub  # allow net production up to UB
                    if kind == "met" and lb > 0:
                        b[0] = lb  # require at least LB produced
    return bounds, missing


def task_name_maps(model: cobra.Model) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build ``name[comp]→[ids]`` and ``comp→[ids]`` lookups for a model's metabolites.

    ``name[comp]`` maps to a *list* because a model can carry several metabolites with
    the same name and compartment. A task referencing it constrains all of them. RAVEN's
    ``checkTasks`` constrains only the first match; both sides warn when a task hits
    duplicates, so the difference is visible instead of silent.
    """
    name_to_ids: dict[str, list[str]] = {}
    comp_to_ids: dict[str, list[str]] = {}
    for m in model.metabolites:
        name_to_ids.setdefault(f"{m.name}[{m.compartment}]".upper(), []).append(m.id)
        comp_to_ids.setdefault((m.compartment or "").upper(), []).append(m.id)
    return name_to_ids, comp_to_ids


def apply_task_constraints(
    model: cobra.Model, task: Task, name_to_id, comp_to_ids
) -> tuple[set[str], str | None]:
    """Apply a task's inputs/outputs/equations/bound-changes to ``model`` in place.

    Sets a feasibility (zero) objective. Returns ``(task_metabolite_ids, error)``;
    ``task_metabolite_ids`` are the model metabolites the task references (RAVEN's
    ``essentialMetsForTasks``). On error the model may be partially modified.
    """
    bounds, missing = _metabolite_bounds(task, name_to_id, comp_to_ids)
    if missing:
        return set(), f"unknown metabolite(s): {sorted(set(missing))}"
    task_mets = {mid for mid in bounds}
    for mid, (lb, ub) in bounds.items():
        if (lb, ub) != (0.0, 0.0):
            _set_constraint_bounds(model.constraints[mid], lb, ub)

    if task.equations:
        existing = {m.id for m in model.metabolites}
        specs = [
            {"id": f"TASK_TMP_{i}", "equation": equ, "bounds": (lb, ub)}
            for i, (equ, lb, ub) in enumerate(task.equations)
        ]
        add_reactions_from_equations(model, specs, mets_by="name", allow_new_mets=True)
        for i in range(len(specs)):
            tmp = model.reactions.get_by_id(f"TASK_TMP_{i}")
            task_mets |= {m.id for m in tmp.metabolites if m.id in existing}

    for rxn_id, new_lb, new_ub in task.changed:
        if rxn_id not in model.reactions:
            return set(), f"CHANGED RXN not in model: {rxn_id!r}"
        rxn = model.reactions.get_by_id(rxn_id)
        # An empty bound in the task list means "leave this bound as it is".
        rxn.bounds = (
            rxn.lower_bound if new_lb is None else new_lb,
            rxn.upper_bound if new_ub is None else new_ub,
        )

    model.objective = model.problem.Objective(Zero, direction="max")  # feasibility only
    return task_mets, None


def _build_task_model(
    base: cobra.Model, task: Task, name_to_id, comp_to_ids
) -> tuple[cobra.Model | None, set[str], str | None]:
    """Copy ``base`` and apply a task's constraints (``model``/``error`` exclusive)."""
    model = base.copy()
    task_mets, error = apply_task_constraints(model, task, name_to_id, comp_to_ids)
    return (None if error else model), task_mets, error


def _run_task(base: cobra.Model, task: Task, name_to_id, comp_to_ids) -> TaskResult:
    """Test one task by applying its constraints to ``base`` in place, then reverting.

    Avoids copying the (genome-scale) model per task — the copy dominates ``check_tasks``
    runtime. ``with base:`` reverts everything ``apply_task_constraints`` does through
    cobra's API (temp reactions/metabolites for equations, reaction bounds, objective);
    the one untracked change — direct metabolite mass-balance (``model.constraints[mid]``)
    bound edits — is snapshotted and restored explicitly. Net result is identical to the
    copy-based version but reuses a single model across all tasks.
    """
    bounds, missing = _metabolite_bounds(task, name_to_id, comp_to_ids)
    if missing:
        return TaskResult(task.id, task.description, False, False,
                          f"unknown metabolite(s): {sorted(set(missing))}")
    saved = {mid: (base.constraints[mid].lb, base.constraints[mid].ub) for mid in bounds}
    try:
        with base:  # reverts temp reactions/mets, reaction bounds, objective on exit
            _, error = apply_task_constraints(base, task, name_to_id, comp_to_ids)
            if error is not None:
                return TaskResult(task.id, task.description, False, False, error)
            base.slim_optimize()
            feasible = base.solver.status == "optimal"
    finally:  # restore the untracked metabolite-constraint bound edits
        for mid, (lb, ub) in saved.items():
            _set_constraint_bounds(base.constraints[mid], lb, ub)
    return TaskResult(task.id, task.description, feasible != task.should_fail, feasible)


def check_tasks(
    model: cobra.Model,
    tasks: str | Iterable[Task],
    *,
    close_boundaries: bool = False,
) -> list[TaskResult]:
    """Run a task list against ``model`` and return a :class:`TaskResult` per task.

    ``tasks`` is a parsed list of :class:`Task` or a path to a task-list file. By
    default (``close_boundaries=False``), existing exchange/sink/demand reactions are
    left open, so a task's inputs/outputs add to whatever the model's own boundary
    already allows, as RAVEN's ``checkTasks`` does on a model without boundary
    metabolites. Pass ``close_boundaries=True`` for the stricter reading, where
    inputs/outputs are defined purely by the tasks, as with RAVEN's
    ``checkTasks(closeModel(model), ...)``.
    """
    tasks = _as_tasks(tasks)
    base, name_to_id, comp_to_ids = _prepare_base(model, close_boundaries)
    return [_run_task(base, task, name_to_id, comp_to_ids) for task in tasks]


def _as_tasks(tasks: str | Iterable[Task]) -> list[Task]:
    if isinstance(tasks, (str, bytes)) or hasattr(tasks, "__fspath__"):
        return parse_task_list(cast("str | Path", tasks))  # guard above ⇒ path-like
    return list(tasks)


def _prepare_base(model: cobra.Model, close_boundaries: bool):
    base = model.copy()
    if close_boundaries:
        # Opt-in only: RAVEN's own checkTasks does not close boundary reactions before
        # applying a task's constraints (confirmed directly — a task's inputs/outputs
        # add to whatever the model's own exchanges already allow, not replace them),
        # so close_model_in_place is skipped by default to match. Kept as an option for
        # callers that want the stricter, task-is-the-complete-boundary reading.
        close_model_in_place(base)
    name_to_id, comp_to_ids = task_name_maps(base)
    return base, name_to_id, comp_to_ids


def _set_deterministic_solver(model: cobra.Model) -> None:
    """Configure the solver to match RAVEN's ``solveLP`` and be deterministic.

    RAVEN's ``getEssentialRxns`` runs ``solveLP`` (Gurobi via ``optimizeProb``) at
    ``FeasibilityTol/OptimalityTol = 1e-9`` and single-threaded. Multi-threaded Gurobi
    picks among the degenerate min-flux optima non-deterministically, so the candidate set
    (and the discovered essential-reaction count) varies run to run; ``Threads=1`` plus a
    fixed ``Seed`` makes it reproducible. Gurobi-specific; a no-op on other backends.
    """
    try:
        model.solver.problem.Params.Threads = 1
        model.solver.problem.Params.Seed = 1234
        model.solver.problem.Params.FeasibilityTol = 1e-9
        model.solver.problem.Params.OptimalityTol = 1e-9
    except Exception:  # noqa: BLE001
        pass


@dataclass
class EssentialReactionsResult:
    """Reactions a model *must* use to perform a task list (RAVEN ``essentialRxns``).

    ``reactions`` maps reaction id → forced flux direction (``+1`` forward, ``-1``
    reverse): the reaction must carry flux of that sign in every feasible solution of
    at least one task. ``per_task`` is the same, split by task id. ``task_metabolites``
    are the model metabolites the tasks reference (RAVEN ``essentialMetsForTasks``,
    protected from removal). ``failed_tasks`` are the ids of tasks that were infeasible or
    malformed and thus skipped (RAVEN drops these from the task list); ``failed_indices`` are
    their positions in the task list, which stay distinct when several tasks share an id.
    """

    reactions: dict[str, int]
    per_task: dict[str, dict[str, int]]
    task_metabolites: set[str]
    failed_tasks: list[str]
    failed_indices: list[int] = field(default_factory=list)


def _task_essential_reactions(
    task_model: cobra.Model, original_ids: set[str]
) -> dict[str, int]:
    """Reactions whose removal makes the task infeasible — faithful port of ``getEssentialRxns``.

    Mirrors RAVEN ``getEssentialRxns.m`` step for step:

    1. **Min-flux solve** (``solveLP(model,1)`` = pFBA, minimise ``Σ|flux|``); candidates are
       the reactions carrying ``|flux| > 1e-12``.
    2. **Shrink loop** (``getEssentialRxns.m:38-55``): minimise, then maximise, the *sum of
       the candidate fluxes*, each time keeping only candidates still carrying
       ``|flux| > 1e-8``; repeat within a phase until it stops shrinking, then switch
       min→max, then stop. This yields a candidate set that does not depend on which
       degenerate min-flux vertex the solver happened to pick.
    3. **Exact test** (``:57-63``): constrain each remaining candidate to 0 and re-solve; if
       the task is then infeasible the reaction is essential. Direction is the sign it
       carries in the min-flux solution (RAVEN's ``essentialFluxes``).

    Returns ``{reaction id: +1 forward | -1 reverse}``.
    """
    fluxes = pfba(task_model).fluxes
    candidates = [rid for rid in original_ids if abs(fluxes.get(rid, 0.0)) > 1e-12]

    n_to_check = len(candidates)
    minimize = True
    while candidates:
        expr = add([task_model.reactions.get_by_id(rid).flux_expression for rid in candidates])
        task_model.objective = task_model.problem.Objective(
            expr, direction="min" if minimize else "max"
        )
        sol = task_model.optimize()
        candidates = [rid for rid in candidates if abs(sol.fluxes[rid]) > 1e-8]
        if len(candidates) >= n_to_check:   # no reduction this pass
            if minimize:
                minimize = False            # switch to the maximise phase
            else:
                break                       # maximise also stable → done
        else:
            n_to_check = len(candidates)

    # Pure feasibility for the constrain-to-0 test (RAVEN solveLP with c = 0).
    task_model.objective = task_model.problem.Objective(Zero, direction="max")
    essential: dict[str, int] = {}
    for rid in candidates:
        rxn = task_model.reactions.get_by_id(rid)
        saved = rxn.bounds
        rxn.bounds = (0.0, 0.0)
        task_model.slim_optimize()
        if task_model.solver.status != "optimal":   # infeasible ⇒ essential
            essential[rid] = 1 if fluxes.get(rid, 0.0) >= 0 else -1
        rxn.bounds = saved
    return essential


def _compute_one_task(
    base: cobra.Model, task: Task, name_to_id, comp_to_ids, original_ids: set[str]
) -> tuple[dict[str, int] | None, set[str], str | None]:
    """Essential reactions for one task, or ``(None, set(), error)`` if it could not run.

    Pulled out of :func:`find_task_essential_reactions` so it can run identically from
    the sequential loop or from a worker process (:func:`_essential_worker`).
    """
    task_model, task_mets, error = _build_task_model(base, task, name_to_id, comp_to_ids)
    if error is not None:
        return None, set(), error
    _set_deterministic_solver(task_model)  # match RAVEN solveLP + reproducibility
    try:
        return _task_essential_reactions(task_model, original_ids), task_mets, None
    except OptimizationError:
        return None, set(), "optimization error"


# Set once per worker process by _init_worker; module-level so ProcessPoolExecutor's
# workers (which import this module fresh rather than inheriting parent state on
# 'spawn') can reach it without re-pickling `base` on every task.
_WORKER: dict = {}


def _init_worker(base: cobra.Model, name_to_id, comp_to_ids, original_ids: set[str]) -> None:
    _WORKER["base"] = base
    _WORKER["name_to_id"] = name_to_id
    _WORKER["comp_to_ids"] = comp_to_ids
    _WORKER["original_ids"] = original_ids


def _essential_worker(i: int, task: Task) -> tuple[int, dict[str, int] | None, set[str], str | None]:
    essential, task_mets, error = _compute_one_task(
        _WORKER["base"], task, _WORKER["name_to_id"], _WORKER["comp_to_ids"], _WORKER["original_ids"]
    )
    return i, essential, task_mets, error


def find_task_essential_reactions(
    model: cobra.Model,
    tasks: str | Iterable[Task],
    *,
    close_boundaries: bool = False,
    cache_path: str | Path | None = None,
    verbose: bool = False,
    processes: int = 1,
    checkpoint_every: int = 10,
) -> EssentialReactionsResult:
    """Find the reactions a model must use to satisfy a task list.

    A faithful port of RAVEN's ``checkTasks(getEssential=true)`` → ``getEssentialRxns``:
    each task is constrained as in :func:`check_tasks`, then :func:`_task_essential_reactions`
    finds the reactions it must use (min-flux solve → shrink loop → exact constrain-to-0
    test). This is the ``prepINITModel`` step that feeds (ft)INIT: essential reactions are
    kept regardless of expression score and made irreversible in their forced direction.
    When a reaction is essential in several tasks with conflicting directions, the majority
    wins (ties → forward), matching RAVEN's ``pos < neg``. ``close_boundaries`` defaults to
    ``False``: a task's inputs/outputs add to whatever the model's own open exchanges
    already allow, rather than replacing them, as in :func:`check_tasks`. RAVEN's
    ``prepINITModel`` calls ``closeModel`` first, which corresponds to
    ``close_boundaries=True``.

    The solver is configured per task to match RAVEN's ``solveLP`` (``FeasibilityTol =
    1e-9``, single-threaded with a fixed seed) so the degenerate min-flux vertex — and thus
    the discovered essential set — is reproducible. ``Threads=1`` only pins *that one solve*;
    it says nothing about the tasks themselves, which are independent of each other.
    ``processes`` (default 1, sequential) parallelises *across* tasks with a
    ``ProcessPoolExecutor`` — each worker still solves single-threaded, so results are
    identical to the sequential run, just computed concurrently. Pick it against available
    cores and memory: each worker holds its own copy of ``model``, sent once at pool
    startup, not per task.

    On a genome-scale model this is slow (a min-flux solve plus a feasibility LP per
    candidate, per task). Pass ``cache_path`` to make it **resumable**: the accumulated
    results are written there (atomically) every ``checkpoint_every`` tasks (across both
    sequential and parallel execution) and once more after the last one, and a re-run skips
    tasks already cached — so it survives interruptions across sessions. Re-serializing the
    whole accumulated result after every single task would make total checkpoint I/O grow
    quadratically in the task count; ``checkpoint_every`` trades a little resumability
    granularity (at most that many completed tasks are redone after an interruption) for
    checkpoint I/O that scales with ``n_tasks / checkpoint_every`` instead. ``verbose``
    prints one line per task as it completes (or is skipped because it is already cached),
    silent by default. In parallel, lines print in completion order, not task order.
    """
    tasks = _as_tasks(tasks)
    base, name_to_id, comp_to_ids = _prepare_base(model, close_boundaries)
    original_ids = {r.id for r in base.reactions}
    n_tasks = len(tasks)

    # Results are tracked by task *position*, not by task id. A task list routinely
    # reuses one id for many distinct tasks (metabolicTasks_Essential.txt has 57 tasks
    # under just 5 ids: ER/BS/SU/IC/…); keying by id lets later same-id tasks overwrite
    # earlier ones and silently drop their essential reactions from the union — which
    # under-counted the essential set by ~35% (259 vs 397 on Human-GEM).
    per_index: dict[int, dict[str, int]] = {}
    failed_index: list[int] = []
    task_metabolites: set[str] = set()
    if cache_path is not None and Path(cache_path).exists():
        cached = pickle.load(open(cache_path, "rb"))
        if "per_index" in cached:  # ignore any pre-fix, id-keyed cache and recompute
            per_index = {int(k): v for k, v in cached["per_index"].items()}
            task_metabolites = set(cached["mets"])
            failed_index = list(cached["failed"])

    def write_checkpoint() -> None:
        assert cache_path is not None  # only called when a cache_path was given
        # The handle must be closed *before* the rename, not merely dropped: a
        # buffered write is only guaranteed flushed on close, and on Windows
        # renaming a file that still has an open handle raises PermissionError.
        # Passing open() straight into pickle.dump left that to refcount timing —
        # it happens to work on CPython, and emitted a ResourceWarning per task.
        tmp = Path(f"{cache_path}.part")
        with open(tmp, "wb") as fh:
            pickle.dump({"per_index": per_index, "mets": task_metabolites,
                         "failed": failed_index}, fh)
        tmp.replace(cache_path)

    since_checkpoint = 0

    def record(i: int, task: Task, essential, task_mets, error) -> None:
        nonlocal since_checkpoint
        if error is not None:
            failed_index.append(i)
            if verbose:
                print(f"[{i + 1}/{n_tasks}] {task.id}: FAILED ({error})", flush=True)
        else:
            task_metabolites.update(task_mets)
            per_index[i] = essential
            if verbose:
                print(f"[{i + 1}/{n_tasks}] {task.id}: "
                      f"{len(essential)} essential rxn(s)", flush=True)
        if cache_path is not None:
            since_checkpoint += 1
            if since_checkpoint >= checkpoint_every:
                write_checkpoint()
                since_checkpoint = 0

    done = set(per_index) | set(failed_index)
    pending = [(i, task) for i, task in enumerate(tasks)
               if not task.should_fail and i not in done]
    if verbose:
        for i, task in enumerate(tasks):
            if i in done and not task.should_fail:
                print(f"[{i + 1}/{n_tasks}] {task.id}: cached", flush=True)

    if processes <= 1 or len(pending) <= 1:
        for i, task in pending:
            essential, task_mets, error = _compute_one_task(
                base, task, name_to_id, comp_to_ids, original_ids
            )
            record(i, task, essential, task_mets, error)
    else:
        with cf.ProcessPoolExecutor(
            max_workers=min(processes, len(pending)),
            initializer=_init_worker, initargs=(base, name_to_id, comp_to_ids, original_ids),
        ) as pool:
            futures = [pool.submit(_essential_worker, i, task) for i, task in pending]
            by_index = dict(pending)
            for fut in cf.as_completed(futures):
                i, essential, task_mets, error = fut.result()
                record(i, by_index[i], essential, task_mets, error)

    if cache_path is not None and since_checkpoint:  # flush any tail not yet checkpointed
        write_checkpoint()

    # Majority direction across *all* tasks; tie (sum == 0) → forward, as RAVEN's `pos < neg`.
    direction_votes: dict[str, int] = {}
    for essential in per_index.values():
        for rxn_id, direction in essential.items():
            direction_votes[rxn_id] = direction_votes.get(rxn_id, 0) + direction
    reactions = {rid: (-1 if votes < 0 else 1) for rid, votes in direction_votes.items()}

    # Public per-task view, merged by id (union of the same-id tasks' essentials). This is
    # not used for the overall result — that comes from `reactions` above — so the merge is
    # only for inspection and stays lossless for the union.
    per_task: dict[str, dict[str, int]] = {}
    for i, essential in per_index.items():
        slot = per_task.setdefault(tasks[i].id, {})
        for rxn_id, direction in essential.items():
            slot[rxn_id] = slot.get(rxn_id, 0) + direction
    per_task = {tid: {r: (-1 if v < 0 else 1) for r, v in d.items()} for tid, d in per_task.items()}

    failed = sorted({tasks[i].id for i in failed_index})
    return EssentialReactionsResult(reactions, per_task, task_metabolites, failed,
                                    sorted(failed_index))
