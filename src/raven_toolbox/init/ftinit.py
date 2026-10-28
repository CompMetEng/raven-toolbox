"""The ftINIT MILP — the faster staged variant of INIT.

ftINIT keeps tINIT's objective — pick the reaction subset best matching expression
scores while staying flux-consistent — but with a cheaper MILP encoding that is the
reason it is *fast*: a **positive-score reaction needs no binary**. Because the
objective *maximises* ``Σ score·y`` with ``score > 0``, the optimiser pushes its
continuous indicator ``y ∈ [0,1]`` to 1, and the gate ``net_flux ≥ force_on·y`` only
lets ``y`` reach 1 if the reaction can actually carry flux. Only *negative*-score
reactions need a true ``{0,1}`` binary (their indicator would otherwise sit at 0 for
free). This roughly halves the integer count — the dominant MILP cost.

Reaction categories (RAVEN's six), by score sign × reversibility:

* **score 0** — left in the model, *not* in the problem: a free flux variable that can
  carry flux for connectivity but is neither scored nor removable.
* **positive, irreversible** — continuous ``y∈[0,1]``; ``v ≥ force_on·y``. No binary.
* **positive, reversible** — split ``v = v⁺ − v⁻``; continuous ``y``; a single
  direction binary keeps one of ``v⁺/v⁻`` at 0 (no fwd/back loop faking "on");
  ``v⁺+v⁻ ≥ force_on·y``.
* **negative, irreversible** — binary ``x∈{0,1}``; ``v ≤ ub·x``.
* **negative, reversible** — split; binary ``x``; ``v⁺+v⁻ ≤ cap·x``.
* **essential** — forced on (``v ≥ force_on_ess``); no indicator. Assumed already
  oriented irreversible in its forced direction (``prepINITModel`` does this).

Objective: **maximise** ``Σ score·indicator`` (``+ Σ prod_weight·mon`` when
``metabolomics`` is given). ftINIT does **not** reward production of every metabolite by
default — ``prod_weight`` applies only to metabolomics-detected metabolites, each getting
one continuous ``mon`` variable capped by the sum of its producers' own indicators (see
``run_ftinit``'s docstring). Connectivity otherwise comes solely from the flux gates plus
any essential reactions. ``allow_excretion`` relaxes ``S·v = 0`` to ``≥ 0``; ``rem_pos_rev``
drops positive reversible reactions from the problem (used in the staging schedule).

Needs a MILP solver (cobra's configured optlang solver; only Gurobi is fully viable at
genome scale — see the `INIT solver benchmark
<https://github.com/CompMetEng/raven-docs/blob/main/docs/parameter-tuning/studies/init-solver-benchmark.md>`_
on raven-docs). Magic numbers (``force_on``/``force_on_ess`` = 0.1, ``big_m`` = 1000) are
exposed and scale-dependent; calibration tables are in the `INIT parameter calibration
study
<https://github.com/CompMetEng/raven-docs/blob/main/docs/parameter-tuning/studies/init-param-calibration.md>`_
on raven-docs. Each scored reaction's on/off (direction) constraints use its own big-M,
as RAVEN's ``ftINITInternalAlg`` does: its flux capacity ``max(min(ub, big_m),
min(-lb, big_m), 0)`` (an infinite bound counts as ``big_m``), raised to ``force_on`` so
the "on" floor stays feasible. ``big_m`` is therefore the cap on that capacity, not a
fixed constant. Free / essential reactions keep their real bounds.

⚠️ **Loops.** The MILP has *no* loopless constraint: an internal
thermodynamically-infeasible cycle is flux-consistent (``S·v = 0``), so if its
reactions carry positive net score the optimiser will "include" them with no real
exchange flux. RAVEN tolerates this — loop-free models come from the staged pipeline
+ exchange handling, and at genome scale real exchange reactions make such cycles not
score-optimal. A loopless option could be layered on later if needed.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import cobra
from cobra.exceptions import OptimizationError
from optlang.interface import Variable  # untyped (resolves to Any); used for container hints
from optlang.symbolics import Real, add, mul

from raven_toolbox.init.genes import remove_low_score_genes
from raven_toolbox.init.hpa import HPAData, HPARnaData
from raven_toolbox.init.merge import group_rxn_scores
from raven_toolbox.init.model_scores import TranscrData, score_for_init
from raven_toolbox.init.steps import get_init_steps
from raven_toolbox.init.taskfill import fill_tasks

_FORCE_ON = 0.1  # min flux for a reaction to count as "on" (RAVEN forceOnLim)
_BIG_M = 1000.0  # cap on a scored reaction's per-reaction big-M (RAVEN's bigMCap)
_EXTRACT_SEED = 1234  # RAVEN optimizeProb Seed (default for the ``seed`` parameter)
_EXTRACT_THREADS = 1  # RAVEN forces single-threaded solving; see the ``threads`` parameter
_EXTRACT_TOL = 1e-9   # RAVEN optimizeProb Feasibility/Optimality/IntFeas tolerances (Gurobi's floor)
_EXTRACT_PRESOLVE = 2  # RAVEN optimizeProb Presolve (aggressive)


def _reaction_big_m(lb: float, ub: float, cap: float, force_on: float) -> float:
    """Big-M of one scored reaction, as RAVEN's ``ftINITInternalAlg`` computes it.

    The reaction's flux capacity in either direction, each capped at ``cap`` (an infinite
    or NaN bound counts as ``cap``), and at least ``force_on`` so that ``|flux| >= force_on``
    stays reachable when the indicator is on. For a forward-only reaction (``lb >= 0``)
    this is RAVEN's ``max(min(ub, cap), forceOnLim)``; for a reversible one it is
    ``max(min(ub, cap), min(-lb, cap), forceOnLim)``. A reverse-only reaction
    (``ub <= 0``) gets its reverse capacity ``min(-lb, cap)``.
    """
    ub_safe = min(ub, cap) if math.isfinite(ub) else cap
    lb_safe = max(min(-lb, cap) if math.isfinite(lb) else cap, 0.0)
    return max(ub_safe, lb_safe, force_on)

# A resolve_ties tie-break phase is adopted whether it proves or times out (see
# _resolve_ties), so it is given a fraction of the primary solve's own budget rather than
# the full amount again: a phase that would not converge within the full budget has been
# observed (on genome-scale Human-GEM) to still not converge, only slower; a phase that
# does converge does so far inside either budget (a pure integer count/id-rank objective
# with a 0.4 absolute gap). Not independently tuned — see the ftINIT reproducibility study
# on raven-docs for the experiment that would calibrate this properly instead of halving
# on general principle.
_TIE_BREAK_TIME_FRACTION = 0.5


def _dbg(msg: str) -> None:
    """Print a diagnostic line to stderr when FTINIT_DEBUG is set (off by default)."""
    import os
    import sys
    if os.environ.get("FTINIT_DEBUG"):
        print(msg, file=sys.stderr, flush=True)


@dataclass
class FtInitResult:
    """Result of :func:`run_ftinit`."""

    model: cobra.Model
    kept_reactions: list[str]
    deleted_reactions: list[str]
    fluxes: dict[str, float]
    objective: float
    on_reactions: set[str] = field(default_factory=set)  # scored reactions turned on (indicator)
    achieved_gap: float | None = None  # solver's final relative MIP gap (RAVEN's multi-run driver)
    status: str | None = None  # solver status of the accepted solve ('optimal', 'time_limit', ...)


def run_ftinit(
    model: cobra.Model,
    rxn_scores: Mapping[str, float] | None = None,
    *,
    essential_rxns: Iterable[str] | None = None,
    essential_directions: Mapping[str, int] | None = None,
    essential_force: Mapping[str, float] | None = None,
    allow_excretion: bool = False,
    rem_pos_rev: bool = False,
    ignore_mets: Iterable[str] = (),
    force_on: float = _FORCE_ON,
    force_on_ess: float = _FORCE_ON,
    big_m: float = _BIG_M,
    mip_gap: float | None = None,
    mip_gap_abs: float | None = None,
    time_limit: float | None = None,
    prove_abs_gap: float | None = None,
    resolve_ties: bool = False,
    rank_keys: Mapping[str, str] | None = None,
    metabolomics: Mapping[str, Iterable[str]] | None = None,
    prod_weight: float = 0.5,
    seed: int = _EXTRACT_SEED,
    threads: int = _EXTRACT_THREADS,
    feas_tol: float = _EXTRACT_TOL,
    opt_tol: float = _EXTRACT_TOL,
    int_feas_tol: float = _EXTRACT_TOL,
    presolve: int = _EXTRACT_PRESOLVE,
) -> FtInitResult:
    """Run the single-step ftINIT MILP and return the extracted model.

    ``rxn_scores`` maps reaction id → score (default 0 → reaction left free in the
    model, not scored or removable). ``essential_rxns`` are forced to carry flux
    (≥ ``force_on_ess``, overridable per reaction via ``essential_force``);
    ``essential_directions`` maps an essential reaction id to ``+1`` (forward) or
    ``-1`` (reverse) for the forced direction (default forward). ``ignore_mets`` are
    metabolite **names** whose mass balance is dropped (RAVEN's per-step "simple
    metabolite" removal, e.g. H2O/H+). See the module docstring for the formulation.
    This is the single-step variant; the staged schedule
    (:func:`raven_toolbox.init.ftinit`) calls it per step.

    ``mip_gap`` / ``time_limit``: the default ``None`` uses the solver's own
    defaults (Gurobi: MIPGap≈1e-4, no time cap), which is fine for a single step
    — solve time here is dominated by model construction, so a tight gap is
    nearly free (measured on genome-scale Human-GEM, see the `INIT parameter
    calibration study
    <https://github.com/CompMetEng/raven-docs/blob/main/docs/parameter-tuning/studies/init-param-calibration.md>`_
    on raven-docs). For the genome-scale staged
    schedule (:func:`raven_toolbox.init.ftinit`), an essential-forced step can
    run away without a cap (one severely-degraded case ran >75 min unbounded);
    set ``time_limit`` ≈120-600 s/step, and loosen ``mip_gap`` to ``0.01`` (or
    ``0.005``) for a ~37% speedup at ≥0.99 Jaccard vs. the tight-gap model.

    ``prove_abs_gap``, when set, proves the solve to this fixed absolute objective
    gap instead of a relative ``mip_gap``/``mip_gap_abs`` — see the ``prove_abs_gap``
    parameter of :func:`ftinit` for when to use it (recommended value: ``1.0``).

    ``resolve_ties`` (opt-in, default off to preserve RAVEN parity) resolves the MILP's
    degeneracy deterministically: after the score optimum is found, a lexicographic
    phase 2 holds the objective and minimises the id-ordered count of "on" reactions, so
    the kept set is the sparsest optimum found, independent of solver seed/version,
    instead of an arbitrary tie-break. At genome scale the phase-2 solves themselves can
    exhaust ``time_limit``; when that happens the (still-adopted) incumbent is an
    unproven tie-break and a warning is raised. The tie-break is skipped when the primary
    solve itself stops at ``time_limit``, since there is then no proven optimum to hold.
    After it, the reported ``fluxes`` are a least-total-flux solution for the selected on
    set (:func:`_min_abs_flux`) and, among those, the one with the least id-rank-weighted
    backward flux (:func:`_min_backward_flux`). See :func:`_resolve_ties`.

    ``rank_keys`` maps a reaction id to the key the id-rank phases of ``resolve_ties``
    sort it by (default: the id itself). :func:`ftinit` passes the smallest original
    reaction id of each merge group, so the order does not depend on which member names
    the merged reaction.

    ``metabolomics`` (already in **this model's** id space — :func:`ftinit` resolves a
    caller-facing metabolite-name list into this once) maps a detected-metabolite label
    to the set of reaction ids that produce it. A metabolite's ``mon`` indicator
    (continuous, ``[0, 1]``) is capped by the sum of its producers' own indicators and
    contributes ``prod_weight`` to the objective when any producer is on — the reward
    for keeping (or bringing back) a reaction the data itself scored badly, purely
    because it makes a detected metabolite producible. A producer with score 0 is
    pulled into the problem (given a continuous indicator, contributing nothing itself)
    so it has something for ``mon`` to reference; a negative-score producer's indicator
    gets an added flux *floor* (mirroring what a positive reaction already has), so "on"
    means the reaction genuinely carries flux, not merely that it is permitted to. A
    detected metabolite is dropped silently if any of its producers is already in
    ``essential_rxns`` — it will be produced regardless (RAVEN
    ``ftINITInternalAlg.m:106-111``). This does not verify the metabolite is actually
    net-produced through that specific reaction, only that the reaction is "on"; a
    reaction whose real network role is consumption can still earn the bonus if it ends
    up carrying flux in reverse (RAVEN's own documented trade-off — the rest of the
    network's mass balance is trusted to make the "on" flag meaningful).

    ``prod_weight`` is the per-metabolite reward in ``metabolomics``'s ``mon`` terms
    (RAVEN's ``ftINITInternalAlg`` default when unset; ``ftINIT.m`` itself calls it with
    5). Unused when ``metabolomics`` is not given.

    ``seed`` is Gurobi's ``Seed`` parameter (RAVEN's 1234). The MILP is degenerate, so the
    seed picks which of many equal-score optima the solver returns; varying it is the cheap
    way to probe how much of a result rests on the tie-break rather than on the data.

    ``feas_tol``, ``opt_tol`` and ``int_feas_tol`` are Gurobi's ``FeasibilityTol``,
    ``OptimalityTol`` and ``IntFeasTol`` (RAVEN's 1e-9 each, Gurobi's minimum; Gurobi's own
    defaults are 1e-6, 1e-6 and 1e-5), and ``presolve`` is Gurobi's ``Presolve`` (RAVEN's 2).
    They are RAVEN parity settings: they steer which optimal vertex the degenerate MILP lands
    on and how binaries round at the 0.5 on/off cut. An "off" binary may sit at up to
    ``int_feas_tol`` and still permit flux up to its reaction's big-M times ``int_feas_tol``,
    so relaxing ``int_feas_tol`` loosens the on/off gate in proportion to that big-M. Their effect on
    runtime and on the extracted model has not been benchmarked; leave them at the defaults
    unless you are measuring that.

    ``threads`` is Gurobi's ``Threads`` (RAVEN's 1). Raising it speeds the solve but
    **forfeits reproducibility**: multi-threaded Gurobi races between equal optima, so two
    runs of the same input can return different reaction sets. Left at 1 unless a caller
    knowingly trades determinism for speed. Measured to have no effect on the *swing*
    between different seeds (raising it is not a substitute for ``resolve_ties``) — see the
    `ftINIT reproducibility study
    <https://github.com/CompMetEng/raven-docs/blob/main/docs/parameter-tuning/studies/ftinit-determinism.md>`_
    on raven-docs.
    """
    scores = dict(rxn_scores or {})
    essential = set(essential_rxns or [])
    directions = dict(essential_directions or {})
    essential_force = dict(essential_force or {})
    ignore_met_names = set(ignore_mets)
    prob = model.problem
    opt = prob.Model()
    on_gurobi = _is_gurobi(prob)
    if prob.__name__ == "optlang.glpk_interface":
        warnings.warn(
            "ftINIT is solving its MILP with GLPK, which rarely solves a genome-scale "
            "ftINIT MILP within time_limit, and the seed, threads, tolerance, presolve and "
            "MIP-gap settings are Gurobi parameters that GLPK ignores. Set "
            "model.solver = 'gurobi' first.",
            stacklevel=2,
        )
    # With presolve "auto" (GLPK's default), optlang solves again from scratch after
    # any non-optimal status, which doubles the cost of a solve that hit time_limit.
    if opt.configuration.presolve == "auto":
        opt.configuration.presolve = True

    # A detected metabolite already produced by an essential reaction needs no bonus
    # (it is produced regardless); dropping it here avoids an unused mon/constraint pair.
    met_producers = {name: set(ids) for name, ids in (metabolomics or {}).items()
                     if not (set(ids) & essential)}
    all_met_producers: set[str] = set().union(*met_producers.values()) if met_producers else set()

    variables: list = []
    constraints: list = []
    flux_terms: dict[str, list[tuple[Variable, float]]] = {}  # rxn id -> [(var, sign)]
    indicators: dict[str, tuple[Variable, float]] = {}  # rxn id -> (indicator var, score)
    free_or_essential: set[str] = set()               # kept regardless of an indicator

    def add_constraint(expr, **kw):
        constraints.append(prob.Constraint(expr, **kw))

    for rxn in model.reactions:
        rid = rxn.id
        lb, ub = rxn.lower_bound, rxn.upper_bound
        score = float(scores.get(rid, 0.0))
        if rem_pos_rev and score > 0 and lb < 0 < ub:
            score = 0.0  # staging step 1: positive reversibles dropped from the problem

        if rid in essential:
            # Forced to carry flux in its forced direction (default forward); respect a
            # stricter native bound if the model already forces more flux. The forced
            # magnitude may be set per reaction (RAVEN's min(0.99·|prev flux|, 0.1), so
            # a reaction is never forced above what it carried before).
            force = essential_force.get(rid, force_on_ess) if essential_force else force_on_ess
            if directions.get(rid, 1) >= 0:
                forced = min(force, ub)  # clamp to capacity so we never make lb > ub
                v = prob.Variable(f"v_{rid}", lb=max(forced, lb, 0.0), ub=ub)
            else:  # reverse: flux ≤ -force
                forced = min(force, -lb)
                v = prob.Variable(f"v_{rid}", lb=lb, ub=min(-forced, ub))
            variables.append(v)
            flux_terms[rid] = [(v, 1.0)]
            free_or_essential.add(rid)
            continue

        if score == 0.0 and rid not in all_met_producers:
            # free: carries flux for connectivity, not scored/removable
            v = prob.Variable(f"v_{rid}", lb=lb, ub=ub)
            variables.append(v)
            flux_terms[rid] = [(v, 1.0)]
            free_or_essential.add(rid)
            continue

        reversible = lb < 0 < ub
        m = _reaction_big_m(lb, ub, big_m, force_on)
        if reversible:
            vp = prob.Variable(f"vp_{rid}", lb=0.0, ub=ub)
            vn = prob.Variable(f"vn_{rid}", lb=0.0, ub=-lb)
            variables += [vp, vn]
            flux_terms[rid] = [(vp, 1.0), (vn, -1.0)]
            total = vp + vn  # |flux| (one of vp/vn pinned to 0 below), used by the gates
        else:  # single-direction: keep the model's own [lb, ub] (incl. any forced lb>0)
            v = prob.Variable(f"v_{rid}", lb=lb, ub=ub)
            variables.append(v)
            flux_terms[rid] = [(v, 1.0)]
            total = v if ub > 0 else -v  # magnitude for a single-direction reaction

        if score == 0.0:
            # A zero-score metabolite producer still gets an indicator below (purely so
            # that metabolite's mon term has something to reference), but it must stay
            # unconditionally kept regardless of what that indicator ends up doing --
            # RAVEN never deletes an exactly-zero-score reaction at all
            # (ftINITInternalAlg.m: "deletedRxns = (onoff<0.5) & (rxnScores~=0)"), and
            # nothing about becoming a candidate producer changes that.
            free_or_essential.add(rid)

        if score > 0 or rid in all_met_producers:
            # A zero-score metabolite producer is pulled in here too (score stays 0, so
            # it contributes nothing itself) purely so it has an indicator for that
            # metabolite's mon term to reference below (RAVEN: such a reaction "can
            # still be seen as positive, since it either does not matter ... or can
            # contribute to an increased score" -- ftINITInternalAlg.m:96-101).
            y = prob.Variable(f"y_{rid}", lb=0.0, ub=1.0)  # continuous indicator, no binary
            variables.append(y)
            indicators[rid] = (y, score)
            add_constraint(total - force_on * y, lb=0.0, name=f"on_{rid}")  # y=1 ⇒ |flux| ≥ force_on
            if reversible:  # one direction binary stops a fwd/back loop faking "on"
                b = prob.Variable(f"b_{rid}", type="binary")
                variables.append(b)
                add_constraint(vp - m * b, ub=0.0, name=f"dirp_{rid}")          # vp ≤ M·b
                add_constraint(vn + m * b, ub=m, name=f"dirn_{rid}")            # vn ≤ M·(1-b)
        else:  # score < 0
            x = prob.Variable(f"x_{rid}", type="binary")
            variables.append(x)
            indicators[rid] = (x, score)
            add_constraint(total - m * x, ub=0.0, name=f"off_{rid}")  # flux>0 ⇒ x=1
            if rid in all_met_producers:
                # The cap above only *permits* flux when x=1; a met-producer also needs
                # a *floor*, or its production bonus below could be earned without the
                # reaction ever actually carrying flux (ftINITInternalAlg.m:187-195).
                add_constraint(total - force_on * x, lb=0.0, name=f"metforce_{rid}")
                if reversible:
                    # Without this, vp and vn could both sit at a small positive value
                    # simultaneously -- a physically meaningless fwd+back "loop" that
                    # still clears the floor above without a genuine net direction.
                    # Positive reversibles already need this guard to stop the same
                    # trick faking "on"; a plain negative reversible never needed it
                    # (nothing rewards faking one), but a met-producer's indicator now
                    # feeds a real reward, so it needs the same guard too.
                    bm = prob.Variable(f"bm_{rid}", type="binary")
                    variables.append(bm)
                    add_constraint(vp - m * bm, ub=0.0, name=f"metdirp_{rid}")
                    add_constraint(vn + m * bm, ub=m, name=f"metdirn_{rid}")

    # One continuous mon ∈ [0, 1] per detected metabolite, capped by the sum of its
    # producers' own indicators (mon ≤ Σ producer indicators) and rewarded prod_weight
    # in the objective below -- the "any producer on" bonus (ftINITInternalAlg.m:177-195).
    # A metabolite with no producer that actually entered the problem (e.g. its only
    # producers were themselves score-0 and non-producing, or filtered out above) gets
    # no mon at all: with an empty sum the cap would pin mon at 0 anyway, so skipping it
    # is just avoiding a dead variable, not a behaviour change.
    met_bonus_terms = []
    for i, (name, ids) in enumerate(met_producers.items()):
        producer_inds = [indicators[rid][0] for rid in ids if rid in indicators]
        if not producer_inds:
            continue
        mon = prob.Variable(f"_mon_{i}", lb=0.0, ub=1.0)
        variables.append(mon)
        add_constraint(mon - add([mul([Real(1.0), ind]) for ind in producer_inds]),
                       ub=0.0, name=f"metprod_{i}")
        met_bonus_terms.append(mul([Real(prod_weight), mon]))
        _dbg(f"[ftinit] metabolomics: {name!r} <- {len(producer_inds)} producer(s)")

    # Steady state S·v {== 0 | >= 0}; ignored metabolites are left unbalanced.
    # NOTE (allow_excretion sign): we relax to S·v >= 0 (net production / excretion
    # allowed), matching the constraint's name and classic INIT (RAVEN runINIT). RAVEN's
    # *ftINIT* builder instead uses csense 'L' → S·v <= 0 (net consumption), which reads as
    # inconsistent with its own "allowExcretion" intent. We deliberately keep the
    # intent-faithful S·v >= 0: allow_excretion is unused in the default '1+1' schedule
    # (both steps have allow_met_secr=False) and only fires in '2+1'/'2+0' step 1, so this
    # does not affect the gene-essentiality pipeline. Revisit only if exact '2+1' parity
    # with RAVEN is required.
    # Build each metabolite's balance as a *flat* list of (coeff·sign)·var terms and sum
    # it with optlang.symbolics.add. Python's builtin sum re-canonicalises a growing
    # sympy expression at every step (O(n²)); for hub metabolites that appear in ~10³
    # reactions that is minutes per constraint. add() builds the sum in one pass.
    met_terms: dict = {m: [] for m in model.metabolites if m.name not in ignore_met_names}
    for rxn in model.reactions:
        terms = flux_terms[rxn.id]
        for met, coeff in rxn.metabolites.items():
            bucket = met_terms.get(met)
            if bucket is None:
                continue
            for var, sign in terms:
                bucket.append(mul([Real(coeff * sign), var]))
    for termlist in met_terms.values():
        if termlist:
            add_constraint(add(termlist), lb=0.0, ub=None if allow_excretion else 0.0)

    opt.add(variables + constraints)
    obj_expr = add([mul([Real(score), ind]) for ind, score in indicators.values()]
                   + met_bonus_terms)
    opt.objective = prob.Objective(obj_expr, direction="max")
    if on_gurobi:
        # Match RAVEN's optimizeProb defaults exactly, because the ftINIT MILP is highly
        # degenerate and the chosen incumbent (hence which reactions are kept) depends on
        # these:
        #   * Threads=1 — RAVEN forces single-threaded solving; multi-threaded Gurobi
        #     picks among equal optima non-deterministically and can even report the MILP
        #     infeasible (RAVEN issue #607). This is the dominant reproducibility lever.
        #   * Presolve=2, FeasibilityTol/OptimalityTol/IntFeasTol=1e-9 — RAVEN's
        #     optimizeProb defaults (the ``presolve``/``feas_tol``/``opt_tol``/
        #     ``int_feas_tol`` parameters); they steer which optimal vertex a degenerate
        #     MILP lands on and how binaries round at the 0.5 on/off cut.
        #   * Seed — fixed by default so tie-breaking is reproducible (see ``seed``).
        opt.problem.Params.Threads = threads
        opt.problem.Params.Presolve = presolve
        opt.problem.Params.FeasibilityTol = feas_tol
        opt.problem.Params.OptimalityTol = opt_tol
        opt.problem.Params.IntFeasTol = int_feas_tol
        opt.problem.Params.Seed = seed
    if prove_abs_gap is not None:
        # One solve proven to this fixed *absolute* gap, replacing the relative-gap
        # escalation below. Measured on genome-scale Human-GEM: the escalation returns a
        # solution 2.0-4.0 below the true optimum (see the raven-docs reproducibility
        # study); any value in 1.0-2.0 recovers the optimum and stays provable within a
        # normal time_limit, while going tighter (<=0.5) stops proving within the time
        # limit and returns the identical model anyway. 1.0 is the recommended value.
        if on_gurobi:
            opt.problem.Params.MIPGap = 0.0
            opt.problem.Params.MIPGapAbs = prove_abs_gap
    elif mip_gap is not None and on_gurobi:
        opt.problem.Params.MIPGap = mip_gap

    if prove_abs_gap is None and mip_gap_abs is not None:
        # RAVEN's multi-run gap strategy (ftINIT.m). The final staged step has a
        # near-zero objective (it mostly removes small negative-score reactions), so a
        # fixed *relative* gap becomes an almost-zero absolute gap the solver cannot
        # reach in the time limit — it stops with extra bypass reactions on, giving a
        # suboptimal model. A fixed *absolute* gap (Gurobi MIPGapAbs) is unsafe: on a
        # small-objective model the solver stops at the first garbage incumbent within
        # it. Instead: a quick estimate solve, then re-solve (warm-started, same model)
        # with a *relative* gap scaled to the objective — max(mip_gap, mip_gap_abs/|obj|)
        # = RAVEN's AbsMIPGap/|objVal|, safe at any scale.
        if time_limit is not None:
            opt.configuration.timeout = max(1, min(int(time_limit) // 10, 30))
        opt.optimize()
        obj = abs(opt.objective.value) if opt.objective.value is not None else 0.0
        _dbg(f"[ftinit] estimate solve: obj={opt.objective.value} status={opt.status}")
        if obj > 0 and on_gurobi:
            eff_gap = min(max(mip_gap or 0.0, mip_gap_abs / obj), 1.0)
            _dbg(f"[ftinit] eff_gap = max({mip_gap}, {mip_gap_abs}/{obj:.1f}) = {eff_gap:.5f}")
            opt.problem.Params.MIPGap = eff_gap

    if time_limit is not None:
        opt.configuration.timeout = int(time_limit)
    opt.optimize()
    _achieved = None
    if on_gurobi:
        try:  # unset when there is no MIP incumbent, e.g. no integer variables at all
            _achieved = opt.problem.MIPGap
        except AttributeError:
            pass
    _status = opt.status  # capture before the tie-resolution phase 2 re-solves in place
    _dbg(f"[ftinit] final solve: obj={opt.objective.value} status={_status} "
         f"achieved_gap={_achieved}")
    # Accept a near-optimal incumbent (when a MIP gap / time limit is set), as RAVEN does,
    # but only if the solver actually holds one to read.
    if opt.status not in ("optimal", "feasible", "suboptimal", "time_limit") \
            or not _has_solution(opt):
        raise OptimizationError(
            f"ftINIT MILP produced no usable solution (status: {opt.status}); "
            "increase time_limit or loosen prove_abs_gap."
        )

    # Report the *primary* (score) objective; the tie-resolution phase replaces it
    # in place, so capture it before that.
    primary_obj = float(opt.objective.value) if opt.objective.value is not None else 0.0

    # RAVEN: a reaction is "on" iff its indicator ≥ 0.5 (positive indicators are
    # continuous and can land fractionally when a reaction can carry only tiny flux).
    def _read_solution():
        on = {rid for rid, (ind, _) in indicators.items() if (ind.primal or 0.0) >= 0.5}
        fluxes = {rid: sum(sign * (var.primal or 0.0) for var, sign in terms)
                  for rid, terms in flux_terms.items()}
        return on, fluxes

    on, fluxes = _read_solution()  # the primary optimum
    # tie resolution is best-effort: keep its result only if phase 2 actually converged.
    # It is skipped when the primary solve stopped at the time limit: phase 2 would pin
    # an unproven incumbent's objective, so it cannot select among true optima, and at
    # genome scale it then only spends its own time budget without finding a solution.
    keys = dict(rank_keys or {})

    def rank_key(rid: str) -> str:
        return keys.get(rid, rid)

    if resolve_ties and indicators and _status == "time_limit":
        _dbg("[ftinit] tie-break skipped: the primary solve stopped at the time limit")
    elif resolve_ties and indicators and _resolve_ties(opt, prob, obj_expr, indicators,
                                                        primary_obj, time_limit, rank_key):
        on, fluxes = _read_solution()
        # Like phases 2a/2b, only when there is a removable (negative-score) reaction.
        held = (_min_abs_flux(opt, prob, indicators, flux_terms, time_limit)
                if any(score < 0 for _, score in indicators.values()) else None)
        if held is not None:
            fluxes = _read_solution()[1]
            abs_expr, total, free_sign = held
            if _min_backward_flux(opt, prob, flux_terms, abs_expr, total, free_sign,
                                  rank_key, time_limit):
                fluxes = _read_solution()[1]

    kept = free_or_essential | on
    deleted = [r.id for r in model.reactions if r.id not in kept]

    out = model.copy()
    out.remove_reactions(deleted, remove_orphans=True)
    return FtInitResult(out, sorted(kept), sorted(deleted), fluxes,
                        primary_obj, on_reactions=on,
                        achieved_gap=_achieved, status=_status)


def _is_gurobi(interface) -> bool:
    """Whether ``interface`` (an optlang interface module) is the Gurobi backend."""
    return interface.__name__ == "optlang.gurobi_interface"


def _has_solution(opt) -> bool:
    """Whether the solver currently holds a readable primal solution.

    A MILP can finish with an accepted status (notably ``time_limit``) yet no incumbent,
    so reading ``.primal`` would raise. On Gurobi this is the solution count; on other
    backends the status is taken as authoritative.
    """
    try:
        return opt.problem.SolCount > 0
    except Exception:  # noqa: BLE001 - non-Gurobi backend
        return True


def _resolve_ties(opt, prob, obj_expr, indicators, primary, time_limit, rank_key) -> bool:
    """Pin ftINIT's degenerate optimum to a single, reproducible solution (on ``opt``).

    The ftINIT MILP is highly degenerate: many reaction subsets reach the same score
    optimum and the solver returns an arbitrary one — reproducible for a fixed solver
    build + seed, but fragile to the Gurobi version or platform (a changed tie-break
    moves ~1-2% of the kept reactions, which flips a handful of gene-essentiality calls).

    This runs a lexicographic phase 2, holding the score objective at its optimum with a
    floor constraint: first minimise the count of kept removable reactions (the sparsest
    optimum), then, among the sparsest, minimise their summed id rank (prefer lower ids;
    ``rank_key`` maps a reaction id to the key it is ranked by). The result is a stable,
    near-unique optimum independent of seed/solver version *when
    both phases converge* — see the caveat below. ``opt`` is left holding the phase-2
    solution, which the caller reads for the "on" set and fluxes; the reported objective
    stays the phase-1 ``primary`` value.

    Only the negative-score reactions carry a true 0/1 "keep" binary (the positive
    indicators are continuous and pinned near 1 by the score objective), so the two phases
    run over those: their count and id-sum are integers, provable with a cheap absolute gap
    below 1 rather than the near-full proof a tiny relative gap would need. This resolves
    the removable-reaction choices, the genuine seed-fragile degeneracy. The flux
    distribution, which the staged schedule reads for the next step's forced directions and
    ``ess_force`` clamp, is pinned afterwards by :func:`_min_abs_flux`.

    Caveat, measured on genome-scale Human-GEM (not visible at toy/unit-test scale, where
    both phases prove in milliseconds): the phases can themselves exhaust ``time_limit``,
    in which case the incumbent adopted is an *unproven* tie-break — it still measurably
    reduces the run-to-run spread relative to no tie-break at all, but it is no longer a
    guarantee. A warning is raised when this happens; see the `ftINIT reproducibility
    study
    <https://github.com/CompMetEng/raven-docs/blob/main/docs/parameter-tuning/studies/ftinit-determinism.md>`_
    on raven-docs.

    Returns ``True`` if phase 2 produced a usable solution (the caller then reads the
    resolved "on" set and fluxes), ``False`` if it did not converge to an incumbent (the
    caller keeps the phase-1 optimum). Best-effort: it never fails the extraction.
    """
    binaries = {rid: ind for rid, (ind, score) in indicators.items() if score < 0}
    if not binaries:  # only positive / free reactions: no removable tie to resolve
        return _has_solution(opt)
    tol = max(abs(primary) * 1e-7, 1e-7)
    opt.add(prob.Constraint(obj_expr, lb=primary - tol, name="_canon_obj_floor"))
    count = add([mul([Real(1.0), ind]) for ind in binaries.values()])
    unproven: list[str] = []

    def _phase(objective, label: str) -> bool:
        opt.objective = objective
        if _is_gurobi(prob):  # integer objective: an absolute gap < 1 proves it cheaply
            opt.problem.Params.MIPGap = 0.0
            opt.problem.Params.MIPGapAbs = 0.4
        if time_limit is not None:
            # A fraction of the primary solve's own budget, not the full amount again —
            # see _TIE_BREAK_TIME_FRACTION.
            opt.configuration.timeout = max(1, int(time_limit * _TIE_BREAK_TIME_FRACTION))
        opt.optimize()
        _dbg(f"[ftinit] tie-break {label}: status={opt.status} obj={opt.objective.value}")
        # A phase that ends at the time limit still holds an incumbent, and adopting it
        # means the tie-break is itself an arbitrary within-gap pick — the very thing this
        # function exists to remove. It is still adopted (it measurably reduces the spread)
        # but it is *recorded*, so "resolve_ties=True" cannot silently mean "tried, failed".
        if opt.status == "time_limit":
            unproven.append(label)
        return (opt.status in ("optimal", "feasible", "suboptimal", "time_limit")
                and _has_solution(opt))

    # Phase 2a — parsimony: the fewest kept removable reactions.
    if not _phase(prob.Objective(count, direction="min"), "phase2a-parsimony"):
        return False
    # A timed-out phase can report a non-finite objective. ``inf + 0.5`` is still ``inf``,
    # which would make the cap below vacuous and silently switch the parsimony pin off, so
    # cap on the incumbent's actual count instead.
    kmin = opt.objective.value
    if kmin is None or not math.isfinite(kmin):
        kmin = float(sum(1 for ind in binaries.values() if (ind.primal or 0.0) >= 0.5))
        unproven.append("phase2a-objective-not-finite")
    # Phase 2b — among the sparsest, prefer lower reaction ids (deterministic tie-break).
    opt.add(prob.Constraint(count, ub=kmin + 0.5, name="_canon_count_cap"))
    ranks = {rid: i for i, rid in enumerate(sorted(binaries, key=rank_key))}
    idsum = add([mul([Real(float(1 + ranks[rid])), ind]) for rid, ind in binaries.items()])
    ok = _phase(prob.Objective(idsum, direction="min"), "phase2b-idrank")
    if ok and unproven:
        warnings.warn(
            f"ftINIT tie resolution did not converge ({', '.join(unproven)}): the "
            "selection among equally score-optimal solutions is itself an unproven "
            "incumbent, so resolve_ties=True has reduced but not removed the "
            "run-to-run spread. Raise time_limit for a proven tie-break. See the "
            "ftINIT reproducibility study on raven-docs "
            "(docs/parameter-tuning/studies/ftinit-determinism.md).",
            stacklevel=3,
        )
    return ok


def _min_abs_flux(opt, prob, indicators, flux_terms, time_limit):
    """Phase 2c of ``resolve_ties``: the least total absolute flux for the tie-broken on set.

    Phases 2a/2b fix which reactions are on but leave the flux distribution free, and the
    staged schedule reads that distribution: the next step forces a reversible reaction in
    the direction of its flux and at ``min(0.99·|flux|, force_on)``. This phase holds the
    on/off state of every indicator (a binary at its value; a continuous indicator that is
    on keeps at least its value, one that is off at most its value) and minimises
    ``Σ|v|`` over all reactions, so the flux read-out comes from a minimal-flux solution
    rather than from whichever vertex the solver happened to reach. Directions of
    reversible scored reactions stay free, so the phase is still a MILP; it stops at a
    relative or absolute gap of 1e-6, with the full ``time_limit`` as its budget (as RAVEN's
    phase 2c).

    Returns ``(abs_expr, total, free_sign)`` if it produced a usable solution (the caller
    then reads its fluxes): the ``Σ|v|`` expression, its value and the ids of the
    reactions whose flux can take either sign. Returns ``None`` otherwise (the caller keeps
    the phase-2b fluxes). It never fails the extraction; a phase ending at the time limit
    is adopted with a warning.
    """
    # Read every value before changing a bound: a bound change discards the solution.
    values = [(ind, ind.primal or 0.0) for ind, _ in indicators.values()]
    for ind, val in values:
        if ind.type == "binary":
            ind.set_bounds(round(val), round(val))
        elif val >= 0.5:
            ind.lb = min(max(val, 0.0), 1.0)
        else:
            ind.ub = min(max(val, 0.0), 1.0)
    abs_terms, extra, free_sign = [], [], []
    for rid, terms in flux_terms.items():
        if len(terms) == 1 and terms[0][0].lb is not None and terms[0][0].lb >= 0:
            abs_terms.append(mul([Real(terms[0][1]), terms[0][0]]))
        elif len(terms) == 1 and terms[0][0].ub is not None and terms[0][0].ub <= 0:
            abs_terms.append(mul([Real(-terms[0][1]), terms[0][0]]))
        else:
            net = add([mul([Real(sign), var]) for var, sign in terms])
            t = prob.Variable(f"_abs_{rid}", lb=0.0)
            extra += [t, prob.Constraint(t - net, lb=0.0, name=f"_absp_{rid}"),
                      prob.Constraint(t + net, lb=0.0, name=f"_absn_{rid}")]
            abs_terms.append(t)
            free_sign.append(rid)
    opt.add(extra)
    abs_expr = add(abs_terms)
    opt.objective = prob.Objective(abs_expr, direction="min")
    if _is_gurobi(prob):
        opt.problem.Params.MIPGap = 1e-6
        opt.problem.Params.MIPGapAbs = 1e-6
    if time_limit is not None:
        opt.configuration.timeout = max(1, int(time_limit))
    opt.optimize()
    _dbg(f"[ftinit] tie-break phase2c-minflux: status={opt.status} obj={opt.objective.value}")
    if opt.status == "time_limit":
        warnings.warn(
            "ftINIT minimal-flux phase (resolve_ties) stopped at the time limit: the flux "
            "distribution, and hence the next step's forced directions, is an unproven "
            "incumbent. Raise time_limit for a proven minimal-flux solution.",
            stacklevel=3,
        )
    if opt.status in ("optimal", "feasible", "suboptimal", "time_limit") and _has_solution(opt):
        return abs_expr, float(opt.objective.value), free_sign
    return None


def _min_backward_flux(opt, prob, flux_terms, abs_expr, total, free_sign, rank_key,
                       time_limit) -> bool:
    """Phase 2d of ``resolve_ties``: fix the directions left open by the minimal total flux.

    Phase 2c can leave the direction of a flux free when two directions carry the same total
    flux, for example a pair of reversible reactions that close a loop in either direction.
    This phase keeps the on/off state of phase 2c, holds ``Σ|v|`` at most at its phase-2c
    value plus ``max(1e-7·Σ|v|, 1e-6)``, and minimises ``Σ w·max(0, -v)`` over the reactions
    whose flux can take either sign. The weight is the reaction's id rank from the end:
    sorted by ``rank_key``, the first of ``n`` reactions gets ``n`` and the last ``1``, so a
    reaction with a lower id prefers its forward direction. It stops at a relative or
    absolute gap of 1e-6, with the full ``time_limit`` as its budget.

    Returns ``True`` if it produced a usable solution (the caller then reads its fluxes),
    ``False`` otherwise (the caller keeps the phase-2c fluxes). It never fails the
    extraction; a phase ending at the time limit is adopted with a warning.
    """
    if not free_sign:
        return False
    tol = max(abs(total) * 1e-7, 1e-6)
    n = len(free_sign)
    weight = {rid: float(n - i) for i, rid in enumerate(sorted(free_sign, key=rank_key))}
    extra, bwd_terms = [prob.Constraint(abs_expr, ub=total + tol, name="_minflux_cap")], []
    for rid in free_sign:
        net = add([mul([Real(sign), var]) for var, sign in flux_terms[rid]])
        b = prob.Variable(f"_bwd_{rid}", lb=0.0)
        extra += [b, prob.Constraint(b + net, lb=0.0, name=f"_bwdc_{rid}")]
        bwd_terms.append(mul([Real(weight[rid]), b]))
    opt.add(extra)
    opt.objective = prob.Objective(add(bwd_terms), direction="min")
    if _is_gurobi(prob):
        opt.problem.Params.MIPGap = 1e-6
        opt.problem.Params.MIPGapAbs = 1e-6
    if time_limit is not None:
        opt.configuration.timeout = max(1, int(time_limit))
    opt.optimize()
    _dbg(f"[ftinit] tie-break phase2d-direction: status={opt.status} obj={opt.objective.value}")
    if opt.status == "time_limit":
        warnings.warn(
            "ftINIT direction phase (resolve_ties) stopped at the time limit: the "
            "directions of fluxes that carry the same total flux either way, and hence the "
            "next step's forced directions, are an unproven incumbent. Raise time_limit "
            "for a proven solution.",
            stacklevel=3,
        )
    return (opt.status in ("optimal", "feasible", "suboptimal", "time_limit")
            and _has_solution(opt))


def _nudge_scores(rxn_scores: Mapping[str, float]) -> dict[str, float]:
    """Push tiny reaction scores off zero (RAVEN ``ftINIT.m:160-161``).

    Scores in ``(-0.1, 0]`` become ``-0.1`` and in ``(0, 0.1)`` become ``0.1`` — near-zero
    scores cause the MILP numerical trouble. Exactly-zero scores become ``-0.1`` (scored,
    removable) here; genuinely ignore-masked reactions are set back to 0 afterwards by
    :func:`group_rxn_scores`.
    """
    nudged: dict[str, float] = {}
    for rid, s in rxn_scores.items():
        if -0.1 < s <= 0:
            s = -0.1
        elif 0 < s < 0.1:
            s = 0.1
        nudged[rid] = s
    return nudged


def _metabolomics_producers(prep, metabolomics: Iterable[str]) -> dict[str, set[str]]:
    """Map each detected metabolite name to its producer reactions in ``prep.min_model``'s
    (merged) id space.

    A producer is any ``prep.ref_model`` reaction with a positive stoichiometric
    coefficient for a metabolite of that name, or any *reversible* reaction with a
    negative one (a producer when run in reverse) — RAVEN's own rule
    (``ftINIT.m:129-152``). Matching is by metabolite *name* (case-insensitive), not id,
    unioning across every metabolite that shares it (e.g. the same compound in several
    compartments) — also RAVEN's rule. Translated through the merge groups the same way
    the (since-removed) ``reference_reactions`` anchoring did: a merged reaction counts
    as a producer if *any* of its original members does — the conservative reading,
    since a group's representative id is otherwise an implementation detail the caller
    should not need to know.

    Warns (does not raise) for a name matching no metabolite at all — likely a typo or a
    name from an unrelated model. A name matching a metabolite with no producer at all is
    not warned about: legitimately nothing to reward, not a usage error.
    """
    ref = prep.ref_model
    by_name: dict[str, list] = {}
    for m in ref.metabolites:
        by_name.setdefault(m.name.upper(), []).append(m)

    group_of = prep.group_of
    members_of_group: dict[int, list[str]] = {}
    for rid, gid in zip(prep.orig_rxn_ids, prep.group_ids, strict=True):
        if gid:
            members_of_group.setdefault(gid, []).append(rid)

    result: dict[str, set[str]] = {}
    for name in metabolomics:
        mets = by_name.get(name.upper())
        if not mets:
            warnings.warn(
                f"metabolomics metabolite {name!r} matched no metabolite in this "
                "template by name; it will have no effect on this build.",
                stacklevel=2,
            )
            continue
        producer_orig_ids: set[str] = set()
        for met in mets:
            for rxn in met.reactions:
                coeff = rxn.get_coefficient(met.id)
                reversible = rxn.lower_bound < 0 < rxn.upper_bound
                if coeff > 0 or (coeff < 0 and reversible):
                    producer_orig_ids.add(rxn.id)
        matched: set[str] = set()
        for r in prep.min_model.reactions:
            gid = group_of.get(r.id, 0)
            members = members_of_group.get(gid, [r.id]) if gid else [r.id]
            if not producer_orig_ids.isdisjoint(members):
                matched.add(r.id)
        result[name] = matched
    return result


def _solve_step(
    min_model, scores, step, *, essential, directions, ess_force, force_on, big_m,
    mip_gap, mip_gap_abs, time_limit, prove_abs_gap=None, resolve_ties=False,
    rank_keys=None, metabolomics=None, prod_weight=0.5,
    seed=_EXTRACT_SEED, threads=_EXTRACT_THREADS,
    feas_tol=_EXTRACT_TOL, opt_tol=_EXTRACT_TOL, int_feas_tol=_EXTRACT_TOL,
    presolve=_EXTRACT_PRESOLVE,
) -> FtInitResult:
    """Solve one ftINIT step, following RAVEN's multi-run gap-escalation schedule.

    RAVEN (``ftINIT.m:227-283``) runs a step up to ``len(step.milp_runs)`` times: run 1 at
    a flat relative gap; each later run at ``max(mip_gap, abs_gap/|obj|)`` — the absolute
    gap made relative, which is safe when the step's objective is near zero — breaking as
    soon as the previous run's achieved gap already meets the next (looser) target. A
    caller ``time_limit`` caps each run's own limit. With no schedule (``step.milp_runs``
    empty, as in an ad-hoc step) this is a single solve at the caller's gap.
    """
    def _run(mg, mga, tl, prove=None):
        return run_ftinit(
            min_model, scores, essential_rxns=essential, essential_directions=directions,
            essential_force=ess_force, allow_excretion=step.allow_met_secr,
            rem_pos_rev=step.pos_rev_off, ignore_mets=step.mets_to_ignore,
            force_on=force_on, force_on_ess=force_on, big_m=big_m,
            mip_gap=mg, mip_gap_abs=mga, time_limit=tl,
            prove_abs_gap=prove, resolve_ties=resolve_ties, rank_keys=rank_keys,
            metabolomics=metabolomics, prod_weight=prod_weight,
            seed=seed, threads=threads, feas_tol=feas_tol, opt_tol=opt_tol,
            int_feas_tol=int_feas_tol, presolve=presolve,
        )

    if prove_abs_gap is not None:
        # One solve per step proven to a fixed absolute gap, bypassing RAVEN's loose
        # relative escalation — whose final near-zero-objective run otherwise accepts an
        # arbitrary within-gap incumbent. Trades runtime for a well-defined optimum (pairs
        # naturally with ``resolve_ties``).
        return _run(None, None, time_limit, prove=prove_abs_gap)

    if not step.milp_runs:
        return _run(mip_gap, mip_gap_abs, time_limit)

    res: FtInitResult | None = None
    last_obj: float | None = None
    achieved: float | None = None
    for i, run in enumerate(step.milp_runs):
        target = run["mip_gap"]
        tl = run["time_limit"] if time_limit is None else min(run["time_limit"], time_limit)
        if i > 0:
            # RAVEN: near-zero objective → abs_gap/|obj| blows up → accept any incumbent.
            target = (min(max(run["mip_gap"], run["abs_gap"] / abs(last_obj)), 1.0)
                      if last_obj else 1.0)
            if achieved is not None and achieved <= target:
                break
        res = _run(target, None, tl)
        last_obj, achieved = res.objective, res.achieved_gap
    assert res is not None  # milp_runs is non-empty here
    return res


def ftinit(
    prep,
    tissue: str | None = None,
    celltype: str | None = None,
    hpa: HPAData | None = None,
    *,
    transcr: TranscrData | HPARnaData | None = None,
    rxn_scores: Mapping[str, float] | None = None,
    gene_scores: Mapping[str, float] | None = None,
    remove_genes: bool = True,
    use_scores_for_tasks: bool = True,
    no_gene_score: float = -2.0,
    isozyme_scoring: str = "max",
    complex_scoring: str = "min",
    multiple_cell_scoring: str = "max",
    hpa_level_scores: Mapping[str, float] | None = None,
    data_precedence: str = "gene",
    series: str = "1+1",
    steps=None,
    fill_gaps: bool = True,
    metabolomics: Iterable[str] | None = None,
    prod_weight: float = 0.5,
    force_on: float = _FORCE_ON,
    big_m: float = _BIG_M,
    mip_gap: float | None = None,
    mip_gap_abs: float | None = 10.0,
    time_limit: float | None = None,
    prove_abs_gap: float | None = None,
    resolve_ties: bool = False,
    seed: int = _EXTRACT_SEED,
    threads: int = _EXTRACT_THREADS,
    feas_tol: float = _EXTRACT_TOL,
    opt_tol: float = _EXTRACT_TOL,
    int_feas_tol: float = _EXTRACT_TOL,
    presolve: int = _EXTRACT_PRESOLVE,
    verbose: bool = False,
) -> cobra.Model:
    """Run the full ftINIT pipeline on prepData and return the context-specific model.

    ``prep`` is a :class:`raven_toolbox.init.PrepData`. Like RAVEN's ``ftINIT(prepData, tissue,
    celltype, hpaData, 'transcrData', ...)``, the scores come from ``hpa`` and/or ``transcr``
    for ``tissue`` (and ``celltype``) on the reference model, by :func:`score_for_init`, which
    also takes the scoring options (``no_gene_score``, ``isozyme_scoring``, ``complex_scoring``,
    ``multiple_cell_scoring``, ``hpa_level_scores``, ``data_precedence``). Alternatively pass
    ``rxn_scores`` (**original** reaction id → score) and no data; ``gene_scores`` then gives
    the gene scores for the final pruning. ``remove_genes`` (default true) prunes low-score
    genes from the GPRs when gene scores exist; ``use_scores_for_tasks`` (default true) uses
    the reaction scores as weights when fitting the tasks.
    ``series`` selects which staged schedule to run — see :func:`get_init_steps` for
    what each option changes (default ``'1+1'``); pass ``steps`` directly to use a
    custom schedule instead. Each step regroups scores under its ``ignore_mask``, fixes
    the reactions turned on by earlier steps as essential (in their flux direction), and
    solves :func:`run_ftinit` on the merged model. Reactions never turned on (and not
    essential or left-in) are removed from the reference model; exchange reactions are
    always kept (RAVEN re-adds them).

    If ``fill_gaps`` and ``prep`` carries tasks, reactions are added back so every task
    is feasible (:func:`raven_toolbox.init.fill_tasks`). If ``gene_scores`` is given,
    negative-scoring genes are pruned from the GPRs at the end
    (:func:`raven_toolbox.init.remove_low_score_genes`).

    Essential reactions are forced to carry ``force_on`` (default 0.1) of flux in the
    forced direction. On genome-scale models a stricter regime is needed (the previous
    step's actual carried flux instead of a flat 0.1) — exposed via per-reaction
    ``essential_force`` on :func:`run_ftinit`.

    ``metabolomics`` is a list of detected metabolite **names** (case-insensitive,
    ``prep.ref_model`` metabolite names — matched and unioned across every metabolite
    sharing that name, e.g. the same compound in several compartments). Each is resolved
    once, before the step loop, into its producer reactions (translated through the
    merge groups so the resolution survives the linear merge collapsing them into
    combined reactions) and rewarded ``prod_weight`` in every step's objective when any
    producer is on — see :func:`run_ftinit`'s own ``metabolomics``/``prod_weight``
    docstring for the MILP mechanics and its one caveat (the bonus checks a producer is
    "on", not that the metabolite is verifiably net-produced through it). A name that
    matches no metabolite in ``prep.ref_model`` warns rather than silently doing nothing.
    Ported from RAVEN 3's ``ftINITInternalAlg``. ``big_m`` (default 1000, RAVEN's
    ``bigMCap``) caps each scored reaction's own big-M, which is otherwise its flux
    capacity; see :func:`run_ftinit` and the module docstring.

    ``mip_gap`` and ``mip_gap_abs`` apply to a step without a run schedule; a step with a
    schedule (every named series, including ``full``) uses the gaps of its own runs
    (``full``: one run at 0.0004 and 10000 s). ``time_limit`` caps the time limit of each
    run of every step, scheduled or not. On
    genome-scale models they are essential for tractability — see the `INIT parameter
    calibration study
    <https://github.com/CompMetEng/raven-docs/blob/main/docs/parameter-tuning/studies/init-param-calibration.md>`_
    on raven-docs for the calibration table.

    ``resolve_ties`` and ``prove_abs_gap`` (both opt-in, default off → exact RAVEN
    behaviour) reduce the arbitrariness of the degenerate MILP's tie-break, yielding a more
    parsimonious and more *reproducible* extracted model. They do **not** make the model
    biologically more accurate: the alternative optima they choose among are equally
    consistent with the expression data, so this only pins *which* optimum is returned. It
    reduces the fragility of the MILP, not its correctness:

      * ``resolve_ties`` adds a lexicographic phase 2 that selects the sparsest (then
        lowest-id) optimum, so the degenerate choice is pinned rather than left to the
        solver — applied both to each extraction step and to the task gap-fill. In each
        extraction step it then takes the fluxes from the least-total-flux solution for
        that on set, so the directions and forced magnitudes the next step reads from
        them do not depend on the solver's vertex either.
      * ``prove_abs_gap`` replaces the loose relative-gap escalation with a single solve per
        step proven to this fixed *absolute* gap, so the kept-set objective is optimal at
        any scale — not an arbitrary within-gap incumbent. **Try 1.0**: on Human-GEM/DLD1
        the escalation returns a solution 2.0 below the optimum at the first stage and 4.0
        below at the second (351 kept reactions instead of 349), while every value in
        1.0-2.0 proves the true optimum at ~2.4x the runtime. Tighter values are
        counter-productive — 0.5 and below stop being provable within the time limit and
        return the same model anyway.

    ``seed`` (RAVEN's 1234), ``threads`` (RAVEN's 1) and the solver tolerances
    ``feas_tol``/``opt_tol``/``int_feas_tol``/``presolve`` (RAVEN's 1e-9 and 2) are
    forwarded to every step; see :func:`run_ftinit` for what they trade off. Neither
    ``seed`` nor ``threads`` makes the extraction *stable*
    under small input changes, which is a separate property — see the `ftINIT
    reproducibility study
    <https://github.com/CompMetEng/raven-docs/blob/main/docs/parameter-tuning/studies/ftinit-determinism.md>`_
    on raven-docs. Measured there on Human-GEM/DLD1: ``resolve_ties`` makes repeated builds
    at a fixed seed identical and halves the seed-to-seed spread in predicted essential
    genes (16 → 8 flips); its own tie-break phases can exhaust ``time_limit`` at genome
    scale, in which case it reduces rather than eliminates that spread and raises a
    warning. Baseline and ``resolve_ties`` on identical inputs already differ by ~10% of
    essential-gene calls, both fully score-optimal — when comparing models before and
    after a curation, apply the edit to the extracted model as a control, not only to the
    template.

    ``verbose`` prints one line per staged step and forwards to :func:`fill_tasks` for
    per-task gap-fill progress (matching RAVEN ``ftINIT``'s console report); silent by
    default.

    A stability-anchoring parameter (``reference_reactions``, biasing a re-extraction
    toward a prior build's reaction choices) was implemented and tested against a real
    curation here; it helped on one cell line and caused a 5x *increase* in spurious
    essential-gene drift on another, traced to a network-topology regime swap the
    reference-matching objective cannot see coming, and was removed. See the
    `reference_reactions postmortem
    <https://github.com/CompMetEng/raven-docs/blob/main/docs/parameter-tuning/studies/ftinit-reference-reactions.md>`_
    on raven-docs for the full account.
    """
    if rxn_scores is not None:
        if hpa is not None or transcr is not None:
            raise ValueError("pass either rxn_scores or hpa/transcr, not both")
    else:
        if tissue is None:
            raise ValueError("tissue is required when scoring from hpa/transcr")
        rxn_scores, gene_scores = score_for_init(
            prep.ref_model, hpa, transcr, tissue, celltype, no_gene_score=no_gene_score,
            isozyme_scoring=isozyme_scoring, complex_scoring=complex_scoring,
            multiple_cell_scoring=multiple_cell_scoring, hpa_level_scores=hpa_level_scores,
            data_precedence=data_precedence)
    steps = steps if steps is not None else get_init_steps(series)
    min_model, group_of = prep.min_model, prep.group_of
    met_producers = (_metabolomics_producers(prep, metabolomics)
                     if metabolomics is not None else None)

    # RAVEN nudges tiny reaction scores off zero once, before per-step grouping.
    rxn_scores = _nudge_scores(rxn_scores)

    turned_on: dict[str, float] = {}   # merged reaction id -> flux (accumulated)
    left_in: set[str] = set()          # merged reactions with score 0 in the last step
    # RAVEN seeds every reaction's "carried flux" at force_on (0.1) and updates it after
    # each step; an essential reaction is forced at min(0.99·|carried flux|, force_on), so
    # it is never forced above what it last carried (ftINIT.m:172,248) — this applies to
    # the permanent (prep) essentials too, not only reactions turned on by a prior step.
    flux_of: dict[str, float] = {r.id: force_on for r in min_model.reactions}
    # resolve_ties ranks a merged reaction by its group's smallest original id, which does
    # not depend on which member names the merged reaction.
    smallest: dict[int, str] = {}
    for rid, gid in zip(prep.orig_rxn_ids, prep.group_ids, strict=True):
        if gid and (gid not in smallest or rid < smallest[gid]):
            smallest[gid] = rid
    rank_keys = {r.id: smallest[group_of[r.id]] for r in min_model.reactions
                 if group_of.get(r.id, 0)}
    n_steps = len(steps)
    for i, step in enumerate(steps):
        if verbose:
            print(f"[ftinit] step {i + 1}/{n_steps}", flush=True)
        to_zero = prep.masks.ignored(step.ignore_mask)
        scores = group_rxn_scores(min_model, rxn_scores, prep.orig_rxn_ids,
                                  prep.group_ids, to_zero)
        essential = set(prep.essential_rxns)  # pre-oriented forward (default direction)
        directions: dict[str, int] = {}
        if step.how_to_use_prev == "essential":
            for rid, flux in turned_on.items():
                essential.add(rid)
                directions[rid] = 1 if flux >= 0 else -1
        ess_force = {rid: min(abs(flux_of.get(rid, force_on)) * 0.99, force_on)
                     for rid in essential}
        res = _solve_step(
            min_model, scores, step, essential=essential, directions=directions,
            ess_force=ess_force, force_on=force_on, big_m=big_m,
            mip_gap=mip_gap, mip_gap_abs=mip_gap_abs, time_limit=time_limit,
            prove_abs_gap=prove_abs_gap, resolve_ties=resolve_ties, rank_keys=rank_keys,
            metabolomics=met_producers, prod_weight=prod_weight,
            seed=seed, threads=threads, feas_tol=feas_tol, opt_tol=opt_tol,
            int_feas_tol=int_feas_tol, presolve=presolve,
        )
        if res.status == "time_limit":
            # The step ran out of wall clock before proving its gap, so the kept set is
            # whichever incumbent the search happened to hold — not a proven optimum. It
            # is accepted (RAVEN does the same), but silently accepting it is what makes
            # a rebuild look like a model change, so say so.
            warnings.warn(
                f"ftINIT step {i} hit the {time_limit}s time limit with an unproven "
                f"MIP gap ({res.achieved_gap}); its kept reactions are an arbitrary "
                "incumbent among the near-optimal solutions and may differ across "
                "solver builds. Raise time_limit, or use resolve_ties=True to pin the "
                "selection. See the ftINIT reproducibility study on raven-docs "
                "(docs/parameter-tuning/studies/ftinit-determinism.md).",
                stacklevel=2,
            )
        for rid in res.on_reactions:
            turned_on[rid] = res.fluxes[rid]
        flux_of.update(res.fluxes)  # carry this step's fluxes into the next
        left_in = {rid for rid, s in scores.items() if s == 0.0}

    # Merged reactions to keep: turned on + permanently essential + left-in (score 0).
    kept_min = set(turned_on) | set(prep.essential_rxns) | left_in
    deleted_min = [r.id for r in min_model.reactions if r.id not in kept_min]

    # Map deleted merged reactions back to all originals in their groups.
    removed_groups = {group_of[rid] for rid in deleted_min if group_of[rid] != 0}
    to_remove = {o for o in prep.orig_rxn_ids if group_of[o] and group_of[o] in removed_groups}
    to_remove |= {rid for rid in deleted_min if group_of[rid] == 0}  # unmerged
    # Keep the surviving originals plus all exchange reactions (always re-added).
    final_kept = (set(prep.orig_rxn_ids) - to_remove) | prep.masks.exchange

    # ref_model carries the template's bounds and directions, so the output does too.
    out = prep.ref_model.copy()
    out.remove_reactions([r.id for r in out.reactions if r.id not in final_kept],
                         remove_orphans=True)

    if fill_gaps and prep.tasks:  # add reactions back so every task is feasible
        if verbose:
            print(f"[ftinit] gap-filling {len(prep.tasks)} task(s)", flush=True)
        # The gap-fill MILP is its own problem (RAVEN ftINITFillGaps); it uses RAVEN's
        # fixed per-task 300 s limit and seed, not the main extraction's time_limit.
        # mutate_in_place=True: `out` is this function's own disposable copy of
        # prep.ref_model, not exposed anywhere else, so fill_tasks gap-filling it directly
        # skips a second full copy of the same (genome-scale) model.
        out = fill_tasks(out, prep.ref_model, prep.tasks, rxn_scores=rxn_scores if use_scores_for_tasks else None,
                         resolve_ties=resolve_ties, mutate_in_place=True,
                         verbose=verbose).model
    # RAVEN removes the reactions with removeReactions(refModel, ..., true, true), which drops
    # every metabolite that has no reaction left, also one that had none in the reference model
    # (a metabolite a task names but the first simplification left without a reaction).
    out.remove_metabolites([m for m in out.metabolites if not m.reactions])
    # removeReactions(..., true, true) also drops every gene that has no reaction left, also
    # one that had none in the reference model (prep_init_model with simplify=False keeps those).
    for gene in [g for g in out.genes if not g.reactions]:
        out.genes.remove(gene)
    if remove_genes and gene_scores is not None:   # prune negative-scoring genes from the GPRs
        out, _ = remove_low_score_genes(out, gene_scores)
    return out
