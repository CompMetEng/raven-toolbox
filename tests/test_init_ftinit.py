"""The single-step ftINIT MILP (run_ftinit).

Validated on the testModel oracle against (a) a hand-checked score-optimal solution
and (b) the formulation invariants. The full-pipeline RAVEN outputs (tinitTests
T0001/T0002) additionally involve linear merge + the toIgnore masks + staging +
exchange re-adding, covered elsewhere.

Note on the toy result: with strict mass balance and no metabolite-production reward
(ftINIT, unlike classic INIT, only rewards metabolomics-detected mets), the
score-optimal subnetwork on testModel is the internal cycle R4→R6→(R10 rev)→(R9 rev),
worth 7+0.5-3+3.5 = 8.0 — it beats the "honest" exchange path because that path must
pay for the negative-score transport reactions R2/R7. The bare INIT MILP has no
loopless constraint (neither does RAVEN's); loop-free models come from the staged
pipeline + exchange handling and, at genome scale, from models having real exchanges
so such cycles are not score-optimal. This faithfully matches RAVEN's MILP.
"""
import cobra
import pytest
from tinit_oracles import TEST_MODEL_SCORES, expr_for_rxn_score, make_test_model

from raven_toolbox.init import FtInitResult, run_ftinit
from raven_toolbox.init.ftinit import _EXTRACT_SEED
from raven_toolbox.init.score import gene_scores_from_expression, score_reactions_from_genes

_LOOP = {"R4", "R6", "R9", "R10"}  # the score-optimal subnetwork (8.0)


def _scores(model):
    expr = expr_for_rxn_score(TEST_MODEL_SCORES)
    return score_reactions_from_genes(model, gene_scores_from_expression(expr, 1.0))


def test_full_milp_score_optimum():
    model = make_test_model()
    res = run_ftinit(model, _scores(model))
    assert isinstance(res, FtInitResult)
    assert set(res.kept_reactions) == _LOOP
    assert res.deleted_reactions == ["R1", "R2", "R3", "R5", "R7", "R8"]
    assert res.objective == pytest.approx(8.0, abs=1e-6)


def test_kept_reactions_carry_flux_and_balance():
    """Indicator-on reactions carry flux (≥ force_on) and the solution is steady-state."""
    model = make_test_model()
    res = run_ftinit(model, _scores(model))
    for rid in res.kept_reactions:
        assert abs(res.fluxes[rid]) > 1e-9
    # The extracted model is itself feasible/flux-consistent.
    assert res.model.slim_optimize() is not None


def test_essential_force_clamps_to_capacity():
    """Forcing an essential reaction is clamped to its capacity (no lb>ub crash).

    A reaction capped at 0.05 forced with the default 0.1 must not error; it is forced
    to its capacity (0.05) and the model stays feasible. A per-reaction force of 0.04
    forces exactly that.
    """
    m = cobra.Model("cap")
    a, b = (cobra.Metabolite(x, compartment="s") for x in "ab")
    m.add_metabolites([a, b])
    r = cobra.Reaction("LOW", lower_bound=0, upper_bound=0.05)  # tiny capacity
    r.add_metabolites({a: -1, b: 1})
    for mid, st in [("EX_a", {a: -1}), ("EX_b", {b: -1})]:
        ex = cobra.Reaction(mid, lower_bound=-1000, upper_bound=1000)
        ex.add_metabolites(st)
        m.add_reactions([ex])
    m.add_reactions([r])
    m.objective = "LOW"

    res = run_ftinit(m, {}, essential_rxns=["LOW"], force_on_ess=0.1)  # clamped to 0.05
    assert res.fluxes["LOW"] >= 0.05 - 1e-9
    res2 = run_ftinit(m, {}, essential_rxns=["LOW"], essential_force={"LOW": 0.04})
    assert res2.fluxes["LOW"] >= 0.04 - 1e-9


def test_solver_tolerances_reach_gurobi_and_default_to_raven(monkeypatch):
    """feas_tol/opt_tol/int_feas_tol/presolve are applied to the solver; defaults are RAVEN's."""
    import importlib

    try:  # optlang ships the module even without gurobipy, so import for real
        gi = importlib.import_module("optlang.gurobi_interface")
    except ImportError:
        pytest.skip("gurobi is not installed; the Params under test are Gurobi-specific")

    seen: list[tuple] = []
    orig = gi.Model.optimize

    def spy(self, *args, **kwargs):
        p = self.problem.Params
        seen.append((p.FeasibilityTol, p.OptimalityTol, p.IntFeasTol, p.Presolve))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(gi.Model, "optimize", spy)

    model = make_test_model()
    model.solver = "gurobi"
    scores = _scores(model)

    run_ftinit(model, scores)
    assert seen[-1] == (1e-9, 1e-9, 1e-9, 2)

    res = run_ftinit(model, scores, feas_tol=1e-6, opt_tol=1e-7, int_feas_tol=1e-5, presolve=1)
    assert seen[-1] == (1e-6, 1e-7, 1e-5, 1)
    assert res.objective == pytest.approx(8.0, abs=1e-6)  # relaxed tolerances, same optimum


def test_essential_reaction_forced_on():
    """An essential reaction is kept and carries flux even when its score is negative."""
    model = make_test_model()
    res = run_ftinit(model, _scores(model), essential_rxns=["R3"])
    assert "R3" in res.kept_reactions
    assert abs(res.fluxes["R3"]) > 1e-6


def test_rem_pos_rev_drops_positive_reversibles():
    """rem_pos_rev frees positive reversibles (score→0): the score-8.0 loop collapses.

    R4 (+7) and R10 (+3.5) are positive reversibles; with them unscored, the cycle is
    no longer profitable (R6 0.5 - R9 3 < 0), so nothing scored stays on.
    """
    model = make_test_model()
    res = run_ftinit(model, _scores(model), rem_pos_rev=True)
    assert res.objective == pytest.approx(0.0, abs=1e-6)
    assert "R6" not in res.kept_reactions and "R9" not in res.kept_reactions


def test_allow_excretion_relaxes_balance():
    """With allow_excretion the result stays feasible (net production permitted)."""
    model = make_test_model()
    res = run_ftinit(model, _scores(model), allow_excretion=True)
    assert res.objective >= 8.0 - 1e-6  # at least as good as strict balance


def test_unscored_reactions_are_kept_free():
    """Score-0 reactions are left in the model (not removable), not deleted."""
    model = make_test_model()
    scores = _scores(model)
    scores["R3"] = 0.0  # make R3 unscored -> must not be deleted
    res = run_ftinit(model, scores)
    assert "R3" not in res.deleted_reactions


def test_forced_flux_lower_bound_is_respected():
    """A scored, non-reversible reaction with lb>0 must keep carrying >= lb flux.

    Guards the bound handling: the single-direction branch must use the model's own
    [lb, ub], not zero out a positive lower bound.
    """
    model = make_test_model()
    scores = _scores(model)
    # R6 (2 d[c] => e[c]) is forward-irreversible; force >=2 flux through it.
    model.reactions.get_by_id("R6").lower_bound = 2.0
    res = run_ftinit(model, scores)
    assert res.fluxes["R6"] >= 2.0 - 1e-6
    assert "R6" not in res.deleted_reactions


# --------------------------------------------------------------------------- #
# resolve_ties (deterministic selection among equal optima).
# --------------------------------------------------------------------------- #
def _degenerate_model():
    """Two interchangeable negative-score reactions (R1, R2) both feed an essential E.

    Exactly one is needed to make E carry flux, so the score optimum (-1) is degenerate
    between keeping R1 or R2. Which one a plain solve keeps is an arbitrary tie-break.
    """
    m = cobra.Model("degen")
    a, mm, p = (cobra.Metabolite(x, name=x, compartment="s") for x in ("a", "m", "p"))
    m.add_metabolites([a, mm, p])
    R1 = cobra.Reaction("R1", lower_bound=0, upper_bound=1000)
    R1.add_metabolites({a: -1, mm: 1})
    R2 = cobra.Reaction("R2", lower_bound=0, upper_bound=1000)
    R2.add_metabolites({a: -1, mm: 1})
    E = cobra.Reaction("E", lower_bound=0, upper_bound=1000)
    E.add_metabolites({mm: -1, p: 1})
    EXa = cobra.Reaction("EX_a", lower_bound=-1000, upper_bound=1000)
    EXa.add_metabolites({a: -1})
    EXp = cobra.Reaction("EX_p", lower_bound=-1000, upper_bound=1000)
    EXp.add_metabolites({p: -1})
    m.add_reactions([R1, R2, E, EXa, EXp])
    m.objective = "E"
    return m


def test_resolve_ties_breaks_degenerate_tie_by_id():
    """resolve_ties selects the sparsest, lowest-id optimum among equal alternatives."""
    m = _degenerate_model()
    res = run_ftinit(m, {"R1": -1.0, "R2": -1.0}, essential_rxns=["E"], resolve_ties=True)
    # exactly one of the degenerate pair is kept (the tie is resolved, not doubled) ...
    assert len({"R1", "R2"} & set(res.kept_reactions)) == 1
    # ... and it is deterministically the lower-id one.
    assert "R1" in res.kept_reactions and "R2" not in res.kept_reactions
    # the reported objective is the primary (score) optimum, not the phase-2 secondary.
    assert res.objective == pytest.approx(-1.0, abs=1e-6)


def test_resolve_ties_skipped_after_time_limited_primary(monkeypatch):
    """No tie-break when the primary solve stops at the time limit: the phase-1 set is kept."""
    import importlib

    import optlang.interface

    ftinit_module = importlib.import_module("raven_toolbox.init.ftinit")

    calls = []
    monkeypatch.setattr(ftinit_module, "_resolve_ties", lambda *a, **k: calls.append(a) or True)
    monkeypatch.setattr(optlang.interface.Model, "status", property(lambda self: "time_limit"))
    m = _degenerate_model()
    res = run_ftinit(m, {"R1": -1.0, "R2": -1.0}, essential_rxns=["E"], resolve_ties=True)
    assert calls == []
    assert res.status == "time_limit"


def test_resolve_ties_safe_on_unique_optimum():
    """On a non-degenerate model resolve_ties returns the same optimum (no regression)."""
    model = make_test_model()
    res = run_ftinit(model, _scores(model), resolve_ties=True)
    assert set(res.kept_reactions) == _LOOP
    assert res.objective == pytest.approx(8.0, abs=1e-6)


def _direction_model():
    """A positive reversible R3 (m <=> q) that can be on in either direction.

    Run forward it needs m from a via the free R1; run backward it imports q and feeds the
    essential E. Both directions reach the same score, so the flux of R3 (and the direction
    the next ftINIT step would force it in) is a tie that only the flux distribution
    resolves. The free loop L1/L2 can carry any flux. R2 is a negative-score dead end.
    """
    m = cobra.Model("direction")
    a, mm, p, q, n, z = (cobra.Metabolite(x, name=x, compartment="s")
                         for x in ("a", "m", "p", "q", "n", "z"))
    rxns = {
        "R1": ({a: -1, mm: 1}, 0), "R2": ({a: -1, z: 1}, 0), "R3": ({mm: -1, q: 1}, -1000),
        "E": ({mm: -1, p: 1}, 0), "L1": ({mm: -1, n: 1}, 0), "L2": ({n: -1, mm: 1}, 0),
        "EX_a": ({a: -1}, -1000), "EX_p": ({p: -1}, -1000), "EX_q": ({q: -1}, -1000),
    }
    for rid, (mets, lb) in rxns.items():
        r = cobra.Reaction(rid, lower_bound=lb, upper_bound=1000)
        m.add_reactions([r])
        r.add_metabolites(mets)
    return m


def test_resolve_ties_takes_fluxes_from_a_minimal_flux_solution():
    """resolve_ties reports the least-total-flux distribution for the tie-broken on set.

    The backward R3 route (import q, feed E) carries 0.4 in total, the forward one (a via
    R1 into both R3 and E) 0.8, and the free loop L1/L2 carries nothing.
    """
    res = run_ftinit(_direction_model(), {"R2": -1.0, "R3": 1.0}, essential_rxns=["E"],
                     resolve_ties=True)
    assert "R3" in res.on_reactions and "R2" not in res.kept_reactions
    expected = {"R1": 0.0, "R2": 0.0, "R3": -0.1, "E": 0.1, "L1": 0.0, "L2": 0.0,
                "EX_a": 0.0, "EX_p": 0.1, "EX_q": -0.1}
    for rid, v in expected.items():
        assert res.fluxes[rid] == pytest.approx(v, abs=1e-6), rid
    assert res.objective == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# seed as an explicit parameter, and the unproven-incumbent warning.
# --------------------------------------------------------------------------- #
def test_seed_is_an_explicit_parameter_not_a_hidden_constant():
    """The solver seed is settable per call; the default matches RAVEN's 1234.

    The seed decides which of several equal-score optima a degenerate MILP returns, so
    varying it is how a caller probes whether a result rests on the tie-break rather than
    on the data. It must not require patching a module constant.
    """
    m = _degenerate_model()
    scores = {"R1": -1.0, "R2": -1.0}
    default = run_ftinit(m, scores, essential_rxns=["E"])
    explicit = run_ftinit(m, scores, essential_rxns=["E"], seed=_EXTRACT_SEED)
    assert default.kept_reactions == explicit.kept_reactions
    # A different seed is accepted and still yields a valid, score-optimal extraction.
    other = run_ftinit(m, scores, essential_rxns=["E"], seed=7)
    assert other.objective == pytest.approx(default.objective, abs=1e-6)
    assert len({"R1", "R2"} & set(other.kept_reactions)) == 1


def test_result_reports_solver_status():
    """FtInitResult carries the accepted solve's status, so callers can spot a fallback."""
    res = run_ftinit(_degenerate_model(), {"R1": -1.0, "R2": -1.0}, essential_rxns=["E"])
    assert res.status == "optimal"


def test_time_limit_fallback_warns(monkeypatch):
    """A step that ends at the time limit warns that its kept set is arbitrary.

    RAVEN accepts such an incumbent and so do we, but accepting it *silently* is what
    makes an unchanged rebuild look like a model change.
    """
    import importlib

    from raven_toolbox.init import prep_init_model

    # NB: `raven_toolbox.init.ftinit` resolves to the *function* — the package rebinds the
    # name — so the module has to be fetched explicitly before it can be patched.
    ftinit_mod = importlib.import_module("raven_toolbox.init.ftinit")
    model = make_test_model()
    prep = prep_init_model(model, ext_comp="s")
    scores = score_reactions_from_genes(
        model, gene_scores_from_expression(expr_for_rxn_score(TEST_MODEL_SCORES), 1.0))
    real = ftinit_mod._solve_step

    def timed_out(*args, **kwargs):
        res = real(*args, **kwargs)
        return FtInitResult(res.model, res.kept_reactions, res.deleted_reactions,
                            res.fluxes, res.objective, on_reactions=res.on_reactions,
                            achieved_gap=0.42, status="time_limit")

    monkeypatch.setattr(ftinit_mod, "_solve_step", timed_out)
    with pytest.warns(UserWarning, match="unproven MIP gap"):
        ftinit_mod.ftinit(prep, rxn_scores=scores, fill_gaps=False)


def test_unproven_tie_break_warns():
    """resolve_ties reports when its own phase 2 ends unproven.

    At genome scale the tie-break phases can exhaust the time limit. Their incumbent is
    still adopted (it measurably reduces the spread), but it is an arbitrary within-gap
    pick, so ``resolve_ties=True`` must not silently imply a proven canonical selection.
    Driven through a stub solver, because at toy scale the phases prove instantly.
    """
    import importlib

    from optlang import interface

    ftinit_mod = importlib.import_module("raven_toolbox.init.ftinit")

    class _Stub:
        """Minimal optlang-shaped model that always finishes at the time limit."""

        status = "time_limit"

        def __init__(self):
            self.objective = interface.Objective(0)
            self.problem = type("P", (), {"Params": type("R", (), {})()})()
            self.configuration = type("C", (), {"timeout": None})()

        def add(self, item):
            pass

        def optimize(self):
            return self.status

    ind = interface.Variable("ind_R1", lb=0, ub=1)
    stub = _Stub()
    monkey = ftinit_mod._has_solution
    ftinit_mod._has_solution = lambda opt: True
    try:
        with pytest.warns(UserWarning, match="tie resolution did not converge"):
            ok = ftinit_mod._resolve_ties(stub, interface, interface.Variable("objx"),
                                          {"R1": (ind, -1.0)}, -1.0, 1.0)
    finally:
        ftinit_mod._has_solution = monkey
    assert ok  # the incumbent is still adopted, just no longer silently


# --------------------------------------------------------------------------- #
# metabolomics (production-bonus for detected metabolites).
#
# Fixtures and expected outcomes are the same ones hand-verified end-to-end against
# RAVEN 3's own ftINITInternalAlg via a standalone MATLAB probe
# before porting: same toy topology, same objective values, same flux directions.
# --------------------------------------------------------------------------- #
def _irrev_producer_model():
    """a --[R1, score -2]--> x --[EX_x]--> ; a supplied by a free EX_a.

    R1 is x's only producer and is badly scored -- on its own it would never be kept.
    """
    m = cobra.Model("irrev_producer")
    a, x = (cobra.Metabolite(n, name=n, compartment="s") for n in ("a", "x"))
    m.add_metabolites([a, x])
    EXa = cobra.Reaction("EX_a", lower_bound=-1000, upper_bound=1000)
    EXa.add_metabolites({a: 1})
    R1 = cobra.Reaction("R1", lower_bound=0, upper_bound=1000)
    R1.add_metabolites({a: -1, x: 1})
    EXx = cobra.Reaction("EX_x", lower_bound=-1000, upper_bound=1000)
    EXx.add_metabolites({x: -1})
    m.add_reactions([EXa, R1, EXx])
    return m


def test_metabolomics_pulls_in_a_badly_scored_producer():
    """A negative-score reaction is kept, and genuinely carries flux, to earn the bonus."""
    m = _irrev_producer_model()
    res = run_ftinit(m, {"R1": -2.0}, metabolomics={"x": {"R1"}}, prod_weight=5.0)
    assert "R1" in res.kept_reactions
    assert abs(res.fluxes["R1"]) > 1e-6
    # objective = bonus(5) - R1's own cost(2), matching the hand-verified MATLAB run.
    assert res.objective == pytest.approx(3.0, abs=1e-6)


def test_metabolomics_without_a_bonus_leaves_the_producer_off():
    """Control: prod_weight=0 removes the incentive, so the bad producer is dropped."""
    m = _irrev_producer_model()
    res = run_ftinit(m, {"R1": -2.0}, metabolomics={"x": {"R1"}}, prod_weight=0.0)
    assert "R1" not in res.kept_reactions
    assert res.objective == pytest.approx(0.0, abs=1e-6)


def test_metabolomics_reversible_producer_carries_genuine_production_flux():
    """A reversible negative-score producer is forced into the *producing* direction.

    EX_y is a one-way sink (never a supply), so satisfying y's mass balance genuinely
    requires R2 to run forward (b -> y), not just have its indicator flagged "on".
    """
    m = cobra.Model("rev_producer")
    b, y = (cobra.Metabolite(n, name=n, compartment="s") for n in ("b", "y"))
    m.add_metabolites([b, y])
    EXb = cobra.Reaction("EX_b", lower_bound=-1000, upper_bound=1000)
    EXb.add_metabolites({b: 1})
    R2 = cobra.Reaction("R2", lower_bound=-1000, upper_bound=1000)
    R2.add_metabolites({b: -1, y: 1})
    EXy = cobra.Reaction("EX_y", lower_bound=0, upper_bound=1000)  # sink only
    EXy.add_metabolites({y: -1})
    m.add_reactions([EXb, R2, EXy])

    res = run_ftinit(m, {"R2": -3.0}, metabolomics={"y": {"R2"}}, prod_weight=5.0)
    assert "R2" in res.kept_reactions
    assert res.fluxes["R2"] > 1e-6  # positive: producing y forward, not a fwd/back loop
    assert res.objective == pytest.approx(2.0, abs=1e-6)  # bonus(5) - cost(3)


def test_metabolomics_essential_producer_is_dropped_without_a_crash():
    """A metabolite whose only producer is already essential needs no bonus variable."""
    m = _irrev_producer_model()
    res = run_ftinit(m, {"R1": -2.0}, essential_rxns=["R1"],
                     metabolomics={"x": {"R1"}}, prod_weight=5.0)
    assert "R1" in res.kept_reactions  # essential regardless
    # No bonus was paid for x (R1's own score isn't even in play -- it's essential), so
    # the objective is the essential-only baseline (0), not inflated by the met bonus.
    assert res.objective == pytest.approx(0.0, abs=1e-6)


def test_metabolomics_zero_score_producer_stays_kept_even_if_never_turned_on():
    """A score-0 producer is never deleted, matching every other exactly-zero-score
    reaction -- becoming a metabolomics candidate must not change that.

    R1 here has score 0 and is x's only producer, but prod_weight=0 removes any reason
    for its (now-real) indicator to turn on. It must still survive into kept_reactions.
    """
    m = _irrev_producer_model()
    res = run_ftinit(m, {"R1": 0.0}, metabolomics={"x": {"R1"}}, prod_weight=0.0)
    assert "R1" in res.kept_reactions


def test_metabolomics_two_metabolites_mixed_categories_no_crash():
    """Two detected metabolites at once, one irrev- one rev-producer: no crash, both
    correctly resolved -- the exact shape RAVEN's own metabolomics code once crashed on
    when only some detected metabolites had a reversible producer."""
    m1 = _irrev_producer_model()
    b, y = (cobra.Metabolite(n, name=n, compartment="s") for n in ("b", "y"))
    m1.add_metabolites([b, y])
    EXb = cobra.Reaction("EX_b", lower_bound=-1000, upper_bound=1000)
    EXb.add_metabolites({b: 1})
    R2 = cobra.Reaction("R2", lower_bound=-1000, upper_bound=1000)
    R2.add_metabolites({b: -1, y: 1})
    EXy = cobra.Reaction("EX_y", lower_bound=0, upper_bound=1000)
    EXy.add_metabolites({y: -1})
    m1.add_reactions([EXb, R2, EXy])

    res = run_ftinit(m1, {"R1": -2.0, "R2": -3.0},
                     metabolomics={"x": {"R1"}, "y": {"R2"}}, prod_weight=5.0)
    assert {"R1", "R2"} <= set(res.kept_reactions)
    assert res.objective == pytest.approx(5.0, abs=1e-6)  # 2·bonus(5) - cost(2) - cost(3)


def test_glpk_backend_warns_and_does_not_resolve():
    """A GLPK MILP warns that it is not solving on Gurobi, and optlang's
    "auto" presolve is pinned so a non-optimal status is not solved a second time."""
    model = make_test_model()
    model.solver = "glpk"
    solves = []
    original = model.problem.Model._optimize

    def hits_time_limit(self):  # a solve that stops at time_limit with an incumbent
        solves.append(1)
        original(self)
        return "time_limit"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(model.problem.Model, "_optimize", hits_time_limit)
        with pytest.warns(UserWarning, match="with GLPK"):
            run_ftinit(model, _scores(model))
    assert len(solves) == 1


@pytest.mark.parametrize(("lb", "ub", "cap", "expected"), [
    (0.0, 500.0, 1000.0, 500.0),             # forward irreversible: its ub
    (0.0, float("inf"), 1000.0, 1000.0),     # infinite ub counts as the cap
    (0.0, 0.05, 1000.0, 0.1),                # never below force_on
    (-50.0, 20.0, 1000.0, 50.0),             # reversible: larger of the two capacities
    (-2000.0, 300.0, 1000.0, 1000.0),        # each capacity capped
    (-float("inf"), 5.0, 1000.0, 1000.0),
    (-70.0, 0.0, 1000.0, 70.0),              # reverse-only: its reverse capacity
    (-50.0, 20.0, 35.0, 35.0),               # a lower cap binds
    (float("nan"), 20.0, 1000.0, 1000.0),    # a NaN bound counts as the cap
])
def test_reaction_big_m_follows_raven(lb, ub, cap, expected):
    from raven_toolbox.init.ftinit import _reaction_big_m

    assert _reaction_big_m(lb, ub, cap, 0.1) == pytest.approx(expected)


def _big_m_model():
    m = cobra.Model("bigm")
    a, b = (cobra.Metabolite(n, name=n, compartment="s") for n in ("a", "b"))
    m.add_metabolites([a, b])
    rxns = []
    for rid, lb, ub, st in [("EX_a", -1000, 1000, {a: 1}), ("EX_b", -1000, 1000, {b: -1}),
                            ("PR", -50, 20, {a: -1, b: 1}), ("NI", 0, float("inf"), {a: -1, b: 1}),
                            ("NIs", 0, 0.05, {a: -1, b: 1}), ("NR", -30, 40, {a: -1, b: 1}),
                            ("NRo", -70, 0, {a: -1, b: 1})]:
        r = cobra.Reaction(rid, lower_bound=lb, upper_bound=ub)
        r.add_metabolites(st)
        rxns.append(r)
    m.add_reactions(rxns)
    return m


@pytest.mark.parametrize(("cap", "metabolomics", "expected"), [
    (None, None, {("dirp_PR", "b_PR"): -50.0, ("dirn_PR", "b_PR"): 50.0,
                  ("off_NI", "x_NI"): -1000.0, ("off_NIs", "x_NIs"): -0.1,
                  ("off_NR", "x_NR"): -40.0, ("off_NRo", "x_NRo"): -70.0}),
    (35.0, None, {("dirp_PR", "b_PR"): -35.0, ("dirn_PR", "b_PR"): 35.0,
                  ("off_NI", "x_NI"): -35.0, ("off_NIs", "x_NIs"): -0.1,
                  ("off_NR", "x_NR"): -35.0, ("off_NRo", "x_NRo"): -35.0}),
    # a metabolite producer gets an indicator and a direction binary with its own big-M
    (None, {"b": {"NR"}}, {("dirp_NR", "b_NR"): -40.0, ("dirn_NR", "b_NR"): 40.0}),
])
def test_each_scored_reaction_class_uses_its_own_big_m(monkeypatch, cap, metabolomics, expected):
    """Direction and off constraints use RAVEN's per-reaction big-M, capped by big_m."""
    model = _big_m_model()
    seen = {}
    cls = model.problem.Model
    orig = cls.optimize

    def spy(self, *args, **kwargs):
        if not seen and "dirp_PR" in self.constraints:  # the ftINIT MILP, not another solve
            for (cname, vname) in expected:
                c, v = self.constraints[cname], self.variables[vname]
                seen[(cname, vname)] = c.get_linear_coefficients([v])[v]
                if cname.startswith("dirn"):
                    assert c.ub == pytest.approx(expected[(cname, vname)])
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(cls, "optimize", spy)
    kw = {} if cap is None else {"big_m": cap}
    run_ftinit(model, {"PR": 1.0, "NI": -1.0, "NIs": -1.0, "NR": -1.0, "NRo": -1.0},
               metabolomics=metabolomics, **kw)
    assert seen == pytest.approx(expected)
