"""Task gap-filling (fill_tasks).

Oracle: RAVEN tinitTests T0003. Remove the exchange reactions and create a gap by
deleting R7 (e[c] -> e[s]); gap-filling against the full reference must add R7 back so
the task 'make e[s] from a[s]' becomes feasible again.
"""
from tinit_oracles import make_test_model, make_test_task

from raven_toolbox.init import TaskFillResult, fill_tasks


def _reference_without_exchanges():
    """testModel with the exchange reactions (R1, R8) removed — the gap-fill template."""
    ref = make_test_model()
    ref.remove_reactions(["R1", "R8"], remove_orphans=False)
    return ref


def test_fills_the_gap_with_r7():
    ref = _reference_without_exchanges()
    gapped = ref.copy()
    gapped.remove_reactions(["R7"], remove_orphans=False)  # the gap
    res = fill_tasks(gapped, ref, [make_test_task()])
    assert isinstance(res, TaskFillResult)
    assert res.added_reactions == ["R7"]
    assert "R7" in {r.id for r in res.model.reactions}
    assert not res.failed_tasks


def test_no_fill_when_already_feasible():
    """A model that can already do the task gets no additions."""
    ref = _reference_without_exchanges()
    res = fill_tasks(ref.copy(), ref, [make_test_task()])
    assert res.added_reactions == []


def test_should_fail_tasks_ignored():
    from raven_toolbox.tasks import Task

    ref = _reference_without_exchanges()
    gapped = ref.copy()
    gapped.remove_reactions(["R7"], remove_orphans=False)
    sf = Task(id="sf", should_fail=True, outputs=[("e[s]", 1.0, 1.0)])
    res = fill_tasks(gapped, ref, [sf])
    assert res.added_reactions == []  # should_fail task drives no gap-filling


def test_open_exchange_does_not_short_circuit_gapfill():
    """Boundaries are closed during gap-filling, so an open exchange can't fake feasibility.

    Give the gapped model an open exchange on e[s]; without closing boundaries the task
    'produce e[s]' would look feasible (free secretion) and R7 would never be added.
    """
    import cobra

    ref = _reference_without_exchanges()
    gapped = ref.copy()
    gapped.remove_reactions(["R7"], remove_orphans=False)
    ex_es = cobra.Reaction("EX_es", lower_bound=-1000, upper_bound=1000)
    ex_es.add_metabolites({gapped.metabolites.es: -1})
    gapped.add_reactions([ex_es])  # open exchange that must be ignored
    res = fill_tasks(gapped, ref, [make_test_task()])
    assert "R7" in res.added_reactions  # gap still detected and filled


def test_prefers_cheaper_reactions_by_score():
    """When two candidates can fill a gap, the higher-scored (cheaper) one is chosen.

    Build a gap that R7 (e[c]->e[s]) OR an alternative ALT (e[c]->e[s]) can fill; give
    ALT a much better score so it is preferred.
    """
    import cobra

    ref = _reference_without_exchanges()
    alt = cobra.Reaction("ALT", lower_bound=0, upper_bound=1000)
    alt.add_metabolites({ref.metabolites.ec: -1, ref.metabolites.es: 1})  # same as R7
    alt.gene_reaction_rule = "gALT"
    ref.add_reactions([alt])
    gapped = ref.copy()
    gapped.remove_reactions(["R7", "ALT"], remove_orphans=False)
    # ALT scored high (cost low), R7 scored low (cost high) → ALT chosen.
    res = fill_tasks(gapped, ref, [make_test_task()], rxn_scores={"ALT": 5.0, "R7": -3.0})
    assert res.added_reactions == ["ALT"]


# --------------------------------------------------------------------------- #
# Regression tests for gap-fill reaction construction: reactions are reconstructed
# rather than deep-copied (cobra Reaction.copy() recurses without bound on
# genome-scale models), candidates are never copied into the model, and
# un-fillable tasks are surfaced.
# --------------------------------------------------------------------------- #
def test_gap_filled_reaction_keeps_its_gpr():
    """A gap-filled reaction retains its GPR, stoichiometry and bounds.

    The gap-fill reconstructs each chosen reaction (id/bounds/stoichiometry/GPR) instead
    of deep-copying it. Dropping the gene rule would make the added reaction look
    gene-less, so every gene controlling only that reaction would be silently missed by
    downstream gene-essentiality analysis.
    """
    ref = _reference_without_exchanges()
    gapped = ref.copy()
    gapped.remove_reactions(["R7"], remove_orphans=False)
    res = fill_tasks(gapped, ref, [make_test_task()])
    r7 = res.model.reactions.get_by_id("R7")
    assert r7.gene_reaction_rule == "G7"                 # gene rule preserved
    assert "G7" in {g.id for g in res.model.genes}       # and its gene is in the model
    assert {m.id: c for m, c in r7.metabolites.items()} == {"ec": -1, "es": 1}  # stoichiometry
    assert (r7.lower_bound, r7.upper_bound) == (0, 1000)  # bounds


def test_add_reference_reactions_recreates_missing_metabolites():
    """``_add_reference_reactions`` creates metabolites a chosen reaction needs but the
    target lacks, and reconstructs the reaction without cobra's recursive deep-copy."""
    from raven_toolbox.init.taskfill import _add_reference_reactions

    ref = make_test_model()                                   # full model: R7 (ec->es, G7)
    target = make_test_model()
    target.remove_reactions(["R7", "R8"], remove_orphans=True)  # e[s] now orphaned → removed
    assert "es" not in {m.id for m in target.metabolites}

    _add_reference_reactions(target, ref, ["R7"])

    assert "es" in {m.id for m in target.metabolites}         # metabolite recreated
    r7 = target.reactions.get_by_id("R7")
    assert r7.gene_reaction_rule == "G7"
    assert {m.id: c for m, c in r7.metabolites.items()} == {"ec": -1, "es": 1}
    assert (r7.lower_bound, r7.upper_bound) == (0, 1000)


def test_unfillable_task_is_reported_and_warns():
    """A task the reference cannot satisfy is returned in ``failed_tasks`` and warned
    about, not silently dropped."""
    import pytest

    ref = _reference_without_exchanges()
    ref.remove_reactions(["R7"], remove_orphans=False)  # reference itself cannot make e[s]
    gapped = ref.copy()
    with pytest.warns(UserWarning, match="could not be gap-filled"):
        res = fill_tasks(gapped, ref, [make_test_task()])
    assert res.failed_tasks == ["Gen e[s] from a[s]"]
    assert res.added_reactions == []


def test_additions_carry_forward_to_later_tasks():
    """A reaction added for one task is present for later tasks (model carried forward),
    so a second task with the same requirement needs no further additions."""
    import math

    from raven_toolbox.tasks import Task

    ref = _reference_without_exchanges()
    gapped = ref.copy()
    gapped.remove_reactions(["R7"], remove_orphans=False)
    first = make_test_task()
    second = Task(id="second", inputs=[("a[s]", 0.0, math.inf)], outputs=[("e[s]", 1.0, 1.0)])
    res = fill_tasks(gapped, ref, [first, second])
    assert res.added_reactions == ["R7"]  # added once; the second task saw R7 already there
    assert not res.failed_tasks


def test_resolve_ties_gap_fill_breaks_tie_by_id():
    """Two equal-cost candidate fills: resolve_ties adds the lower-id one deterministically."""
    import cobra

    from raven_toolbox.tasks import Task

    ref = cobra.Model("ref")
    A, M, P = (cobra.Metabolite(x, name=x, compartment="s") for x in ("A", "M", "P"))
    ref.add_metabolites([A, M, P])
    RA = cobra.Reaction("RA", lower_bound=0, upper_bound=1000)
    RA.add_metabolites({A: -1, M: 1})
    RB = cobra.Reaction("RB", lower_bound=0, upper_bound=1000)
    RB.add_metabolites({A: -1, M: 1})
    RP = cobra.Reaction("RP", lower_bound=0, upper_bound=1000)
    RP.add_metabolites({M: -1, P: 1})
    ref.add_reactions([RA, RB, RP])
    task = Task(id="mkP", inputs=[("A[s]", 0.0, 1000.0)], outputs=[("P[s]", 1.0, 1000.0)])

    gapped = ref.copy()
    gapped.remove_reactions(["RA", "RB"], remove_orphans=False)  # both routes to M removed
    # exactly one equal-cost route is needed; resolve_ties must pick the lower id (RA).
    res = fill_tasks(gapped, ref, [task], resolve_ties=True)
    assert res.added_reactions == ["RA"]
    assert not res.failed_tasks


def test_reference_model_is_unchanged_and_reusable_after_gap_fill():
    """reference_model is prep.ref_model in real use -- shared and reused across every
    sample built from the same PrepData. _gap_fill_task mutates it in place instead of
    copying it (for speed on a genome-scale model), so it must come back exactly as it
    went in: no leaked _fill_* variables/constraints, no changed reaction bounds or
    objective, no leaked Gurobi solver parameters -- and still gap-fill correctly a
    second time, proving nothing from the first call stuck around to corrupt it."""
    import importlib

    import pytest

    try:  # real import, not find_spec: optlang ships the module even without gurobipy
        importlib.import_module("optlang.gurobi_interface")
    except ImportError:
        pytest.skip("gurobi is not installed; this test pins Gurobi-specific solver state")

    ref = _reference_without_exchanges()
    ref.solver = "gurobi"  # the state this test pins (.problem.Params) is Gurobi-specific
    ref.solver.problem.Params.Threads = 3  # distinct from _set_fill_solver's own Threads=1

    bounds_before = {r.id: r.bounds for r in ref.reactions}
    n_constraints_before = len(ref.constraints)
    n_variables_before = len(ref.variables)
    objective_before = str(ref.objective.expression)

    gapped = ref.copy()
    gapped.remove_reactions(["R7"], remove_orphans=False)
    res = fill_tasks(gapped, ref, [make_test_task()])
    assert res.added_reactions == ["R7"]

    assert {r.id: r.bounds for r in ref.reactions} == bounds_before
    assert len(ref.constraints) == n_constraints_before
    assert len(ref.variables) == n_variables_before
    assert str(ref.objective.expression) == objective_before
    assert ref.solver.problem.Params.Threads == 3

    # ref must still work correctly a second time.
    gapped_again = ref.copy()
    gapped_again.remove_reactions(["R7"], remove_orphans=False)
    res2 = fill_tasks(gapped_again, ref, [make_test_task()])
    assert res2.added_reactions == ["R7"]
    assert not res2.failed_tasks


def test_resolve_ties_restores_the_gap_parameters_of_the_reference_model():
    """fill_tasks(resolve_ties=True) leaves MIPGap and MIPGapAbs of reference_model as they were."""
    import importlib

    import pytest

    try:
        importlib.import_module("optlang.gurobi_interface")
    except ImportError:
        pytest.skip("gurobi is not installed; this test pins Gurobi-specific solver state")

    ref = _reference_without_exchanges()
    ref.solver = "gurobi"
    ref.solver.problem.Params.MIPGap = 0.01
    ref.solver.problem.Params.MIPGapAbs = 0.25

    for _ in range(2):  # the second call starts from the state the first one left
        gapped = ref.copy()
        gapped.remove_reactions(["R7"], remove_orphans=False)
        res = fill_tasks(gapped, ref, [make_test_task()], resolve_ties=True)
        assert res.added_reactions == ["R7"]
        assert ref.solver.problem.Params.MIPGap == 0.01
        assert ref.solver.problem.Params.MIPGapAbs == 0.25
