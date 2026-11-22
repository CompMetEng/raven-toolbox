"""Tests for the scipy (HiGHS) support in raven_toolbox.utils.solver.

optlang's scipy interface keeps a variable index map that goes wrong after a removal,
which makes cobra read fluxes from the wrong columns, and it raises for duals.
raven_toolbox.utils.solver repairs the first and reads the HiGHS marginals for the second.
"""
import cobra
import numpy as np
import pytest

from raven_toolbox.utils.solver import (
    edit_on_fast_solver,
    repair_scipy_problem,
    switch_to_scipy,
)


@pytest.fixture
def scipy_default():
    """cobra builds every new model on scipy while the test runs."""
    config = cobra.Configuration()
    previous = config.solver
    config.solver = "scipy"
    yield
    config.solver = previous


def _columns_are_consistent(model):
    problem = model.solver.problem
    return list(problem._variables.values()) == list(range(len(problem._variables)))


def _ring(solver):
    """A plain cobra model on ``solver`` with more variables than constraints."""
    config = cobra.Configuration()
    previous = config.solver
    config.solver = solver
    try:
        model = cobra.Model("plain")
        mets = [cobra.Metabolite(f"m{i}") for i in range(3)]
        rxns = []
        for i in range(6):
            rxn = cobra.Reaction(f"r{i}")
            rxn.bounds = (0, 10)
            rxn.add_metabolites({mets[i % 3]: 1.0, mets[(i + 1) % 3]: -1.0} if i else {mets[0]: 1.0})
            rxns.append(rxn)
        model.add_reactions(rxns)
    finally:
        config.solver = previous
    model.objective = "r5"
    return model


def test_a_plain_cobra_model_is_repaired_on_request():
    model, reference = _ring("scipy"), _ring("glpk")
    for m in (model, reference):
        m.remove_reactions(["r1", "r2"])
        m.reactions.get_by_id("r4").bounds = (0, 3)
    assert repair_scipy_problem(model)
    assert _columns_are_consistent(model)
    assert model.slim_optimize() == pytest.approx(reference.slim_optimize())
    assert model.reactions.get_by_id("r5").flux == pytest.approx(reference.reactions.get_by_id("r5").flux)


def test_repair_does_nothing_on_other_solvers():
    model = cobra.Model("m")
    model.solver = "glpk"
    assert repair_scipy_problem(model) is False


def _columns_ring(solver):
    """Three metabolites, twelve reactions: removals leave late variables with stale columns."""
    config = cobra.Configuration()
    previous = config.solver
    config.solver = solver
    try:
        model = cobra.Model("ring")
        mets = [cobra.Metabolite(f"m{i}") for i in range(3)]
        rxns = []
        for i in range(12):
            rxn = cobra.Reaction(f"r{i}")
            rxn.bounds = (0, 10 + i)
            if i == 0:
                rxn.add_metabolites({mets[0]: 1.0})
            else:
                rxn.add_metabolites({mets[i % 3]: -1.0, mets[(i + 1) % 3]: 0.5 + 0.1 * i})
            rxns.append(rxn)
        model.add_reactions(rxns)
    finally:
        config.solver = previous
    model.objective = "r11"
    return model


def test_removals_after_the_install_keep_every_column_right():
    model, reference = _columns_ring("scipy"), _columns_ring("glpk")
    assert repair_scipy_problem(model)
    for rid in ("r3", "r7", "r9", "r10"):
        model.reactions.get_by_id(rid).delete()
        reference.reactions.get_by_id(rid).delete()
    assert _columns_are_consistent(model)
    assert model.slim_optimize() == pytest.approx(reference.slim_optimize())
    solution = model.optimize()
    expected = reference.optimize()
    assert solution.objective_value == pytest.approx(expected.objective_value)
    for rid in solution.fluxes.index:
        assert model.reactions.get_by_id(rid).flux == pytest.approx(solution.fluxes[rid])


def test_a_map_that_is_already_stale_is_rebuilt_from_the_expressions():
    model, reference = _columns_ring("scipy"), _columns_ring("glpk")
    model.reactions.get_by_id("r5").delete()
    reference.reactions.get_by_id("r5").delete()
    model.solver.update()
    assert not _columns_are_consistent(model)
    assert repair_scipy_problem(model)
    assert _columns_are_consistent(model)
    assert model.slim_optimize() == pytest.approx(reference.slim_optimize())


def test_the_sparse_problem_hands_highs_the_matrix_of_the_dense_problem():
    model = _columns_ring("scipy")
    dense = model.solver.problem.A.copy()
    assert repair_scipy_problem(model)
    assert (model.solver.problem.A.toarray() == dense).all()


def test_switch_to_scipy_keeps_the_objective_and_its_direction():
    model, reference = _columns_ring("glpk"), _columns_ring("glpk")
    for m in (model, reference):
        m.objective = "r8"
        m.objective_direction = "min"
        m.reactions.get_by_id("r1").lower_bound = 2.0
    switch_to_scipy(model)
    assert model.solver.interface.__name__.endswith("scipy_interface")
    assert model.objective_direction == "min"
    assert model.slim_optimize() == pytest.approx(reference.slim_optimize())
    assert model.reactions.get_by_id("r8").flux == pytest.approx(model.slim_optimize())


@pytest.mark.parametrize("status_case", ["infeasible", "unbounded"])
def test_a_solve_without_optimum_returns_a_solution_with_its_status(status_case):
    config = cobra.Configuration()
    previous = config.solver
    config.solver = "scipy"
    try:
        model = cobra.Model("chain")
        met = cobra.Metabolite("m")
        source, sink = cobra.Reaction("source"), cobra.Reaction("sink")
        source.add_metabolites({met: 1.0})
        sink.add_metabolites({met: -1.0})
        source.bounds = (0, 10) if status_case == "infeasible" else (0, 1000.0)
        sink.bounds = (20, 30) if status_case == "infeasible" else (0, 1000.0)
        model.add_reactions([source, sink])
    finally:
        config.solver = previous
    model.objective = "sink"
    if status_case == "unbounded":
        source.upper_bound = float("inf")
        sink.upper_bound = float("inf")
    assert repair_scipy_problem(model)
    if status_case == "unbounded":
        # cobra raises for this status on every solver
        with pytest.raises(cobra.exceptions.OptimizationError, match="unbounded"):
            model.optimize(raise_error=False)
        return
    solution = model.optimize(raise_error=False)
    assert solution.status == status_case
    assert np.isnan(solution.objective_value)
    assert solution.fluxes.isna().all()


def test_removing_several_constraints_at_once_on_the_sparse_problem():
    model = _columns_ring("scipy")
    repair_scipy_problem(model)
    ids = [m.id for m in list(model.metabolites)[:3]]
    model.remove_cons_vars([model.constraints[i] for i in ids])
    model.solver.update()
    assert len(model.solver.problem._constraints) == 2 * len(model.constraints)
    assert not set(ids) & set(model.constraints)


def test_edit_on_fast_solver_returns_to_scipy_with_the_same_problem():
    model = _columns_ring("scipy")
    repair_scipy_problem(model)
    before = model.slim_optimize()
    with edit_on_fast_solver(model) as fast:
        assert fast
        assert model.solver.interface.__name__.endswith("glpk_interface")
        model.reactions.get_by_id("r1").bounds = (0.0, 5.0)
    assert model.solver.interface.__name__.endswith("scipy_interface")
    assert model.reactions.get_by_id("r1").upper_bound == 5.0
    assert model.slim_optimize() <= before
    reference = _columns_ring("glpk")
    reference.reactions.get_by_id("r1").bounds = (0.0, 5.0)
    assert model.slim_optimize() == pytest.approx(reference.slim_optimize())


def test_edit_on_fast_solver_changes_nothing_on_glpk():
    model = _columns_ring("glpk")
    with edit_on_fast_solver(model) as fast:
        assert not fast
    assert model.solver.interface.__name__.endswith("glpk_interface")


def test_model_tolerance_reaches_highs_and_survives_a_copy(monkeypatch):
    import pickle

    from scipy import optimize

    model = _ring("scipy")
    repair_scipy_problem(model)
    model.tolerance = 1e-9
    assert model.solver.configuration.tolerances.feasibility == 1e-9
    assert model.solver.configuration.tolerances.optimality == 1e-9

    seen = []
    real = optimize.linprog

    def spy(*args, **kwargs):
        seen.append(kwargs["options"])
        return real(*args, **kwargs)

    for m in (model, pickle.loads(pickle.dumps(model)), model.copy()):
        monkeypatch.setattr("raven_toolbox.utils.solver.linprog", spy)
        m.slim_optimize()
    assert len(seen) == 3
    assert all(o["primal_feasibility_tolerance"] == 1e-9 for o in seen)
    assert all(o["dual_feasibility_tolerance"] == 1e-9 for o in seen)


def test_a_solve_with_numerical_difficulties_is_repeated_at_the_default_tolerance(monkeypatch):
    from scipy import optimize

    model = _ring("scipy")
    repair_scipy_problem(model)
    expected = model.slim_optimize()
    model.tolerance = 1e-9
    real = optimize.linprog
    calls = []

    def numerical_trouble_when_tight(*args, **kwargs):
        calls.append(kwargs["options"])
        if "primal_feasibility_tolerance" in kwargs["options"]:
            return optimize.OptimizeResult(status=4, fun=None, x=None, slack=None)
        return real(*args, **kwargs)

    monkeypatch.setattr("raven_toolbox.utils.solver.linprog", numerical_trouble_when_tight)
    assert model.slim_optimize() == pytest.approx(expected)
    assert len(calls) == 2

    def always_numerical_trouble(*args, **kwargs):
        return optimize.OptimizeResult(status=4, fun=None, x=None, slack=None)

    monkeypatch.setattr("raven_toolbox.utils.solver.linprog", always_numerical_trouble)
    assert model.optimize(raise_error=False).status == "numeric"
