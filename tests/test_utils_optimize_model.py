"""Tests for optimize_model."""
import math

import cobra
import pytest

from raven_toolbox.utils.solver import optimize_model


def _infeasible_model(solver: str) -> cobra.Model:
    model = cobra.Model("m")
    a = cobra.Metabolite("A", compartment="c")
    rxn = cobra.Reaction("R1")
    rxn.add_metabolites({a: 1.0})
    rxn.bounds = (0.0, 10.0)
    sink = cobra.Reaction("S1")
    sink.add_metabolites({a: -1.0})
    sink.bounds = (20.0, 30.0)
    model.add_reactions([rxn, sink])
    model.objective = "R1"
    model.solver = solver
    return model


@pytest.mark.parametrize("solver", ["glpk", "scipy"])
def test_infeasible_lp_gives_an_infeasible_solution(solver):
    solution = optimize_model(_infeasible_model(solver))
    assert solution.status == "infeasible"


def test_feasible_lp_matches_model_optimize():
    model = _infeasible_model("glpk")
    model.reactions.get_by_id("S1").bounds = (0.0, 5.0)
    solution = optimize_model(model)
    assert solution.status == "optimal"
    assert solution.objective_value == pytest.approx(5.0)


def test_gurobi_presolved_infeasible_lp_gives_an_infeasible_solution():
    """Gurobi holds no primal values for an infeasible LP that it presolved, which makes
    ``model.optimize()`` raise ``GurobiError``."""
    try:
        model = _infeasible_model("gurobi")
    except Exception as exc:
        pytest.skip(f"gurobi unusable: {exc}")
    model.solver.problem.Params.Presolve = 2
    solution = optimize_model(model)
    assert solution.status == "infeasible"
    assert math.isnan(solution.objective_value)
    assert set(solution.fluxes.index) == {"R1", "S1"}


def test_other_errors_are_raised():
    class Broken:
        class solver:
            status = "optimal"

        def optimize(self, **kwargs):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        optimize_model(Broken())
