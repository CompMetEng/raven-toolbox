"""Tests for growth-floor MILP gap-filling (gapfilling/kumar_milp.py)."""
import cobra
import pytest

from raven_toolbox.gapfilling import KumarGapFillResult, fill_gaps_kumar_milp


def _met(mid):
    return cobra.Metabolite(mid, name=mid, compartment="c")


@pytest.fixture
def growth_gap():
    """Draft model that cannot grow because the biomass precursor B is unreachable.

    draft: EX_A (uptake of A), biomass (consumes B). Producing B requires
    r_missing: A -> B, which only the template supplies.
    """
    A, B = _met("A_c"), _met("B_c")
    draft = cobra.Model("draft")
    exa = cobra.Reaction("EX_A", lower_bound=-10, upper_bound=1000)
    exa.add_metabolites({A: -1})
    bio = cobra.Reaction("biomass", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({B: -1})
    draft.add_reactions([exa, bio])
    draft.objective = "biomass"

    template = cobra.Model("template")
    r_missing = cobra.Reaction("r_missing", lower_bound=0, upper_bound=1000)
    r_missing.add_metabolites({_met("A_c"): -1, _met("B_c"): 1})
    template.add_reactions([r_missing])

    return draft, template


@pytest.fixture
def directionality_gap():
    """Draft where a reaction is irreversible in the wrong direction.

    Chain: EX_B (lb<0, uptake of B) and then we need B -> A (reverse of r1).
    r1 is currently forward-only (A -> B, lb=0).
    biomass needs A.
    """
    A, B = _met("A_c"), _met("B_c")
    draft = cobra.Model("draft")
    exb = cobra.Reaction("EX_B", lower_bound=-10, upper_bound=1000)
    exb.add_metabolites({B: -1})   # B can be taken up
    r1 = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({A: -1, B: 1})  # A -> B (irreversible; wrong direction)
    bio = cobra.Reaction("biomass", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({A: -1})
    draft.add_reactions([exb, r1, bio])
    draft.objective = "biomass"

    # Empty template — reversal is the only repair
    template = cobra.Model("template_empty")
    return draft, template


def test_result_type(growth_gap):
    draft, template = growth_gap
    result = fill_gaps_kumar_milp(draft, template, verbose=False)
    assert isinstance(result, KumarGapFillResult)


def test_added_reactions_list(growth_gap):
    draft, template = growth_gap
    result = fill_gaps_kumar_milp(draft, template, verbose=False)
    assert isinstance(result.added_reactions, list)


def test_reversed_reactions_list(growth_gap):
    draft, template = growth_gap
    result = fill_gaps_kumar_milp(draft, template, verbose=False)
    assert isinstance(result.reversed_reactions, list)


def test_model_returned(growth_gap):
    draft, template = growth_gap
    result = fill_gaps_kumar_milp(draft, template, verbose=False)
    assert isinstance(result.model, cobra.Model)


def test_gap_filled_enables_growth(growth_gap):
    """After adding r_missing, the model should be able to produce biomass."""
    draft, template = growth_gap
    result = fill_gaps_kumar_milp(draft, template, verbose=False)
    if result.exit_status != "optimal":
        pytest.skip(f"Solver status: {result.exit_status}")
    val = result.model.slim_optimize()
    assert val is not None and val > 0


def test_r_missing_added(growth_gap):
    """r_missing should be in the returned model."""
    draft, template = growth_gap
    result = fill_gaps_kumar_milp(draft, template, verbose=False)
    if result.exit_status != "optimal":
        pytest.skip(f"Solver status: {result.exit_status}")
    assert "r_missing" in result.added_reactions


def test_directionality_reversal(directionality_gap):
    """r1 should be reversed when that is the only repair option."""
    draft, template = directionality_gap
    result = fill_gaps_kumar_milp(draft, template, verbose=False)
    if result.exit_status != "optimal":
        pytest.skip(f"Solver status: {result.exit_status}")
    assert "r1" in result.reversed_reactions


def test_no_objective_raises(growth_gap):
    """A model without an objective should raise ValueError."""
    draft, template = growth_gap
    draft.objective = {}   # clear objective
    with pytest.raises(ValueError, match="no objective"):
        fill_gaps_kumar_milp(draft, template, verbose=False)


def test_exit_status_on_infeasible():
    """If the merged model cannot achieve growth, exit_status should not be 'optimal'."""
    # A model that is structurally infeasible (no exchange reactions)
    m = cobra.Model("infeasible")
    A = _met("A_c")
    bio = cobra.Reaction("bio", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({A: -1})
    m.add_reactions([bio])
    m.objective = "bio"
    template = cobra.Model("empty_template")
    result = fill_gaps_kumar_milp(m, template, verbose=False)
    assert result.exit_status != "optimal"


def _solvers():
    out = ["glpk"]
    try:
        import gurobipy  # noqa: F401
        out.append("gurobi")
    except ImportError:
        pass
    return out


@pytest.mark.parametrize("solver", _solvers())
def test_small_floor_is_not_met_by_a_fractional_indicator(growth_gap, solver):
    """With a floor below 1e-5 x big-M, a near-zero indicator must not pass as 'not added'."""
    draft, template = growth_gap
    draft.solver = solver
    result = fill_gaps_kumar_milp(draft, template, min_growth=0.005, verbose=False)
    assert result.exit_status == "optimal"
    assert result.added_reactions == ["r_missing"]
    assert result.model.slim_optimize() >= 0.005


def test_selection_that_does_not_restore_growth_is_reported(growth_gap, monkeypatch):
    """A rounded MILP selection is checked on the filled model."""
    from raven_toolbox.gapfilling import kumar_milp

    draft, template = growth_gap
    draft.solver = "glpk"
    # A loose integrality tolerance lets the indicator stay fractional.
    monkeypatch.setattr(kumar_milp, "_INTEGRALITY_TOL", 1e-4)
    result = fill_gaps_kumar_milp(draft, template, min_growth=0.005, verbose=False)
    assert result.added_reactions == []
    assert result.exit_status == "not_restored"
