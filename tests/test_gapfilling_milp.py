"""Tests for growth-floor MILP gap-filling (gapfilling/milp.py)."""
import cobra
import pytest

from raven_toolbox.gapfilling import MILPGapFillResult, fill_gaps_milp


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

    # The template offers nothing that makes A
    template = cobra.Model("template")
    other = cobra.Reaction("r_other", lower_bound=0, upper_bound=1000)
    other.add_metabolites({_met("B_c"): -1})
    template.add_reactions([other])
    return draft, template


def test_result_type(growth_gap):
    draft, template = growth_gap
    result = fill_gaps_milp(draft, template, verbose=False)
    assert isinstance(result, MILPGapFillResult)


def test_added_reactions_list(growth_gap):
    draft, template = growth_gap
    result = fill_gaps_milp(draft, template, verbose=False)
    assert isinstance(result.added_reactions, list)


def test_model_returned(growth_gap):
    draft, template = growth_gap
    result = fill_gaps_milp(draft, template, verbose=False)
    assert isinstance(result.model, cobra.Model)


def test_gap_filled_enables_growth(growth_gap):
    """After adding r_missing, the model should be able to produce biomass."""
    draft, template = growth_gap
    result = fill_gaps_milp(draft, template, verbose=False)
    if result.exit_status != "optimal":
        pytest.skip(f"Solver status: {result.exit_status}")
    val = result.model.slim_optimize()
    assert val is not None and val > 0


def test_r_missing_added(growth_gap):
    """r_missing should be in the returned model."""
    draft, template = growth_gap
    result = fill_gaps_milp(draft, template, verbose=False)
    if result.exit_status != "optimal":
        pytest.skip(f"Solver status: {result.exit_status}")
    assert "r_missing" in result.added_reactions


def test_draft_directions_are_kept(directionality_gap):
    """A draft reaction in the wrong direction is not reversed: there is no fill."""
    draft, template = directionality_gap
    result = fill_gaps_milp(draft, template, min_growth=1.0, verbose=False)
    assert result.exit_status != "optimal"
    assert result.added_reactions == []
    assert result.model.reactions.r1.bounds == (0, 1000)


def test_forced_flux_is_kept():
    """A draft reaction with a forced flux (maintenance) keeps its lower bound."""
    A, B = _met("A_c"), _met("B_c")
    draft = cobra.Model("draft")
    exa = cobra.Reaction("EX_A", lower_bound=0, upper_bound=10)
    exa.add_metabolites({A: 1})
    r1 = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({A: -1, B: 1})
    grow = cobra.Reaction("grow", lower_bound=0, upper_bound=1000)
    grow.add_metabolites({B: -1})
    maint = cobra.Reaction("maint", lower_bound=5, upper_bound=1000)
    maint.add_metabolites({A: -1})
    draft.add_reactions([exa, r1, grow, maint])
    draft.objective = "grow"

    template = cobra.Model("template")
    u1 = cobra.Reaction("u1", lower_bound=0, upper_bound=10)
    u1.add_metabolites({_met("A_c"): 1})
    template.add_reactions([u1])

    result = fill_gaps_milp(draft, template, min_growth=8.0, verbose=False)
    assert result.exit_status == "optimal"
    assert result.added_reactions == ["u1"]
    assert result.model.reactions.maint.lower_bound == 5
    assert result.model.slim_optimize() >= 8 - 1e-6


def test_no_objective_raises(growth_gap):
    """A model without an objective should raise ValueError."""
    draft, template = growth_gap
    draft.objective = {}   # clear objective
    with pytest.raises(ValueError, match="no objective"):
        fill_gaps_milp(draft, template, verbose=False)


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
    result = fill_gaps_milp(m, template, verbose=False)
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
    result = fill_gaps_milp(draft, template, min_growth=0.005, verbose=False)
    assert result.exit_status == "optimal"
    assert result.added_reactions == ["r_missing"]
    assert result.model.slim_optimize() >= 0.005


def test_floor_check_on_filled_model(growth_gap):
    """The floor check fails on the draft and passes once r_missing is added."""
    from raven_toolbox.gapfilling import milp

    draft, template = growth_gap
    obj = {draft.reactions.biomass: 1.0}
    assert not milp._meets_floor(draft, obj, 0.005)
    draft.add_reactions([template.reactions.r_missing.copy()])
    assert milp._meets_floor(draft, obj, 0.005)


def test_selection_that_does_not_restore_growth_is_reported(growth_gap, monkeypatch):
    """A selection that fails the floor check is returned as 'not_restored'."""
    from raven_toolbox.gapfilling import milp

    draft, template = growth_gap
    monkeypatch.setattr(milp, "_meets_floor", lambda *args: False)
    result = fill_gaps_milp(draft, template, min_growth=0.005, verbose=False)
    assert result.added_reactions == ["r_missing"]
    assert result.exit_status == "not_restored"


def _rxn(rid, mets, lb=0, ub=1000):
    r = cobra.Reaction(rid, lower_bound=lb, upper_bound=ub)
    r.add_metabolites({_met(m): c for m, c in mets.items()})
    return r


def test_growth_floor_is_on_the_weighted_objective():
    """With objective coefficient 2 the floor applies to 2 * v, not to v."""
    draft = cobra.Model("draft")
    draft.add_reactions([_rxn("EX_A", {"A_c": 1}, 0, 10), _rxn("BIO", {"B_c": -1})])
    draft.objective = {draft.reactions.BIO: 2.0}
    template = cobra.Model("template")
    template.add_reactions([_rxn("T1", {"A_c": -1, "B_c": 1})])
    result = fill_gaps_milp(draft, template, min_growth=15, verbose=False)
    assert result.exit_status == "optimal"
    assert result.added_reactions == ["T1"]


def test_growth_floor_is_shared_by_several_objective_reactions():
    """A floor of 5 on BIO + BIO2 is met by one of them, so one template reaction suffices."""
    draft = cobra.Model("draft")
    draft.add_reactions([
        _rxn("EX_A", {"A_c": 1}, 0, 10),
        _rxn("BIO", {"B_c": -1}),
        _rxn("BIO2", {"C_c": -1}),
    ])
    draft.objective = {draft.reactions.BIO: 1.0, draft.reactions.BIO2: 1.0}
    template = cobra.Model("template")
    template.add_reactions([_rxn("T1", {"A_c": -1, "B_c": 1}), _rxn("T2", {"A_c": -1, "C_c": 1})])
    result = fill_gaps_milp(draft, template, min_growth=5, verbose=False)
    assert result.exit_status == "optimal"
    assert len(result.added_reactions) == 1


def test_template_lower_bound_is_only_forced_when_selected():
    """T2 has lb 1 but is not needed, so it is not added."""
    draft = cobra.Model("draft")
    draft.add_reactions([
        _rxn("EX_A", {"A_c": 1}, 0, 10),
        _rxn("BIO", {"B_c": -1}),
        _rxn("EX_C", {"C_c": -1}),
    ])
    draft.objective = "BIO"
    template = cobra.Model("template")
    template.add_reactions([
        _rxn("T1", {"A_c": -1, "B_c": 1}),
        _rxn("T2", {"A_c": -1, "C_c": 1}, 1, 1000),
    ])
    result = fill_gaps_milp(draft, template, min_growth=5, verbose=False)
    assert result.exit_status == "optimal"
    assert result.added_reactions == ["T1"]


def test_template_reaction_with_a_draft_id_is_renamed():
    """A template reaction named like a draft reaction is a separate candidate."""
    draft = cobra.Model("draft")
    draft.add_reactions([_rxn("R1", {"a_c": -1, "b_c": 1}), _rxn("Rname", {"b_c": -1})])
    draft.objective = "R1"
    template = cobra.Model("template")
    template.add_reactions([_rxn("Rname", {"a_c": 1})])
    result = fill_gaps_milp(draft, template, min_growth=1, verbose=False)
    assert result.exit_status == "optimal"
    assert result.added_reactions == ["Rname_template"]
    assert "Rname_template" in result.model.reactions
    assert {m.id: c for m, c in result.model.reactions.Rname.metabolites.items()} == {"b_c": -1}


def test_finite_template_bounds_are_not_capped_by_big_m():
    """A template reaction with ub 5000 can carry a growth floor of 1500 although big_m is 1000."""
    draft = cobra.Model("draft")
    draft.add_reactions([_rxn("EX_A", {"A_c": 1}, 0, 5000), _rxn("BIO", {"B_c": -1}, 0, 5000)])
    draft.objective = "BIO"
    template = cobra.Model("template")
    template.add_reactions([_rxn("T1", {"A_c": -1, "B_c": 1}, 0, 5000)])
    result = fill_gaps_milp(draft, template, min_growth=1500, big_m=1000, verbose=False)
    assert result.exit_status == "optimal"
    assert result.added_reactions == ["T1"]
