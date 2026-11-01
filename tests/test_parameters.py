"""Tests for set_variance_bounds (the var mode of setParam) and set_exchange_bounds."""
import cobra
import pytest

from raven_toolbox.manipulation import (
    add_reactions_from_equations,
    set_exchange_bounds,
    set_variance_bounds,
)


@pytest.fixture
def model():
    m = cobra.Model("t")
    m.add_metabolites(
        [cobra.Metabolite("a_c", compartment="c"), cobra.Metabolite("b_c", compartment="c")]
    )
    add_reactions_from_equations(
        m,
        [
            {"id": "R1", "equation": "a_c <=> b_c"},
            {"id": "R2", "equation": "a_c <=> b_c"},
        ],
    )
    return m


def test_band_positive(model):
    set_variance_bounds(model, "R1", 100, 5)  # 97.5 .. 102.5
    lb, ub = model.reactions.get_by_id("R1").bounds
    assert lb == pytest.approx(97.5)
    assert ub == pytest.approx(102.5)


def test_band_negative_is_ordered(model):
    set_variance_bounds(model, "R1", -100, 5)
    lb, ub = model.reactions.get_by_id("R1").bounds
    assert lb == pytest.approx(-102.5)
    assert ub == pytest.approx(-97.5)
    assert lb <= ub


def test_broadcast_scalar(model):
    set_variance_bounds(model, ["R1", "R2"], 50, 10)
    for rid in ("R1", "R2"):
        lb, ub = model.reactions.get_by_id(rid).bounds
        assert lb == pytest.approx(47.5)
        assert ub == pytest.approx(52.5)


def test_per_reaction_values(model):
    set_variance_bounds(model, ["R1", "R2"], [100, 200], 0)
    assert model.reactions.get_by_id("R1").bounds == pytest.approx((100, 100))
    assert model.reactions.get_by_id("R2").bounds == pytest.approx((200, 200))


def test_length_mismatch_raises(model):
    with pytest.raises(ValueError, match="to match the reactions"):
        set_variance_bounds(model, ["R1", "R2"], [1, 2, 3], 5)


def test_unknown_reaction_raises(model):
    with pytest.raises(ValueError, match="not found"):
        set_variance_bounds(model, "NOPE", 1, 5)


@pytest.fixture
def exchange_model():
    m = cobra.Model("t")
    m.compartments = {"e": "extracellular", "c": "cytosol"}
    a_e = cobra.Metabolite("a_e", compartment="e")
    b_e = cobra.Metabolite("b_e", compartment="e")
    c_c = cobra.Metabolite("c_c", compartment="c")
    m.add_metabolites([a_e, b_e, c_c])
    # Consistent direction: the metabolite is a reactant (coeff -1) in every
    # exchange, so positive flux always means export, negative always import.
    ex_a = cobra.Reaction("EX_a", lower_bound=0, upper_bound=1000)
    ex_a.add_metabolites({a_e: -1})
    ex_b = cobra.Reaction("EX_b", lower_bound=0, upper_bound=1000)
    ex_b.add_metabolites({b_e: -1})
    sk_c = cobra.Reaction("SK_c", lower_bound=0, upper_bound=1000)
    sk_c.add_metabolites({c_c: -1})
    m.add_reactions([ex_a, ex_b, sk_c])
    return m


def test_exchange_bounds_basic_and_close_others(exchange_model):
    unused = set_exchange_bounds(exchange_model, "a_e", -10, 5)
    assert unused == []
    assert exchange_model.reactions.EX_a.bounds == (-10, 5)
    # direction is "backward" (import = negative flux); close_others closes
    # only import, i.e. sets lb=0, leaving export (ub) untouched.
    assert exchange_model.reactions.EX_b.bounds == (0, 1000)
    assert exchange_model.reactions.SK_c.bounds == (0, 1000)


def test_exchange_bounds_media_only(exchange_model):
    set_exchange_bounds(exchange_model, None, -5, 5, media_only=True)
    assert exchange_model.reactions.EX_a.bounds == (-5, 5)
    assert exchange_model.reactions.EX_b.bounds == (-5, 5)
    # SK_c's metabolite (c_c) is cytosolic, not extracellular: untouched.
    assert exchange_model.reactions.SK_c.bounds == (0, 1000)


def test_exchange_bounds_media_only_requires_extracellular_compartment():
    m = cobra.Model("t")
    m.compartments = {"c": "cytosol"}
    met = cobra.Metabolite("a_c", compartment="c")
    m.add_metabolites([met])
    rxn = cobra.Reaction("EX_a", lower_bound=0, upper_bound=1000)
    rxn.add_metabolites({met: -1})
    m.add_reactions([rxn])
    with pytest.raises(ValueError, match="extracellular"):
        set_exchange_bounds(m, None, -5, 5, media_only=True)


def test_exchange_bounds_mixed_direction_disables_close_others(exchange_model):
    # Flip EX_b to the opposite sign convention (metabolite as product).
    exchange_model.reactions.EX_b.add_metabolites({exchange_model.metabolites.b_e: 2}, combine=True)
    with pytest.warns(UserWarning, match="differ in direction"):
        set_exchange_bounds(exchange_model, "a_e", -10, 5, close_others=True)
    assert exchange_model.reactions.EX_a.bounds == (-10, 5)
    # close_others was forced off: EX_b and SK_c keep their original bounds.
    assert exchange_model.reactions.EX_b.bounds == (0, 1000)
    assert exchange_model.reactions.SK_c.bounds == (0, 1000)


def test_exchange_bounds_unused_mets(exchange_model):
    unused = set_exchange_bounds(exchange_model, ["a_e", "nonexistent"], [-10, -5], [0, 0])
    assert unused == ["nonexistent"]
    assert exchange_model.reactions.EX_a.bounds == (-10, 0)


def test_exchange_bounds_no_mets_requires_scalar(exchange_model):
    with pytest.raises(ValueError, match="Only one upper and one lower bound"):
        set_exchange_bounds(exchange_model, None, [-10, -5], [0, 0])


def test_exchange_bounds_no_mets_applies_to_all(exchange_model):
    set_exchange_bounds(exchange_model, None, -5, 5)
    for rid in ("EX_a", "EX_b", "SK_c"):
        assert exchange_model.reactions.get_by_id(rid).bounds == (-5, 5)


def test_exchange_bounds_warns_on_metabolite_in_multiple_exchanges(exchange_model):
    extra = cobra.Reaction("EX_a2", lower_bound=0, upper_bound=1000)
    extra.add_metabolites({exchange_model.metabolites.a_e: -1})
    exchange_model.add_reactions([extra])
    with pytest.warns(UserWarning, match="more than one exchange reaction"):
        set_exchange_bounds(exchange_model, "a_e", -10, 5)
    # both reactions exchanging a_e get the requested bounds.
    assert exchange_model.reactions.EX_a.bounds == (-10, 5)
    assert exchange_model.reactions.EX_a2.bounds == (-10, 5)


def test_exchange_bounds_name_entries(exchange_model):
    exchange_model.metabolites.a_e.name = "Alpha"
    unused = set_exchange_bounds(exchange_model, ["ALPHA", "No Such Name"], [-10, -5], [3, 3])
    assert exchange_model.reactions.EX_a.bounds == (-10, 3)
    assert unused == ["No Such Name"]
