"""Tests for sort_model (RAVEN's sortModel)."""
import cobra
import pytest

from raven_toolbox.utils import sort_model, subsystem_order_scores


def _toy():
    """One reversible reaction A <=> B with the names A and B."""
    m = cobra.Model("toy")
    a = cobra.Metabolite("A_c", name="A", compartment="c")
    b = cobra.Metabolite("B_c", name="B", compartment="c")
    m.add_metabolites([a, b])
    r = cobra.Reaction("R1")
    m.add_reactions([r])
    r.add_metabolites({a: 1, b: -1})
    r.bounds = (-1000, 1000)
    return m


def _coefficients(m, rid):
    return {met.id: c for met, c in m.reactions.get_by_id(rid).metabolites.items()}


def _chain(subsystem="S"):
    """Reactions listed in the order that consumes before it produces: R3 B->C, R2 A->B, R1 -> A."""
    m = cobra.Model("chain")
    mets = {x: cobra.Metabolite(f"{x}_c", name=x, compartment="c") for x in "ABC"}
    m.add_metabolites(list(mets.values()))
    for rid, stoich in (("R3", {"B": -1, "C": 1}), ("R2", {"A": -1, "B": 1}), ("R1", {"A": 1})):
        r = cobra.Reaction(rid)
        m.add_reactions([r])
        r.add_metabolites({mets[k]: v for k, v in stoich.items()})
        r.bounds = (0, 1000)
        r.subsystem = subsystem
        r.gene_reaction_rule = f"g_{rid}"
    return m


def test_default_leaves_directions_alone():
    m = sort_model(_toy())
    assert _coefficients(m, "R1") == {"A_c": 1, "B_c": -1}


def test_sort_reversible_makes_first_sorted_metabolite_a_reactant():
    m = sort_model(_toy(), sort_reversible=True)
    assert _coefficients(m, "R1") == {"A_c": -1, "B_c": 1}
    assert m.reactions.R1.bounds == (-1000, 1000)


def test_sort_reversible_never_flips_irreversible_reactions():
    m = _toy()
    m.reactions.R1.bounds = (0, 1000)
    sort_model(m, sort_reversible=True)
    assert _coefficients(m, "R1") == {"A_c": 1, "B_c": -1}


def test_sort_reversible_key_is_name_plus_compartment():
    m = cobra.Model("k")
    a = cobra.Metabolite("zz", name="B", compartment="c")
    b = cobra.Metabolite("aa", name="A", compartment="m")
    m.add_metabolites([a, b])
    r = cobra.Reaction("R1")
    m.add_reactions([r])
    r.add_metabolites({a: -1, b: 1})
    r.bounds = (-1000, 1000)
    sort_model(m, sort_reversible=True)
    assert _coefficients(m, "R1") == {"zz": 1, "aa": -1}


def test_sort_met_name_orders_by_name_and_compartment():
    m = cobra.Model("n")
    m.add_metabolites(
        [
            cobra.Metabolite("x1", name="B", compartment="c"),
            cobra.Metabolite("x2", name="A", compartment="m"),
            cobra.Metabolite("x3", name="A", compartment="c"),
        ]
    )
    sort_model(m, sort_met_name=True)
    assert [x.id for x in m.metabolites] == ["x3", "x2", "x1"]


def test_sort_reaction_order_does_not_lower_the_score_and_keeps_the_rest():
    m = _chain()
    before = subsystem_order_scores(m)["S"]
    sort_model(m, sort_reaction_order=True, seed=1)
    after = subsystem_order_scores(m)["S"]
    assert after >= before
    assert sorted(r.id for r in m.reactions) == ["R1", "R2", "R3"]
    assert [r.id for r in m.reactions] == ["R1", "R2", "R3"]
    assert after == 3
    for r in m.reactions:
        assert r.gene_reaction_rule == f"g_{r.id}"
        assert r.bounds == (0, 1000)


def test_sort_reaction_order_is_reproducible_with_a_seed():
    orders = []
    for _ in range(2):
        m = _chain()
        sort_model(m, sort_reaction_order=True, seed=7)
        orders.append([r.id for r in m.reactions])
    assert orders[0] == orders[1]


def test_sort_reaction_order_never_flips_irreversible_reactions():
    m = _chain()
    sort_model(m, sort_reaction_order=True, seed=3)
    assert _coefficients(m, "R2") == {"A_c": -1, "B_c": 1}
    assert _coefficients(m, "R3") == {"B_c": -1, "C_c": 1}


def test_sort_reaction_order_skips_reactions_without_subsystem_and_single_reaction_subsystems():
    m = _chain(subsystem="")
    sort_model(m, sort_reaction_order=True, seed=1)
    assert [r.id for r in m.reactions] == ["R3", "R2", "R1"]
    assert subsystem_order_scores(m) == {}


def test_subsystem_entries_are_matched_whole():
    m = _chain()
    m.reactions.R3.subsystem = "S;Other"
    m.reactions.R2.subsystem = "Other S"
    # "S" lists R1 and R3; "Other" and "Other S" have one reaction each and are skipped.
    assert set(subsystem_order_scores(m)) == {"S"}


def _reversible_chain():
    m = cobra.Model("rev")
    mets = {x: cobra.Metabolite(f"{x}_c", name=x, compartment="c") for x in "ABC"}
    m.add_metabolites(list(mets.values()))
    for rid, stoich, bounds in (
        ("R1", {"A": 1}, (0, 1000)),
        ("R2", {"A": 1, "B": -1}, (-1000, 1000)),
        ("R3", {"B": 1, "C": -1}, (-1000, 1000)),
    ):
        r = cobra.Reaction(rid)
        m.add_reactions([r])
        r.add_metabolites({mets[k]: v for k, v in stoich.items()})
        r.bounds = bounds
        r.subsystem = "S"
    return m


def test_reaction_order_search_reverses_reversible_reactions_when_it_helps():
    start = subsystem_order_scores(_reversible_chain())["S"]
    scores = []
    for seed in range(5):
        m = sort_model(_reversible_chain(), sort_reaction_order=True, seed=seed)
        scores.append(subsystem_order_scores(m)["S"])
        assert scores[-1] >= start
    assert max(scores) == 3


def test_reaction_order_search_keeps_a_single_reversible_reaction_as_it_is():
    # As in sortModel.m, the search reverses a reaction only when the subsystem has more than one reversible reaction.
    m = _reversible_chain()
    m.reactions.R3.bounds = (0, 1000)
    for seed in range(5):
        c = sort_model(m.copy(), sort_reaction_order=True, seed=seed)
        assert _coefficients(c, "R2") == {"A_c": 1, "B_c": -1}


@pytest.mark.parametrize("flag", ["sort_reversible", "sort_met_name"])
def test_returns_the_same_model(flag):
    m = _toy()
    assert sort_model(m, **{flag: True}) is m
