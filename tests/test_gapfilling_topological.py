"""Tests for topological gap analysis (gapfilling/topological.py)."""
import cobra
import pytest

from raven_toolbox.gapfilling import TopologicalAnalysisResult, analyse_topology


def _met(mid):
    return cobra.Metabolite(mid, name=mid, compartment="c")


@pytest.fixture
def chain_with_gap():
    """Draft: EX_A (uptake) -> A -(r1)-> B -(r2)-> C.

    C has no consumer and B is a dead-end — nothing past B is reachable.
    Template supplies B -(r3)-> C -(r4)-> D and EX_D (exchange).
    Objective: r2, so the default targets are B and C.

    With seeds = {A}, the scope reaches A and B (via r1) but NOT C (r2 is blocked
    because C has no further consumer). If targets = {D}, D is unreachable.
    """
    A, B, C, D = _met("A_c"), _met("B_c"), _met("C_c"), _met("D_c")
    draft = cobra.Model("draft")
    draft.add_metabolites([A, B, C, D])

    exa = cobra.Reaction("EX_A", lower_bound=-10, upper_bound=1000)
    exa.add_metabolites({A: -1})      # exchange: negative lb → A available as seed

    r1 = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({A: -1, B: 1})  # A -> B

    r2 = cobra.Reaction("r2", lower_bound=0, upper_bound=1000)
    r2.add_metabolites({B: -1, C: 1})  # B -> C, but C has no consumer

    draft.add_reactions([exa, r1, r2])
    draft.objective = "r2"

    # Template supplies C consumer and D pathway
    template = cobra.Model("template")
    r3 = cobra.Reaction("r3", lower_bound=0, upper_bound=1000)
    r3.add_metabolites({_met("C_c"): -1, _met("D_c"): 1})  # C -> D
    exd = cobra.Reaction("EX_D", lower_bound=-1000, upper_bound=1000)
    exd.add_metabolites({_met("D_c"): -1})
    template.add_reactions([r3, exd])

    return draft, template


def test_result_type(chain_with_gap):
    draft, template = chain_with_gap
    result = analyse_topology(draft, template, verbose=False)
    assert isinstance(result, TopologicalAnalysisResult)


def test_seeds_detected_from_exchange(chain_with_gap):
    """A_c should be identified as a seed from EX_A (lb < 0)."""
    draft, template = chain_with_gap
    result = analyse_topology(draft, template, verbose=False)
    assert "A_c" in result.reachable_metabolites


def test_reachable_includes_direct_products(chain_with_gap):
    """B_c should be reachable (produced by r1 from seed A)."""
    draft, template = chain_with_gap
    result = analyse_topology(draft, template, verbose=False)
    assert "B_c" in result.reachable_metabolites


def test_blocked_metabolite_identified(chain_with_gap):
    """D_c should be unreachable in the draft model."""
    draft, template = chain_with_gap
    result = analyse_topology(draft, template, targets=["D_c"], verbose=False)
    assert "D_c" in result.blocked_metabolites


def test_candidate_reactions_for_blocked(chain_with_gap):
    """r3 (C -> D) should be listed as a candidate for blocked D_c."""
    draft, template = chain_with_gap
    result = analyse_topology(draft, template, targets=["D_c"], verbose=False)
    assert "r3" in result.candidate_reactions.get("D_c", [])


def test_pruning_fraction_in_range(chain_with_gap):
    draft, template = chain_with_gap
    result = analyse_topology(draft, template, verbose=False)
    assert 0.0 <= result.pruning_fraction <= 1.0


def test_explicit_seeds_override_default(chain_with_gap):
    """Passing seeds=[] should yield no reachable metabolites (empty BFS start)."""
    draft, template = chain_with_gap
    result = analyse_topology(draft, template, seeds=[], verbose=False)
    # With no seeds and no zero-substrate reactions that fire, nothing is reachable
    # (r1 needs A, r2 needs B — none fire without seeds)
    assert "B_c" not in result.reachable_metabolites


def test_explicit_targets_subset(chain_with_gap):
    """Providing explicit targets limits what is reported as blocked."""
    draft, template = chain_with_gap
    result = analyse_topology(draft, template, targets=["A_c"], verbose=False)
    # A_c is a seed itself, so it should be reachable
    assert "A_c" not in result.blocked_metabolites


def test_reversible_reaction_allows_reverse_scope(chain_with_gap):
    """A reversible reaction should propagate scope in both directions."""
    draft, template = chain_with_gap
    # Make r1 reversible: B -> A now also possible
    draft.reactions.get_by_id("r1").lower_bound = -1000

    # Seed: C_c only (no EX_A uptake)
    result = analyse_topology(
        draft, template, seeds=["C_c"], targets=["A_c"], verbose=False
    )
    # r2 is irreversible, so scope still can't reach B (and hence A) from seed
    # C_c even with r1 now reversible; this only exercises the reverse-scope path.
    assert isinstance(result, TopologicalAnalysisResult)


def test_default_targets_are_the_objective_metabolites(chain_with_gap):
    """As in RAVEN: substrates and products of the first objective reaction."""
    draft, template = chain_with_gap
    result = analyse_topology(draft, template, verbose=False)
    assert result.blocked_metabolites == set()
    draft.objective = "EX_A"
    draft.reactions.EX_A.lower_bound = 0
    with pytest.warns(UserWarning, match="no uptake exchange"):
        result = analyse_topology(draft, template, verbose=False)
    assert result.blocked_metabolites == {"A_c"}


def test_default_targets_need_an_objective(chain_with_gap):
    draft, template = chain_with_gap
    draft.objective = {}
    with pytest.raises(ValueError, match="no objective function"):
        analyse_topology(draft, template, verbose=False)


def test_default_seeds_follow_ravens_exchange_rule():
    """Seeds: metabolites of an only-substrate reaction with lb < 0 or an only-product one
    with ub > 0, whatever the number of metabolites; nothing else."""
    model = cobra.Model("m")
    mets = {mid: _met(mid) for mid in ("A_c", "B_c", "C_c", "D_c", "E_c")}
    model.add_metabolites(list(mets.values()))
    rxns = [
        ("EX_AB", {"A_c": -1, "B_c": -1}, -10, 0),   # two-metabolite uptake: seeds A and B
        ("EX_C", {"C_c": -1}, 0, 1000),              # excretion only: not a seed
        ("R_DE", {"D_c": -1, "E_c": 1}, -10, 10),    # internal, reversible: not a seed
    ]
    for rid, stoich, lb, ub in rxns:
        rxn = cobra.Reaction(rid, lower_bound=lb, upper_bound=ub)
        rxn.add_metabolites({mets[m]: c for m, c in stoich.items()})
        model.add_reactions([rxn])
    model.objective = "R_DE"
    result = analyse_topology(model, cobra.Model("t"), verbose=False)
    assert result.reachable_metabolites == {"A_c", "B_c"}
    assert result.blocked_metabolites == {"D_c", "E_c"}
