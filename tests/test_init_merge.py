"""Linear reaction merging (merge_linear + group_rxn_scores).

Oracles: RAVEN tinitTests T0004. testModel merges {R1,R2},{R3,R5},{R4,R6},{R7,R8},
{R9,R10}; testModel4 merges {R5,R6},{R7,R8},{R9,R10} with two reactions flipped.
"""
import cobra
import pytest
from tinit_oracles import (
    TEST_MODEL4_GROUP_IDS,
    TEST_MODEL4_MERGED_REV,
    TEST_MODEL4_REVERSED_RXNS,
    TEST_MODEL_GROUP_IDS,
    TEST_MODEL_GROUPED_SCORES,
    TEST_MODEL_MERGED_LB,
    TEST_MODEL_MERGED_REV,
    TEST_MODEL_SCORES,
    make_test_model,
    make_test_model4,
)

from raven_toolbox.init import group_rxn_scores, merge_linear


def test_test_model_group_ids():
    _, orig_ids, group_ids, _ = merge_linear(make_test_model())
    assert orig_ids == [f"R{i}" for i in range(1, 11)]
    assert group_ids == TEST_MODEL_GROUP_IDS  # [1,1,2,3,2,3,4,4,5,5]


def test_test_model_reduced_shape():
    reduced, _, _, _ = merge_linear(make_test_model())
    # Five merged reactions, survivors keep the producer's id, original order.
    assert [r.id for r in reduced.reactions] == ["R1", "R3", "R4", "R7", "R9"]
    assert [int(r.lower_bound < 0) for r in reduced.reactions] == TEST_MODEL_MERGED_REV
    assert [r.lower_bound for r in reduced.reactions] == TEST_MODEL_MERGED_LB


def test_test_model_grouped_scores():
    reduced, orig_ids, group_ids, _ = merge_linear(make_test_model())
    scores = dict(zip(orig_ids, TEST_MODEL_SCORES, strict=True))
    grouped = group_rxn_scores(reduced, scores, orig_ids, group_ids,
                               to_zero={"R1", "R2", "R8"})
    got = [grouped[r.id] for r in reduced.reactions]
    assert got == pytest.approx(TEST_MODEL_GROUPED_SCORES)  # [0,-0.5,7.5,-1,0.5]


def test_test_model4_group_ids_and_flips():
    reduced, orig_ids, group_ids, reversed_rxns = merge_linear(make_test_model4())
    assert group_ids == TEST_MODEL4_GROUP_IDS  # [0,0,0,0,1,1,2,2,3,3,0]
    assert [int(r.lower_bound < 0) for r in reduced.reactions] == TEST_MODEL4_MERGED_REV
    flipped = {oid for oid, rev in zip(orig_ids, reversed_rxns, strict=True) if rev}
    assert flipped == set(TEST_MODEL4_REVERSED_RXNS)  # {R6, R9}


def test_merge_preserves_feasible_space():
    """The reduced model admits flux through the merged export path, like the original.

    The reduced model carries no objective (merging drops genes and objective; ftINIT
    sets its own from scores), so we set one on the surviving export reaction. R8
    (e[s]=>) was merged into R7 (grp4), so R7 is the reduced export.
    """
    original = make_test_model()
    assert original.slim_optimize() > 1e-9  # exports e via R8
    reduced, _, _, _ = merge_linear(original)
    reduced.objective = "R7"
    assert reduced.slim_optimize() > 1e-9


def test_no_merge_blocks_merging():
    """A reaction in no_merge keeps its own group (id 0) and is not contracted."""
    _, orig_ids, group_ids, _ = merge_linear(make_test_model(), no_merge=["R2"])
    g = dict(zip(orig_ids, group_ids, strict=True))
    assert g["R2"] == 0  # R2 never merged
    # R1 was only mergeable with R2, so it stays unmerged too.
    assert g["R1"] == 0


def test_multipass_chain_collapses_to_one_group():
    """A 3-reaction chain A→X→Y→Z collapses to one reaction (exercises multi-pass).

    X is degree-2 (r1,r2), Y degree-2 (r2,r3); A and Z are degree-1 (retained). Merging
    X makes Y newly degree-2 with the survivor, caught on a later pass. Confluence: all
    three reactions end in one group, leaving the net A→Z reaction.
    """
    import cobra

    m = cobra.Model("chain")
    A, X, Y, Z = (cobra.Metabolite(i, name=i, compartment="c") for i in "AXYZ")
    m.add_metabolites([A, X, Y, Z])
    for rid, stoich in [("r1", {A: -1, X: 1}), ("r2", {X: -1, Y: 1}), ("r3", {Y: -1, Z: 1})]:
        r = cobra.Reaction(rid, lower_bound=0, upper_bound=1000)
        r.add_metabolites(stoich)
        m.add_reactions([r])

    reduced, orig_ids, group_ids, _ = merge_linear(m)
    assert len(reduced.reactions) == 1                      # collapsed to net A -> Z
    assert len(set(group_ids)) == 1 and group_ids[0] != 0   # all three in one group
    only = reduced.reactions[0]
    assert {mt.id: c for mt, c in only.metabolites.items()} == {"A": -1.0, "Z": 1.0}


def test_groups_are_numbered_by_their_first_reaction():
    """Group numbers follow the first reaction of each group in the original model, not the
    merge order. Here R3+R4 merge before R1+R2's group reaches them, and the merge of two
    groups keeps the larger number, so numbering by merge order gives 2, 2, 2, 2, 3, 3."""
    m = cobra.Model("numbering")
    mets = {i: cobra.Metabolite(i, name=i, compartment="c") for i in ("x", "a", "m", "b", "y", "z", "c", "w")}
    m.add_metabolites(list(mets.values()))
    chain = [("R1", "x", "a"), ("R2", "a", "m"), ("R3", "m", "b"), ("R4", "b", "y"),
             ("R5", "z", "c"), ("R6", "c", "w")]
    for rid, sub, prod in chain:
        r = cobra.Reaction(rid, lower_bound=0, upper_bound=1000)
        r.add_metabolites({mets[sub]: -1, mets[prod]: 1})
        m.add_reactions([r])

    reduced, orig_ids, group_ids, _ = merge_linear(m)
    assert orig_ids == ["R1", "R2", "R3", "R4", "R5", "R6"]
    assert group_ids == [1, 1, 1, 1, 2, 2]
    assert sorted(r.id for r in reduced.reactions) == ["R1", "R5"]


def test_reaction_flipped_in_later_pass_is_marked_reversed():
    """R1 and R2 merge through A in the first pass; the second pass merges R1 and R4 through
    Y and flips R4, so R4 (not R3, the third reaction) is the reversed one."""
    m = cobra.Model("lateflip")
    mets = {i: cobra.Metabolite(i, name=i, compartment="c") for i in "AYQW"}
    spec = [("R1", {"A": 1, "Y": 1}, 0), ("R2", {"A": -1, "Y": 1}, 0), ("R3", {"Q": -1}, 0),
            ("R4", {"W": -1, "Y": 1}, -1000), ("R5", {"W": 1}, 0), ("R6", {"W": -1}, 0)]
    for rid, stoichiometry, lb in spec:
        r = cobra.Reaction(rid, lower_bound=lb, upper_bound=1000)
        r.add_metabolites({mets[k]: v for k, v in stoichiometry.items()})
        m.add_reactions([r])

    reduced, orig_ids, group_ids, reversed_rxns = merge_linear(m)
    assert group_ids == [1, 1, 0, 1, 0, 0]
    assert [rid for rid, flag in zip(orig_ids, reversed_rxns, strict=True) if flag] == ["R4"]
    assert sorted(r.id for r in reduced.reactions) == ["R1", "R3", "R5", "R6"]


def test_group_scores_zero_handling():
    """Genuine-zero score → 0.01; a group cancelling to zero with nonzero members → 0.01."""
    reduced, orig_ids, group_ids, _ = merge_linear(make_test_model())
    # Give group {R3,R5} scores that cancel: R3=+1, R5=-1 -> sum 0 but members nonzero.
    scores = dict.fromkeys(orig_ids, 0.0)
    scores["R3"], scores["R5"] = 1.0, -1.0
    grouped = group_rxn_scores(reduced, scores, orig_ids, group_ids)
    assert grouped["R3"] == pytest.approx(0.01)        # cancelled group rescued
    assert grouped["R4"] == pytest.approx(0.02)         # {R4,R6} both genuine-0 → 0.01+0.01


@pytest.mark.parametrize("default", ["glpk", "gurobi"])
def test_reduced_model_keeps_template_solver(default):
    """The reduced model uses the template's solver, not cobra's global default."""
    pytest.importorskip("gurobipy")
    template = make_test_model()
    template.solver = "gurobi" if default == "glpk" else "glpk"
    config = cobra.Configuration()
    previous = config.solver
    config.solver = default
    try:
        reduced, _, _, _ = merge_linear(template)
    finally:
        config.solver = previous
    assert reduced.problem is template.problem
