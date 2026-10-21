"""remove_low_score_genes — the three RAVEN docstring oracle cases.

Scores use distinct values in the oracle cases. Ties are tested separately: when
all isozyme alternatives are negative and equal, the one listed first in the rule is
kept, on both sides (RAVEN's removeLowScoreGenes does the same).
"""
import cobra

from raven_toolbox.init import remove_low_score_genes


def _model(rule: str) -> cobra.Model:
    m = cobra.Model("g")
    a = cobra.Metabolite("a", compartment="c")
    b = cobra.Metabolite("b", compartment="c")
    r = cobra.Reaction("R", lower_bound=0, upper_bound=1000)
    r.add_metabolites({a: -1, b: 1})
    m.add_reactions([r])
    r.gene_reaction_rule = rule
    return m


def _norm(rule: str) -> str:
    """cobra's normalized form of a GPR string, for order/paren-insensitive comparison."""
    return _model(rule).reactions.R.gene_reaction_rule


def _result(rule: str, scores: dict) -> str:
    out, _ = remove_low_score_genes(_model(rule), scores)
    return out.reactions.R.gene_reaction_rule


def test_case1_isozyme_vs_complex():
    """G1 or (G2 and G3 and G4); G1,G2 negative → keep the complex."""
    # G1 more negative than G2 so the complex (= G2's score under min) is least-negative.
    scores = {"G1": -2.0, "G2": -1.0, "G3": 1.0, "G4": 1.0}
    assert _result("G1 or (G2 and G3 and G4)", scores) == _norm("G2 and G3 and G4")


def test_case2_two_complexes():
    """G1 or (G2 and G3) or (G4 and G5); G1,G2 negative → keep the positive complex."""
    scores = {"G1": -1.0, "G2": -1.0, "G3": 1.0, "G4": 1.0, "G5": 1.0}
    assert _result("G1 or (G2 and G3) or (G4 and G5)", scores) == _norm("G4 and G5")


def test_case3_nested_isozyme_in_complex():
    """(G1 and (G2 or G3) and G4); G2 negative → prune G2 from the inner isozyme group."""
    scores = {"G1": 1.0, "G2": -1.0, "G3": 1.0, "G4": 1.0}
    assert _result("G1 and (G2 or G3) and G4", scores) == _norm("G1 and G3 and G4")


def test_complex_subunit_not_removed_individually():
    """A negative subunit of a pure complex stays (the whole complex is kept)."""
    scores = {"G1": 1.0, "G2": -1.0}
    assert _result("G1 and G2", scores) == _norm("G1 and G2")


def test_single_negative_gene_kept():
    """A reaction's only gene is never removed (≥1 must remain)."""
    assert _result("G1", {"G1": -5.0}) == "G1"


def test_unscored_genes_not_removed():
    """Genes absent from the score map are treated as unscored and not removed."""
    scores = {"G1": -1.0}  # G2 unscored
    assert _result("G1 or G2", scores) == _norm("G2")  # only the negative G1 dropped


def test_removed_genes_reported_and_pruned():
    out, removed = remove_low_score_genes(_model("G1 or G2"), {"G1": -1.0, "G2": 1.0})
    assert removed == ["G1"]
    assert "G1" not in {g.id for g in out.genes}


def test_tie_between_negative_isozymes_keeps_the_first_in_the_rule():
    # G2 comes first in the rule although G1 comes first alphabetically.
    assert _result("G2 or G1", {"G1": -1.0, "G2": -1.0}) == "G2"
    assert _result("G4 or G3 or G2", {"G2": -1.0, "G3": 1.0, "G4": -1.0}) == "G3"


def test_tie_between_negative_complexes_keeps_the_first_in_the_rule():
    scores = {"G1": -1.0, "G2": -1.0, "G3": -1.0, "G4": -1.0}
    assert _result("(G3 and G4) or (G1 and G2)", scores) == _norm("G3 and G4")


def test_positive_isozymes_keep_rule_order():
    scores = {"G1": 1.0, "G2": 1.0, "G3": 1.0}
    assert _result("G3 or G1 or G2", scores) == _norm("G3 or G1 or G2")
