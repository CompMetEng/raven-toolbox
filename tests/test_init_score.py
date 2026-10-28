"""Tests for gene/reaction scoring (init/score.py)."""
import math

import cobra
import pytest

from raven_toolbox.init import gene_scores_from_expression, score_reactions_from_genes


# --------------------------------------------------------------------------- #
# score_reactions_from_genes
# --------------------------------------------------------------------------- #
@pytest.fixture
def gpr_model():
    m = cobra.Model("g")
    a = cobra.Metabolite("a_c", compartment="c")
    b = cobra.Metabolite("b_c", compartment="c")
    m.add_metabolites([a, b])
    r_complex = cobra.Reaction("r_complex")  # (g1 and g2) or g3
    r_complex.add_metabolites({a: -1, b: 1})
    m.add_reactions([r_complex])
    r_complex.gene_reaction_rule = "(g1 and g2) or g3"
    r_nogene = cobra.Reaction("r_nogene")
    r_nogene.add_metabolites({b: -1})
    m.add_reactions([r_nogene])
    return m


def test_score_isozyme_max_complex_min(gpr_model):
    # (g1 and g2) or g3 -> max(min(1, 4), 3) = max(1, 3) = 3
    scores = score_reactions_from_genes(gpr_model, {"g1": 1.0, "g2": 4.0, "g3": 3.0})
    assert scores["r_complex"] == 3.0


def test_score_no_gene_reaction_gets_default(gpr_model):
    scores = score_reactions_from_genes(gpr_model, {"g1": 1, "g2": 1, "g3": 1}, no_gene_score=-2.0)
    assert scores["r_nogene"] == -2.0


def test_score_missing_genes_omitted(gpr_model):
    # g2 missing -> complex (g1 and g2) collapses to g1=1; OR with g3=3 -> max(1,3)=3
    scores = score_reactions_from_genes(gpr_model, {"g1": 1.0, "g3": 3.0})
    assert scores["r_complex"] == 3.0
    # all genes missing -> no_gene_score
    assert score_reactions_from_genes(gpr_model, {})["r_complex"] == -2.0


def test_score_invalid_method(gpr_model):
    with pytest.raises(ValueError, match="isozyme_scoring"):
        score_reactions_from_genes(gpr_model, {}, isozyme_scoring="nonsense")


# --------------------------------------------------------------------------- #
# gene_scores_from_expression (RNA-seq path)
# --------------------------------------------------------------------------- #
def test_expression_scores_sign_and_clamp():
    expr = {"hi": 100.0, "lo": 1.0, "mid": 10.0, "zero": 0.0}
    ref = 10.0  # threshold/reference
    s = gene_scores_from_expression(expr, ref)
    assert s["hi"] == pytest.approx(min(5 * math.log(10), 10.0))  # above ref -> positive
    assert s["lo"] == pytest.approx(max(5 * math.log(0.1), -5.0))  # below ref -> negative
    assert s["mid"] == pytest.approx(0.0)  # at ref -> 0
    assert s["zero"] == -5.0  # non-positive -> floor


def test_expression_per_gene_reference():
    expr = {"g": 20.0}
    s = gene_scores_from_expression(expr, {"g": 5.0})
    assert s["g"] == pytest.approx(5 * math.log(4))


def test_expression_zero_reference_scores_like_raven():
    s = gene_scores_from_expression({"zero": 0.0, "one": 1.0, "ten": 10.0}, 0.0)
    assert s == {"zero": -5.0, "one": 10.0, "ten": 10.0}


def test_expression_missing_level_gets_no_score():
    s = gene_scores_from_expression({"g": math.nan, "h": None, "k": 2.0}, 1.0)  # type: ignore[dict-item]
    assert set(s) == {"k"}


def test_nan_gene_score_is_left_out_of_the_rule(gpr_model):
    scores = score_reactions_from_genes(gpr_model, {"g1": math.nan, "g2": math.nan, "g3": math.nan})
    assert set(scores.values()) == {-2.0}
    partial = score_reactions_from_genes(gpr_model, {"g1": 1.0, "g2": math.nan, "g3": 3.0})
    assert not any(math.isnan(v) for v in partial.values())
