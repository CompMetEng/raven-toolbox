"""Scoring of a model from HPA and expression data for ftINIT, as RAVEN's scoreModel."""
import math

import cobra
import numpy as np
import pandas as pd
import pytest

from raven_toolbox.init import HPAData, TranscrData, ftinit, prep_init_model, score_for_init


def _model():
    m = cobra.Model("m")
    rxns = []
    for rid, gpr in [("R1", "G1"), ("R2", "G1 and G2"), ("R3", "G1 or G3"), ("R4", ""), ("R5", "G4")]:
        r = cobra.Reaction(rid)
        r.gene_reaction_rule = gpr
        rxns.append(r)
    m.add_reactions(rxns)
    return m


def _transcr(threshold=None):
    levels = np.array([[8.0, 1.0], [1.0, 1.0], [0.0, 2.0], [np.nan, np.nan]])
    return TranscrData(["G1", "G2", "G3", "G4"], ["liver", "brain"], ["", ""], levels, threshold)


def _hpa(rows):
    return HPAData(pd.DataFrame(rows, columns=["gene_id", "gene_name", "tissue", "celltype", "level", "reliability"]))


def test_expression_scores_use_the_mean_over_tissues():
    rxn, genes = score_for_init(_model(), None, _transcr(), "liver")
    assert genes["G1"] == pytest.approx(5 * math.log(8 / 4.5))
    assert genes["G2"] == 0.0
    assert genes["G3"] == -5.0
    assert "G4" not in genes
    assert rxn["R1"] == pytest.approx(genes["G1"])
    assert rxn["R2"] == 0.0
    assert rxn["R3"] == pytest.approx(genes["G1"])
    assert rxn["R4"] == -2.0 and rxn["R5"] == -2.0


def test_threshold_replaces_the_mean_and_tissue_match_ignores_case():
    _, genes = score_for_init(_model(), None, _transcr(threshold=2.0), "LIVER")
    assert genes["G1"] == pytest.approx(5 * math.log(4))
    assert genes["G2"] == pytest.approx(5 * math.log(0.5))


def test_hpa_replaces_expression_per_gene_and_per_reaction():
    hpa = _hpa([["G2", "g2", "liver", "hepatocytes", "High", "Supportive"]])
    rxn, genes = score_for_init(_model(), hpa, _transcr(), "liver")
    assert genes["G2"] == 20.0
    assert rxn["R2"] == pytest.approx(5 * math.log(8 / 4.5))
    rxn, _ = score_for_init(_model(), hpa, _transcr(), "liver", data_precedence="reaction")
    assert rxn["R2"] == 20.0
    assert rxn["R1"] == pytest.approx(5 * math.log(8 / 4.5))


def test_multiple_cell_types_take_the_max_or_the_average():
    hpa = _hpa([["G1", "g1", "liver", "a", "High", ""], ["G1", "g1", "liver", "b", "Not detected", ""]])
    assert score_for_init(_model(), hpa, None, "liver")[1]["G1"] == 20.0
    assert score_for_init(_model(), hpa, None, "liver", multiple_cell_scoring="average")[1]["G1"] == 6.0
    assert score_for_init(_model(), hpa, None, "liver", "b")[1]["G1"] == -8.0


@pytest.mark.parametrize("kwargs", [
    {"hpa": None, "transcr": None, "tissue": "liver"},
    {"hpa": None, "transcr": "t", "tissue": "heart"},
    {"hpa": None, "transcr": "t", "tissue": "liver", "celltype": "nope"},
])
def test_bad_input_is_an_error(kwargs):
    transcr = _transcr() if kwargs["transcr"] else None
    with pytest.raises(ValueError):
        score_for_init(_model(), kwargs["hpa"], transcr, kwargs["tissue"], kwargs.get("celltype"))


def test_one_tissue_needs_a_threshold():
    t = TranscrData(["G1"], ["liver"], [""], np.array([[3.0]]))
    with pytest.raises(ValueError):
        score_for_init(_model(), None, t, "liver")
    t.threshold = 3.0
    assert score_for_init(_model(), None, t, "liver")[1]["G1"] == 0.0


def test_unknown_hpa_level_in_another_tissue_is_an_error():
    rows = [["G1", "g1", "liver", "", "High", ""], ["G1", "g1", "kidney", "", "Mixed", ""]]
    with pytest.raises(ValueError):
        score_for_init(_model(), _hpa(rows), None, "liver")


def test_hpa_levels_that_differ_in_case_use_the_first_table_entry():
    rows = [["G1", "g1", "liver", "", "High", ""]]
    scores = score_for_init(_model(), _hpa(rows), None, "liver",
                            hpa_level_scores={"High": 1.0, "HIGH": 9.0})[1]
    assert scores["G1"] == 1.0


def test_unknown_hpa_level_is_an_error():
    with pytest.raises(ValueError):
        score_for_init(_model(), _hpa([["G1", "g1", "liver", "", "Mixed", ""]]), None, "liver")


def test_ftinit_scores_from_data_equal_the_reaction_score_route():
    from tinit_oracles import make_test_model

    model = make_test_model()
    prep = prep_init_model(model, ext_comp="s")
    genes = [g.id for g in model.genes]
    levels = np.array([[5.0 if i % 2 == 0 else 0.2, 1.0] for i in range(len(genes))])
    transcr = TranscrData(genes, ["x", "y"], ["", ""], levels)
    rxn_scores, gene_scores = score_for_init(prep.ref_model, None, transcr, "x")
    from_data = ftinit(prep, "x", transcr=transcr, fill_gaps=False)
    from_scores = ftinit(prep, rxn_scores=rxn_scores, gene_scores=gene_scores, fill_gaps=False)
    assert {r.id for r in from_data.reactions} == {r.id for r in from_scores.reactions}
    assert {g.id for g in from_data.genes} == {g.id for g in from_scores.genes}
    unpruned = ftinit(prep, "x", transcr=transcr, fill_gaps=False, remove_genes=False)
    assert {g.id for g in from_data.genes} <= {g.id for g in unpruned.genes}


def test_ftinit_needs_exactly_one_score_source():
    prep = prep_init_model(_model(), ext_comp="s")
    with pytest.raises(ValueError):
        ftinit(prep)
    with pytest.raises(ValueError):
        ftinit(prep, "x", transcr=_transcr(), rxn_scores={})
