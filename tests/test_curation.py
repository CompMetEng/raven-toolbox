"""Tests for raven_toolbox.curation.batch."""
from __future__ import annotations

import math
import re

import cobra
import pandas as pd
import pytest

from raven_toolbox.curation import (
    CurationResult,
    batch_curate,
    batch_curate_from_tsv,
)


def _base_model() -> cobra.Model:
    """Small yeast-flavoured model with a couple of mets/genes/rxns."""
    m = cobra.Model("base")
    m.compartments = {"c": "cytoplasm", "e": "extracellular"}
    atp = cobra.Metabolite("s_0001", name="ATP", compartment="c",
                            formula="C10H12N5O13P3", charge=-4)
    adp = cobra.Metabolite("s_0002", name="ADP", compartment="c",
                            formula="C10H12N5O10P2", charge=-3)
    glc_e = cobra.Metabolite("s_0003", name="glucose", compartment="e",
                              formula="C6H12O6", charge=0)
    m.add_metabolites([atp, adp, glc_e])
    from cobra.core.gene import Gene
    m.genes.append(Gene("YAL001C", name="A1"))
    r = cobra.Reaction("r_0001", lower_bound=0, upper_bound=1000)
    r.name = "ATP hydrolysis"
    r.add_metabolites({atp: -1, adp: 1})
    r.gene_reaction_rule = "YAL001C"
    m.add_reactions([r])
    return m


# --- metabolites ------------------------------------------------------


def _genes_df(*ids):
    """A genes table that lists ``ids``: the gene rule of a new reaction may only name genes the
    model or the genes table has."""
    return pd.DataFrame({"genes": list(ids)})


def test_add_new_metabolite():
    m = _base_model()
    df = pd.DataFrame([
        {"metNames": "NADH", "comps": "c", "formula": "C21H27N7O14P2",
         "charge": -2, "inchi": "", "metNotes": "added by curation"},
    ])
    result = batch_curate(m, mets_df=df, met_id_prefix="s_")
    assert isinstance(result, CurationResult)
    assert result.added_metabolites == ["s_0004"]
    assert len(result.updated_metabolites) == 0
    new = m.metabolites.get_by_id("s_0004")
    assert new.name == "NADH"
    assert new.compartment == "c"
    assert new.formula == "C21H27N7O14P2"
    assert new.charge == -2
    assert new.notes.get("note") == "added by curation"


def test_update_existing_metabolite():
    m = _base_model()
    df = pd.DataFrame([
        {"metNames": "ATP", "comps": "c", "formula": "C10H16N5O13P3",
         "charge": -3, "inchi": "InChI=ATP", "metNotes": ""},
    ])
    with pytest.warns(UserWarning, match="overwritten"):
        result = batch_curate(m, mets_df=df, met_id_prefix="s_")
    assert result.added_metabolites == []
    assert result.updated_metabolites == ["s_0001"]
    atp = m.metabolites.get_by_id("s_0001")
    assert atp.formula == "C10H16N5O13P3"  # overwritten
    assert atp.charge == -3
    assert atp.notes["inchis"] == "InChI=ATP"


def test_update_keeps_an_annotated_inchi_in_step():
    m = _base_model()
    atp = m.metabolites.get_by_id("s_0001")
    atp.annotation["inchi"] = "InChI=old"
    df = pd.DataFrame([
        {"metNames": "ATP", "comps": "c", "formula": "", "charge": "",
         "inchi": "InChI=ATP", "metNotes": ""},
    ])
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=df, met_id_prefix="s_")
    assert atp.notes["inchis"] == "InChI=ATP"
    assert atp.annotation["inchi"] == "InChI=ATP"


def test_curated_notes_are_written_as_raven_yaml_fields(tmp_path):
    """Notes, references, confidence scores and InChIs reach YAML as RAVEN's own fields."""
    from cobra.io.yaml import yaml as cobra_yaml

    from raven_toolbox.io import read_yaml_model, write_yaml_model

    m = _base_model()
    mets_df = pd.DataFrame([
        {"metNames": "ATP", "comps": "c", "formula": "", "charge": "",
         "inchi": "InChI=ATP", "metNotes": "charge assigned at pH 7.3"},
    ])
    rxns_df = pd.DataFrame([
        {"rxnNames": "ATP hydrolysis", "grRules": "", "lb": "", "ub": "",
         "rev": "", "subSystems": "", "eccodes": "", "rxnNotes": "curated",
         "rxnReferences": "PMID:1", "rxnConfidenceScores": 2},
    ])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "ATP hydrolysis", "metNames": "ATP", "comps": "c",
         "coefficient": -1.0},
        {"rxnNames": "ATP hydrolysis", "metNames": "ADP", "comps": "c",
         "coefficient": 1.0},
    ])
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=mets_df, rxns_df=rxns_df,
                     rxns_coeffs_df=coeffs_df, met_id_prefix="s_",
                     rxn_id_prefix="r_")
    out = tmp_path / "curated.yml"
    write_yaml_model(m, out)

    with open(out, encoding="utf-8") as fh:
        raw = cobra_yaml.load(fh)
    met = next(x for x in raw["metabolites"] if x["id"] == "s_0001")
    assert met["notes"] == "charge assigned at pH 7.3"
    assert met["inchis"] == "InChI=ATP"
    rxn = next(x for x in raw["reactions"] if x["id"] == "r_0001")
    assert rxn["notes"] == "curated"
    assert rxn["references"] == "PMID:1"
    assert rxn["confidence_score"] == 2

    back = read_yaml_model(out)
    assert back.metabolites.get_by_id("s_0001").notes["note"] == "charge assigned at pH 7.3"
    assert back.reactions.get_by_id("r_0001").notes["confidence_score"] == 2


def test_miriam_columns_auto_detected():
    m = _base_model()
    df = pd.DataFrame([
        {"metNames": "NADH", "comps": "c", "formula": "C21H27N7O14P2",
         "charge": -2, "inchi": "", "metNotes": "",
         "kegg.compound": "C00004", "chebi": "CHEBI:16908"},
    ])
    batch_curate(m, mets_df=df, met_id_prefix="s_")
    new = m.metabolites.get_by_id("s_0004")
    assert new.annotation["kegg.compound"] == "C00004"
    assert new.annotation["chebi"] == "CHEBI:16908"


def test_new_metabolite_id_increment_preserves_width():
    """If existing ids are s_0001, s_0002, s_0003 (width 4), new ids
    should be s_0004, s_0005, … not s_4 / s_5."""
    m = _base_model()
    df = pd.DataFrame([
        {"metNames": "X", "comps": "c"},
        {"metNames": "Y", "comps": "c"},
    ])
    result = batch_curate(m, mets_df=df, met_id_prefix="s_")
    assert result.added_metabolites == ["s_0004", "s_0005"]


# --- genes ------------------------------------------------------------

def test_add_new_gene():
    m = _base_model()
    df = pd.DataFrame([
        {"genes": "YBR123C", "geneShortNames": "B2", "uniprot": "P12345"},
    ])
    result = batch_curate(m, genes_df=df)
    assert result.added_genes == ["YBR123C"]
    g = m.genes.get_by_id("YBR123C")
    assert g.name == "B2"
    assert g.annotation["uniprot"] == "P12345"


def test_update_existing_gene():
    m = _base_model()
    df = pd.DataFrame([
        {"genes": "YAL001C", "geneShortNames": "A1_NEW", "uniprot": "P99999"},
    ])
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, genes_df=df)
    g = m.genes.get_by_id("YAL001C")
    assert g.name == "A1_NEW"
    assert g.annotation["uniprot"] == "P99999"


# --- reactions --------------------------------------------------------

def test_add_new_reaction_with_existing_mets():
    m = _base_model()
    rxns_df = pd.DataFrame([
        {"rxnNames": "ADP phosphorylation", "grRules": "YBR456W",
         "lb": -1000, "ub": 1000, "rev": 1, "subSystems": "energy",
         "eccodes": "2.7.4.6", "rxnNotes": "", "rxnReferences": "",
         "rxnConfidenceScores": 3, "kegg.reaction": "R00187"},
    ])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "ADP phosphorylation", "metNames": "ADP", "comps": "c",
         "coefficient": -1.0},
        {"rxnNames": "ADP phosphorylation", "metNames": "ATP", "comps": "c",
         "coefficient": 1.0},
    ])
    result = batch_curate(m, genes_df=_genes_df("YBR456W"), rxns_df=rxns_df,
                          rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert result.added_reactions == ["r_0002"]
    new = m.reactions.get_by_id("r_0002")
    assert new.name == "ADP phosphorylation"
    assert new.gene_reaction_rule == "YBR456W"
    assert new.annotation["ec-code"] == "2.7.4.6"
    assert new.annotation["kegg.reaction"] == "R00187"
    assert new.notes.get("confidence_score") == 3


def test_new_reaction_without_ub_takes_the_stored_default_ub():
    def curate(model):
        rxns_df = pd.DataFrame([
            {"rxnNames": "ADP phosphorylation", "grRules": "", "lb": "", "ub": "", "rev": 1,
             "subSystems": "", "eccodes": "", "rxnNotes": "", "rxnReferences": ""},
        ])
        coeffs_df = pd.DataFrame([
            {"rxnNames": "ADP phosphorylation", "metNames": "ADP", "comps": "c", "coefficient": -1.0},
            {"rxnNames": "ADP phosphorylation", "metNames": "ATP", "comps": "c", "coefficient": 1.0},
        ])
        batch_curate(model, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
        return model.reactions.get_by_id("r_0002")

    plain = curate(_base_model())
    assert plain.upper_bound == float("inf")
    m = _base_model()
    m.notes["metaData"] = {"defaultUB": 1000.0}
    stored = curate(m)
    assert stored.upper_bound == 1000.0
    assert stored.lower_bound == float("-inf")
    annotated = _base_model()
    annotated.annotation["defaultUB"] = 500.0
    assert curate(annotated).upper_bound == 500.0
    precedence = _base_model()
    precedence.notes["metaData"] = {"defaultUB": 1000.0}
    precedence.annotation["defaultUB"] = 500.0
    assert curate(precedence).upper_bound == 1000.0
    both = _base_model()
    both.notes["metaData"] = {"defaultLB": -1000.0, "defaultUB": 1000.0}
    new = curate(both)
    # addRxns gives reactions added from mets and stoichCoeffs lb -inf before it reads defaultLB.
    assert (new.lower_bound, new.upper_bound) == (float("-inf"), 1000.0)


def test_add_new_reaction_joins_list_subsystems():
    # RAVEN allows a reaction in several subsystems (a list). The new-reaction
    # path must ``;``-join it like the update path does, not emit a str(list) repr.
    m = _base_model()
    rxns_df = pd.DataFrame([
        {"rxnNames": "ADP phosphorylation", "grRules": "YBR456W",
         "lb": -1000, "ub": 1000, "rev": 1,
         "subSystems": ["glycolysis", "energy"],
         "eccodes": "2.7.4.6", "rxnNotes": "", "rxnReferences": "",
         "rxnConfidenceScores": 3, "kegg.reaction": "R00187"},
    ])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "ADP phosphorylation", "metNames": "ADP", "comps": "c",
         "coefficient": -1.0},
        {"rxnNames": "ADP phosphorylation", "metNames": "ATP", "comps": "c",
         "coefficient": 1.0},
    ])
    result = batch_curate(m, genes_df=_genes_df("YBR456W"), rxns_df=rxns_df,
                          rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert result.added_reactions == ["r_0002"]
    assert m.reactions.get_by_id("r_0002").subsystem == "glycolysis;energy"


def test_update_existing_reaction_by_stoichiometry():
    m = _base_model()
    rxns_df = pd.DataFrame([
        {"rxnNames": "ATP hydrolysis renamed", "grRules": "YAL999W",
         "lb": -100, "ub": 100, "rev": 1,
         "subSystems": "", "eccodes": "3.6.1.3", "rxnNotes": "updated",
         "rxnReferences": "", "rxnConfidenceScores": 2},
    ])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "ATP hydrolysis renamed", "metNames": "ATP",
         "comps": "c", "coefficient": -1.0},
        {"rxnNames": "ATP hydrolysis renamed", "metNames": "ADP",
         "comps": "c", "coefficient": 1.0},
    ])
    with pytest.warns(UserWarning, match="overwritten"):
        result = batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df,
                              rxn_id_prefix="r_")
    assert result.updated_reactions == ["r_0001"]
    rxn = m.reactions.get_by_id("r_0001")
    assert rxn.name == "ATP hydrolysis renamed"
    assert rxn.gene_reaction_rule == "YAL999W"
    assert rxn.lower_bound == -100
    assert rxn.notes.get("note") == "updated"


def test_rxn_with_unknown_met_raises():
    m = _base_model()
    rxns_df = pd.DataFrame([
        {"rxnNames": "uses unknown", "grRules": "", "lb": 0, "ub": 1000,
         "rev": 0, "subSystems": "", "eccodes": "", "rxnNotes": "",
         "rxnReferences": "", "rxnConfidenceScores": ""},
    ])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "uses unknown", "metNames": "mystery", "comps": "c",
         "coefficient": -1.0},
    ])
    with pytest.raises(ValueError, match="mystery"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df,
                     rxn_id_prefix="r_")


def test_must_provide_both_rxn_tables():
    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnNames": "X"}])
    with pytest.raises(ValueError, match="must be provided together"):
        batch_curate(m, rxns_df=rxns_df)


def test_add_new_met_and_use_in_new_rxn():
    """Common curation pattern: introduce a new metabolite + a new
    reaction that uses it, in a single call."""
    m = _base_model()
    mets_df = pd.DataFrame([
        {"metNames": "NADH", "comps": "c", "formula": "C21H27N7O14P2",
         "charge": -2, "inchi": "", "metNotes": ""},
    ])
    rxns_df = pd.DataFrame([
        {"rxnNames": "made up", "grRules": "", "lb": 0, "ub": 1000,
         "rev": 0, "subSystems": "", "eccodes": "", "rxnNotes": "",
         "rxnReferences": "", "rxnConfidenceScores": ""},
    ])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "made up", "metNames": "ATP", "comps": "c",
         "coefficient": -1.0},
        {"rxnNames": "made up", "metNames": "NADH", "comps": "c",
         "coefficient": 1.0},
    ])
    result = batch_curate(
        m, mets_df=mets_df, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df,
        met_id_prefix="s_", rxn_id_prefix="r_",
    )
    assert result.added_metabolites == ["s_0004"]
    assert result.added_reactions == ["r_0002"]
    new_rxn = m.reactions.get_by_id("r_0002")
    new_met = m.metabolites.get_by_id("s_0004")
    assert new_met in new_rxn.metabolites


# --- from_tsv ---------------------------------------------------------

def test_from_tsv_round_trip(tmp_path):
    m = _base_model()
    mets_path = tmp_path / "mets.tsv"
    mets_path.write_text(
        "metNames\tcomps\tformula\tcharge\tinchi\tmetNotes\tkegg.compound\n"
        "NADH\tc\tC21H27N7O14P2\t-2\t\t\tC00004\n"
    )
    result = batch_curate_from_tsv(m, mets_tsv=mets_path, met_id_prefix="s_")
    assert result.added_metabolites == ["s_0004"]
    new = m.metabolites.get_by_id("s_0004")
    assert new.annotation["kegg.compound"] == "C00004"


def test_empty_call_no_op():
    m = _base_model()
    result = batch_curate(m)
    assert not result
    assert result.added_metabolites == []


# --- RAVEN parity -----------------------------------------------------

def _two_hexokinases() -> tuple[pd.DataFrame, pd.DataFrame]:
    rxns_df = pd.DataFrame([
        {"rxnIdx": 1, "rxnNames": "hexokinase", "grRules": "G1"},
        {"rxnIdx": 2, "rxnNames": "hexokinase", "grRules": "G2"},
    ])
    coeffs_df = pd.DataFrame([
        {"rxnIdx": 1, "rxnNames": "hexokinase", "metNames": "ATP", "comps": "c", "coefficient": -1.0},
        {"rxnIdx": 1, "rxnNames": "hexokinase", "metNames": "ADP", "comps": "c", "coefficient": 3.0},
        {"rxnIdx": 2, "rxnNames": "hexokinase", "metNames": "ATP", "comps": "c", "coefficient": -2.0},
        {"rxnIdx": 2, "rxnNames": "hexokinase", "metNames": "ADP", "comps": "c", "coefficient": 2.0},
    ])
    return rxns_df, coeffs_df


def test_reactions_sharing_a_name_are_told_apart_by_rxn_idx():
    m = _base_model()
    rxns_df, coeffs_df = _two_hexokinases()
    result = batch_curate(m, genes_df=_genes_df("G1", "G2"), rxns_df=rxns_df,
                          rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert result.added_reactions == ["r_0002", "r_0003"]
    first, second = (m.reactions.get_by_id(i) for i in result.added_reactions)
    assert {met.name: c for met, c in first.metabolites.items()} == {"ATP": -1.0, "ADP": 3.0}
    assert {met.name: c for met, c in second.metabolites.items()} == {"ATP": -2.0, "ADP": 2.0}
    assert first.gene_reaction_rule == "G1"
    assert second.gene_reaction_rule == "G2"


def test_index_header_is_read_as_rxn_idx():
    m = _base_model()
    rxns_df, coeffs_df = _two_hexokinases()
    result = batch_curate(
        m,
        genes_df=_genes_df("G1", "G2"),
        rxns_df=rxns_df.rename(columns={"rxnIdx": "index"}),
        rxns_coeffs_df=coeffs_df.rename(columns={"rxnIdx": "Index"}),
        rxn_id_prefix="r_",
    )
    assert len(result.added_reactions) == 2


def test_new_reaction_without_bounds_is_unbounded():
    m = _base_model()
    rxns_df, coeffs_df = _two_hexokinases()
    batch_curate(m, genes_df=_genes_df("G1", "G2"), rxns_df=rxns_df, rxns_coeffs_df=coeffs_df,
                 rxn_id_prefix="r_")
    rxn = m.reactions.get_by_id("r_0002")
    assert rxn.lower_bound == float("-inf")
    assert rxn.upper_bound == float("inf")


def test_raven_header_spellings_are_read():
    m = _base_model()
    mets_df = pd.DataFrame([
        {"metName": "NADH", "comp": "c", "metFormulas": "C21H27N7O14P2", "metCharges": -2},
    ])
    genes_df = pd.DataFrame([{"Gene": "G9", "geneShortName": "NINE"}])
    batch_curate(m, mets_df=mets_df, genes_df=genes_df, met_id_prefix="s_")
    nadh = m.metabolites.get_by_id("s_0004")
    assert nadh.formula == "C21H27N7O14P2"
    assert nadh.charge == -2
    assert m.genes.get_by_id("G9").name == "NINE"


def test_new_ids_use_four_digits_when_no_id_has_the_prefix():
    m = cobra.Model("empty")
    mets_df = pd.DataFrame([{"metNames": "A", "comps": "c"}])
    result = batch_curate(m, mets_df=mets_df)
    assert result.added_metabolites == ["M_0001"]


def test_new_ids_follow_the_numerically_largest_id():
    m = cobra.Model("unpadded")
    m.add_reactions([cobra.Reaction("R_9"), cobra.Reaction("R_10")])
    m.add_metabolites([cobra.Metabolite("M_1", name="A", compartment="c")])
    rxns_df = pd.DataFrame([{"rxnNames": "x"}])
    coeffs_df = pd.DataFrame([{"rxnNames": "x", "metNames": "A", "comps": "c", "coefficient": -1.0}])
    result = batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df)
    assert result.added_reactions == ["R_11"]


# --- empty cells of existing entities ----------------------------------

def _empty_mets_table():
    return pd.DataFrame([
        {"metNames": "ATP", "comps": "c", "formula": "", "charge": float("nan"), "metNotes": ""},
        {"metNames": "ADP", "comps": "c", "formula": "C10H12N5O10P2", "charge": float("nan"),
         "metNotes": ""},
        {"metNames": "glucose", "comps": "e", "formula": "C6H12O6", "charge": -1.0, "metNotes": ""},
    ])


def test_empty_metabolite_cells_overwrite_by_default():
    m = _base_model()
    m.metabolites.get_by_id("s_0001").notes["note"] = "kept?"
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=_empty_mets_table(), met_id_prefix="s_")
    atp, adp, glc = (m.metabolites.get_by_id(i) for i in ("s_0001", "s_0002", "s_0003"))
    assert atp.formula == "" and atp.charge == 0 and "note" not in atp.notes
    assert adp.formula == "C10H12N5O10P2" and adp.charge == 0
    assert glc.charge == -1


def test_empty_metabolite_cells_are_kept_without_overwrite_with_empty():
    m = _base_model()
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=_empty_mets_table(), met_id_prefix="s_",
                     overwrite_with_empty=False)
    atp, adp, glc = (m.metabolites.get_by_id(i) for i in ("s_0001", "s_0002", "s_0003"))
    assert atp.formula == "C10H12N5O13P3" and atp.charge == -4
    assert adp.formula == "C10H12N5O10P2" and adp.charge == -3
    assert glc.charge == -1


def test_a_charge_column_without_any_value_is_not_written():
    m = _base_model()
    df = pd.DataFrame([{"metNames": "ATP", "comps": "c", "formula": "X", "charge": float("nan")}])
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=df, met_id_prefix="s_")
    assert m.metabolites.get_by_id("s_0001").charge == -4


def test_an_absent_column_is_not_written():
    m = _base_model()
    df = pd.DataFrame([{"metNames": "ATP", "comps": "c", "charge": -2}])
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=df, met_id_prefix="s_")
    atp = m.metabolites.get_by_id("s_0001")
    assert atp.formula == "C10H12N5O13P3" and atp.charge == -2


@pytest.mark.parametrize("overwrite, name", [(True, ""), (False, "A1")])
def test_empty_gene_short_name(overwrite, name):
    m = _base_model()
    df = pd.DataFrame([{"genes": "YAL001C", "geneShortNames": ""}])
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, genes_df=df, overwrite_with_empty=overwrite)
    assert m.genes.get_by_id("YAL001C").name == name


def _atp_tables(**cells):
    rxns_df = pd.DataFrame([{"rxnNames": "ATP hydrolysis", **cells}])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "ATP hydrolysis", "metNames": "ATP", "comps": "c", "coefficient": -1.0},
        {"rxnNames": "ATP hydrolysis", "metNames": "ADP", "comps": "c", "coefficient": 1.0},
    ])
    return rxns_df, coeffs_df


@pytest.mark.parametrize("overwrite", [True, False])
def test_empty_bounds_and_gene_rule_never_change_an_existing_reaction(overwrite):
    m = _base_model()
    rxns_df, coeffs_df = _atp_tables(lb=float("nan"), ub="", grRules="")
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_",
                     overwrite_with_empty=overwrite)
    rxn = m.reactions.get_by_id("r_0001")
    assert rxn.bounds == (0, 1000)
    assert rxn.gene_reaction_rule == "YAL001C"


@pytest.mark.parametrize("overwrite, subsystem, ec", [(True, "", None), (False, "Energy", "3.6.1.3")])
def test_empty_annotation_cells_of_an_existing_reaction(overwrite, subsystem, ec):
    m = _base_model()
    rxn = m.reactions.get_by_id("r_0001")
    rxn.subsystem = "Energy"
    rxn.annotation["ec-code"] = "3.6.1.3"
    rxns_df, coeffs_df = _atp_tables(subSystems="", eccodes="")
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_",
                     overwrite_with_empty=overwrite)
    assert rxn.subsystem == subsystem
    assert rxn.annotation.get("ec-code") == ec


def test_new_reaction_bounds_with_an_empty_cell():
    m = _base_model()
    rxns_df = pd.DataFrame([
        {"rxnNames": "bounded", "lb": -5, "ub": 7},
        {"rxnNames": "openub", "lb": 0, "ub": float("nan")},
    ])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "bounded", "metNames": "ATP", "comps": "c", "coefficient": -2.0},
        {"rxnNames": "bounded", "metNames": "ADP", "comps": "c", "coefficient": 3.0},
        {"rxnNames": "openub", "metNames": "ATP", "comps": "c", "coefficient": -3.0},
        {"rxnNames": "openub", "metNames": "ADP", "comps": "c", "coefficient": 5.0},
    ])
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    bounded = next(r for r in m.reactions if r.name == "bounded")
    openub = next(r for r in m.reactions if r.name == "openub")
    assert bounded.bounds == (-5, 7)
    assert openub.lower_bound == 0 and openub.upper_bound == float("inf")


# --- MIRIAM, zero fill, text cells ------------------------------------

def _annotated_model() -> cobra.Model:
    m = _base_model()
    m.metabolites.get_by_id("s_0001").annotation["kegg.compound"] = "C1"
    m.metabolites.get_by_id("s_0002").annotation["kegg.compound"] = "C2"
    m.genes.get_by_id("YAL001C").annotation["ncbigene"] = "11"
    m.reactions.get_by_id("r_0001").annotation["kegg.reaction"] = "R1"
    return m


def _kegg_mets_table(first, second):
    return pd.DataFrame([
        {"metNames": "ATP", "comps": "c", "kegg.compound": first},
        {"metNames": "ADP", "comps": "c", "kegg.compound": second},
    ])


@pytest.mark.parametrize("overwrite, kept", [(True, False), (False, True)])
def test_an_empty_miriam_cell_removes_the_namespace_unless_kept(overwrite, kept):
    m = _annotated_model()
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=_kegg_mets_table("", "C9"), overwrite_with_empty=overwrite)
    atp, adp = m.metabolites.get_by_id("s_0001"), m.metabolites.get_by_id("s_0002")
    assert adp.annotation["kegg.compound"] == "C9"
    assert ("kegg.compound" in atp.annotation) is kept


@pytest.mark.parametrize("overwrite", [True, False])
def test_a_miriam_column_without_any_value_is_not_written(overwrite):
    # Pin: a column without any value is not written, in both modes.
    m = _annotated_model()
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=_kegg_mets_table("", ""), overwrite_with_empty=overwrite)
    assert m.metabolites.get_by_id("s_0001").annotation["kegg.compound"] == "C1"
    assert m.metabolites.get_by_id("s_0002").annotation["kegg.compound"] == "C2"


def test_a_miriam_column_counts_only_the_existing_entities():
    # Pin: the existing entity keeps its namespace because the only value of the column is on a new one.
    m = _annotated_model()
    df = pd.DataFrame([
        {"metNames": "ATP", "comps": "c", "kegg.compound": ""},
        {"metNames": "NADH", "comps": "c", "kegg.compound": "C4"},
    ])
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=df, met_id_prefix="s_")
    assert m.metabolites.get_by_id("s_0001").annotation["kegg.compound"] == "C1"
    assert m.metabolites.get_by_id("s_0004").annotation["kegg.compound"] == "C4"


def test_an_empty_gene_miriam_cell_removes_the_namespace():
    m = _annotated_model()
    from cobra.core.gene import Gene

    m.genes.append(Gene("YAL002W"))
    m.genes.get_by_id("YAL002W").annotation["ncbigene"] = "22"
    df = pd.DataFrame([{"genes": "YAL001C", "ncbigene": ""}, {"genes": "YAL002W", "ncbigene": "33"}])
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, genes_df=df)
    assert "ncbigene" not in m.genes.get_by_id("YAL001C").annotation
    assert m.genes.get_by_id("YAL002W").annotation["ncbigene"] == "33"


def test_an_empty_reaction_miriam_cell_removes_the_namespace():
    m = _annotated_model()
    atp, adp = m.metabolites.get_by_id("s_0001"), m.metabolites.get_by_id("s_0002")
    reverse = cobra.Reaction("r_0002", name="ATP synthesis")
    m.add_reactions([reverse])
    reverse.add_metabolites({atp: 1, adp: -1})
    reverse.annotation["kegg.reaction"] = "R2"
    rxns_df = pd.DataFrame([
        {"rxnNames": "ATP hydrolysis", "kegg.reaction": "", "metacyc": "RXN-1"},
        {"rxnNames": "ATP synthesis", "kegg.reaction": "R5", "metacyc": ""},
    ])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "ATP hydrolysis", "metNames": "ATP", "comps": "c", "coefficient": -1.0},
        {"rxnNames": "ATP hydrolysis", "metNames": "ADP", "comps": "c", "coefficient": 1.0},
        {"rxnNames": "ATP synthesis", "metNames": "ATP", "comps": "c", "coefficient": 1.0},
        {"rxnNames": "ATP synthesis", "metNames": "ADP", "comps": "c", "coefficient": -1.0},
    ])
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    hydrolysis = m.reactions.get_by_id("r_0001")
    assert "kegg.reaction" not in hydrolysis.annotation
    assert hydrolysis.annotation["metacyc"] == "RXN-1"
    assert reverse.annotation["kegg.reaction"] == "R5"
    assert "metacyc" not in reverse.annotation


def test_a_new_metabolite_without_a_charge_takes_zero_when_the_column_has_a_value():
    m = _base_model()
    df = pd.DataFrame([
        {"metNames": "NADH", "comps": "c", "charge": float("nan")},
        {"metNames": "NAD", "comps": "c", "charge": -1.0},
    ])
    batch_curate(m, mets_df=df, met_id_prefix="s_")
    charges = {met.name: met.charge for met in m.metabolites if met.name in ("NADH", "NAD")}
    assert charges == {"NADH": 0, "NAD": -1}


def test_a_new_reaction_without_a_confidence_score_takes_zero_when_the_column_has_a_value():
    m = _base_model()
    rxns_df = pd.DataFrame([
        {"rxnNames": "scored", "rxnConfidenceScores": 3},
        {"rxnNames": "unscored", "rxnConfidenceScores": float("nan")},
    ])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "scored", "metNames": "ATP", "comps": "c", "coefficient": -2.0},
        {"rxnNames": "scored", "metNames": "ADP", "comps": "c", "coefficient": 3.0},
        {"rxnNames": "unscored", "metNames": "ATP", "comps": "c", "coefficient": -3.0},
        {"rxnNames": "unscored", "metNames": "ADP", "comps": "c", "coefficient": 5.0},
    ])
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    scores = {r.name: r.notes.get("confidence_score") for r in m.reactions if r.name in ("scored", "unscored")}
    assert scores == {"scored": 3, "unscored": 0}


def test_tsv_cells_such_as_na_stay_text_and_numbers_stay_as_written(tmp_path):
    m = _base_model()
    genes = tmp_path / "genes.tsv"
    genes.write_text("genes\tgeneShortNames\tncbigene\nYAL001C\tNA\t22\n")
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate_from_tsv(m, genes_tsv=genes)
    gene = m.genes.get_by_id("YAL001C")
    assert gene.name == "NA"
    assert gene.annotation["ncbigene"] == "22"


def test_tsv_blank_cells_are_empty(tmp_path):
    # Pin: a cell with only blanks is empty, as RAVEN reads it.
    m = _base_model()
    genes = tmp_path / "genes.tsv"
    genes.write_text("genes\tgeneShortNames\nYAL001C\t  \n")
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate_from_tsv(m, genes_tsv=genes)
    assert m.genes.get_by_id("YAL001C").name == ""


def test_nullable_pandas_dtypes_are_read_as_empty():
    m = _base_model()
    df = pd.DataFrame([
        {"metNames": "ATP", "comps": "c", "charge": pd.NA},
        {"metNames": "ADP", "comps": "c", "charge": -2},
    ]).convert_dtypes()
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=df)
    assert m.metabolites.get_by_id("s_0001").charge == 0
    assert m.metabolites.get_by_id("s_0002").charge == -2


@pytest.mark.parametrize("value", ["false", 0, 1, None, "True"])
def test_overwrite_with_empty_must_be_a_bool(value):
    with pytest.raises(TypeError, match="overwrite_with_empty must be a bool"):
        batch_curate(_base_model(), overwrite_with_empty=value)


def test_overwrite_with_empty_accepts_a_numpy_bool():
    import numpy as np

    m = _annotated_model()
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=_kegg_mets_table("", "C9"), overwrite_with_empty=np.False_)
    assert m.metabolites.get_by_id("s_0001").annotation["kegg.compound"] == "C1"
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=_kegg_mets_table("", "C9"), overwrite_with_empty=np.True_)
    assert "kegg.compound" not in m.metabolites.get_by_id("s_0001").annotation


def test_the_overwrite_with_empty_error_names_both_bool_types():
    with pytest.raises(TypeError, match=r"bool or numpy\.bool_"):
        batch_curate(_base_model(), overwrite_with_empty="no")


@pytest.mark.parametrize("blank", [" ", "   "])
@pytest.mark.parametrize("overwrite, kept", [(True, False), (False, True)])
def test_a_blank_only_miriam_cell_is_empty_in_both_modes(blank, overwrite, kept):
    m = _annotated_model()
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=_kegg_mets_table(blank, "C9"), overwrite_with_empty=overwrite)
    assert ("kegg.compound" in m.metabolites.get_by_id("s_0001").annotation) is kept


# --- numeric cells ----------------------------------------------------

def _bound_tables(**cells):
    rxns_df = pd.DataFrame([
        {"rxnNames": "ATP hydrolysis", **{k: v[0] for k, v in cells.items()}},
        {"rxnNames": "fresh", **{k: v[1] for k, v in cells.items()}},
    ])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "ATP hydrolysis", "metNames": "ATP", "comps": "c", "coefficient": -1.0},
        {"rxnNames": "ATP hydrolysis", "metNames": "ADP", "comps": "c", "coefficient": 1.0},
        {"rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": -3.0},
        {"rxnNames": "fresh", "metNames": "ADP", "comps": "c", "coefficient": 5.0},
    ])
    return rxns_df, coeffs_df


def _by_name(m, name):
    return next(r for r in m.reactions if r.name == name)


@pytest.mark.parametrize("nan", ["NaN", "nan", " NAN ", float("nan")])
def test_a_nan_bound_is_not_given_for_existing_and_new_reactions(nan):
    m = _base_model()
    old = _by_name(m, "ATP hydrolysis").bounds
    rxns_df, coeffs_df = _bound_tables(lb=(nan, nan), ub=(nan, nan))
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert _by_name(m, "ATP hydrolysis").bounds == old
    assert _by_name(m, "fresh").bounds == (-math.inf, math.inf)


@pytest.mark.parametrize("text, value", [("true", 1.0), ("FALSE", 0.0), (" True ", 1.0)])
def test_true_and_false_cells_are_read_as_one_and_zero(text, value):
    m = _base_model()
    rxns_df, coeffs_df = _bound_tables(lb=(text, text), ub=("1000", "1000"), rev=(text, text))
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert _by_name(m, "ATP hydrolysis").lower_bound == value
    assert _by_name(m, "fresh").lower_bound == value


@pytest.mark.parametrize("column", ["lb", "ub", "rev"])
@pytest.mark.parametrize("text", ["abc", "NA", "None", "1 0", "pi", "1i", "1,5", "1,000", "0x10",
                                  "1d3", "1_0", "Infinity", "٣", "１"])
def test_a_reaction_cell_that_is_not_a_number_is_refused(column, text):
    m = _base_model()
    rxns_df, coeffs_df = _bound_tables(**{column: ("1", text)})
    with pytest.raises(ValueError, match=f'{column} value "{text}"'):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")


@pytest.mark.parametrize("text", ["abc", "NA", "None", "pi", "1i", "1,5", "0x10"])
def test_a_charge_that_is_not_a_number_is_refused(text):
    df = pd.DataFrame([{"metNames": "ATP", "comps": "c", "charge": text},
                       {"metNames": "ADP", "comps": "c", "charge": -2}])
    with pytest.raises(ValueError, match=f'charge value "{text}"'):
        batch_curate(_base_model(), mets_df=df)


@pytest.mark.parametrize("text", ["0.5", "2.5", "-1.5", "1e-3", "Inf", "-inf", "+Inf"])
def test_a_charge_that_is_not_a_finite_whole_number_is_refused(text):
    df = pd.DataFrame([{"metNames": "ATP", "comps": "c", "charge": text},
                       {"metNames": "ADP", "comps": "c", "charge": -2}])
    with pytest.raises(ValueError, match=re.escape(f'charge value "{text}"') + ".*finite whole number"):
        batch_curate(_base_model(), mets_df=df)


# Pin: whole-number charges in these notations were read before the check; this keeps them read.
@pytest.mark.parametrize("text, charge", [("2.0", 2), ("2e0", 2), ("-3", -3), ("+1", 1),
                                         (" 4 ", 4), (2.0, 2), ("-0", 0)])
def test_a_whole_number_charge_is_accepted_in_any_notation(text, charge):
    df = pd.DataFrame([{"metNames": "ATP", "comps": "c", "charge": text},
                       {"metNames": "ADP", "comps": "c", "charge": -2}])
    m = _base_model()
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=df)
    assert m.metabolites.get_by_id("s_0001").charge == charge


@pytest.mark.parametrize("text", ["2", "0.5", "Inf", "-inf", "-1", "1.5", "2e0"])
def test_a_reversibility_cell_other_than_zero_one_true_false_is_refused(text):
    m = _base_model()
    rxns_df, coeffs_df = _bound_tables(rev=("1", text))
    with pytest.raises(ValueError, match=re.escape(f'rev value "{text}"') + ".*0, 1, true or false"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")


@pytest.mark.parametrize("table", ["info", "coeffs"])
@pytest.mark.parametrize("text", ["1.5", "Inf", "0", "-1", "-inf", "0.5"])
def test_an_rxn_index_that_is_not_a_whole_number_of_at_least_one_is_refused(table, text):
    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnIdx": "1", "rxnNames": "fresh", "lb": "0"}])
    coeffs_df = pd.DataFrame([{"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ATP", "comps": "c",
                               "coefficient": "-1"}])
    (rxns_df if table == "info" else coeffs_df).loc[0, "rxnIdx"] = text
    label = "rxnsInfo" if table == "info" else "rxnsCoeffs"
    with pytest.raises(ValueError, match=re.escape(f'rxnIdx value "{text}"') + f".*{label}"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")


# Pin: whole-number indices were accepted before the check; this keeps them accepted.
@pytest.mark.parametrize("text", ["1", "2", "3.0", "1e0", " 4 "])
def test_an_rxn_index_that_is_a_whole_number_of_at_least_one_is_accepted(text):
    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnIdx": text, "rxnNames": "fresh", "lb": "0"}])
    coeffs_df = pd.DataFrame([{"rxnIdx": text, "rxnNames": "fresh", "metNames": "ATP", "comps": "c",
                               "coefficient": "-1"}])
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert _by_name(m, "fresh") is not None


def test_a_coefficient_row_without_any_content_is_ignored():
    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnIdx": "1", "rxnNames": "fresh", "lb": "0"}])
    coeffs_df = pd.DataFrame([
        {"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": "-1"},
        {"rxnIdx": "", "rxnNames": "", "metNames": "", "comps": "", "coefficient": ""}])
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert len(_by_name(m, "fresh").metabolites) == 1


def test_a_coefficients_table_without_a_coefficient_column_names_the_missing_columns():
    rxns_df = pd.DataFrame([{"rxnNames": "fresh", "lb": "0"}])
    coeffs_df = pd.DataFrame([{"rxnNames": "fresh", "metNames": "ATP", "comps": "c"}])
    with pytest.raises(ValueError, match="rxns_coeffs_df is missing required columns"):
        batch_curate(_base_model(), rxns_df=rxns_df, rxns_coeffs_df=coeffs_df)


def _snapshot(model):
    return (
        [(m.id, m.name, m.formula, m.charge, dict(m.annotation)) for m in model.metabolites],
        [(r.id, r.name, r.bounds, dict(r.notes)) for r in model.reactions],
        [(g.id, g.name) for g in model.genes],
    )


def _failing_tables():
    ok_mets = pd.DataFrame([{"metNames": "ATP", "comps": "c", "formula": "X", "charge": "7"}])
    return {
        "charge": dict(mets_df=pd.DataFrame([
            {"metNames": "ATP", "comps": "c", "charge": "7"},
            {"metNames": "ADP", "comps": "c", "charge": "Inf"}])),
        "rev": dict(mets_df=ok_mets, rxns_df=pd.DataFrame([
            {"rxnNames": "fresh", "lb": "0", "rev": "2"}]),
            rxns_coeffs_df=pd.DataFrame([
                {"rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": "-1"}])),
        "unknown metabolite": dict(mets_df=ok_mets, rxns_df=pd.DataFrame([
            {"rxnNames": "fresh", "lb": "0"}]),
            rxns_coeffs_df=pd.DataFrame([
                {"rxnNames": "fresh", "metNames": "nothing", "comps": "c", "coefficient": "-1"}])),
        "no coefficient rows": dict(mets_df=ok_mets, rxns_df=pd.DataFrame([
            {"rxnNames": "fresh", "lb": "0"}]),
            rxns_coeffs_df=pd.DataFrame([
                {"rxnNames": "other", "metNames": "ATP", "comps": "c", "coefficient": "-1"}])),
        "tables not given together": dict(mets_df=ok_mets, rxns_df=pd.DataFrame([
            {"rxnNames": "fresh", "lb": "0"}])),
    }


@pytest.mark.parametrize("case", list(_failing_tables()))
def test_a_table_that_fails_leaves_the_model_unchanged(case):
    m = _base_model()
    before = _snapshot(m)
    with pytest.raises(ValueError):
        batch_curate(m, **_failing_tables()[case])
    assert _snapshot(m) == before


# Pin: these rev cells were accepted before the check; this keeps them accepted.
@pytest.mark.parametrize("text", ["0", "1", "1.0", "true", "FALSE", " True ", "NaN", ""])
def test_a_reversibility_cell_of_zero_one_true_false_or_nothing_is_accepted(text):
    m = _base_model()
    rxns_df, coeffs_df = _bound_tables(rev=(text, text))
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")


@pytest.mark.parametrize("text", ["abc", "NA", "None"])
def test_a_confidence_score_that_is_not_a_number_is_refused(text):
    m = _base_model()
    rxns_df, coeffs_df = _bound_tables(rxnConfidenceScores=("3", text))
    with pytest.raises(ValueError, match="rxnConfidenceScores value"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")


def test_charges_bounds_and_scores_are_read_as_plain_decimal_numbers():
    df = pd.DataFrame([{"metNames": "ATP", "comps": "c", "charge": "-4"},
                       {"metNames": "ADP", "comps": "c", "charge": " 2 "}])
    m = _base_model()
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=df)
    assert [m.metabolites.get_by_id(i).charge for i in ("s_0001", "s_0002")] == [-4, 2]
    m = _base_model()
    rxns_df, coeffs_df = _bound_tables(lb=(".5", "1e2"), ub=("Inf", "+inf"),
                                       rxnConfidenceScores=("2", " 1e0 "))
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert _by_name(m, "ATP hydrolysis").bounds == (0.5, math.inf)
    assert _by_name(m, "fresh").bounds == (100.0, math.inf)
    assert _by_name(m, "ATP hydrolysis").notes["confidence_score"] == 2
    assert _by_name(m, "fresh").notes["confidence_score"] == 1


def test_a_non_breaking_space_is_not_trimmed_from_a_cell():
    rxns_df, coeffs_df = _bound_tables(lb=("1", "5\u00a0"))
    with pytest.raises(ValueError, match="lb value"):
        batch_curate(_base_model(), rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    m = _base_model()
    df = pd.DataFrame([{"metNames": "ATP", "comps": "c", "formula": "\u00a0"}])
    batch_curate(m, mets_df=df, overwrite_with_empty=False)
    assert m.metabolites.get_by_id("s_0001").formula == "\u00a0"
    m = _base_model()
    df = pd.DataFrame([{"metNames": "ATP", "comps": "c", "formula": "\t \n"}])
    batch_curate(m, mets_df=df, overwrite_with_empty=False)
    assert m.metabolites.get_by_id("s_0001").formula == "C10H12N5O13P3"


@pytest.mark.parametrize("value", [math.inf, -math.inf, "Inf", "-inf"])
def test_a_charge_that_is_not_finite_is_refused(value):
    df = pd.DataFrame([{"metNames": "ATP", "comps": "c", "charge": value}])
    with pytest.raises(ValueError, match="not a finite whole number"):
        batch_curate(_base_model(), mets_df=df)


@pytest.mark.parametrize("coefficient", ["abc", "1,5", "0x10", "", "NaN"])
def test_a_coefficient_that_is_not_a_number_is_refused(coefficient):
    rxns_df, coeffs_df = _bound_tables(lb=("0", "0"))
    coeffs_df = coeffs_df.astype({"coefficient": object})
    coeffs_df.loc[3, "coefficient"] = coefficient
    with pytest.raises(ValueError, match="coefficient|number"):
        batch_curate(_base_model(), rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")


def test_a_reaction_index_that_is_not_a_number_is_refused():
    rxns_df, coeffs_df = _bound_tables(lb=("0", "0"))
    rxns_df["rxnIdx"] = ["1", "x2"]
    coeffs_df["rxnIdx"] = ["1", "1", "x2", "x2"]
    with pytest.raises(ValueError, match="rxnIdx value"):
        batch_curate(_base_model(), rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")


def test_a_nul_character_in_a_table_file_stays_in_the_cell(tmp_path):
    path = tmp_path / "mets.tsv"
    path.write_bytes(b"metNames\tcomps\tformula\tcharge\nATP\tc\tCH4\t5\x00\n")
    with pytest.raises(ValueError, match="charge value"):
        batch_curate_from_tsv(_base_model(), mets_tsv=path)


# --- table layout, bounds and reaction tables -----------------------------

def _write(path, text):
    path.write_bytes(text.encode("utf-8"))
    return path


def _two_tables(rxns_rows, coeff_rows, rxns_header="rxnIdx\trxnNames\tlb\tub",
                coeff_header="rxnIdx\trxnNames\tmetNames\tcomps\tcoefficient"):
    rxns = pd.DataFrame([r.split("\t") for r in rxns_rows], columns=rxns_header.split("\t"))
    coeffs = pd.DataFrame([r.split("\t") for r in coeff_rows], columns=coeff_header.split("\t"))
    return rxns, coeffs


def test_both_bounds_of_an_existing_reaction_are_set_together():
    m = _base_model()
    m.reactions.r_0001.bounds = (0, 10)
    rxns_df, coeffs_df = _bound_tables(lb=("20", "0"), ub=("30", "1"))
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert _by_name(m, "ATP hydrolysis").bounds == (20.0, 30.0)


@pytest.mark.parametrize("lb, ub", [("5", "1"), ("2000", "")])
def test_a_lower_bound_above_the_upper_bound_of_an_existing_reaction_is_refused_before_any_change(lb, ub):
    m = _base_model()
    rxns_df, coeffs_df = _bound_tables(lb=(lb, "0"), ub=(ub, "1"))
    mets_df = pd.DataFrame([{"metNames": "NADH", "comps": "c", "formula": "X"}])
    before = _snapshot(m)
    with pytest.raises(ValueError, match="lower bound .* larger than its upper bound"):
        batch_curate(m, mets_df=mets_df, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert _snapshot(m) == before


def test_a_lower_bound_above_the_upper_bound_of_a_new_reaction_is_refused_before_any_change():
    m = _base_model()
    rxns_df, coeffs_df = _bound_tables(lb=("", "5"), ub=("", "1"))
    before = _snapshot(m)
    with pytest.raises(ValueError, match="'fresh'"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert _snapshot(m) == before


def test_a_trailing_tab_on_every_row_does_not_shift_the_columns(tmp_path):
    m = _base_model()
    path = _write(tmp_path / "mets.tsv", "metNames\tcomps\tformula\nNADH\tc\tX\t\nNAD\tc\tY\t\n")
    batch_curate_from_tsv(m, mets_tsv=path, met_id_prefix="s_")
    assert {(x.name, x.compartment, x.formula) for x in m.metabolites if x.name.startswith("NAD")} == {
        ("NADH", "c", "X"), ("NAD", "c", "Y")}


def test_a_row_with_more_cells_than_the_header_is_refused(tmp_path):
    path = _write(tmp_path / "mets.tsv", "metNames\tcomps\tformula\nNADH\tc\tX\tZ\nNAD\tc\tY\n")
    before = _snapshot(_base_model())
    m = _base_model()
    with pytest.raises(ValueError, match="Line 2 .* 4 cells but the header has 3"):
        batch_curate_from_tsv(m, mets_tsv=path, met_id_prefix="s_")
    assert _snapshot(m) == before


def test_spaces_before_a_text_cell_are_dropped_and_spaces_after_it_are_kept(tmp_path):
    m = _base_model()
    path = _write(tmp_path / "mets.tsv",
                  'metNames\tcomps\tformula\n C\t c\t H2O \n  \"D\"\tc\t \" X \"\n')
    batch_curate_from_tsv(m, mets_tsv=path, met_id_prefix="s_")
    assert {(x.name, x.compartment, x.formula) for x in m.metabolites if x.id > "s_0003"} == {
        ("C", "c", "H2O "), ("D", "c", " X ")}


def test_a_coefficient_row_names_its_metabolite_without_leading_spaces(tmp_path):
    m = _base_model()
    rxns = _write(tmp_path / "rxns.tsv", "rxnIdx\trxnNames\tlb\n1\tfresh\t0\n")
    coeffs = _write(tmp_path / "coeffs.tsv",
                    "rxnIdx\trxnNames\tmetNames\tcomps\tcoefficient\n1\tfresh\t ATP\t c\t-2\n"
                    "1\tfresh\tADP\tc\t2\n")
    result = batch_curate_from_tsv(m, rxns_tsv=rxns, rxns_coeffs_tsv=coeffs, rxn_id_prefix="r_")
    assert len(result.added_reactions) == 1


def test_a_very_long_cell_and_a_quote_followed_by_text_are_read(tmp_path):
    m = _base_model()
    path = _write(tmp_path / "mets.tsv",
                  "metNames\tcomps\tformula\nNADH\tc\t" + "C" * 200000 + '\n"NA"D\tc\tY\n')
    batch_curate_from_tsv(m, mets_tsv=path, met_id_prefix="s_")
    formulas = {x.name: x.formula for x in m.metabolites if x.id > "s_0003"}
    assert len(formulas["NADH"]) == 200000
    assert formulas["NAD"] == "Y"


def test_an_unmatched_opening_quote_takes_the_rest_of_the_file_as_one_cell(tmp_path):
    path = _write(tmp_path / "mets.tsv", 'metNames\tcomps\tformula\nNADH\tc\t"X\nNAD\tc\tY\n')
    m = _base_model()
    batch_curate_from_tsv(m, mets_tsv=path, met_id_prefix="s_")
    assert [(x.name, x.formula) for x in m.metabolites if x.id > "s_0003"] == [
        ("NADH", "X\nNAD\tc\tY\n")]


# Pin: the numbered name is pandas' own convention for a repeated header.
def test_a_repeated_header_is_stored_under_a_numbered_name(tmp_path):
    path = _write(tmp_path / "mets.tsv", "metNames\tcomps\tchebi\tchebi\nNADH\tc\t1\t2\n")
    m = _base_model()
    batch_curate_from_tsv(m, mets_tsv=path, met_id_prefix="s_")
    assert m.metabolites.s_0004.annotation == {"chebi": "1", "chebi.1": "2"}


def test_an_empty_table_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="is empty"):
        batch_curate_from_tsv(_base_model(), mets_tsv=_write(tmp_path / "mets.tsv", ""))


def test_a_metabolite_row_of_tabs_only_is_refused():
    m = _base_model()
    mets_df = pd.DataFrame([{"metNames": "NADH", "comps": "c"}, {"metNames": "", "comps": ""}])
    before = _snapshot(m)
    with pytest.raises(ValueError, match="Row 2 .* no compartment"):
        batch_curate(m, mets_df=mets_df)
    assert _snapshot(m) == before


@pytest.mark.parametrize("table, columns", [
    ("mets_df", {"metNames": ["NADH"], "formula": ["X"]}),
    ("genes_df", {"geneShortNames": ["a"]}),
])
def test_a_table_without_its_key_column_is_refused_before_any_change(table, columns):
    m = _base_model()
    first = pd.DataFrame([{"metNames": "NADH", "comps": "c"}])
    kwargs = {"mets_df": first} if table == "genes_df" else {}
    kwargs[table] = pd.DataFrame(columns)
    before = _snapshot(m)
    with pytest.raises(ValueError, match="missing required columns"):
        batch_curate(m, **kwargs)
    assert _snapshot(m) == before


def test_a_coefficient_row_with_only_a_coefficient_or_only_an_extra_column_is_ignored():
    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnIdx": "1", "rxnNames": "fresh"}])
    coeffs_df = pd.DataFrame([
        {"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": "-2", "note": ""},
        {"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ADP", "comps": "c", "coefficient": "2", "note": ""},
        {"rxnIdx": "", "rxnNames": "", "metNames": "", "comps": "", "coefficient": "7", "note": ""},
        {"rxnIdx": "", "rxnNames": "", "metNames": "", "comps": "", "coefficient": "", "note": "remark"},
    ])
    result = batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert len(result.added_reactions) == 1


def test_a_reaction_row_of_tabs_only_is_ignored():
    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnIdx": "1", "rxnNames": "fresh"}, {"rxnIdx": "", "rxnNames": ""}])
    coeffs_df = pd.DataFrame([
        {"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": "-2"},
        {"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ADP", "comps": "c", "coefficient": "2"},
    ])
    result = batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert len(result.added_reactions) == 1


@pytest.mark.parametrize("table", ["rxns", "coeffs"])
def test_a_row_with_content_but_no_rxn_idx_is_refused(table):
    rxns_df, coeffs_df = _two_tables(
        ["1\tfresh\t0\t5"],
        ["1\tfresh\tATP\tc\t-2", "1\tfresh\tADP\tc\t2"])
    if table == "rxns":
        rxns_df.loc[0, "rxnIdx"] = ""
    else:
        coeffs_df.loc[1, "rxnIdx"] = ""
    m = _base_model()
    before = _snapshot(m)
    with pytest.raises(ValueError, match="has values but its rxnIdx is empty or NaN"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert _snapshot(m) == before


def test_two_coefficient_rows_for_one_metabolite_add_up():
    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnIdx": "1", "rxnNames": "fresh"}])
    coeffs_df = pd.DataFrame([
        {"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": "-1"},
        {"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": "-2"},
        {"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ADP", "comps": "c", "coefficient": "2"},
    ])
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert {x.name: c for x, c in _by_name(m, "fresh").metabolites.items()} == {"ATP": -3.0, "ADP": 2.0}


@pytest.mark.parametrize("rxn_idx, coeff_idx", [(1.0, 1), ("1", "1.0"), ("1e0", "1"), (1, "1.0")])
def test_rxn_idx_is_matched_by_value_in_both_tables(rxn_idx, coeff_idx):
    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnIdx": rxn_idx, "rxnNames": "fresh"}])
    coeffs_df = pd.DataFrame([
        {"rxnIdx": coeff_idx, "rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": -2.0},
        {"rxnIdx": coeff_idx, "rxnNames": "fresh", "metNames": "ADP", "comps": "c", "coefficient": 2.0},
    ])
    result = batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert len(result.added_reactions) == 1


def test_a_reaction_without_a_row_in_the_other_table_is_refused_with_its_identifiers():
    rxns_df, coeffs_df = _two_tables(
        ["1\tfresh\t0\t5"],
        ["1\tfresh\tATP\tc\t-2", "2\tstray\tADP\tc\t2"])
    with pytest.raises(ValueError, match=r"'stray' \(rxnIdx 2\) in rxns_coeffs_df has no matching row"):
        batch_curate(_base_model(), rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    rxns_df, coeffs_df = _two_tables(
        ["1\tfresh\t0\t5", "3\tlost\t0\t5"], ["1\tfresh\tATP\tc\t-2"])
    with pytest.raises(ValueError, match=r"'lost' \(rxnIdx 3\) in rxns_df has no matching"):
        batch_curate(_base_model(), rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")


@pytest.mark.parametrize("column, table", [
    ("charge", "mets"), ("rxnIdx", "rxns"), ("coefficient", "coeffs"), ("rxnConfidenceScores", "rxns")])
@pytest.mark.parametrize("text", ["true", "FALSE"])
def test_true_and_false_are_not_numbers_outside_the_bound_and_rev_columns(column, table, text):
    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnIdx": "1", "rxnNames": "fresh"}])
    coeffs_df = pd.DataFrame([
        {"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": "-2"},
        {"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ADP", "comps": "c", "coefficient": "2"}])
    kwargs = {}
    if table == "mets":
        kwargs["mets_df"] = pd.DataFrame([{"metNames": "ATP", "comps": "c", "charge": text}])
    else:
        {"rxns": rxns_df, "coeffs": coeffs_df}[table][column] = text
        kwargs.update(rxns_df=rxns_df, rxns_coeffs_df=coeffs_df)
    with pytest.raises(ValueError, match="is not a number"):
        batch_curate(m, rxn_id_prefix="r_", **kwargs)


@pytest.mark.parametrize("text", ["0", "Inf", "-inf", "0.0"])
def test_a_coefficient_that_is_zero_or_infinite_is_refused(text):
    rxns_df = pd.DataFrame([{"rxnIdx": "1", "rxnNames": "fresh"}])
    coeffs_df = pd.DataFrame([
        {"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": text},
        {"rxnIdx": "1", "rxnNames": "fresh", "metNames": "ADP", "comps": "c", "coefficient": "2"}])
    with pytest.raises(ValueError, match="finite number other than 0"):
        batch_curate(_base_model(), rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")


# Pin: an all-empty MIRIAM column was never written.
def test_an_all_empty_miriam_column_removes_no_annotation():
    m = _annotated_model()
    df = pd.DataFrame([
        {"metNames": "ATP", "comps": "c", "kegg.compound": ""},
        {"metNames": "ADP", "comps": "c", "kegg.compound": ""}])
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=df)
    assert m.metabolites.get_by_id("s_0001").annotation["kegg.compound"] == "C1"


@pytest.mark.parametrize("rule", [
    "G1 and and (", "(G1 or G2", "G1 or", "G1 G2", "and G1", "()", "G1 and (G2 or )", "G1)", "G1 and or G2",
    "or", "(", "(G1) (G2)", "G1 (G2)", "(G1 and G2) G3", "G1 and ()", "G1 xor G2", "G1 && || G2", "& G1", "(G1 and) G2"])
@pytest.mark.parametrize("existing", [False, True])
def test_a_malformed_gene_rule_is_refused_before_any_change(rule, existing):
    m = _base_model()
    name, ctable = ("ATP hydrolysis", ("-1", "1")) if existing else ("fresh", ("-2", "2"))
    rxns_df = pd.DataFrame([{"rxnNames": name, "grRules": rule}])
    coeffs_df = pd.DataFrame([
        {"rxnNames": name, "metNames": "ATP", "comps": "c", "coefficient": ctable[0]},
        {"rxnNames": name, "metNames": "ADP", "comps": "c", "coefficient": ctable[1]}])
    mets_df = pd.DataFrame([{"metNames": "NADH", "comps": "c"}])
    before = _snapshot(m)
    with pytest.raises(ValueError, match="is not a valid gene rule"):
        batch_curate(m, mets_df=mets_df, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert _snapshot(m) == before


@pytest.mark.parametrize("rule, genes", [
    ("G1 AND (G2 or G3)", ["G1", "G2", "G3"]),
    ("G1 And G2", ["G1", "G2"]),
    ("G1 && G2", ["G1", "G2"]),
    ("G1 || G2", ["G1", "G2"]),
    ("G1 & G2", ["G1", "G2"]),
    ("G1 | G2", ["G1", "G2"]),
    ("G1 and(G2 or G3)", ["G1", "G2", "G3"]),
    ("(G1)and(G2)", ["G1", "G2"]),
    ("G1 AND G2 OR G3", ["G1", "G2", "G3"]),
    ("  G1  ", ["G1"]),
    ("((G1))", ["G1"]),
])
def test_a_valid_gene_rule_in_any_spelling_is_stored(rule, genes):
    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnNames": "fresh", "grRules": rule}])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": "-2"},
        {"rxnNames": "fresh", "metNames": "ADP", "comps": "c", "coefficient": "2"}])
    batch_curate(m, genes_df=_genes_df("G1", "G2", "G3"), rxns_df=rxns_df,
                 rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert sorted(g.id for g in _by_name(m, "fresh").genes) == genes


@pytest.mark.parametrize("name, ctable", [("fresh", ("-2", "2")), ("ATP hydrolysis", ("-1", "1"))])
@pytest.mark.parametrize("rule, genes", [
    ("G1&G2 or G3", ["G1&G2", "G3"]),
    ("G1|G2", ["G1|G2"]),
])
def test_an_operator_glued_to_a_gene_id_stays_part_of_the_stored_gene(name, ctable, rule, genes):
    """RAVEN's parseGrRule reads G1&G2 as one gene id; the stored rule has the same genes."""
    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnNames": name, "grRules": rule}])
    coeffs_df = pd.DataFrame([
        {"rxnNames": name, "metNames": "ATP", "comps": "c", "coefficient": ctable[0]},
        {"rxnNames": name, "metNames": "ADP", "comps": "c", "coefficient": ctable[1]}])
    batch_curate(m, genes_df=_genes_df(*genes), rxns_df=rxns_df,
                 rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    rxn = _by_name(m, name)
    assert sorted(g.id for g in rxn.genes) == genes
    assert rxn.gene_reaction_rule == rule
    assert not {"G1", "G2"} & {g.id for g in m.genes}


def test_a_glued_gene_id_stays_one_gene_through_the_yaml_files(tmp_path):
    from raven_toolbox.io import read_yaml_model, write_yaml_model

    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnNames": "fresh", "grRules": "G1&G2 or G3"}])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": "-1"},
        {"rxnNames": "fresh", "metNames": "ADP", "comps": "c", "coefficient": "1"}])
    batch_curate(m, genes_df=_genes_df("G1&G2", "G3"), rxns_df=rxns_df,
                 rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    write_yaml_model(m, tmp_path / "m.yml")
    back = read_yaml_model(tmp_path / "m.yml")
    rxn = _by_name(back, "fresh")
    assert sorted(g.id for g in rxn.genes) == ["G1&G2", "G3"]
    assert rxn.gene_reaction_rule == "G1&G2 or G3"
    assert not {"G1", "G2"} & {g.id for g in back.genes}


# --- header-only tables, atomicity of several rows, genes of new reactions, headers ---------

_RXN_HEADER = "rxnIdx\trxnNames\tlb\tub\n"
_COEFF_HEADER = "rxnIdx\trxnNames\tmetNames\tcomps\tcoefficient\n"


def test_header_only_reaction_tables_change_nothing(tmp_path):
    m = _base_model()
    before = _snapshot(m)
    result = batch_curate_from_tsv(
        m, rxns_tsv=_write(tmp_path / "r.tsv", _RXN_HEADER),
        rxns_coeffs_tsv=_write(tmp_path / "c.tsv", _COEFF_HEADER))
    assert not result
    assert _snapshot(m) == before


def test_header_only_reaction_data_frames_change_nothing():
    m = _base_model()
    before = _snapshot(m)
    rxns_df = pd.DataFrame(columns=["rxnIdx", "rxnNames", "lb"])
    coeffs_df = pd.DataFrame(columns=["rxnIdx", "rxnNames", "metNames", "comps", "coefficient"])
    assert not batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df)
    assert _snapshot(m) == before


# Pin: the error for a header-only reaction table next to coefficient rows was already raised.
def test_a_header_only_reaction_table_beside_coefficient_rows_is_refused(tmp_path):
    m = _base_model()
    before = _snapshot(m)
    with pytest.raises(ValueError, match="no matching row"):
        batch_curate_from_tsv(
            m, rxns_tsv=_write(tmp_path / "r.tsv", _RXN_HEADER),
            rxns_coeffs_tsv=_write(tmp_path / "c.tsv", _COEFF_HEADER + "1\tfresh\tATP\tc\t-1\n"))
    assert _snapshot(m) == before


def test_two_rows_that_update_one_reaction_are_checked_one_after_the_other():
    m = _base_model()
    before = _snapshot(m)
    rxns_df, coeffs_df = _two_tables(
        ["1\tfirst\t500\t", "2\tsecond\t\t300"],
        ["1\tfirst\tATP\tc\t-1", "1\tfirst\tADP\tc\t1", "2\tsecond\tATP\tc\t-1", "2\tsecond\tADP\tc\t1"])
    with pytest.raises(ValueError, match="lower bound 500 .* upper bound 300"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df)
    assert _snapshot(m) == before


# Pin: two valid rows for one reaction were already applied one after the other.
def test_two_rows_that_update_one_reaction_are_applied_in_turn():
    m = _base_model()
    rxns_df, coeffs_df = _two_tables(
        ["1\tfirst\t200\t", "2\tsecond\t\t900"],
        ["1\tfirst\tATP\tc\t-1", "1\tfirst\tADP\tc\t1", "2\tsecond\tATP\tc\t-1", "2\tsecond\tADP\tc\t1"])
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df)
    rxn = m.reactions.get_by_id("r_0001")
    assert (rxn.name, rxn.bounds) == ("second", (200.0, 900.0))


def test_a_new_gene_listed_twice_is_refused_before_any_change():
    m = _base_model()
    before = _snapshot(m)
    mets_df = pd.DataFrame([{"metNames": "NADH", "comps": "c"}])
    genes_df = pd.DataFrame({"genes": ["G9", "G9"]})
    with pytest.raises(ValueError, match="listed more than once.*G9"):
        batch_curate(m, mets_df=mets_df, genes_df=genes_df)
    assert _snapshot(m) == before


def test_a_gene_of_the_model_listed_twice_is_refused_before_any_change():
    m = _base_model()
    before = _snapshot(m)
    genes_df = pd.DataFrame({"genes": ["YAL001C", "YAL001C"], "geneShortNames": ["a", "b"]})
    with pytest.raises(ValueError, match="listed more than once.*YAL001C"):
        batch_curate(m, genes_df=genes_df)
    assert _snapshot(m) == before


def test_a_new_reaction_with_a_gene_that_nothing_defines_is_refused_before_any_change():
    m = _base_model()
    before = _snapshot(m)
    mets_df = pd.DataFrame([{"metNames": "NADH", "comps": "c"}])
    rxns_df = pd.DataFrame([{"rxnNames": "fresh", "grRules": "G1 and YAL001C"}])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "fresh", "metNames": "ATP", "comps": "c", "coefficient": "-2"},
        {"rxnNames": "fresh", "metNames": "ADP", "comps": "c", "coefficient": "2"}])
    with pytest.raises(ValueError, match=r"not in the model or in genes_df: \['G1'\]"):
        batch_curate(m, mets_df=mets_df, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df)
    assert _snapshot(m) == before


# Pin: the rule of an existing reaction may still name a gene the model does not have.
def test_the_gene_rule_of_an_existing_reaction_may_name_a_new_gene():
    m = _base_model()
    rxns_df = pd.DataFrame([{"rxnNames": "ATP hydrolysis", "grRules": "G7"}])
    coeffs_df = pd.DataFrame([
        {"rxnNames": "ATP hydrolysis", "metNames": "ATP", "comps": "c", "coefficient": "-1"},
        {"rxnNames": "ATP hydrolysis", "metNames": "ADP", "comps": "c", "coefficient": "1"}])
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df)
    assert m.reactions.r_0001.gene_reaction_rule == "G7"


@pytest.mark.parametrize("header, row, expected", [
    ("metNames\tcomps\tmetNames", "NADH\tc\tother", "NADH"),
    ("metNames\tcomps\tformula\tmetFormulas", "NADH\tc\tH\tHH", "H"),
    ("Comps\tmetNames\tcomps", "c\tNADH\tother", "NADH"),
])
def test_a_core_header_that_is_repeated_or_spelled_twice_uses_the_first_column(
        tmp_path, header, row, expected):
    path = _write(tmp_path / "mets.tsv", header + "\n" + row + "\n")
    m = _base_model()
    batch_curate_from_tsv(m, mets_tsv=path, met_id_prefix="s_")
    new = m.metabolites.s_0004
    assert new.name == "NADH" and new.compartment == "c"
    assert new.annotation == {}
    assert new.formula in ("", None, "H")
    if expected == "H":
        assert new.formula == "H"


def test_a_lower_case_and_an_upper_case_lb_header_use_the_first_column(tmp_path):
    rxns = _write(tmp_path / "r.tsv", "rxnIdx\trxnNames\tLB\tlb\n1\tATP hydrolysis\t10\t20\n")
    coeffs = _write(tmp_path / "c.tsv", _COEFF_HEADER + "1\tATP hydrolysis\tATP\tc\t-1\n"
                    "1\tATP hydrolysis\tADP\tc\t1\n")
    m = _base_model()
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate_from_tsv(m, rxns_tsv=rxns, rxns_coeffs_tsv=coeffs)
    assert m.reactions.r_0001.lower_bound == 10.0


def test_header_names_are_trimmed_and_one_pair_of_quotes_is_removed(tmp_path):
    path = _write(tmp_path / "mets.tsv",
                  'metNames\t comps \t" a "\t"b""c"\t"d"x\n' "NADH\tc\t0\t1\t2\n")
    m = _base_model()
    batch_curate_from_tsv(m, mets_tsv=path, met_id_prefix="s_")
    assert m.metabolites.s_0004.annotation == {" a ": "0", 'b""c': "1", '"d"x': "2"}


def test_a_column_with_an_empty_header_is_ignored(tmp_path):
    path = _write(tmp_path / "mets.tsv", "metNames\tcomps\t\nNADH\tc\tvalue\n")
    m = _base_model()
    batch_curate_from_tsv(m, mets_tsv=path, met_id_prefix="s_")
    assert m.metabolites.s_0004.annotation == {}


# Pin: repeated header names already got distinct names.
def test_repeated_header_names_never_collide(tmp_path):
    path = _write(tmp_path / "mets.tsv", "metNames\tcomps\ta\ta.1\ta\ta\nNADH\tc\t1\t2\t3\t4\n")
    m = _base_model()
    batch_curate_from_tsv(m, mets_tsv=path, met_id_prefix="s_")
    annotation = m.metabolites.s_0004.annotation
    assert sorted(annotation.values()) == ["1", "2", "3", "4"]


def test_a_blank_first_line_is_the_header_and_the_file_is_refused(tmp_path):
    path = _write(tmp_path / "mets.tsv", "\nmetNames\tcomps\nNADH\tc\n")
    with pytest.raises(ValueError, match="1 columns"):
        batch_curate_from_tsv(_base_model(), mets_tsv=path)


def test_text_cells_of_a_data_frame_with_numbers_are_not_upcast():
    m = _base_model()
    rxns_df = pd.DataFrame({"rxnIdx": [1], "rxnNames": [7], "lb": [0.0]})
    coeffs_df = pd.DataFrame({"rxnIdx": [1], "rxnNames": ["7"], "metNames": ["ATP"],
                              "comps": ["c"], "coefficient": [-1.0]})
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df)
    assert [r.name for r in m.reactions][-1] == "7"


def test_a_reaction_name_that_is_nan_in_both_tables_is_the_empty_name():
    m = _base_model()
    rxns_df = pd.DataFrame({"rxnIdx": [1], "rxnNames": [float("nan")], "lb": [0.0]})
    coeffs_df = pd.DataFrame({"rxnIdx": [1], "rxnNames": [float("nan")], "metNames": ["ATP"],
                              "comps": ["c"], "coefficient": [-1.0]})
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df)
    assert [r.name for r in m.reactions][-1] == ""


# --- a repeated rxnIdx in the reaction table, padded quoted headers -------------------------


@pytest.mark.parametrize("second", ["1", "1.0", "1e0"])
def test_a_repeated_rxn_idx_in_the_reaction_table_is_refused_before_any_change(second):
    m = _base_model()
    rxns_df, coeffs_df = _two_tables(
        ["1\tnewA\t\t", f"{second}\tnewB\t\t"],
        ["1\tnewA\tATP\tc\t-1", "1\tnewA\tADP\tc\t1", "1\tnewB\tATP\tc\t-1", "1\tnewB\tADP\tc\t1"])
    mets_df = pd.DataFrame([{"metNames": "NADH", "comps": "c", "formula": "X"}])
    before = _snapshot(m)
    with pytest.raises(ValueError, match="rxnIdx 1 is used by data rows 1, 2"):
        batch_curate(m, mets_df=mets_df, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert _snapshot(m) == before


# Pin: Python already ignored spaces inside the quotes of a header cell; RAVEN does since the same change.
@pytest.mark.parametrize("header", [
    '"  metNames"\t" comps "\t formula ',
    '"metNames"\t"comps"\t"formula"',
    ' "metNames" \t "comps" \t "formula" ',
])
def test_spaces_inside_the_quotes_of_a_header_cell_are_ignored(tmp_path, header):
    path = _write(tmp_path / "mets.tsv", header + "\nNADH\tc\tH2O\n")
    m = _base_model()
    batch_curate_from_tsv(m, mets_tsv=path, met_id_prefix="s_")
    new = m.metabolites.s_0004
    assert new.name == "NADH" and new.compartment == "c" and new.formula == "H2O"
