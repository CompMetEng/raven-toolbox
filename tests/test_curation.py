"""Tests for raven_toolbox.curation.batch."""
from __future__ import annotations

import math

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
    result = batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df,
                          rxn_id_prefix="r_")
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
    result = batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df,
                          rxn_id_prefix="r_")
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
    result = batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
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
        rxns_df=rxns_df.rename(columns={"rxnIdx": "index"}),
        rxns_coeffs_df=coeffs_df.rename(columns={"rxnIdx": "Index"}),
        rxn_id_prefix="r_",
    )
    assert len(result.added_reactions) == 2


def test_new_reaction_without_bounds_is_unbounded():
    m = _base_model()
    rxns_df, coeffs_df = _two_hexokinases()
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
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
    # Pin: a blank-only cell was empty before, and RAVEN reads it as empty as well.
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
    with pytest.raises(TypeError, match="overwrite_with_empty"):
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
@pytest.mark.parametrize("text", ["abc", "NA", "None", "1 0", "pi", "1i"])
def test_a_reaction_cell_that_is_not_a_number_is_refused(column, text):
    m = _base_model()
    rxns_df, coeffs_df = _bound_tables(**{column: ("1", text)})
    with pytest.raises(ValueError, match=f'{column} value "{text}"'):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")


@pytest.mark.parametrize("text", ["abc", "NA", "None", "pi", "1i"])
def test_a_charge_that_is_not_a_number_is_refused(text):
    df = pd.DataFrame([{"metNames": "ATP", "comps": "c", "charge": text},
                       {"metNames": "ADP", "comps": "c", "charge": -2}])
    with pytest.raises(ValueError, match=f'charge value "{text}"'):
        batch_curate(_base_model(), mets_df=df)


@pytest.mark.parametrize("text", ["abc", "NA", "None"])
def test_a_confidence_score_that_is_not_a_number_is_refused(text):
    m = _base_model()
    rxns_df, coeffs_df = _bound_tables(rxnConfidenceScores=("3", text))
    with pytest.raises(ValueError, match="rxnConfidenceScores value"):
        batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")


def test_charges_and_scores_are_read_from_text_with_a_separator_or_an_exponent():
    df = pd.DataFrame([{"metNames": "ATP", "comps": "c", "charge": "-4"},
                       {"metNames": "ADP", "comps": "c", "charge": " 2 "}])
    m = _base_model()
    with pytest.warns(UserWarning, match="overwritten"):
        batch_curate(m, mets_df=df)
    assert [m.metabolites.get_by_id(i).charge for i in ("s_0001", "s_0002")] == [-4, 2]
    m = _base_model()
    rxns_df, coeffs_df = _bound_tables(lb=("1,000", "1e2"), ub=("Inf", "+inf"))
    batch_curate(m, rxns_df=rxns_df, rxns_coeffs_df=coeffs_df, rxn_id_prefix="r_")
    assert _by_name(m, "ATP hydrolysis").bounds == (1000.0, math.inf)
    assert _by_name(m, "fresh").bounds == (100.0, math.inf)
