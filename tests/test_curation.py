"""Tests for raven_toolbox.curation.batch."""
from __future__ import annotations

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
