"""Tests for raven_toolbox.io.yaml's GECKO ec-model support.

Covers:
- model.ec populated from `ec-rxns` / `ec-enzymes` / `gecko_light` sections;
- model.ec serialised back to those sections (overwriting any stale
  `_yaml_sections` entries);
- numpy/sparse coercion happens at the boundary (writer must not see
  numpy scalars);
- legacy quirks: top-level smiles, reverse-direction usage_prot_*, bare-`-`
  document root;
- malformed inputs: half a pair of ec-* sections, dangling enzyme reference.
"""
from __future__ import annotations

from pathlib import Path

import cobra
import numpy as np
import pytest
from cobra.io.yaml import yaml as cobra_yaml
from scipy import sparse

from raven_toolbox.io import EcData, read_yaml_model, write_yaml_model
from raven_toolbox.io.ec_data import _eccodes_to_yaml
from raven_toolbox.io.yaml import model_from_yaml_data

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _minimal_model_doc() -> dict:
    """A cobra-shaped doc with two mets, one rxn, one gene — minimum needed
    for cobra to parse a model. Tests add the ec-* sections as needed."""
    return {
        "metabolites": [
            {"id": "a", "name": "A", "compartment": "c"},
            {"id": "b", "name": "B", "compartment": "c"},
        ],
        "reactions": [
            {
                "id": "R1",
                "metabolites": {"a": -1, "b": 1},
                "lower_bound": 0.0,
                "upper_bound": 1000.0,
                "gene_reaction_rule": "G1",
            },
        ],
        "genes": [{"id": "G1", "name": "g one"}],
        "id": "m",
        "compartments": {"c": "cytoplasm"},
    }


def _write_yaml(doc: dict, path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        cobra_yaml.dump(doc, fh)
    return path


# --------------------------------------------------------------------------- #
# Load: ec sections populate model.ec
# --------------------------------------------------------------------------- #

def test_load_populates_ec_data(tmp_path):
    doc = _minimal_model_doc()
    doc["ec-rxns"] = [
        {"id": "R1", "kcat": 12.5, "source": "manual", "enzymes": {"P1": 1.0}},
    ]
    doc["ec-enzymes"] = [
        {"genes": "G1", "enzymes": "P1", "mw": 30000.0, "sequence": "MAGIC"},
    ]
    doc["gecko_light"] = False
    model = read_yaml_model(_write_yaml(doc, tmp_path / "m.yml"))

    assert isinstance(model.ec, EcData)
    assert model.ec.gecko_light is False
    assert model.ec.rxns == ["R1"]
    assert model.ec.kcat[0] == 12.5
    assert model.ec.source[0] == "manual"
    assert model.ec.enzymes == ["P1"]
    assert model.ec.mw[0] == 30000.0
    assert model.ec.rxn_enz_mat.shape == (1, 1)
    assert model.ec.rxn_enz_mat[0, 0] == 1.0


def test_load_missing_optional_fields_get_sentinels(tmp_path):
    """Optional fields omitted from a row come back as the right sentinels
    on the loaded EcData."""
    doc = _minimal_model_doc()
    doc["ec-rxns"] = [{"id": "R1", "kcat": 0.0, "enzymes": {"P1": 1.0}}]
    doc["ec-enzymes"] = [{"genes": "G1", "enzymes": "P1"}]  # no mw, sequence, concs
    model = read_yaml_model(_write_yaml(doc, tmp_path / "m.yml"))

    assert model.ec.source == [""]
    assert model.ec.notes == [""]
    assert model.ec.eccodes == [""]
    assert np.isnan(model.ec.mw[0])
    assert model.ec.sequence == [""]
    assert np.isnan(model.ec.concs[0])


def test_load_gecko_light_flag(tmp_path):
    doc = _minimal_model_doc()
    doc["ec-rxns"] = [
        {"id": "001_R1", "kcat": 5.0, "enzymes": {"P1": 1.0}},
    ]
    doc["ec-enzymes"] = [{"genes": "G1", "enzymes": "P1", "mw": 100.0}]
    doc["gecko_light"] = True
    model = read_yaml_model(_write_yaml(doc, tmp_path / "m.yml"))
    assert model.ec.gecko_light is True


def test_load_without_ec_sections_leaves_model_without_ec(tmp_path):
    """Non-ecmodel YAML loads as a plain cobra.Model — no model.ec."""
    model = read_yaml_model(_write_yaml(_minimal_model_doc(), tmp_path / "m.yml"))
    assert not hasattr(model, "ec") or model.ec is None or model.ec == EcData()
    # The attribute is simply not set in the no-ec case.
    assert "ec" not in vars(model)


def test_load_eccodes_scalar_or_list_both_round_trip(tmp_path):
    """The schema accepts a scalar string OR a list of strings for `eccodes`;
    both flavours land in the same `;`-joined internal form."""
    doc = _minimal_model_doc()
    doc["ec-rxns"] = [
        {"id": "R1", "kcat": 1.0, "eccodes": "1.1.1.1", "enzymes": {"P1": 1.0}},
    ]
    doc["ec-enzymes"] = [{"genes": "G1", "enzymes": "P1", "mw": 1.0}]
    m1 = read_yaml_model(_write_yaml(doc, tmp_path / "m1.yml"))
    assert m1.ec.eccodes == ["1.1.1.1"]

    doc["ec-rxns"][0]["eccodes"] = ["1.1.1.1", "1.1.99.40"]
    m2 = read_yaml_model(_write_yaml(doc, tmp_path / "m2.yml"))
    assert m2.ec.eccodes == ["1.1.1.1;1.1.99.40"]


def test_eccodes_to_yaml_strips_trailing_separator():
    # The write-path helper must clean the internal `;`-joined form: a single EC
    # round-trips to a bare scalar, and stray separators never leak into the YAML.
    assert _eccodes_to_yaml("1.1.1.1") == "1.1.1.1"
    assert _eccodes_to_yaml("1.1.1.1;") == "1.1.1.1"
    assert _eccodes_to_yaml(";") == ""
    assert _eccodes_to_yaml("") == ""
    assert _eccodes_to_yaml("1.1.1.1;1.1.99.40") == ["1.1.1.1", "1.1.99.40"]


# --------------------------------------------------------------------------- #
# Save: model.ec serialised back to top-level sections
# --------------------------------------------------------------------------- #

def _make_ec_model() -> cobra.Model:
    """A minimal cobra.Model with a populated model.ec attached by hand."""
    model = cobra.Model("m")
    a = cobra.Metabolite("a", compartment="c")
    b = cobra.Metabolite("b", compartment="c")
    model.add_metabolites([a, b])
    r = cobra.Reaction("R1", lower_bound=0.0, upper_bound=1000.0)
    r.add_metabolites({a: -1, b: 1})
    r.gene_reaction_rule = "G1"
    model.add_reactions([r])

    mat = sparse.lil_matrix((1, 1), dtype=float)
    mat[0, 0] = 1.0
    model.ec = EcData(
        gecko_light=False,
        rxns=["R1"],
        kcat=np.array([42.0]),
        source=["manual"],
        notes=[""],
        eccodes=["1.1.1.1"],
        genes=["G1"],
        enzymes=["P1"],
        mw=np.array([12345.0]),
        sequence=["MAGIC"],
        concs=np.array([np.nan]),
        rxn_enz_mat=mat.tocsr(),
    )
    return model


def test_save_emits_ec_sections(tmp_path):
    out = tmp_path / "out.yml"
    write_yaml_model(_make_ec_model(), out)
    text = out.read_text()
    assert "ec-rxns:" in text
    assert "ec-enzymes:" in text
    assert "gecko_light:" in text
    assert "42" in text and "12345" in text and "MAGIC" in text


def test_save_round_trip_preserves_all_ec_fields(tmp_path):
    out = tmp_path / "out.yml"
    write_yaml_model(_make_ec_model(), out)
    reloaded = read_yaml_model(out)
    assert reloaded.ec.rxns == ["R1"]
    assert reloaded.ec.kcat[0] == 42.0
    assert reloaded.ec.source == ["manual"]
    assert reloaded.ec.eccodes == ["1.1.1.1"]
    assert reloaded.ec.enzymes == ["P1"]
    assert reloaded.ec.mw[0] == 12345.0
    assert reloaded.ec.sequence == ["MAGIC"]
    assert np.isnan(reloaded.ec.concs[0])  # NaN omitted on write, restored on load
    assert reloaded.ec.rxn_enz_mat[0, 0] == 1.0


def test_save_skips_nan_and_empty_optional_fields(tmp_path):
    """NaN mw/concs and empty source/notes/eccodes/sequence get omitted
    from the YAML to keep files compact."""
    model = _make_ec_model()
    model.ec.source = [""]
    model.ec.notes = [""]
    model.ec.eccodes = [""]
    model.ec.sequence = [""]
    model.ec.mw = np.array([np.nan])
    model.ec.concs = np.array([np.nan])

    out = tmp_path / "out.yml"
    write_yaml_model(model, out)
    text = out.read_text()
    assert "source:" not in text
    assert "notes:" not in text
    assert "eccodes:" not in text
    assert "sequence:" not in text
    # kcat is always written (even 0); mw / concs omitted when NaN.
    assert "kcat:" in text


def test_save_coerces_numpy_scalars(tmp_path):
    """ec arrays hold numpy types; the writer must coerce so the YAML
    dumper never sees an np.float64 or np.int64."""
    model = _make_ec_model()
    model.ec.kcat = np.array([np.float32(7.5)])
    model.ec.mw = np.array([np.int64(20000)])
    out = tmp_path / "out.yml"
    write_yaml_model(model, out)  # must not raise
    reloaded = read_yaml_model(out)
    assert reloaded.ec.kcat[0] == pytest.approx(7.5)
    assert reloaded.ec.mw[0] == 20000.0


def test_save_without_ec_omits_sections(tmp_path):
    model = cobra.Model("m")
    a = cobra.Metabolite("a", compartment="c")
    model.add_metabolites([a])
    out = tmp_path / "out.yml"
    write_yaml_model(model, out)
    text = out.read_text()
    assert "ec-rxns:" not in text
    assert "ec-enzymes:" not in text
    assert "gecko_light:" not in text


def test_save_overrides_stale_yaml_sections_for_ec_keys(tmp_path):
    """If a loaded model carried stale ec-* in _yaml_sections AND also has
    a populated model.ec (shouldn't happen via the normal load path,
    but a caller could construct it), the writer uses model.ec and
    drops the stale stash so the file isn't ambiguous."""
    model = _make_ec_model()
    model.notes["_yaml_sections"] = {
        "ec-rxns": [{"id": "STALE", "kcat": 999.0}],
        "ec-enzymes": [{"genes": "GHOST", "enzymes": "PGHOST"}],
    }
    out = tmp_path / "out.yml"
    write_yaml_model(model, out)
    reloaded = read_yaml_model(out)
    assert reloaded.ec.rxns == ["R1"]  # not "STALE"
    assert reloaded.ec.enzymes == ["P1"]  # not "PGHOST"


# --------------------------------------------------------------------------- #
# Legacy quirks
# --------------------------------------------------------------------------- #

def test_legacy_top_level_smiles_lifted_to_annotation(tmp_path):
    """Old MATLAB GECKO ecModels put SMILES at the metabolite top level
    rather than inside annotation. The loader normalises both flavours."""
    doc = _minimal_model_doc()
    doc["metabolites"][0]["smiles"] = "CC(=O)O"
    model = read_yaml_model(_write_yaml(doc, tmp_path / "m.yml"))
    a = model.metabolites.get_by_id("a")
    assert a.annotation["smiles"] == ["CC(=O)O"]
    # Should NOT survive as a stray notes key.
    assert "smiles" not in a.notes


def test_legacy_reverse_direction_prot_flipped(tmp_path):
    """`usage_prot_*` reactions with a negative lower bound and swapped
    stoichiometry get flipped to the forward convention on load."""
    doc = _minimal_model_doc()
    doc["metabolites"].append({"id": "prot_P1", "name": "prot", "compartment": "c"})
    doc["metabolites"].append({"id": "prot_pool", "name": "pool", "compartment": "c"})
    doc["reactions"].append(
        {
            "id": "usage_prot_P1",
            "metabolites": {"prot_P1": -1, "prot_pool": 1},  # swapped signs
            "lower_bound": -1000.0,
            "upper_bound": 0.0,
        }
    )
    with pytest.warns(UserWarning, match="reverse-direction"):
        model = read_yaml_model(_write_yaml(doc, tmp_path / "m.yml"))
    r = model.reactions.get_by_id("usage_prot_P1")
    assert r.lower_bound == 0.0
    assert r.upper_bound == 1000.0
    # Stoichiometry signs flipped.
    coefs = {m.id: c for m, c in r.metabolites.items()}
    assert coefs["prot_P1"] == 1.0
    assert coefs["prot_pool"] == -1.0


def test_legacy_bare_sequence_root_merged(tmp_path):
    """Very old RAVEN files were written as a bare `-` sequence of
    single-key mappings; reader merges to one dict."""
    legacy_text = (
        "- metabolites:\n"
        "  - id: a\n"
        "    name: A\n"
        "    compartment: c\n"
        "- reactions: []\n"
        "- genes: []\n"
        "- id: legacy_m\n"
        "- compartments:\n"
        "    c: cyt\n"
    )
    p = tmp_path / "bare.yml"
    p.write_text(legacy_text)
    model = read_yaml_model(p)
    assert model.id == "legacy_m"
    assert {m.id for m in model.metabolites} == {"a"}


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #

def test_load_rxns_only_without_enzymes_raises(tmp_path):
    doc = _minimal_model_doc()
    doc["ec-rxns"] = [{"id": "R1", "kcat": 1.0}]
    with pytest.raises(ValueError, match="ec-enzymes"):
        read_yaml_model(_write_yaml(doc, tmp_path / "m.yml"))


def test_load_enzymes_only_without_rxns_raises(tmp_path):
    doc = _minimal_model_doc()
    doc["ec-enzymes"] = [{"genes": "G1", "enzymes": "P1"}]
    with pytest.raises(ValueError, match="ec-rxns"):
        read_yaml_model(_write_yaml(doc, tmp_path / "m.yml"))


def test_load_dangling_enzyme_reference_raises(tmp_path):
    """An ec-rxns row whose enzymes mapping references an accession not
    listed in ec-enzymes is a hard error — catches the common authoring
    bug where the two sections drifted apart."""
    doc = _minimal_model_doc()
    doc["ec-rxns"] = [
        {"id": "R1", "kcat": 1.0, "enzymes": {"PGHOST": 1.0}},
    ]
    doc["ec-enzymes"] = [{"genes": "G1", "enzymes": "P1", "mw": 1.0}]
    with pytest.raises(ValueError, match="PGHOST"):
        read_yaml_model(_write_yaml(doc, tmp_path / "m.yml"))


def test_load_all_zero_coupling_matrix_warns(tmp_path):
    """ec-rxns present but none reference an enzyme -> the coupling matrix is
    all-zero (likely malformed). Warn rather than build a silently empty model."""
    doc = _minimal_model_doc()
    doc["ec-rxns"] = [{"id": "R1", "kcat": 1.0, "enzymes": {}}]
    doc["ec-enzymes"] = [{"genes": "G1", "enzymes": "P1", "mw": 1.0}]
    with pytest.warns(UserWarning, match="all-zero"):
        read_yaml_model(_write_yaml(doc, tmp_path / "m.yml"))


# --------------------------------------------------------------------------- #
# In-memory model_from_yaml_data (no file I/O)
# --------------------------------------------------------------------------- #

def test_model_from_yaml_data_mutates_in_place():
    """`model_from_yaml_data` pops sections off its dict input. Documented
    behaviour — verify callers passing a fresh dict see it drained."""
    doc = _minimal_model_doc()
    doc["ec-rxns"] = [{"id": "R1", "kcat": 1.0, "enzymes": {"P1": 1.0}}]
    doc["ec-enzymes"] = [{"genes": "G1", "enzymes": "P1", "mw": 1.0}]
    doc["gecko_light"] = False
    model_from_yaml_data(doc)
    assert "ec-rxns" not in doc
    assert "ec-enzymes" not in doc
    assert "gecko_light" not in doc


# --------------------------------------------------------------------------- #
# EcData.validate / EcData.empty
# --------------------------------------------------------------------------- #

def test_empty_has_canonical_sentinels():
    """`EcData.empty(n, m)` preallocates with the documented sentinels."""
    ec = EcData.empty(3, 2)
    assert ec.n_rxns == 3
    assert ec.n_enzymes == 2
    assert ec.rxns == ["", "", ""]
    assert (ec.kcat == 0).all()
    assert np.isnan(ec.mw).all()
    assert np.isnan(ec.concs).all()
    assert ec.rxn_enz_mat.shape == (3, 2)
    assert ec.rxn_enz_mat.nnz == 0


def test_empty_round_trips_through_validate():
    EcData.empty(5, 4).validate()  # must not raise


def test_validate_catches_per_rxn_length_drift():
    ec = EcData.empty(3, 2)
    ec.kcat = np.array([1.0, 2.0])  # length 2, should be 3
    with pytest.raises(ValueError, match="ec.kcat has length 2, expected 3"):
        ec.validate()


def test_validate_catches_per_enzyme_length_drift():
    ec = EcData.empty(3, 2)
    ec.mw = np.array([1.0])  # length 1, should be 2
    with pytest.raises(ValueError, match="ec.mw has length 1, expected 2"):
        ec.validate()


def test_validate_catches_coupling_matrix_shape_drift():
    ec = EcData.empty(3, 2)
    ec.rxn_enz_mat = sparse.csr_matrix((3, 5), dtype=float)
    with pytest.raises(ValueError, match=r"ec.rxn_enz_mat has shape \(3, 5\)"):
        ec.validate()


def test_empty_gecko_light_flag_propagates():
    assert EcData.empty(1, 1, gecko_light=True).gecko_light is True
    assert EcData.empty(1, 1).gecko_light is False


# --------------------------------------------------------------------------- #
# Files as GECKO and RAVEN write them
# --------------------------------------------------------------------------- #

_GECKO_FILE = """\
---
!!omap
- metabolites:
  - !!omap
    - id: "a"
    - name: "A"
    - compartment: "c"
- reactions:
  - !!omap
    - id: "R1"
    - lower_bound: 0
    - upper_bound: 1000
    - metabolites: !!omap
      - a: -1
- genes: []
- compartments: !!omap
  - c: "cytoplasm"
- ec-rxns:
  - !!omap
    - id: "std"
    - kcat: 10
    - source: "custom"
    - notes: {note}
    - enzymes: !!omap{empty}
  - !!omap
    - id: "enz"
    - kcat: 5
    - enzymes: !!omap
      - P1: 1
- ec-enzymes:
  - !!omap
    - genes: "G1"
    - enzymes: "P1"
    - mw: 30000
    - sequence: "MAGIC"
"""


@pytest.mark.parametrize("empty", ["", " []"])
def test_load_reaction_with_empty_enzymes_omap(tmp_path, empty):
    # GECKO and RAVEN write `- enzymes: !!omap` with nothing under it for an ec
    # reaction without enzymes; raven-toolbox writes `!!omap []`. Both read as no enzymes.
    path = tmp_path / "m.yml"
    path.write_text(_GECKO_FILE.format(note='"ok"', empty=empty), encoding="utf-8")
    model = read_yaml_model(path)
    assert model.ec.rxns == ["std", "enz"]
    assert model.ec.enzymes == ["P1"]
    assert model.ec.rxn_enz_mat.toarray().tolist() == [[0.0], [1.0]]


def test_load_notes_in_doubled_quotes_keep_one_pair(tmp_path):
    # An older writer put quotes around a note that carried its own, giving `""text""`,
    # which is not valid YAML. readYAMLmodel.m strips one pair and keeps the other.
    path = tmp_path / "m.yml"
    path.write_text(_GECKO_FILE.format(note='""Putative: lowest kcat.""', empty=""), encoding="utf-8")
    model = read_yaml_model(path)
    assert model.ec.notes[0] == '"Putative: lowest kcat."'
    assert model.ec.rxns == ["std", "enz"]


def test_doubled_quotes_unescape_like_matlab_and_keep_apostrophes(tmp_path):
    path = tmp_path / "m.yml"
    note = '""it\'s a \\\\ and \\"inner\\" one""'
    path.write_text(_GECKO_FILE.format(note=note, empty=""), encoding="utf-8")
    model = read_yaml_model(path)
    assert model.ec.notes[0] == '"it\'s a \\ and "inner" one"'


def test_invalid_yaml_other_than_doubled_quotes_still_raises(tmp_path):
    path = tmp_path / "m.yml"
    path.write_text(_GECKO_FILE.format(note='"ok"', empty="").replace("kcat: 10", "kcat: [10"), encoding="utf-8")
    with pytest.raises(Exception):
        read_yaml_model(path)
