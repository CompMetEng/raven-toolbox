"""Tests for raven_toolbox.io.yaml against the RAVEN cobra-native (!!omap) schema."""
from pathlib import Path

import cobra
import pytest
from cobra.io.yaml import yaml as cobra_yaml

from raven_toolbox.io import read_yaml_model, write_yaml_model

# A model laid out exactly as RAVEN writeYAMLmodel emits: cobra-native
# structure, RAVEN-only fields as top-level per-entry keys, smiles/ec-code inside
# the annotation block, metaData provenance-only, id/name/version top-level,
# plus the GECKO ec-* sections that populate `model.ec`.
RAVEN_DOC = {
    "metabolites": [
        {
            "id": "s_0001",
            "name": "ATP",
            "compartment": "c",
            "formula": "C10H16N5O13P3",
            "charge": -4,
            "inchis": "InChI=1S/CH4",
            "deltaG": 12.5,
            "notes": "a metabolite note",
            "metFrom": "KEGG",
            "annotation": {"kegg.compound": ["C00002"], "smiles": ["C1=NC2"]},
        },
        {"id": "s_0002", "name": "ADP", "compartment": "c"},
    ],
    "reactions": [
        {
            "id": "R1",
            "name": "rxn one",
            "metabolites": {"s_0001": -1, "s_0002": 1},
            "lower_bound": -1000.0,
            "upper_bound": 1000.0,
            "gene_reaction_rule": "G1",
            "subsystem": "glycolysis",
            "confidence_score": 2,
            "references": "PMID:123",
            "rxnFrom": "manual",
            "notes": "a reaction note",
            "deltaG": -5.0,
            "annotation": {"ec-code": ["1.1.1.1"]},
        }
    ],
    "genes": [
        {"id": "G1", "name": "gene one", "protein": "P12345", "annotation": {"uniprot": ["P12345"]}}
    ],
    "id": "testModel",
    "name": "Test Model",
    "compartments": {"c": "cytoplasm"},
    "version": "1.0",
    "metaData": {"date": "2026-05-23", "taxonomy": "taxonomy/559292", "defaultLB": "-1000"},
    "gecko_light": False,
    "ec-rxns": [
        {"id": "R1", "kcat": 100.0, "source": "brenda", "enzymes": {"P12345": 1.0}},
    ],
    "ec-enzymes": [
        {"genes": "G1", "enzymes": "P12345", "mw": 50000.0, "sequence": "MAGIC"},
    ],
}


@pytest.fixture
def yaml_file(tmp_path) -> Path:
    p = tmp_path / "model.yml"
    with open(p, "w", encoding="utf-8") as fh:
        cobra_yaml.dump(RAVEN_DOC, fh)
    return p


def test_standard_content(yaml_file):
    model = read_yaml_model(yaml_file)
    assert model.id == "testModel"
    assert model.name == "Test Model"
    assert {m.id for m in model.metabolites} == {"s_0001", "s_0002"}
    r = model.reactions.get_by_id("R1")
    assert r.bounds == (-1000.0, 1000.0)
    assert r.subsystem == "glycolysis"
    assert r.gene_reaction_rule == "G1"


def test_annotation_owned_by_cobra(yaml_file):
    # smiles / ec-code / miriam live in the annotation block (cobra reads them)
    model = read_yaml_model(yaml_file)
    assert model.metabolites.get_by_id("s_0001").annotation["smiles"] == ["C1=NC2"]
    assert model.metabolites.get_by_id("s_0001").annotation["kegg.compound"] == ["C00002"]
    assert model.reactions.get_by_id("R1").annotation["ec-code"] == ["1.1.1.1"]
    assert model.genes.get_by_id("G1").annotation["uniprot"] == ["P12345"]


def test_raven_only_fields_captured(yaml_file):
    model = read_yaml_model(yaml_file)
    a = model.metabolites.get_by_id("s_0001")
    assert a.notes["inchis"] == "InChI=1S/CH4"
    assert a.notes["deltaG"] == 12.5
    assert a.notes["note"] == "a metabolite note"  # RAVEN metNotes string, no crash
    assert a.notes["metFrom"] == "KEGG"
    assert "smiles" not in a.notes  # smiles stays in annotation
    r = model.reactions.get_by_id("R1")
    assert r.notes["confidence_score"] == 2
    assert r.notes["references"] == "PMID:123"
    assert r.notes["rxnFrom"] == "manual"
    assert r.notes["note"] == "a reaction note"
    assert r.notes["deltaG"] == -5.0
    assert model.genes.get_by_id("G1").notes["protein"] == "P12345"


def test_model_level_extras(yaml_file):
    """metaData / version round-trip via model.notes; the ec-* sections
    are consumed into a typed model.ec, not stashed in _yaml_sections."""
    model = read_yaml_model(yaml_file)
    assert model.notes["metaData"]["taxonomy"] == "taxonomy/559292"
    assert model.notes["version"] == "1.0"
    # ec sections live on model.ec, not in _yaml_sections.
    assert "_yaml_sections" not in model.notes


def test_round_trip(yaml_file, tmp_path):
    model = read_yaml_model(yaml_file)
    out = tmp_path / "out.yml"
    write_yaml_model(model, out)
    reloaded = read_yaml_model(out)

    assert reloaded.id == "testModel"
    assert reloaded.notes["version"] == "1.0"
    assert reloaded.notes["metaData"]["taxonomy"] == "taxonomy/559292"
    a = reloaded.metabolites.get_by_id("s_0001")
    assert a.notes["deltaG"] == 12.5
    assert a.notes["note"] == "a metabolite note"
    assert a.annotation["smiles"] == ["C1=NC2"]
    r = reloaded.reactions.get_by_id("R1")
    assert r.notes["confidence_score"] == 2
    assert reloaded.genes.get_by_id("G1").notes["protein"] == "P12345"
    # ec round-trip: same kcat, same coupling entry.
    assert reloaded.ec.rxns == ["R1"]
    assert reloaded.ec.kcat[0] == 100.0
    assert reloaded.ec.source[0] == "brenda"
    assert reloaded.ec.enzymes == ["P12345"]
    assert reloaded.ec.mw[0] == 50000.0


def test_round_trip_preserves_unknown_metadata_field(yaml_file, tmp_path):
    """A metaData key with no RAVEN-defined slot (e.g. geckopy's
    geckopy_version) survives a write/read round trip alongside a normal
    RAVEN annotation field (taxonomy), instead of being silently dropped."""
    model = read_yaml_model(yaml_file)
    model.notes["metaData"]["geckopy_version"] = "0.2.1"
    out = tmp_path / "out.yml"
    write_yaml_model(model, out)
    reloaded = read_yaml_model(out)

    assert reloaded.notes["metaData"]["geckopy_version"] == "0.2.1"
    assert reloaded.notes["metaData"]["taxonomy"] == "taxonomy/559292"


def test_extra_notes_not_dropped_when_free_text_note_present(yaml_file, tmp_path):
    """An entry with both a RAVEN free-text note and an extra note keeps both on write."""
    model = read_yaml_model(yaml_file)
    a = model.metabolites.get_by_id("s_0001")
    a.notes["note"] = "free text"
    a.notes["custom"] = "extra value"  # a non-RAVEN note that must not be silently lost
    out = tmp_path / "out.yml"
    write_yaml_model(model, out)
    text = out.read_text()
    assert "extra value" in text  # the leftover note survives serialization


def test_gzipped_round_trip(yaml_file, tmp_path):
    # A .yml.gz path is transparently gzipped on write and read.
    model = read_yaml_model(yaml_file)
    out = tmp_path / "out.yml.gz"
    write_yaml_model(model, out)
    assert out.read_bytes()[:2] == b"\x1f\x8b"  # gzip magic
    reloaded = read_yaml_model(out)
    assert reloaded.id == "testModel"
    assert {m.id for m in reloaded.metabolites} == {"s_0001", "s_0002"}


def test_output_is_cobra_readable(yaml_file, tmp_path):
    """Cobrapy must be able to parse the file (no syntax error) and
    recover the metabolites, reactions, and the cobrapy-canonical
    annotation block. Model-level id / name / version live inside the
    metaData section (RAVEN convention) — cobrapy doesn't know about
    metaData, so cobra_model.id is None here. raven_toolbox recovers
    them; cobrapy ignores them gracefully."""
    model = read_yaml_model(yaml_file)
    out = tmp_path / "out.yml"
    write_yaml_model(model, out)
    cobra_model = cobra.io.load_yaml_model(str(out))
    assert {m.id for m in cobra_model.metabolites} == {"s_0001", "s_0002"}
    # RAVEN-only fields land in cobra notes; smiles in annotation
    assert cobra_model.metabolites.get_by_id("s_0001").annotation["smiles"] == ["C1=NC2"]


def test_write_emits_raven_top_level_keys(yaml_file, tmp_path):
    model = read_yaml_model(yaml_file)
    out = tmp_path / "out.yml"
    write_yaml_model(model, out)
    text = out.read_text()
    # RAVEN-only fields are lifted back to top-level entry keys, not buried in notes
    assert "inchis:" in text
    assert "deltaG:" in text
    assert "confidence_score:" in text
    assert "metaData:" in text


def test_write_emits_yaml_infinity_token_for_unbounded_reaction(tmp_path):
    """cobra's own model_to_dict stringifies an infinite or NaN bound as
    e.g. "inf" (JSON has no literal for either) --- its own reader tolerates
    this, unconditionally casting both bound fields through float(v). Left
    as a string rather than converted back, ruamel has no reason to emit
    the YAML 1.1 infinity token (.inf) and writes the bare word "inf"
    instead, which is not that token and parses back as the *string* "inf"
    on any YAML reader that doesn't share cobra's specific convention
    (confirmed against ruamel's own round-trip and safe loaders, and
    PyYAML's safe_load: bare "inf" parses as a string on every one of
    them). writeYAMLmodel.m emits .inf here regardless."""
    m = cobra.Model("t")
    a = cobra.Metabolite("a_c", compartment="c")
    m.add_metabolites([a])
    r = cobra.Reaction("R1", lower_bound=0, upper_bound=float("inf"))
    r.add_metabolites({a: 1})
    m.add_reactions([r])

    out = tmp_path / "m.yml"
    write_yaml_model(m, out)
    text = out.read_text()
    assert "upper_bound: .inf" in text
    assert "upper_bound: inf\n" not in text  # the malformed, non-spec token

    reloaded = read_yaml_model(out)
    assert reloaded.reactions.get_by_id("R1").upper_bound == float("inf")


def test_legacy_id_in_metadata(tmp_path):
    # Older RAVEN files nest id/name under metaData and have no top-level id.
    legacy = {
        "metabolites": [{"id": "a_c", "name": "A", "compartment": "c"}],
        "reactions": [],
        "genes": [],
        "compartments": {"c": "cyt"},
        "metaData": {"id": "legacyModel", "name": "Legacy"},
    }
    p = tmp_path / "legacy.yml"
    with open(p, "w", encoding="utf-8") as fh:
        cobra_yaml.dump(legacy, fh)
    model = read_yaml_model(p)
    assert model.id == "legacyModel"
    assert model.name == "Legacy"


# Optional smoke test against a real model file if present.
_YEAST = Path("/home/eduardk/github/GECKO/tutorials/full_ecModel/models/yeast-GEM.yml")


@pytest.mark.skipif(not _YEAST.exists(), reason="real yeast-GEM.yml not available")
def test_real_yeast_gem_loads():
    model = read_yaml_model(_YEAST)
    assert len(model.reactions) > 1000
    # legacy file: identity comes from metaData
    assert model.id


def test_code_built_model_round_trips(tmp_path):
    """A model built from cobra objects (not parsed from a doc) survives a
    write->read cycle intact — including the objective, which no other I/O
    round-trip test asserts is preserved."""
    m = cobra.Model("built")
    m.compartments = {"c": "cytoplasm"}
    a = cobra.Metabolite("a_c", name="A", compartment="c", formula="C6H12O6", charge=0)
    b = cobra.Metabolite("b_c", name="B", compartment="c", formula="C3H4O3", charge=-1)
    a.annotation = {"kegg.compound": ["C00031"]}
    m.add_metabolites([a, b])
    r = cobra.Reaction("R1", lower_bound=0, upper_bound=1000)
    r.add_metabolites({a: -1, b: 2})
    r.gene_reaction_rule = "G1"
    r.subsystem = "glycolysis"
    bio = cobra.Reaction("BIOMASS", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({b: -1})
    m.add_reactions([r, bio])
    m.objective = "BIOMASS"

    out = tmp_path / "built.yml"
    write_yaml_model(m, out)
    back = read_yaml_model(out)

    assert {x.id for x in back.metabolites} == {"a_c", "b_c"}
    assert {x.id for x in back.reactions} == {"R1", "BIOMASS"}
    rr = back.reactions.get_by_id("R1")
    assert rr.bounds == (0.0, 1000.0)
    assert rr.get_coefficient("a_c") == -1
    assert rr.get_coefficient("b_c") == 2
    assert rr.gene_reaction_rule == "G1"
    # RAVEN/raven-toolbox's shared format always writes subsystem as a
    # list (a reaction can carry more than one), even for the single-
    # subsystem case set here as a plain string — so it round-trips as
    # a one-item list, not the original bare string.
    assert rr.subsystem == ["glycolysis"]
    ma = back.metabolites.get_by_id("a_c")
    assert ma.formula == "C6H12O6"
    assert ma.annotation["kegg.compound"] == ["C00031"]
    # The objective survives the round-trip (no other I/O test checks this).
    assert {x.id for x in back.reactions if x.objective_coefficient != 0} == {"BIOMASS"}
    assert back.reactions.get_by_id("BIOMASS").objective_coefficient == 1.0


def test_read_yaml_model_gives_integer_charges(tmp_path):
    """Charges written as floats (RAVEN 3 format) are read as ints and reach SBML."""
    path = tmp_path / "float_charges.yml"
    path.write_text(
        "!!omap\n"
        "- metabolites:\n"
        "  - !!omap\n"
        "    - id: a_c\n"
        "    - name: A\n"
        "    - compartment: c\n"
        "    - charge: -1.0\n"
        "    - formula: C2H3O2\n"
        "  - !!omap\n"
        "    - id: b_c\n"
        "    - name: B\n"
        "    - compartment: c\n"
        "    - charge: 0.5\n"
        "- reactions:\n"
        "  - !!omap\n"
        "    - id: r1\n"
        "    - metabolites: !!omap\n"
        "      - a_c: -1.0\n"
        "      - b_c: 1.0\n"
        "    - lower_bound: 0.0\n"
        "    - upper_bound: 1000.0\n"
        "- genes: []\n"
        "- compartments: !!omap\n"
        "  - c: cytosol\n",
        encoding="utf-8",
    )
    model = read_yaml_model(path)
    a = model.metabolites.get_by_id("a_c")
    assert a.charge == -1 and isinstance(a.charge, int)
    # a fractional charge is kept as it is
    assert model.metabolites.get_by_id("b_c").charge == 0.5

    sbml = tmp_path / "float_charges.xml"
    cobra.io.write_sbml_model(model, str(sbml))
    assert cobra.io.read_sbml_model(str(sbml)).metabolites.get_by_id("a_c").charge == -1


def test_sbo_term_is_a_string_after_yaml_round_trip(tmp_path):
    """``annotation['sbo']`` is a bare string after a YAML round trip, as after add_sbo_terms or SBML read."""
    m = cobra.Model("sbo")
    a = cobra.Metabolite("a_c", compartment="c")
    a.annotation["sbo"] = "SBO:0000247"
    a.annotation["kegg.compound"] = ["C00031"]
    r = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r.add_metabolites({a: -1})
    r.annotation["sbo"] = "SBO:0000627"
    m.add_reactions([r])
    out = tmp_path / "sbo.yml"
    write_yaml_model(m, out)
    back = read_yaml_model(out)
    assert back.reactions.get_by_id("r1").annotation["sbo"] == "SBO:0000627"
    assert back.metabolites.get_by_id("a_c").annotation["sbo"] == "SBO:0000247"
    assert back.metabolites.get_by_id("a_c").annotation["kegg.compound"] == ["C00031"]


def test_sbo_term_written_as_a_list_is_read_as_a_string(tmp_path):
    path = tmp_path / "sbo_list.yml"
    path.write_text(
        "!!omap\n"
        "- metabolites:\n"
        "  - !!omap\n"
        "    - id: a_c\n"
        "    - compartment: c\n"
        "- reactions:\n"
        "  - !!omap\n"
        "    - id: r1\n"
        "    - metabolites: !!omap\n"
        "      - a_c: -1.0\n"
        "    - lower_bound: 0.0\n"
        "    - upper_bound: 1000.0\n"
        "    - annotation: !!omap\n"
        "      - sbo:\n"
        "        - SBO:0000627\n"
        "- genes: []\n"
        "- compartments: !!omap\n"
        "  - c: cytosol\n",
        encoding="utf-8",
    )
    model = read_yaml_model(path)
    assert model.reactions.get_by_id("r1").annotation["sbo"] == "SBO:0000627"


def test_write_warns_on_empty_metabolite_names(tmp_path):
    """RAVEN's writeYAMLmodel refuses empty metabolite names; Python writes but warns."""
    m = cobra.Model("m")
    m.add_metabolites(
        [
            cobra.Metabolite("a_c", name="A", compartment="c"),
            cobra.Metabolite("b_c", name="", compartment="c"),
        ]
    )
    with pytest.warns(UserWarning, match=r"1 metabolite\(s\) have an empty name \(b_c\)"):
        write_yaml_model(m, tmp_path / "out.yml")
    assert (tmp_path / "out.yml").exists()


def test_write_is_silent_when_all_metabolites_are_named(tmp_path, recwarn):
    m = cobra.Model("m")
    m.add_metabolites([cobra.Metabolite("a_c", name="A", compartment="c")])
    write_yaml_model(m, tmp_path / "out.yml")
    assert not [w for w in recwarn if "empty name" in str(w.message)]


def test_write_quotes_yaml_1_1_booleans(tmp_path):
    """Nitric oxide (name and formula NO), formula Y and compartment n are
    YAML 1.1 booleans, which PyYAML reads as True or False when unquoted."""
    m = cobra.Model("t")
    m.compartments = {"n": "nucleus"}
    no = cobra.Metabolite("no_n", name="NO", formula="NO", compartment="n")
    y = cobra.Metabolite("y_n", name="on", formula="Y", compartment="n")
    m.add_metabolites([no, y])
    r = cobra.Reaction("R1", name="yes")
    r.add_metabolites({no: -1, y: 1})
    m.add_reactions([r])

    out = tmp_path / "m.yml"
    write_yaml_model(m, out)
    text = out.read_text()
    for line in ('name: "NO"', 'formula: "NO"', 'formula: "Y"', 'compartment: "n"',
                 'name: "on"', 'name: "yes"'):
        assert line in text
    assert "compartment: n\n" not in text

    back = read_yaml_model(out)
    assert back.metabolites.no_n.name == "NO"
    assert back.metabolites.no_n.formula == "NO"
    assert back.metabolites.y_n.formula == "Y"
    assert back.metabolites.y_n.compartment == "n"
    assert back.reactions.R1.name == "yes"


@pytest.mark.parametrize("value", ["y", "N", "Yes", "nO", "ON", "Off"])
def test_needs_quote_yaml_1_1_booleans_any_case(value):
    from raven_toolbox.io.yaml import _needs_quote

    assert _needs_quote(value)


@pytest.mark.parametrize("value", ["NOS1", "Yn", "onion", "ny", "off-target"])
def test_needs_quote_leaves_longer_words_plain(value):
    from raven_toolbox.io.yaml import _needs_quote

    assert not _needs_quote(value)


def _edge_model():
    m = cobra.Model("edge")
    a = cobra.Metabolite("A_c", name="A", compartment="c")
    b = cobra.Metabolite("B_c", name="B", compartment="c")
    r1 = cobra.Reaction("R1", name="named")
    r2 = cobra.Reaction("R2", name="")
    m.add_reactions([r1, r2])
    r1.add_metabolites({a: -1, b: 1})
    r2.add_metabolites({b: -1})
    return m


def test_unnamed_reaction_has_no_name_key(tmp_path):
    out = tmp_path / "m.yml"
    write_yaml_model(_edge_model(), out)
    text = out.read_text()
    assert 'name: ""' not in text and "name: ''" not in text
    assert read_yaml_model(out).reactions.get_by_id("R2").name == ""


def test_default_bounds_are_written_only_when_the_model_stores_them(tmp_path):
    plain = tmp_path / "plain.yml"
    write_yaml_model(_edge_model(), plain)
    assert "defaultLB" not in plain.read_text()
    m = _edge_model()
    m.notes["metaData"] = {"defaultLB": -1000.0, "defaultUB": 1000.0}
    stored = tmp_path / "stored.yml"
    write_yaml_model(m, stored)
    assert "defaultLB" in stored.read_text()


def test_duplicate_metabolite_names_warn(tmp_path):
    m = cobra.Model("dup")
    x = cobra.Metabolite("X_c", name="dup", compartment="c")
    y = cobra.Metabolite("Y_c", name="dup", compartment="c")
    other = cobra.Metabolite("Z_m", name="dup", compartment="m")
    r = cobra.Reaction("Q")
    m.add_reactions([r])
    r.add_metabolites({x: -1, y: 1, other: 1})
    with pytest.warns(UserWarning, match=r"1 metabolite\(s\) share a name and compartment.*Y_c and X_c"):
        write_yaml_model(m, tmp_path / "m.yml")


def test_only_the_stored_default_bound_is_written(tmp_path):
    m = _edge_model()
    m.notes["metaData"] = {"defaultLB": -1000.0}
    out = tmp_path / "lb_only.yml"
    write_yaml_model(m, out)
    text = out.read_text()
    assert "defaultLB" in text
    assert "defaultUB" not in text
