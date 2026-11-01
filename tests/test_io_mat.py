"""Tests for raven_toolbox.io.mat (RAVEN-format .mat I/O)."""
import cobra
import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from raven_toolbox.io import (
    get_model_metadata,
    read_matlab_model,
    set_model_metadata,
    write_matlab_model,
)
from raven_toolbox.manipulation import add_reactions_from_equations

# Numeric fields RAVEN keeps as doubles. An integer column here is what makes
# mergeLinear fail on `S(...).' > 0 | rev`, so they are asserted by class.
# metComps is deliberately not among them - see test_metcomps_is_integer.
DOUBLE_FIELDS = ("lb", "ub", "rev", "c", "b", "metCharges")


@pytest.fixture
def model():
    m = cobra.Model("testGEM")
    m.name = "A test model"
    m.compartments = {"c": "cytoplasm", "e": "extracellular"}
    atp = cobra.Metabolite("atp_c", name="ATP", compartment="c", charge=-4)
    atp.formula = "C10H12N5O13P3"
    atp.annotation = {"kegg.compound": ["C00002"], "smiles": ["NC1=NC=NC2=C1N=CN2"]}
    atp.notes = {"metFrom": "testDB", "inchis": "InChI=1S/C10H16N5O13P3"}
    adp = cobra.Metabolite("adp_c", name="ADP", compartment="c", charge=-3)
    glc = cobra.Metabolite("glc_e", name="glucose", compartment="e", charge=0)
    m.add_metabolites([atp, adp, glc])
    add_reactions_from_equations(
        m,
        [
            {"id": "R1", "equation": "atp_c <=> adp_c"},
            {"id": "R2", "equation": "glc_e -> atp_c"},
        ],
    )
    r1 = m.reactions.R1
    r1.name = "ATP hydrolysis"
    r1.subsystem = ["Energy metabolism", "Transport"]
    r1.gene_reaction_rule = "g1 or g2"
    r1.annotation = {
        "ec-code": ["3.6.1.3", "3.6.1.-"],
        "kegg.reaction": ["R00086"],
        "kegg.pathway": ["map00190", "map00195"],
    }
    r1.notes = {
        "note": "added during curation",
        "references": "Agren et al 2013",
        "rxnFrom": "testDB",
        "confidence_score": 3,
        "deltaG": -30.5,
    }
    m.genes.g1.name = "ATPase1"
    m.genes.g1.annotation = {"uniprot": ["P12345"]}
    set_model_metadata(
        m, version="1.2.3", date="2026-10-07", taxonomy="taxonomy/9606",
        note="a note", organization="Chalmers",
    )
    return m


def _struct(path, varname):
    raw = scipy.io.loadmat(str(path), mat_dtype=True)
    return raw[varname]


def test_read_model_has_raven_solver_tolerance(model, tmp_path):
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    assert read_matlab_model(path).tolerance == 1e-9


def test_writes_raven_field_names(model, tmp_path):
    """RAVEN's names, not COBRA's: id/name/eccodes/geneShortNames."""
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    fields = _struct(path, "m").dtype.names
    for expected in ("id", "name", "version", "date", "annotation", "eccodes",
                     "geneShortNames", "rev", "subSystems"):
        assert expected in fields, expected
    for cobra_only in ("description_cobra", "modelName", "rxnECNumbers", "geneNames"):
        assert cobra_only not in fields


def test_numeric_fields_are_double(model, tmp_path):
    """An integer column breaks RAVEN's mergeLinear; assert the classes."""
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    struct = _struct(path, "m")
    for field in DOUBLE_FIELDS:
        assert struct[field][0, 0].dtype == np.float64, field


def test_metcomps_is_integer(model, tmp_path):
    """A 1-based index column, so readers that subtract 1 and index get an int.

    Signed, so a metabolite with no compartment (0) goes negative on that
    subtraction rather than wrapping round to the type's maximum.
    """
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    met_comps = _struct(path, "m")["metComps"][0, 0]
    assert met_comps.dtype.kind == "i"
    assert [int(x) for x in met_comps.ravel()] == [1, 1, 2]


def test_cobra_can_load_the_file(model, tmp_path):
    """cobra's loader indexes comps with metComps, so it needs an integer there."""
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    back = cobra.io.load_matlab_model(str(path))
    assert len(back.reactions) == len(model.reactions)
    assert back.metabolites.get_by_id("atp_c").compartment == "c"
    assert back.metabolites.get_by_id("glc_e").compartment == "e"


def test_rev_matches_reversibility(model, tmp_path):
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    rev = _struct(path, "m")["rev"][0, 0].ravel()
    assert list(rev) == [1.0, 0.0]


def test_matrices_are_sparse(model, tmp_path):
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    struct = _struct(path, "m")
    assert sp.issparse(struct["S"][0, 0])
    assert sp.issparse(struct["rxnGeneMat"][0, 0])


def test_subsystems_are_nested_cells(model, tmp_path):
    """RAVEN stores each reaction's subsystems as a cell, not a joined string."""
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    entry = _struct(path, "m")["subSystems"][0, 0][0, 0]
    assert entry.shape == (2, 1)
    assert [str(x[0]) for x in entry.ravel()] == ["Energy metabolism", "Transport"]


def test_miriams_struct_shape(model, tmp_path):
    """Cross-references go into a name/value struct, one row per value."""
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    entry = _struct(path, "m")["rxnMiriams"][0, 0][0, 0]
    names = [str(x[0]) for x in entry["name"][0, 0].ravel()]
    values = [str(x[0]) for x in entry["value"][0, 0].ravel()]
    assert names == ["kegg.reaction", "kegg.pathway", "kegg.pathway"]
    assert values == ["R00086", "map00190", "map00195"]
    # ec-code has its own RAVEN field and stays out of the struct.
    assert "ec-code" not in names


def test_eccodes_joined(model, tmp_path):
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    eccodes = _struct(path, "m")["eccodes"][0, 0]
    assert str(eccodes[0, 0][0]) == "3.6.1.3;3.6.1.-"


def test_varname_defaults_to_model_id(model, tmp_path):
    path = write_matlab_model(model, tmp_path / "m.mat")
    raw = scipy.io.loadmat(str(path))
    assert [k for k in raw if not k.startswith("__")] == ["testGEM"]


def test_optional_fields_omitted_when_empty(tmp_path):
    """RAVEN leaves a field out rather than writing a column of blanks."""
    bare = cobra.Model("bare")
    bare.add_metabolites([cobra.Metabolite("a_c", compartment="c")])
    add_reactions_from_equations(bare, [{"id": "R1", "equation": " -> a_c"}])
    path = write_matlab_model(bare, tmp_path / "bare.mat", varname="m")
    fields = _struct(path, "m").dtype.names
    for optional in ("eccodes", "rxnMiriams", "metMiriams", "rxnNotes",
                     "rxnReferences", "metFrom", "rxnFrom", "inchis"):
        assert optional not in fields


def test_round_trip_structure(model, tmp_path):
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    back = read_matlab_model(path)
    assert back.id == "testGEM"
    assert back.name == "A test model"
    assert len(back.reactions) == len(model.reactions)
    assert len(back.metabolites) == len(model.metabolites)
    assert len(back.genes) == len(model.genes)
    for rxn in model.reactions:
        other = back.reactions.get_by_id(rxn.id)
        assert other.lower_bound == rxn.lower_bound
        assert other.upper_bound == rxn.upper_bound
        assert other.gene_reaction_rule == rxn.gene_reaction_rule
        assert {m.id: c for m, c in other.metabolites.items()} == {
            m.id: c for m, c in rxn.metabolites.items()
        }


def test_round_trip_metabolite_detail(model, tmp_path):
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    atp = read_matlab_model(path).metabolites.atp_c
    assert atp.name == "ATP"
    assert atp.formula == "C10H12N5O13P3"
    assert atp.charge == -4
    assert atp.compartment == "c"
    assert atp.annotation["kegg.compound"] == ["C00002"]
    assert atp.annotation["smiles"] == ["NC1=NC=NC2=C1N=CN2"]
    assert atp.notes["metFrom"] == "testDB"
    assert atp.notes["inchis"] == "InChI=1S/C10H16N5O13P3"


def test_round_trip_reaction_detail(model, tmp_path):
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    r1 = read_matlab_model(path).reactions.R1
    assert r1.subsystem == ["Energy metabolism", "Transport"]
    assert r1.annotation["ec-code"] == ["3.6.1.3", "3.6.1.-"]
    assert r1.annotation["kegg.pathway"] == ["map00190", "map00195"]
    assert r1.notes["note"] == "added during curation"
    assert r1.notes["references"] == "Agren et al 2013"
    assert r1.notes["rxnFrom"] == "testDB"
    assert r1.notes["confidence_score"] == 3
    assert r1.notes["deltaG"] == pytest.approx(-30.5)


def test_round_trip_gene_detail(model, tmp_path):
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    g1 = read_matlab_model(path).genes.g1
    assert g1.name == "ATPase1"
    assert g1.annotation["uniprot"] == ["P12345"]


def test_round_trip_metadata(model, tmp_path):
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    meta = get_model_metadata(read_matlab_model(path))
    assert meta["version"] == "1.2.3"
    assert meta["date"] == "2026-10-07"
    assert meta["taxonomy"] == "taxonomy/9606"
    assert meta["note"] == "a note"
    assert meta["organization"] == "Chalmers"
    assert meta["defaultLB"] == -1000.0


def test_compartments_round_trip(model, tmp_path):
    path = write_matlab_model(model, tmp_path / "m.mat", varname="m")
    back = read_matlab_model(path)
    assert back.compartments == {"c": "cytoplasm", "e": "extracellular"}


def test_read_rejects_ambiguous_file(model, tmp_path):
    path = tmp_path / "two.mat"
    scipy.io.savemat(str(path), {"a": {"id": "x"}, "b": {"id": "y"}})
    with pytest.raises(KeyError, match="varname"):
        read_matlab_model(path)


def test_read_honours_varname(model, tmp_path):
    path = write_matlab_model(model, tmp_path / "m.mat", varname="humanGEM")
    assert read_matlab_model(path, varname="humanGEM").id == "testGEM"
    with pytest.raises(KeyError, match="no variable"):
        read_matlab_model(path, varname="nope")


def _cobra_writable(model):
    """A copy whose subsystems are strings, which is all cobra's writer takes."""
    out = model.copy()
    for rxn in out.reactions:
        if isinstance(rxn.subsystem, (list, tuple)):
            rxn.subsystem = ";".join(rxn.subsystem)
    return out


def test_reads_a_cobra_structure(model, tmp_path):
    """A COBRA .mat is recognised and read through cobra, not half-read.

    Human-GEM 2.0.1 and 2.1.0 shipped such a file, so this is the way back
    from those releases.
    """
    path = tmp_path / "cobra.mat"
    cobra.io.save_matlab_model(_cobra_writable(model), str(path))
    back = read_matlab_model(path)
    assert len(back.reactions) == len(model.reactions)
    assert len(back.metabolites) == len(model.metabolites)
    # the fields COBRA renames, which a RAVEN-only read would drop
    assert back.id == model.id
    assert back.reactions.R1.annotation["ec-code"] == ["3.6.1.3", "3.6.1.-"]
    assert back.genes.g1.name == "ATPase1"
    assert back.metabolites.atp_c.annotation["kegg.compound"] == ["C00002"]


def test_cobra_annotations_are_lists(model, tmp_path):
    """Single cross-references come back as lists, as from the RAVEN path."""
    path = tmp_path / "cobra.mat"
    cobra.io.save_matlab_model(_cobra_writable(model), str(path))
    annotation = read_matlab_model(path).reactions.R1.annotation
    assert all(isinstance(v, list) for v in annotation.values()), annotation


def test_rejects_a_struct_that_is_neither(tmp_path):
    path = tmp_path / "odd.mat"
    scipy.io.savemat(str(path), {"thing": {"alpha": [1.0], "beta": [2.0]}})
    with pytest.raises(ValueError, match="neither RAVEN nor COBRA"):
        read_matlab_model(path)
