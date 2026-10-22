"""Tests for raven_toolbox.io.git (exportForGit port)."""
import cobra
import libsbml
import pytest
from cobra.core import Group

from raven_toolbox.io import export_for_git, read_matlab_model, set_model_metadata
from raven_toolbox.io.git import _for_sbml
from raven_toolbox.manipulation import add_reactions_from_equations


@pytest.fixture
def model():
    m = cobra.Model("yeastGEM")
    m.compartments = {"c": "cytoplasm"}
    m.add_metabolites(
        [cobra.Metabolite("atp_c", name="ATP", compartment="c"),
         cobra.Metabolite("adp_c", name="ADP", compartment="c")]
    )
    add_reactions_from_equations(m, [{"id": "R1", "equation": "atp_c <=> adp_c"}])
    return m


def test_standard_gem_layout(model, tmp_path):
    root = export_for_git(model, tmp_path, prefix="yeast", formats=("yml", "xml", "mat", "xlsx", "txt"))
    assert root == tmp_path / "model"
    assert (root / "yml" / "yeast.yml").exists()
    assert (root / "xml" / "yeast.xml").exists()
    assert (root / "mat" / "yeast.mat").exists()
    assert (root / "xlsx" / "yeast.xlsx").exists()
    assert (root / "txt" / "yeast.txt").exists()
    assert (root / "dependencies.txt").exists()


def test_dependencies_file(model, tmp_path):
    root = export_for_git(model, tmp_path, formats=("yml",))
    deps = (root / "dependencies.txt").read_text()
    assert "python\t" in deps
    assert "cobra\t" in deps
    assert "raven_toolbox\t" in deps


def test_flat_layout(model, tmp_path):
    root = export_for_git(model, tmp_path, formats=("yml",), sub_dirs=False)
    assert root == tmp_path
    assert (tmp_path / "model.yml").exists()


def test_subset_of_formats(model, tmp_path):
    root = export_for_git(model, tmp_path, formats=("yml", "xml"))
    assert (root / "yml" / "model.yml").exists()
    assert not (root / "mat").exists()
    assert not (root / "xlsx").exists()


def test_does_not_mutate_model(model, tmp_path):
    order_before = [r.id for r in model.reactions]
    export_for_git(model, tmp_path, formats=("yml",))
    assert [r.id for r in model.reactions] == order_before


def test_txt_table_content(model, tmp_path):
    root = export_for_git(model, tmp_path, formats=("txt",))
    txt = (root / "txt" / "model.txt").read_text()
    assert txt.splitlines()[0] == "Rxn id\tFormula\tGene-reaction association\tLB\tUB\tObjective"
    assert "R1" in txt
    assert "ATP[c]" in txt


def test_txt_bounds_are_written_exactly(model, tmp_path):
    reaction = model.reactions.get_by_id("R1")
    reaction.bounds = (-1 / 3, 0.1)
    reaction.objective_coefficient = 1e-7
    root = export_for_git(model, tmp_path, formats=("txt",))
    fields = (root / "txt" / "model.txt").read_text().splitlines()[1].split("\t")
    assert fields[3:] == ["-0.3333333333333333", "0.1", "1e-07"]
    assert (float(fields[3]), float(fields[4]), float(fields[5])) == (-1 / 3, 0.1, 1e-7)
    assert not fields[1].endswith(" ")


def test_txt_whole_and_infinite_bounds(model, tmp_path):
    model.reactions.get_by_id("R1").bounds = (-1000.0, float("inf"))
    root = export_for_git(model, tmp_path, formats=("txt",))
    fields = (root / "txt" / "model.txt").read_text().splitlines()[1].split("\t")
    assert fields[3:5] == ["-1000", "Inf"]


def test_all_five_formats_are_written_by_default(model, tmp_path):
    root = export_for_git(model, tmp_path)
    for fmt in ("yml", "xml", "mat", "xlsx", "txt"):
        assert (root / fmt / f"model.{fmt}").exists()


def test_dependencies_are_ordered_and_repeatable(model, tmp_path):
    set_model_metadata(model, version="1.2.0", date="2026-10-05")
    first = export_for_git(model, tmp_path / "a", formats=("yml",))
    second = export_for_git(model, tmp_path / "b", formats=("yml",))
    text = (first / "dependencies.txt").read_text()
    assert text == (second / "dependencies.txt").read_text()
    names = [line.split("\t")[0] for line in text.splitlines()]
    assert names == [
        "python", "cobra", "libSBML", "ruamel.yaml", "scipy", "openpyxl", "raven_toolbox",
        "model_version", "model_date",
    ]
    assert "model_version\t1.2.0" in text


def test_bad_format(model, tmp_path):
    with pytest.raises(ValueError, match="Unknown format"):
        export_for_git(model, tmp_path, formats=("yml", "json"))


def test_mat_export_with_subsystem_lists(tmp_path):
    """A model read from YAML keeps subsystems as lists; the MATLAB export keeps
    them as a cell per reaction, the way RAVEN stores them, without changing the
    model passed in."""
    m = cobra.Model("m")
    a = cobra.Metabolite("a_c", compartment="c")
    b = cobra.Metabolite("b_c", compartment="c")
    r1, r2 = cobra.Reaction("r1"), cobra.Reaction("r2")
    r1.add_metabolites({a: -1, b: 1})
    r2.add_metabolites({b: -1})
    m.add_reactions([r1, r2])
    r1.subsystem = ["Glycolysis"]
    r2.subsystem = ["Transport", "Exchange"]
    export_for_git(m, tmp_path, prefix="m", formats=("mat",), sub_dirs=False)
    back = read_matlab_model(tmp_path / "m.mat")
    assert back.reactions.get_by_id("r1").subsystem == ["Glycolysis"]
    assert back.reactions.get_by_id("r2").subsystem == ["Transport", "Exchange"]
    assert r1.subsystem == ["Glycolysis"]


def _sbml_model():
    m = cobra.Model("sbmlGEM")
    m.compartments = {"c": "cytoplasm"}
    m.add_metabolites(
        [cobra.Metabolite("a_c", compartment="c"),
         cobra.Metabolite("b_c", compartment="c")]
    )
    add_reactions_from_equations(
        m, [{"id": "R1", "equation": "a_c <=> b_c"}, {"id": "R2", "equation": "b_c -> "}]
    )
    m.reactions.R1.subsystem = ["Glycolysis", "Transport"]
    m.reactions.R2.subsystem = ["Transport"]
    m.reactions.R1.notes = {
        "note": "curated", "references": "PMID:1", "confidence_score": 3,
        "rxnFrom": "testDB",
    }
    set_model_metadata(m, taxonomy="9606")
    return m


def test_sbml_builds_groups_from_subsystems():
    """RAVEN reads subSystems from the groups package, which cobra writes
    from model.groups; a model read from YAML has them on the reaction."""
    shaped = _for_sbml(_sbml_model())
    by_name = {g.name: {r.id for r in g.members} for g in shaped.groups}
    assert by_name == {"Glycolysis": {"R1"}, "Transport": {"R1", "R2"}}
    assert all(g.kind == "partonomy" for g in shaped.groups)


def test_sbml_renames_notes_to_raven_labels():
    """importModel's parseNote looks for its own labels, not cobra's keys."""
    notes = _for_sbml(_sbml_model()).reactions.R1.notes
    assert notes["NOTES"] == "curated"
    assert notes["AUTHORS"] == "PMID:1"
    assert notes["Confidence Level"] == 3
    assert notes["rxnFrom"] == "testDB"  # no RAVEN label, passed through
    for cobra_key in ("note", "references", "confidence_score"):
        assert cobra_key not in notes


def test_sbml_sets_taxonomy_annotation():
    """RAVEN recovers annotation.taxonomy from the identifiers.org URL."""
    assert _for_sbml(_sbml_model()).annotation["taxonomy"] == "9606"


def test_sbml_shaping_leaves_the_caller_model_alone():
    m = _sbml_model()
    _for_sbml(m)
    assert not m.groups
    assert "note" in m.reactions.R1.notes
    assert "taxonomy" not in (m.annotation or {})


def test_sbml_keeps_existing_groups():
    m = _sbml_model()
    group = Group(id="g1", name="Preset", kind="partonomy")
    group.add_members([m.reactions.R1])
    m.add_groups([group])
    shaped = _for_sbml(m)
    assert [g.name for g in shaped.groups] == ["Preset"]


def test_sbml_exposes_the_model_note():
    """RAVEN reads annotation.note; cobra only writes what is in model.notes."""
    m = _sbml_model()
    set_model_metadata(m, note="a generic human cell")
    assert _for_sbml(m).notes["note"] == "a generic human cell"


def test_sbml_notes_drop_non_strings():
    """cobra serialises a notes value with str(), so a dict would be published
    as its Python repr; read_yaml_model parks the metaData block there."""
    m = _sbml_model()
    m.notes = {**(m.notes or {}), "metaData": {"id": "x", "defaultLB": -1000.0}}
    notes = _for_sbml(m).notes
    assert "metaData" not in notes
    assert all(isinstance(v, str) for v in notes.values())


def test_sbml_notes_keep_the_model_note():
    m = _sbml_model()
    set_model_metadata(m, note="a generic human cell")
    assert _for_sbml(m).notes["note"] == "a generic human cell"


def _consistency(path):
    doc = libsbml.readSBMLFromFile(str(path))
    doc.checkConsistency()
    return sorted(
        (doc.getError(i).getSeverityAsString(), doc.getError(i).getErrorId())
        for i in range(doc.getNumErrors())
    )


def test_sbml_gene_label_is_the_gene_name(model, tmp_path):
    """RAVEN's importModel reads geneShortNames from fbc:label before fbc:name."""
    model.reactions.R1.gene_reaction_rule = "ENSG1 or ENSG2"
    model.genes.ENSG1.name = "DPM1"
    root = export_for_git(model, tmp_path, formats=("xml",))
    path = root / "xml" / "model.xml"

    doc = libsbml.readSBMLFromFile(str(path))
    products = doc.getModel().getPlugin("fbc").getListOfGeneProducts()
    labels = {gp.getId(): (gp.getName(), gp.getLabel()) for gp in products}
    assert labels == {"G_ENSG1": ("DPM1", "DPM1"), "G_ENSG2": ("G_ENSG2", "G_ENSG2")}

    plain = tmp_path / "plain.xml"
    cobra.io.write_sbml_model(_for_sbml(model), str(plain))
    assert _consistency(path) == _consistency(plain)

    back, plain_back = cobra.io.read_sbml_model(str(path)), cobra.io.read_sbml_model(str(plain))
    assert sorted((g.id, g.name) for g in back.genes) == sorted(
        (g.id, g.name) for g in plain_back.genes
    )
    assert back.reactions.R1.gene_reaction_rule == "ENSG1 or ENSG2"


def test_equation_leaves_out_a_coefficient_of_one(model, tmp_path):
    model.reactions.get_by_id("R1").add_metabolites({model.metabolites.atp_c: -1})
    root = export_for_git(model, tmp_path, formats=("txt",))
    equation = (root / "txt" / "model.txt").read_text().splitlines()[1].split("\t")[1]
    assert equation == "2 ATP[c] <=> ADP[c]"


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (2.777777777e-05, "2.7778e-05"),
        (0.123456, "0.12346"),
        (1234.56789, "1234.5679"),
        (1.2345678, "1.2346"),
        (2.0, "2"),
        (1000000.0, "1000000"),
    ],
)
def test_coefficients_are_written_as_matlab_num2str(value, text):
    from raven_toolbox.io.excel import _num2str

    assert _num2str(value) == text
    assert _num2str(-value) == text
