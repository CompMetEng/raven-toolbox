"""Tests for raven_toolbox.io.git (exportForGit port)."""
import re
from pathlib import Path

import cobra
import libsbml
import pytest
from cobra.core import Group

from raven_toolbox.io import export_for_git, read_matlab_model, read_yaml_model, set_model_metadata
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


def test_dependencies_list_the_toolchain_only_by_default(model, tmp_path):
    set_model_metadata(model, version="1.2.0", date="2026-10-05")
    root = export_for_git(model, tmp_path, formats=("yml",))
    names = [line.split("\t")[0] for line in (root / "dependencies.txt").read_text().splitlines()]
    assert names == ["python", "cobra", "libSBML", "ruamel.yaml", "scipy", "openpyxl", "raven_toolbox"]


def test_dependencies_are_ordered_and_repeatable(model, tmp_path):
    set_model_metadata(model, version="1.2.0", date="2026-10-05")
    first = export_for_git(model, tmp_path / "a", formats=("yml",), model_version_in_dependencies=True)
    second = export_for_git(model, tmp_path / "b", formats=("yml",), model_version_in_dependencies=True)
    text = (first / "dependencies.txt").read_text()
    assert text == (second / "dependencies.txt").read_text()
    names = [line.split("\t")[0] for line in text.splitlines()]
    assert names == [
        "python", "cobra", "libSBML", "ruamel.yaml", "scipy", "openpyxl", "raven_toolbox",
        "model_version", "model_date",
    ]
    assert "model_version\t1.2.0" in text


@pytest.mark.parametrize(
    "flag", ["main_branch_flag", "sub_dirs", "model_version_in_dependencies", "model_version_in_sbml"]
)
@pytest.mark.parametrize("value", ["false", "true", "", 0, 1, None])
def test_export_for_git_refuses_a_flag_that_is_not_a_bool(model, tmp_path, flag, value):
    with pytest.raises(TypeError, match=flag):
        export_for_git(model, tmp_path, formats=("yml",), **{flag: value})
    assert not (tmp_path / "model").exists()


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


def _group_members(path):
    doc = libsbml.readSBMLFromFile(str(path))
    groups = doc.getModel().getPlugin("groups").getListOfGroups()
    return {g.getName(): [m.getIdRef() for m in g.getListOfMembers()] for g in groups}


def test_sbml_group_members_follow_the_reaction_order(tmp_path):
    """cobra keeps the members of a group in a set of objects hashed by identity, so the order it writes
    changes from one process to the next. RAVEN's exportModel writes them in the order of the reactions."""
    m = cobra.Model("groupGEM")
    m.compartments = {"c": "cytoplasm"}
    met = cobra.Metabolite("a_c", compartment="c")
    reactions = [cobra.Reaction(f"R{i:03d}") for i in range(200)]
    m.add_reactions(reactions)
    for i, rxn in enumerate(reactions):
        rxn.add_metabolites({met: -1})
        rxn.subsystem = ["Transport"] if i % 2 else ["Transport", "Glycolysis"]
    export_for_git(m, tmp_path, prefix="m", formats=("xml",), sub_dirs=False)
    order = [f"R_{r.id}" for r in m.reactions]
    assert _group_members(tmp_path / "m.xml") == {
        "Transport": order,
        "Glycolysis": [r for i, r in enumerate(order) if not i % 2],
    }


def test_sbml_members_of_an_existing_group_follow_the_reaction_order(tmp_path):
    """A gene member goes after the reactions, sorted by id."""
    m = _sbml_model()
    m.reactions.R1.gene_reaction_rule = "g1"
    group = Group(id="g1", name="Preset", kind="partonomy")
    group.add_members([m.genes.g1, m.reactions.R2, m.reactions.R1])
    m.add_groups([group])
    export_for_git(m, tmp_path, prefix="m", formats=("xml",), sub_dirs=False)
    assert _group_members(tmp_path / "m.xml") == {"Preset": ["R_R1", "R_R2", "G_g1"]}


def _annotation_keys(path, reaction_id):
    doc = libsbml.readSBMLFromFile(str(path))
    xml = doc.getModel().getReaction(f"R_{reaction_id}").getAnnotationString()
    keys = []
    for key in re.findall(r'identifiers\.org/([^/"]+)/', xml):
        if key not in keys:
            keys.append(key)
    return keys


def _annotated_model():
    m = cobra.Model("annotatedGEM", name="annotated model")
    m.compartments = {"c": "cytoplasm"}
    m.add_metabolites([cobra.Metabolite("a_c", compartment="c"), cobra.Metabolite("b_c", compartment="c")])
    add_reactions_from_equations(m, [{"id": "R1", "equation": "a_c <=> b_c"}])
    r1 = m.reactions.R1
    r1.annotation["kegg.reaction"] = "R00001"
    r1.annotation["biocyc"] = "META:RXN-1"
    r1.annotation["ec-code"] = ["1.1.1.1", "1.1.1.2"]
    a_c = m.metabolites.a_c
    a_c.annotation["kegg.compound"] = "C00001"
    a_c.annotation["chebi"] = "CHEBI:15377"
    return m


def test_sbml_annotation_has_the_ec_code_first_and_the_other_keys_sorted(tmp_path):
    """RAVEN's exportModel writes model.eccodes before rxnMiriams; cobra writes the keys of an annotation
    in the order of its dict, which depends on the order in which they were added."""
    m = _annotated_model()
    export_for_git(m, tmp_path, prefix="m", formats=("xml",), sub_dirs=False)
    assert _annotation_keys(tmp_path / "m.xml", "R1") == ["ec-code", "biocyc", "kegg.reaction"]
    assert list(m.reactions.R1.annotation) == ["kegg.reaction", "biocyc", "ec-code"]
    assert list(m.metabolites.a_c.annotation) == ["kegg.compound", "chebi"]


def test_sbml_file_is_the_same_when_written_from_the_yaml_file(tmp_path):
    """The YAML file lists the annotation keys sorted, so a model read back from it has them in another
    order than the model that was exported."""
    export_for_git(_annotated_model(), tmp_path / "a", prefix="m", formats=("yml", "xml"), sub_dirs=False)
    back = read_yaml_model(tmp_path / "a" / "m.yml")
    export_for_git(back, tmp_path / "b", prefix="m", formats=("xml",), sub_dirs=False)
    assert (tmp_path / "a" / "m.xml").read_text() == (tmp_path / "b" / "m.xml").read_text()


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
    """RAVEN's importModel reads geneShortNames from fbc:label and proteins from fbc:name.

    A gene without a protein has no fbc:name, and its label is the short name or the gene id.
    """
    model.reactions.R1.gene_reaction_rule = "ENSG1 or ENSG2"
    model.genes.ENSG1.name = "DPM1"
    root = export_for_git(model, tmp_path, formats=("xml",))
    path = root / "xml" / "model.xml"

    doc = libsbml.readSBMLFromFile(str(path))
    products = doc.getModel().getPlugin("fbc").getListOfGeneProducts()
    labels = {gp.getId(): (gp.getName(), gp.getLabel()) for gp in products}
    assert labels == {"G_ENSG1": ("", "DPM1"), "G_ENSG2": ("", "ENSG2")}

    plain = tmp_path / "plain.xml"
    cobra.io.write_sbml_model(_for_sbml(model), str(plain))
    assert _consistency(path) == _consistency(plain)

    back = cobra.io.read_sbml_model(str(path))
    assert sorted(g.id for g in back.genes) == ["ENSG1", "ENSG2"]
    assert back.reactions.R1.gene_reaction_rule == "ENSG1 or ENSG2"


def test_sbml_protein_is_the_gene_product_name(model, tmp_path):
    model.reactions.R1.gene_reaction_rule = "ENSG1 or ENSG2"
    model.genes.ENSG1.name = "DPM1"
    model.genes.ENSG1.notes = {"protein": "dolichol-phosphate mannosyltransferase"}
    root = export_for_git(model, tmp_path, formats=("xml",))
    doc = libsbml.readSBMLFromFile(str(root / "xml" / "model.xml"))
    products = {gp.getId(): gp for gp in doc.getModel().getPlugin("fbc").getListOfGeneProducts()}
    assert products["G_ENSG1"].getName() == "dolichol-phosphate mannosyltransferase"
    assert products["G_ENSG1"].getLabel() == "DPM1"
    assert not products["G_ENSG2"].isSetName()
    assert model.genes.ENSG1.notes == {"protein": "dolichol-phosphate mannosyltransferase"}


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
        (1000000000000000.125, "1000000000000000"),
        (2.0**53, "9007199254740992"),
        (1e20, "1e+20"),
        (1e300, "1e+300"),
        (float("inf"), "Inf"),
    ],
)
def test_coefficients_are_written_as_matlab_num2str(value, text):
    from raven_toolbox.io.excel import _num2str

    assert _num2str(value) == text
    assert _num2str(-value) == "-" + text


def test_num2str_of_infinity_and_negative_zero():
    from raven_toolbox.io.excel import _num2str

    assert _num2str(float("-inf")) == "-Inf"
    assert _num2str(-0.0) == "0"


def test_main_branch_flag_runs_the_check_only_when_set(model, tmp_path, monkeypatch):
    calls = []

    def refuse():
        calls.append(1)
        raise RuntimeError("raven_toolbox is on branch 'feature'")

    monkeypatch.setattr("raven_toolbox.io.git._check_main_branch", refuse)
    export_for_git(model, tmp_path, formats=["txt"])
    assert not calls
    with pytest.raises(RuntimeError, match="feature"):
        export_for_git(model, tmp_path, formats=["txt"], main_branch_flag=True)


def _git(cwd, *args):
    import subprocess as sp
    sp.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
           check=True, capture_output=True)


def _copy_package(dst):
    import shutil

    import raven_toolbox
    shutil.copytree(Path(raven_toolbox.__file__).parent, dst,
                    ignore=shutil.ignore_patterns("__pycache__"))


def _run_check(package_dir):
    import subprocess as sp
    import sys
    code = ("import sys; sys.path.insert(0, sys.argv[1]);"
            "from raven_toolbox.io.git import _check_main_branch as c; c()")
    return sp.run([sys.executable, "-c", code, str(package_dir)],
                  capture_output=True, text=True)


def test_main_branch_check_in_checkout_layout(tmp_path):
    repo = tmp_path / "checkout"
    _copy_package(repo / "src" / "raven_toolbox")
    (repo / "pyproject.toml").write_text('[project]\nname = "raven-toolbox"\n')
    _git(repo, "init", "-q", "-b", "feature")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "x")
    out = _run_check(repo / "src")
    assert out.returncode != 0 and "feature" in out.stderr
    _git(repo, "checkout", "-q", "-b", "main")
    assert _run_check(repo / "src").returncode == 0


def test_main_branch_check_passes_in_unrelated_repository(tmp_path):
    repo = tmp_path / "project"
    _copy_package(repo / ".venv" / "lib" / "raven_toolbox")
    (repo / "pyproject.toml").write_text('[project]\nname = "analysis"\n')
    _git(repo, "init", "-q", "-b", "analysis")
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-qm", "x")
    assert _run_check(repo / ".venv" / "lib").returncode == 0


def _checkout(tmp_path, *, layout, name):
    repo = tmp_path / "repo"
    pkg = repo / "src" / "raven_toolbox" if layout else repo / "lib" / "raven_toolbox"
    _copy_package(pkg)
    (repo / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
    _git(repo, "init", "-q", "-b", "feature")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "x")
    return repo / ("src" if layout else "lib")


def test_main_branch_check_passes_with_layout_but_other_project_name(tmp_path):
    assert _run_check(_checkout(tmp_path, layout=True, name="analysis")).returncode == 0


def test_main_branch_check_passes_with_project_name_but_other_layout(tmp_path):
    assert _run_check(_checkout(tmp_path, layout=False, name="raven-toolbox")).returncode == 0


def test_main_branch_check_raises_with_layout_and_project_name(tmp_path):
    out = _run_check(_checkout(tmp_path, layout=True, name="raven-toolbox"))
    assert out.returncode != 0 and "feature" in out.stderr


@pytest.mark.parametrize(
    "flag", ["main_branch_flag", "sub_dirs", "model_version_in_dependencies", "model_version_in_sbml"]
)
def test_export_for_git_accepts_numpy_booleans(model, tmp_path, flag):
    import numpy as np

    kwargs = {"main_branch_flag": False, "sub_dirs": False, flag: np.False_}
    export_for_git(model, tmp_path, formats=("yml",), **kwargs)
    assert (tmp_path / "model.yml").exists()


def test_export_for_git_flag_error_names_the_type(model, tmp_path):
    with pytest.raises(TypeError, match=r"got builtins\.str"):
        export_for_git(model, tmp_path, formats=("yml",), sub_dirs="no")
