"""The files export_for_git writes follow RAVEN's exportForGit where RAVEN's importModel reads them.

The toy model has a model id that is not a valid SBML SId, three genes (a protein and a short name,
a short name only, neither) and compartments that cobra lists as c, p, e.
"""
import cobra
import libsbml
import pytest

from raven_toolbox.io import export_for_git


def _model(model_id: str = "toy_v1.0.0") -> cobra.Model:
    m = cobra.Model(model_id)
    m.compartments = {"c": "cytosol", "e": "extracellular space", "p": "periplasm"}
    x_c, x_p, y_e = (cobra.Metabolite(i, name=i[0], compartment=i[-1]) for i in ("x_c", "x_p", "y_e"))
    r1, r2 = cobra.Reaction("R1"), cobra.Reaction("R2")
    m.add_reactions([r1, r2])
    r1.add_metabolites({x_c: -1, x_p: 1})
    r2.add_metabolites({x_p: -1, y_e: 1})
    r1.gene_reaction_rule = "g1 and (g2 or g3)"
    g1, g2 = m.genes.get_by_id("g1"), m.genes.get_by_id("g2")
    g1.name, g1.notes = "short1", {"protein": "Pg1"}
    g2.name = "short2"
    m.notes = {"version": "1.0.7"}
    return m


@pytest.fixture
def out(tmp_path):
    return export_for_git(_model(), tmp_path, prefix="toy", sub_dirs=False)


# libSBML frees a model together with its document, so the document must outlive the test.
_KEEP_ALIVE = []


def _sbml(root):
    doc = libsbml.readSBMLFromFile(str(root / "toy.xml"))
    _KEEP_ALIVE.append(doc)
    return doc.getModel()


def _products(root):
    return {p.getId(): p for p in _sbml(root).getPlugin("fbc").getListOfGeneProducts()}


def test_model_id_is_escaped_as_in_ravens_exportmodel(out):
    assert _sbml(out).getIdAttribute() == "toy_v1__46__0__46__0"


def test_model_id_that_is_a_valid_sid_is_unchanged(tmp_path):
    root = export_for_git(_model("plain_id"), tmp_path, prefix="toy", sub_dirs=False, formats=("xml",))
    assert _sbml(root).getIdAttribute() == "plain_id"


def test_gene_product_name_is_the_protein_and_label_the_short_name_or_gene_id(out):
    products = _products(out)
    assert (products["G_g1"].getName(), products["G_g1"].getLabel()) == ("Pg1", "short1")
    assert (products["G_g2"].isSetName(), products["G_g2"].getLabel()) == (False, "short2")
    assert (products["G_g3"].isSetName(), products["G_g3"].getLabel()) == (False, "g3")


def test_protein_is_not_also_written_as_a_note(out):
    assert "protein:" not in (out / "toy.xml").read_text(encoding="utf-8")


def test_gene_products_carry_sbo_243(out):
    assert {p.getSBOTermID() for p in _products(out).values()} == {"SBO:0000243"}


def test_own_sbo_term_of_a_gene_is_kept(tmp_path):
    m = _model()
    m.genes.get_by_id("g2").annotation["sbo"] = "SBO:0000010"
    root = export_for_git(m, tmp_path, prefix="toy", sub_dirs=False, formats=("xml",))
    products = _products(root)
    assert products["G_g2"].getSBOTermID() == "SBO:0000010"
    assert products["G_g1"].getSBOTermID() == "SBO:0000243"


def test_genes_and_id_of_the_caller_are_not_changed(tmp_path):
    """cobra's Model.copy shares the gene notes and annotation dicts, so a change in place leaks."""
    m = _model()
    export_for_git(m, tmp_path, prefix="toy", sub_dirs=False)
    assert m.genes.get_by_id("g1").notes == {"protein": "Pg1"}
    assert all(not gene.annotation for gene in m.genes)
    assert m.id == "toy_v1.0.0"


def test_sbml_reads_back_with_cobra(out):
    back = cobra.io.read_sbml_model(str(out / "toy.xml"))
    assert (len(back.genes), len(back.reactions), len(back.metabolites)) == (3, 2, 3)
    assert set(back.compartments) == {"c", "e", "p"}
