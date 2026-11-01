"""Tests for raven_toolbox.manipulation.remove (removeMets/removeGenes ports)."""
import cobra
import pytest

from raven_toolbox.manipulation import (
    add_reactions_from_equations,
    remove_genes,
    remove_metabolites,
)


@pytest.fixture
def model():
    m = cobra.Model("t")
    m.add_metabolites(
        [
            cobra.Metabolite("atp_c", name="ATP", compartment="c"),
            cobra.Metabolite("atp_m", name="ATP", compartment="m"),
            cobra.Metabolite("adp_c", name="ADP", compartment="c"),
            cobra.Metabolite("x_c", name="X", compartment="c"),
        ]
    )
    add_reactions_from_equations(
        m,
        [
            {"id": "R1", "equation": "atp_c --> adp_c", "gene_reaction_rule": "G1 and G2"},
            {"id": "R2", "equation": "atp_c --> x_c", "gene_reaction_rule": "G3 or G4"},
            {"id": "R3", "equation": "atp_m --> adp_c"},  # no GPR (spontaneous)
        ],
    )
    return m


# --- remove_metabolites ----------------------------------------------------

def test_remove_metabolites_by_id(model):
    remove_metabolites(model, ["x_c"])
    assert "x_c" not in model.metabolites
    # reaction kept, just lost the metabolite
    assert "R2" in model.reactions


def test_remove_metabolites_by_name_across_compartments(model):
    # "ATP" exists in c and m; by_name removes both at once.
    remove_metabolites(model, ["ATP"], by_name=True)
    assert "atp_c" not in model.metabolites
    assert "atp_m" not in model.metabolites
    assert "adp_c" in model.metabolites


def test_remove_metabolites_destructive(model):
    remove_metabolites(model, ["adp_c"], destructive=True)
    # R1 and R3 both produced adp_c -> removed
    assert "adp_c" not in model.metabolites
    assert "R1" not in model.reactions and "R3" not in model.reactions


# --- remove_genes ----------------------------------------------------------

def test_remove_genes_remove_mode(model):
    blocked = remove_genes(model, ["G1"], blocked_reactions="remove")
    # R1 = "G1 and G2": removing G1 breaks the complex -> blocked -> removed
    assert blocked == ["R1"]
    assert "R1" not in model.reactions
    assert "R2" in model.reactions  # OR rule unaffected


def test_remove_genes_constrain_mode(model):
    blocked = remove_genes(model, ["G1"], blocked_reactions="constrain")
    assert blocked == ["R1"]
    r1 = model.reactions.get_by_id("R1")
    assert r1.bounds == (0, 0)  # kept but constrained, per RAVEN default
    assert r1.gene_reaction_rule == ""


def test_remove_genes_keep_mode(model):
    blocked = remove_genes(model, ["G1"], blocked_reactions="keep")
    assert blocked == ["R1"]
    r1 = model.reactions.get_by_id("R1")
    assert r1.gene_reaction_rule == ""
    assert r1.bounds != (0, 0)  # left untouched


def test_remove_genes_or_rule_not_blocked(model):
    blocked = remove_genes(model, ["G3"], blocked_reactions="remove")
    # R2 = "G3 or G4": removing G3 leaves G4 -> not blocked
    assert blocked == []
    assert model.reactions.get_by_id("R2").gene_reaction_rule == "G4"


def test_remove_genes_absent_gene_is_noop(model):
    assert remove_genes(model, ["NOPE"]) == []


def test_remove_genes_bad_policy(model):
    with pytest.raises(ValueError, match="blocked_reactions"):
        remove_genes(model, ["G1"], blocked_reactions="explode")


def test_remove_genes_default_constrains_as_raven(model):
    n_reactions = len(model.reactions)
    blocked = remove_genes(model, ["G1"])
    assert blocked == ["R1"]
    assert len(model.reactions) == n_reactions
    r1 = model.reactions.get_by_id("R1")
    assert r1.bounds == (0, 0)
    assert r1.gene_reaction_rule == ""


def test_remove_genes_drops_unused_genes(model):
    model.genes.append(cobra.Gene("G9"))  # orphan gene, not requested
    remove_genes(model, ["G1"])
    # G2 was named by R1, which is kept (constrained), so it stays; G9 goes.
    assert sorted(g.id for g in model.genes) == ["G2", "G3", "G4"]


def test_remove_genes_remove_mode_drops_genes_of_deleted_reactions(model):
    model.genes.append(cobra.Gene("G9"))
    remove_genes(model, ["G1"], blocked_reactions="remove")
    assert sorted(g.id for g in model.genes) == ["G3", "G4"]
