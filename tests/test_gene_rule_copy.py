"""Every function that copies or builds a gene rule keeps RAVEN's reading of it.

In RAVEN's grammar ``G1&G2`` is one gene id (``&`` is an operator only as a whole word);
cobra's string parser would split it into ``G1`` and ``G2``.
"""
import cobra
import pytest

from raven_toolbox.io.yaml import read_yaml_model, write_yaml_model
from raven_toolbox.utils import parse_gr_rule

RULE = "G1&G2 or G3"
GENES = ["G1&G2", "G3"]


def _model(tmp_path):
    """A model whose R1 has RAVEN's rule ``G1&G2 or G3``, read back from YAML."""
    m = cobra.Model("y")
    a = cobra.Metabolite("A_c", name="A", compartment="c")
    b = cobra.Metabolite("B_c", name="B", compartment="c")
    m.add_metabolites([a, b])
    r = cobra.Reaction("R1", lower_bound=-10, upper_bound=10)
    m.add_reactions([r])
    r.add_metabolites({a: -1, b: 1})
    r.gpr = parse_gr_rule(RULE)
    m.compartments = {"c": "cytosol", "m": "mito"}
    path = tmp_path / "m.yml"
    write_yaml_model(m, path)
    return read_yaml_model(path)


def _merge_compartments(m, tmp_path):
    from raven_toolbox.manipulation import merge_compartments
    return merge_compartments(m)[0], "R1"


def _copy_to_compartment(m, tmp_path):
    from raven_toolbox.manipulation import copy_to_compartment
    out, new, _ = copy_to_compartment(m, ["R1"], "m")
    return out, new[0]


def _apply_assignment_duplicate(m, tmp_path):
    from raven_toolbox.localization.assign import AssignmentProposal, apply_assignment
    return apply_assignment(m, AssignmentProposal(placements={"R1": ["c", "m"]})), "R1_m"


def _apply_assignment_universal(m, tmp_path):
    from raven_toolbox.localization.assign import AssignmentProposal, apply_assignment
    universal = m.copy()
    universal.reactions.R1.id = "U1"
    universal.repair()
    m.remove_reactions(["R1"], remove_orphans=True)
    return apply_assignment(m, AssignmentProposal(added_reactions=["U1"]),
                            universal=universal), "U1"


def _convert_to_irreversible(m, tmp_path):
    from raven_toolbox.manipulation import convert_to_irreversible
    convert_to_irreversible(m)
    return m, "R1_REV"


def _merge_models(m, tmp_path):
    from raven_toolbox.manipulation import merge_models
    return merge_models([m, cobra.Model("z")]), "R1"


def _task_fill_reference(m, tmp_path):
    from raven_toolbox.init.taskfill import _add_reference_reactions
    target = cobra.Model("t")
    _add_reference_reactions(target, m, ["R1"])
    return target, "R1"


def _add_reactions_from_model(m, tmp_path):
    from raven_toolbox.manipulation import add_reactions_from_model
    target = cobra.Model("t")
    target.add_metabolites([cobra.Metabolite("A_c", name="A", compartment="c"),
                            cobra.Metabolite("B_c", name="B", compartment="c")])
    add_reactions_from_model(target, m, ["R1"], genes=True)
    return target, "R1"


def _matlab_round_trip(m, tmp_path):
    from raven_toolbox.io.mat import read_matlab_model, write_matlab_model
    path = tmp_path / "m.mat"
    write_matlab_model(m, path)
    return read_matlab_model(path), "R1"


def _change_rules(m, tmp_path):
    from raven_toolbox.manipulation import change_gene_reaction_rules
    change_gene_reaction_rules(m, {"R1": RULE})
    return m, "R1"


def _change_rules_append(m, tmp_path):
    from raven_toolbox.manipulation import change_gene_reaction_rules
    m.reactions.R1.gpr = parse_gr_rule("G1&G2")
    change_gene_reaction_rules(m, {"R1": "G3"}, replace=False)
    return m, "R1"


def _add_from_equations(m, tmp_path):
    from raven_toolbox.manipulation import add_reactions_from_equations
    add_reactions_from_equations(m, [{"id": "R9", "equation": "A_c => B_c",
                                      "gene_reaction_rule": RULE}])
    return m, "R9"


def _remove_low_score_genes(m, tmp_path):
    from raven_toolbox.init.genes import remove_low_score_genes
    m.reactions.R1.gpr = parse_gr_rule("G1&G2 or G3 or G4")
    out, _ = remove_low_score_genes(m, {"G1&G2": 1.0, "G3": 1.0, "G4": -5.0})
    return out, "R1"


@pytest.mark.parametrize("operation", [
    _merge_compartments, _copy_to_compartment, _apply_assignment_duplicate,
    _apply_assignment_universal, _convert_to_irreversible, _merge_models,
    _task_fill_reference, _add_reactions_from_model, _matlab_round_trip, _change_rules,
    _change_rules_append, _add_from_equations, _remove_low_score_genes,
], ids=lambda f: f.__name__.lstrip("_"))
def test_operation_keeps_raven_gene_ids(operation, tmp_path):
    out, rid = operation(_model(tmp_path), tmp_path)
    assert sorted(g.id for g in out.reactions.get_by_id(rid).genes) == GENES
    assert {"G1", "G2"}.isdisjoint(g.id for g in out.genes)


def test_expand_model_keeps_raven_gene_ids(tmp_path):
    from raven_toolbox.manipulation import expand_model
    m = _model(tmp_path)
    m.reactions.R1.gpr = parse_gr_rule("(G1&G2 and G3) or G4")
    expand_model(m)
    expanded = [r for r in m.reactions if r.id.startswith("R1_EXP")]
    assert sorted({g.id for r in expanded for g in r.genes}) == ["G1&G2", "G3", "G4"]


def test_set_gene_rule_copies_a_gpr_and_reads_strings_in_raven_grammar():
    from raven_toolbox.utils.gpr import set_gene_rule

    source = cobra.Reaction("S")
    source.gpr = parse_gr_rule(RULE)
    target = cobra.Reaction("T")
    set_gene_rule(target, source.gpr)
    assert sorted(g.id for g in target.genes) == GENES
    assert target.gpr is not source.gpr
    set_gene_rule(target, "")
    assert target.gene_reaction_rule == "" and not target.genes
    set_gene_rule(target, RULE)
    assert sorted(g.id for g in target.genes) == GENES


@pytest.mark.filterwarnings("ignore:Malformed gene_reaction_rule")
def test_rule_to_gpr_falls_back_to_cobra_for_a_rule_raven_refuses():
    from cobra.core.gene import GPR

    from raven_toolbox.utils.gpr import rule_to_gpr

    rule = "G1 G2 or G3"
    assert rule_to_gpr(rule).to_string() == GPR.from_string(rule).to_string()
    assert rule_to_gpr(None).body is None
