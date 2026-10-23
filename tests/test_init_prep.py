import cobra
import pytest

from raven_toolbox.init.prep import prep_init_model, rescale_for_init
from raven_toolbox.tasks import Task


def _model():
    m = cobra.Model("m")
    for i in "ABC":
        m.add_metabolites([cobra.Metabolite(i)])
    r = cobra.Reaction("R1", lower_bound=-5.0, upper_bound=20.0)
    m.add_reactions([r])
    r.add_metabolites({m.metabolites.A: -100.0, m.metabolites.B: 1.0, m.metabolites.C: 2.0})
    return m


def test_rescale_leaves_bounds():
    m = _model()
    rescale_for_init(m)
    assert m.reactions.R1.bounds == (-5.0, 20.0)
    assert abs(sum(abs(c) for c in m.reactions.R1.metabolites.values()) - 3.0) < 1e-12


def test_prep_init_model_resets_bounds():
    m = cobra.Model("t")
    for i in "ABC":
        m.add_metabolites([cobra.Metabolite(i + "_c", compartment="c")])
    ex = cobra.Reaction("EX_A", lower_bound=0, upper_bound=5)
    r1 = cobra.Reaction("R1", lower_bound=0, upper_bound=50)
    r2 = cobra.Reaction("R2", lower_bound=0, upper_bound=50)
    m.add_reactions([ex, r1, r2])
    ex.add_metabolites({m.metabolites.A_c: 1})
    r1.add_metabolites({m.metabolites.A_c: -1, m.metabolites.B_c: 1})
    r2.add_metabolites({m.metabolites.B_c: -1, m.metabolites.C_c: 1})
    prep = prep_init_model(m, [], ext_comp="e", simplify=False, scale=True)
    for r in prep.min_model.reactions:
        assert r.upper_bound in (0.0, 1000.0)


def test_prep_init_model_resets_reversible_lower_bound():
    m = cobra.Model("t")
    for i in "ABCD":
        m.add_metabolites([cobra.Metabolite(i + "_c", compartment="c")])
    rxns = {
        "EX_A": (-5, 5, {"A_c": 1}),
        "R1": (-50, 50, {"A_c": -1, "B_c": 1}),
        "R2": (-50, 50, {"B_c": -1, "C_c": 1}),
        "R3": (-50, 50, {"B_c": -1, "D_c": 1}),
        "EX_C": (-10, 10, {"C_c": -1}),
        "EX_D": (-10, 10, {"D_c": -1}),
    }
    for rid, (lb, ub, stoich) in rxns.items():
        r = cobra.Reaction(rid, lower_bound=lb, upper_bound=ub)
        m.add_reactions([r])
        r.add_metabolites({m.metabolites.get_by_id(k): v for k, v in stoich.items()})
    prep = prep_init_model(m, [], ext_comp="e", simplify=False, scale=True)
    assert any(r.lower_bound < 0 for r in prep.min_model.reactions)
    for r in prep.min_model.reactions:
        assert r.lower_bound in (0.0, -1000.0)
        assert r.upper_bound in (0.0, 1000.0)




def _gene_model():
    m = cobra.Model("g")
    for i in "ABD":
        m.add_metabolites([cobra.Metabolite(i + "_c", compartment="c")])
    rxns = {
        "EX_A": (0, 5, {"A_c": 1}, ""),
        "R1": (0, 50, {"A_c": -1, "B_c": 1}, "g1"),
        "EX_B": (0, 10, {"B_c": -1}, ""),
        "RD": (0, 50, {"B_c": -1, "D_c": 1}, "gd"),
    }
    for rid, (lb, ub, stoich, rule) in rxns.items():
        r = cobra.Reaction(rid, lower_bound=lb, upper_bound=ub)
        m.add_reactions([r])
        r.add_metabolites({m.metabolites.get_by_id(k): v for k, v in stoich.items()})
        r.gene_reaction_rule = rule
    m.genes.append(cobra.Gene("gunused"))
    return m


def test_first_simplification_removes_every_gene_without_a_reaction():
    m = _gene_model()
    prep = prep_init_model(m, [], ext_comp="e", simplify=True, scale=False)
    assert "RD" not in prep.ref_model.reactions
    assert sorted(g.id for g in prep.ref_model.genes) == ["g1"]
    assert sorted(g.id for g in m.genes) == ["g1", "gd", "gunused"]


def test_prep_init_model_keeps_genes_without_simplification():
    prep = prep_init_model(_gene_model(), [], ext_comp="e", simplify=False, scale=False)
    assert sorted(g.id for g in prep.ref_model.genes) == ["g1", "gd", "gunused"]


def test_prep_init_model_refuses_duplicate_task_ids():
    tasks = [Task(id="T1"), Task(id="T2"), Task(id="T1"), Task(id="T2"), Task(id="T3")]
    with pytest.raises(ValueError, match="T1, T2"):
        prep_init_model(_gene_model(), tasks, ext_comp="e", simplify=False, scale=False)
