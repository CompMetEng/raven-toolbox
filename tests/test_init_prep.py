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


def test_prep_init_model_accepts_repeated_task_ids_and_drops_failed_tasks_by_position():
    # Ids may repeat (category codes in Human-GEM's task list) and may be empty. The task that
    # cannot be performed is removed by its position, so the task with the same id stays.
    def task(task_id, description, inputs=True):
        return Task(id=task_id, description=description,
                    inputs=[("A[e]", 0.0, 1000.0)] if inputs else [],
                    outputs=[("C[c]", 1.0, 1.0)])
    tasks = [task("BS", "first"), task("ER", "second"), task("BS", "cannot be performed", False),
             task("", "empty id"), task("", "empty id again")]
    prep = prep_init_model(_task_met_model(), tasks, ext_comp="e", scale=False)
    assert [t.description for t in prep.tasks] == ["first", "second", "empty id", "empty id again"]
    assert [t.id for t in prep.tasks] == ["BS", "ER", "", ""]


@pytest.mark.parametrize("ids", [[None, None], [None, "T1"], [1, 1]])
def test_prep_init_model_refuses_task_ids_that_are_not_text(ids):
    tasks = [Task(id=i) for i in ids]
    with pytest.raises(ValueError, match="Task ids must be text"):
        prep_init_model(_gene_model(), tasks, ext_comp="e", simplify=False, scale=False)


def _task_met_model():
    m = cobra.Model("t")
    for name, comp in (("A", "e"), ("B", "c"), ("C", "c"), ("T", "c"), ("V", "c"), ("U", "c")):
        m.add_metabolites([cobra.Metabolite(f"{name}_{comp}", name=name, compartment=comp)])
    spec = [("EXA", (0, 1000), {"A_e": 1}), ("R1", (0, 1000), {"A_e": -1, "B_c": 1}),
            ("R2", (-1000, 1000), {"B_c": -1, "C_c": 1}), ("R3", (0, 1000), {"B_c": -1, "T_c": 1}),
            ("EXC", (0, 1000), {"C_c": -1})]
    for rid, bounds, stoich in spec:
        r = cobra.Reaction(rid)
        m.add_reactions([r])
        r.bounds = bounds
        r.add_metabolites({m.metabolites.get_by_id(k): v for k, v in stoich.items()})
    return m


def _mets(model):
    return [f"{m.name}[{m.compartment}]" for m in model.metabolites]


def test_first_simplification_keeps_the_metabolites_that_tasks_name():
    task = Task(id="MAKE_T", inputs=[("A[e]", 0.0, 1000.0)], outputs=[("T[c]", 0.0, 1000.0)],
                equations=[("1 B[c] => 2 V[c]", 0.0, 1000.0)])
    prep = prep_init_model(_task_met_model(), [task], ext_comp="e", scale=False)
    assert "R3" not in prep.ref_model.reactions
    assert _mets(prep.ref_model) == ["A[e]", "B[c]", "C[c]", "T[c]", "V[c]"]
    assert [t.id for t in prep.tasks] == ["MAKE_T"]


def test_first_simplification_removes_unused_metabolites_without_tasks():
    prep = prep_init_model(_task_met_model(), [], ext_comp="e", scale=False)
    assert _mets(prep.ref_model) == ["A[e]", "B[c]", "C[c]"]


def test_task_metabolites_are_matched_by_exact_name_and_compartment():
    task = Task(id="X", outputs=[("t[c]", 0.0, 1000.0), ("T[e]", 0.0, 1000.0)])
    prep = prep_init_model(_task_met_model(), [task], ext_comp="e", scale=False)
    assert _mets(prep.ref_model) == ["A[e]", "B[c]", "C[c]"]


def test_a_task_naming_a_metabolite_in_other_letter_case_is_still_checked_against_it():
    # Every metabolite is present while the tasks are checked, so the task is kept; the metabolite
    # is removed afterwards because its name does not match exactly.
    task = Task(id="X", outputs=[("t[c]", 0.0, 1000.0)])
    prep = prep_init_model(_task_met_model(), [task], ext_comp="e", scale=False)
    assert [t.id for t in prep.tasks] == ["X"]
    assert _mets(prep.ref_model) == ["A[e]", "B[c]", "C[c]"]


def test_of_metabolites_sharing_a_name_and_compartment_only_the_first_is_kept():
    model = _task_met_model()
    model.add_metabolites([cobra.Metabolite("T2_c", name="T", compartment="c")])
    task = Task(id="MAKE_T", outputs=[("T[c]", 0.0, 1000.0)])
    prep = prep_init_model(model, [task], ext_comp="e", scale=False)
    assert [m.id for m in prep.ref_model.metabolites] == ["A_e", "B_c", "C_c", "T_c"]
