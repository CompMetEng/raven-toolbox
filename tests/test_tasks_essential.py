"""Phase 4d.1: essential-reaction discovery for tasks (find_task_essential_reactions).

Oracle: RAVEN tinitTests T0002 — for testModel + the "make e[s] from a[s]" task, the
pre-merge essential reactions are R2 (the only a[s]<->a[c] link) and R7 (the only
e[c]->e[s] producer); the alternative internal paths make nothing else essential.
"""
import cobra
import pytest
from tinit_oracles import (
    TEST_MODEL_TASK_ESSENTIAL_PREMERGE,
    make_test_model,
    make_test_task,
)

from raven_toolbox.tasks import (
    EssentialReactionsResult,
    Task,
    find_task_essential_reactions,
)


def test_essential_reactions_match_oracle():
    res = find_task_essential_reactions(make_test_model(), [make_test_task()])
    assert isinstance(res, EssentialReactionsResult)
    assert sorted(res.reactions) == TEST_MODEL_TASK_ESSENTIAL_PREMERGE  # ['R2', 'R7']
    assert not res.failed_tasks


def test_essential_directions_are_forward():
    """R2 (a[s]->a[c]) and R7 (e[c]->e[s]) both carry positive flux for this task."""
    res = find_task_essential_reactions(make_test_model(), [make_test_task()])
    assert res.reactions == {"R2": 1, "R7": 1}


def test_task_metabolites_collected():
    """a[s] and e[s] are referenced by the task and must be protected from removal."""
    res = find_task_essential_reactions(make_test_model(), [make_test_task()])
    m = make_test_model()
    names = {res_id: f"{m.metabolites.get_by_id(res_id).name}"
             f"[{m.metabolites.get_by_id(res_id).compartment}]" for res_id in res.task_metabolites}
    assert set(names.values()) == {"a[s]", "e[s]"}


def test_no_task_no_essentials():
    res = find_task_essential_reactions(make_test_model(), [])
    assert res.reactions == {} and res.per_task == {}


def test_cache_path_checkpoints_and_resumes(tmp_path, recwarn):
    """cache_path must checkpoint cleanly and let a re-run resume from it.

    This path had no coverage at all, despite being what makes a genome-scale prep
    survive an interruption. Two things are pinned here:

    * the checkpoint leaves no ``.part`` file behind and emits no ResourceWarning --
      the write used to hand ``open()`` straight to ``pickle.dump`` and rename the file
      on the next line, leaving the flush-before-rename to refcount timing (it emitted a
      ResourceWarning per task, 57 per Human-GEM prep, and on Windows renaming a file
      with a live handle raises PermissionError);
    * a second call reading that cache returns the same answer, which is the whole point
      of the feature.
    """
    cache = tmp_path / "essential.pkl"

    first = find_task_essential_reactions(
        make_test_model(), [make_test_task()], cache_path=cache)

    assert cache.exists(), "the checkpoint was never written"
    assert not (tmp_path / "essential.pkl.part").exists(), "a .part file was left behind"
    assert not [w for w in recwarn if issubclass(w.category, ResourceWarning)], (
        "the checkpoint leaked an unclosed file handle"
    )

    # Resume: the cached per-task results are reused and must reproduce the answer.
    resumed = find_task_essential_reactions(
        make_test_model(), [make_test_task()], cache_path=cache)
    assert resumed.reactions == first.reactions
    assert sorted(resumed.reactions) == TEST_MODEL_TASK_ESSENTIAL_PREMERGE


def test_equation_metabolites_are_protected():
    """A task equation's metabolites count as task metabolites (protected from removal)."""
    m = make_test_model()
    task = Task(
        id="equ",
        inputs=[("a[s]", 0.0, 1000.0)],
        outputs=[("e[c]", 1.0, 1.0)],
        equations=[("a[c] => e[c]", 0.0, 1000.0)],  # references a[c], which is not an I/O met
    )
    res = find_task_essential_reactions(m, [task])
    names = {f"{m.metabolites.get_by_id(i).name}[{m.metabolites.get_by_id(i).compartment}]"
             for i in res.task_metabolites}
    assert {"a[c]", "e[c]"} <= names and "equ" not in res.failed_tasks


def test_infeasible_task_is_reported_failed():
    """A task requiring an impossible output is dropped, not crashed."""
    impossible = Task(id="bad", outputs=[("z[s]", 1.0, 1.0)])
    # z[s] doesn't exist -> unknown metabolite -> failed.
    res = find_task_essential_reactions(make_test_model(), [impossible])
    assert res.failed_tasks == ["bad"] and res.reactions == {}


def test_should_fail_task_defines_no_essentials():
    res = find_task_essential_reactions(
        make_test_model(), [Task(id="sf", should_fail=True, outputs=[("e[s]", 1.0, 1.0)])]
    )
    assert res.reactions == {} and "sf" not in res.per_task


def test_direction_majority_across_tasks():
    """A reaction essential reverse in two tasks and forward in one is recorded reverse."""
    # Build a tiny model where a single reaction must run in a chosen direction.
    m = cobra.Model("dir")
    a, b = (cobra.Metabolite(x, name=x, compartment="s") for x in "ab")
    m.add_metabolites([a, b])
    r = cobra.Reaction("REV", lower_bound=-1000, upper_bound=1000)
    r.add_metabolites({a: -1, b: 1})  # a <=> b
    m.add_reactions([r])
    m.objective = "REV"
    # Task forcing net production of b from a -> REV forward (+1).
    fwd = Task(id="fwd", inputs=[("a[s]", 0.0, 1000.0)], outputs=[("b[s]", 1.0, 1.0)])
    # Two tasks forcing net production of a from b -> REV reverse (-1).
    rev1 = Task(id="rev1", inputs=[("b[s]", 0.0, 1000.0)], outputs=[("a[s]", 1.0, 1.0)])
    rev2 = Task(id="rev2", inputs=[("b[s]", 0.0, 1000.0)], outputs=[("a[s]", 1.0, 1.0)])
    res = find_task_essential_reactions(m, [rev1, rev2, fwd])
    assert res.reactions["REV"] == -1  # two reverse votes beat one forward


def test_duplicate_task_ids_all_contribute():
    """Tasks that share an id must each contribute to the union, not overwrite each other.

    Real task lists reuse a handful of ids across many tasks (metabolicTasks_Essential.txt
    has 57 tasks under 5 ids). Keying results by id would drop all but the last task per
    id, under-counting the essential set. Here two tasks share id 't', each making a
    different reaction essential; both must appear.
    """
    m = cobra.Model("dupid")
    a, b, c, d = (cobra.Metabolite(x, name=x, compartment="s") for x in "abcd")
    m.add_metabolites([a, b, c, d])
    r1 = cobra.Reaction("R1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({a: -1, b: 1})
    r2 = cobra.Reaction("R2", lower_bound=0, upper_bound=1000)
    r2.add_metabolites({c: -1, d: 1})
    m.add_reactions([r1, r2])
    m.objective = "R1"
    t_r1 = Task(id="t", inputs=[("a[s]", 0.0, 1000.0)], outputs=[("b[s]", 1.0, 1.0)])
    t_r2 = Task(id="t", inputs=[("c[s]", 0.0, 1000.0)], outputs=[("d[s]", 1.0, 1.0)])
    res = find_task_essential_reactions(m, [t_r1, t_r2])
    assert "R1" in res.reactions and "R2" in res.reactions  # neither overwritten
    # The per-task view merges the same-id tasks (union of their essentials), so the
    # earlier task's reaction is not lost there either.
    assert res.per_task["t"] == {"R1": 1, "R2": 1}


def test_processes_matches_sequential():
    """processes>1 (ProcessPoolExecutor) finds exactly what the sequential loop finds.

    Includes a should_fail task, a failing task and duplicate-id tasks, so the worker
    path is exercised on every branch record() handles, not just the success case.
    """
    m = cobra.Model("dir")
    a, b = (cobra.Metabolite(x, name=x, compartment="s") for x in "ab")
    m.add_metabolites([a, b])
    r = cobra.Reaction("REV", lower_bound=-1000, upper_bound=1000)
    r.add_metabolites({a: -1, b: 1})
    m.add_reactions([r])
    m.objective = "REV"
    fwd = Task(id="fwd", inputs=[("a[s]", 0.0, 1000.0)], outputs=[("b[s]", 1.0, 1.0)])
    rev1 = Task(id="rev1", inputs=[("b[s]", 0.0, 1000.0)], outputs=[("a[s]", 1.0, 1.0)])
    rev2 = Task(id="rev2", inputs=[("b[s]", 0.0, 1000.0)], outputs=[("a[s]", 1.0, 1.0)])
    sf = Task(id="sf", should_fail=True, outputs=[("b[s]", 1.0, 1.0)])
    bad = Task(id="bad", outputs=[("z[s]", 1.0, 1.0)])  # unknown metabolite -> failed
    tasks = [rev1, sf, rev2, bad, fwd]

    seq = find_task_essential_reactions(m, tasks, processes=1)
    par = find_task_essential_reactions(m, tasks, processes=4)

    assert par.reactions == seq.reactions == {"REV": -1}
    assert par.failed_tasks == seq.failed_tasks == ["bad"]
    assert par.task_metabolites == seq.task_metabolites
    assert par.per_task == seq.per_task


def test_duplicate_name_comp_metabolites_both_constrained():
    """A task referencing a name[comp] shared by two metabolites resolves (not 'missing')."""
    m = cobra.Model("dup")
    # Two distinct metabolites with the SAME name and compartment.
    a1 = cobra.Metabolite("a1", name="a", compartment="s")
    a2 = cobra.Metabolite("a2", name="a", compartment="s")
    b = cobra.Metabolite("b", name="b", compartment="s")
    m.add_metabolites([a1, a2, b])
    r1 = cobra.Reaction("R1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({a1: -1, b: 1})  # only a1 feeds b
    m.add_reactions([r1])
    m.objective = "R1"
    # Output b from input a -> 'a[s]' matches both a1 and a2; must not be reported missing.
    task = Task(id="t", inputs=[("a[s]", 0.0, 1000.0)], outputs=[("b[s]", 1.0, 1.0)])
    with pytest.warns(UserWarning, match="matches 2 model metabolites"):
        res = find_task_essential_reactions(m, [task])
    assert res.failed_tasks == []  # 'a[s]' resolved (to both a1 and a2), task feasible
    assert "R1" in res.reactions
