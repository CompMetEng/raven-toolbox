"""Phase 4d.1: essential-reaction discovery for tasks (find_task_essential_reactions).

Oracle: RAVEN tinitTests T0002 — for testModel + the "make e[s] from a[s]" task, the
pre-merge essential reactions are R2 (the only a[s]<->a[c] link) and R7 (the only
e[c]->e[s] producer); the alternative internal paths make nothing else essential.
"""
import os
import subprocess
import sys
from pathlib import Path

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

    * the checkpoint leaves no ``.part`` file behind and emits no ResourceWarning (on
      Windows renaming a file with a live handle raises PermissionError);
    * a second call reading that cache returns the same answer, which is the whole point
      of the feature.
    """
    cache = tmp_path / "essential.json"

    first = find_task_essential_reactions(
        make_test_model(), [make_test_task()], cache_path=cache)

    assert cache.exists(), "the checkpoint was never written"
    assert not (tmp_path / "essential.json.part").exists(), "a .part file was left behind"
    assert not [w for w in recwarn if issubclass(w.category, ResourceWarning)], (
        "the checkpoint leaked an unclosed file handle"
    )

    # Resume: the cached per-task results are reused and must reproduce the answer.
    resumed = find_task_essential_reactions(
        make_test_model(), [make_test_task()], cache_path=cache)
    assert resumed.reactions == first.reactions
    assert sorted(resumed.reactions) == TEST_MODEL_TASK_ESSENTIAL_PREMERGE


def test_cache_is_json_and_is_read_without_warning(tmp_path, recwarn):
    import json

    cache = tmp_path / "essential.json"
    first = find_task_essential_reactions(
        make_test_model(), [make_test_task()], cache_path=cache)
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert data["version"] == 1 and set(data) == {"version", "per_index", "mets", "failed"}
    assert sorted(data["per_index"]["0"]) == TEST_MODEL_TASK_ESSENTIAL_PREMERGE

    # A cache with a result for task 0 is used: the result comes from the file, not a solve.
    data["per_index"]["0"] = {"R2": 1}
    cache.write_text(json.dumps(data), encoding="utf-8")
    resumed = find_task_essential_reactions(
        make_test_model(), [make_test_task()], cache_path=cache)
    assert resumed.reactions == {"R2": 1}
    assert first.reactions != resumed.reactions
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_pickle_cache_is_ignored_with_a_warning(tmp_path):
    import pickle

    cache = tmp_path / "essential.pkl"
    cache.write_bytes(pickle.dumps({"per_index": {0: {"R2": 1}}, "mets": set(), "failed": []}))
    with pytest.warns(UserWarning, match="ignored"):
        res = find_task_essential_reactions(
            make_test_model(), [make_test_task()], cache_path=cache)
    assert sorted(res.reactions) == TEST_MODEL_TASK_ESSENTIAL_PREMERGE
    assert cache.read_bytes()[:1] == b"{"  # overwritten by the JSON checkpoint


@pytest.mark.parametrize("bad", [0, -3])
def test_checkpoint_every_below_one_raises(bad):
    with pytest.raises(ValueError, match="checkpoint_every"):
        find_task_essential_reactions(make_test_model(), [make_test_task()], checkpoint_every=bad)


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


def test_failed_tasks_are_reported_by_id_and_by_position():
    """Two tasks share an id; only the second cannot be performed.

    ``failed_tasks`` lists the id, ``failed_indices`` the position, so a caller can drop the
    failing task and keep the one with the same id.
    """
    good = Task(id="t", inputs=[("a[s]", 0.0, 1000.0)], outputs=[("e[s]", 1.0, 1.0)])
    bad = Task(id="t", outputs=[("z[s]", 1.0, 1.0)])
    res = find_task_essential_reactions(make_test_model(), [good, bad])
    assert res.failed_tasks == ["t"] and res.failed_indices == [1]


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


_ORDER_SCRIPT = """
from tinit_oracles import make_test_model, make_test_task
from raven_toolbox.tasks import find_task_essential_reactions
res = find_task_essential_reactions(make_test_model(), [make_test_task()])
print(",".join(res.reactions))
"""


def test_essential_set_and_order_do_not_depend_on_hash_seed():
    """Candidates are taken in model reaction order, whatever PYTHONHASHSEED is."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parent)] + [p for p in sys.path if p]
    )
    outputs = set()
    for seed in ("0", "1", "2", "3", "4", "5"):
        env["PYTHONHASHSEED"] = seed
        run = subprocess.run([sys.executable, "-c", _ORDER_SCRIPT], env=env,
                             capture_output=True, text=True, check=True)
        outputs.add(run.stdout.strip().splitlines()[-1])
    assert outputs == {"R2,R7"}


class _FakeProblem:
    """Solver stand-in whose stored basis makes a feasible knock-out look infeasible."""

    def __init__(self, feasible: bool, warm_status: str):
        self.feasible, self.warm_status, self.warm, self.resets = feasible, warm_status, True, 0

    def reset(self):
        self.warm, self.resets = False, self.resets + 1


class _FakeModel:
    def __init__(self, problem: _FakeProblem):
        self.solver = type("Solver", (), {})()
        self.solver.problem, self.solver.status = problem, None

    def slim_optimize(self):
        p = self.solver.problem
        if p.warm:
            self.solver.status = p.warm_status
        else:
            self.solver.status = "optimal" if p.feasible else "infeasible"


def test_knockout_rechecked_cold_after_infeasible_warm_start():
    """A warm-started 'infeasible' that a cold solve finds feasible is not essential."""
    from raven_toolbox.tasks.check import _knockout_feasible
    spurious = _FakeProblem(feasible=True, warm_status="infeasible")
    assert _knockout_feasible(_FakeModel(spurious)) is True
    assert spurious.resets == 1
    real = _FakeProblem(feasible=False, warm_status="infeasible")
    assert _knockout_feasible(_FakeModel(real)) is False


def test_knockout_feasible_statuses_follow_raven_check_solution():
    """Optimal and suboptimal count as feasible without a retry; time_limit gets one."""
    from raven_toolbox.tasks.check import _knockout_feasible
    for status in ("optimal", "suboptimal"):
        p = _FakeProblem(feasible=False, warm_status=status)
        assert _knockout_feasible(_FakeModel(p)) is True
        assert p.resets == 0
    p = _FakeProblem(feasible=False, warm_status="time_limit")
    assert _knockout_feasible(_FakeModel(p)) is False
    assert p.resets == 1


def test_knockout_feasible_on_solver_model():
    """On a real solver: blocking the only route to a required output is infeasible."""
    from raven_toolbox.tasks.check import _knockout_feasible
    m = cobra.Model("ko")
    a, b = cobra.Metabolite("a"), cobra.Metabolite("b")
    uptake, r1, r2, out = (cobra.Reaction(i) for i in ("EX_a", "R1", "R2", "EX_b"))
    uptake.add_metabolites({a: 1})
    r1.add_metabolites({a: -1, b: 1})
    r2.add_metabolites({a: -1, b: 1})
    out.add_metabolites({b: -1})
    m.add_reactions([uptake, r1, r2, out])
    uptake.bounds, r1.bounds, r2.bounds, out.bounds = (0, 10), (0, 10), (0, 10), (1, 10)
    r1.bounds = (0.0, 0.0)
    assert _knockout_feasible(m) is True       # R2 still carries the flux
    r2.bounds = (0.0, 0.0)
    assert _knockout_feasible(m) is False
