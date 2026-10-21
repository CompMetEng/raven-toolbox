"""Tests for metabolic tasks (Phase 4a): parse_task_list + check_tasks."""
import cobra
import pytest

from raven_toolbox.tasks import Task, check_tasks, parse_task_list

TASK_TSV = (
    "ID\tDESCRIPTION\tIN\tIN UB\tOUT\tOUT LB\tEQU\tSHOULD FAIL\n"
    "T1\tgrowth\tglc[e];o2[e]\t10\tbio[c]\t1\t\t\n"
    "T2\tinfeasible\t\t\tatp[c]\t1\t\ttrue\n"
    "\t\t\t\tnadh[c]\t\t\t\n"
    "T3\twithequ\tA[c]\t\tB[c]\t\tA[c] <=> B[c]\t\n"
)


# --------------------------------------------------------------------------- #
# parse_task_list
# --------------------------------------------------------------------------- #
@pytest.fixture
def task_file(tmp_path):
    p = tmp_path / "tasks.txt"
    p.write_text(TASK_TSV)
    return p


def test_parse_basic_and_defaults(task_file):
    tasks = parse_task_list(task_file)
    assert [t.id for t in tasks] == ["T1", "T2", "T3"]
    t1 = tasks[0]
    assert t1.description == "growth"
    # ';' splits mets sharing the row's bounds; IN LB defaults 0, IN UB from cell.
    assert t1.inputs == [("glc[e]", 0.0, 10.0), ("o2[e]", 0.0, 10.0)]
    assert t1.outputs == [("bio[c]", 1.0, 1000.0)]  # OUT UB defaults 1000


def test_parse_should_fail_and_continuation(task_file):
    t2 = parse_task_list(task_file)[1]
    assert t2.should_fail is True
    # continuation row (empty ID) appends nadh[c] to the same task's outputs
    assert t2.outputs == [("atp[c]", 1.0, 1000.0), ("nadh[c]", 0.0, 1000.0)]


def test_parse_equation_default_bounds(task_file):
    t3 = parse_task_list(task_file)[2]
    # reversible '<=>' -> EQU LB defaults -1000, UB 1000
    assert t3.equations == [("A[c] <=> B[c]", -1000.0, 1000.0)]


def test_parse_missing_id_column(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("FOO\tBAR\nx\ty\n")
    with pytest.raises(ValueError, match="ID"):
        parse_task_list(p)


def test_parse_warns_on_data_row_before_first_id(tmp_path):
    """Continuation rows appearing before the first task ID are malformed input;
    parse_task_list warns instead of silently dropping them."""
    p = tmp_path / "orphan.txt"
    p.write_text(
        "ID\tDESCRIPTION\tIN\tIN UB\tOUT\tOUT UB\tSHOULD FAIL\n"
        "\t\tglc[e]\t10\t\t\t\n"        # orphan data row, no ID seen yet
        "T1\tgrowth\t\t\tbio[c]\t1\t\n"
    )
    with pytest.warns(UserWarning, match="no task ID has been seen yet"):
        tasks = parse_task_list(p)
    assert [t.id for t in tasks] == ["T1"]
    # The orphan row's data isn't grafted onto T1 either.
    assert tasks[0].inputs == []


def test_parse_task_list_xlsx_missing_tasks_sheet(tmp_path):
    """A .xlsx without a 'TASKS' sheet raises a ValueError naming the sheets
    that are actually present."""
    pytest.importorskip("openpyxl")
    from openpyxl import Workbook

    wb = Workbook()
    wb.active.title = "NotTasks"
    p = tmp_path / "wrong.xlsx"
    wb.save(p)
    with pytest.raises(ValueError, match="no sheet named 'TASKS'"):
        parse_task_list(p)


# --------------------------------------------------------------------------- #
# check_tasks
# --------------------------------------------------------------------------- #
def _met(mid, name, comp="c"):
    return cobra.Metabolite(mid, name=name, compartment=comp)


@pytest.fixture
def model():
    """Closed model: A -> B (r1); D present but unproduced."""
    m = cobra.Model("t")
    A, B, D = _met("A_c", "A"), _met("B_c", "B"), _met("D_c", "D")
    m.add_metabolites([A, B, D])
    r1 = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({A: -1, B: 1})
    m.add_reactions([r1])
    return m


def _by_id(results):
    return {r.id: r for r in results}


def test_feasible_task_passes(model):
    # OUT LB=1 requires producing B (LB=0 would pass trivially via zero flux).
    task = Task("make_B", inputs=[("A[c]", 0, 1000)], outputs=[("B[c]", 1, 1000)])
    (res,) = check_tasks(model, [task])
    assert res.feasible and res.passed


def test_should_fail_task_passes_when_infeasible(model):
    # Require producing B with no input -> infeasible -> should_fail makes it pass.
    task = Task("no_input", outputs=[("B[c]", 1, 1000)], should_fail=True)
    (res,) = check_tasks(model, [task])
    assert not res.feasible and res.passed


def test_unsatisfiable_task_fails(model):
    task = Task("need_B", outputs=[("B[c]", 1, 1000)])  # no input, not should_fail
    (res,) = check_tasks(model, [task])
    assert not res.feasible and not res.passed


def test_equation_adds_pathway(model):
    # Model can't make D; the task's extra reaction B -> D enables output of D.
    task = Task(
        "make_D",
        inputs=[("A[c]", 0, 1000)],
        outputs=[("D[c]", 1, 1000)],
        equations=[("B[c] => D[c]", 0.0, 1000.0)],
    )
    (res,) = check_tasks(model, [task])
    assert res.passed
    # without the extra reaction D cannot be made
    (res2,) = check_tasks(model, [Task("make_D2", inputs=[("A[c]", 0, 1000)], outputs=[("D[c]", 1, 1000)])])
    assert not res2.passed


def test_changed_bounds_block_reaction(model):
    # Blocking r1 makes B unproducible.
    task = Task(
        "block_r1",
        inputs=[("A[c]", 0, 1000)],
        outputs=[("B[c]", 1, 1000)],
        changed=[("r1", 0.0, 0.0)],
    )
    (res,) = check_tasks(model, [task])
    assert not res.passed


def test_allmets_output(model):
    # Force uptake of A (IN LB=1); the only fate is A->B, so B must be excreted.
    # ALLMETS output permits that, making the task feasible; without it B accumulates.
    task = Task("sink_all", inputs=[("A[c]", 1, 1000)], outputs=[("ALLMETS", 0, 1000)])
    (res,) = check_tasks(model, [task])
    assert res.passed
    (res2,) = check_tasks(model, [Task("forced_no_out", inputs=[("A[c]", 1, 1000)])])
    assert not res2.passed  # forced A uptake but nowhere for B to go


def test_unknown_metabolite_reported(model):
    task = Task("typo", inputs=[("Z[c]", 0, 1000)], outputs=[("B[c]", 0, 1000)])
    (res,) = check_tasks(model, [task])
    assert not res.passed and "unknown metabolite" in res.error


def test_open_exchange_is_closed_so_task_controls_io(model):
    # An open sink would let B leave for free, but check_tasks closes all boundary
    # reactions first. The task defines input A but no output for B, so B must still
    # balance even though the sink existed.
    model.add_boundary(model.metabolites.B_c, type="sink")  # open B sink
    task = Task("need_D_out", inputs=[("A[c]", 0, 1000)], outputs=[("D[c]", 1, 1000)])
    (res,) = check_tasks(model, [task])
    assert not res.passed  # D still cannot be produced despite the (now-closed) B sink


def test_check_tasks_accepts_a_file_path(model, tmp_path):
    p = tmp_path / "t.txt"
    p.write_text(
        "ID\tDESCRIPTION\tIN\tOUT\tOUT LB\n"
        "make_B\tconvert\tA[c]\tB[c]\t1\n"
    )
    results = check_tasks(model, p)  # path, parsed internally
    assert _by_id(results)["make_B"].passed


# --------------------------------------------------------------------------- #
# Parity with RAVEN parseTaskList / checkTasks (flags, lists, CHANGED, duplicates)
# --------------------------------------------------------------------------- #
def test_parse_flags_false_texts(tmp_path):
    p = tmp_path / "flags.txt"
    p.write_text(
        "ID\tSHOULD FAIL\tPRINT FLUX\n"
        "A\tfalse\t0\n"
        "B\tNo\tFALSE\n"
        "C\tyes\t1\n"
        "D\t\tTRUE\n"
    )
    flags = {t.id: (t.should_fail, t.print_fluxes) for t in parse_task_list(p)}
    assert flags == {"A": (False, False), "B": (False, False), "C": (True, True), "D": (False, True)}


def test_parse_semicolon_lists_are_trimmed_and_drop_empty_items(tmp_path):
    p = tmp_path / "lists.txt"
    p.write_text(
        "ID\tIN\tIN UB\tCHANGED RXN\tCHANGED UB\n"
        "T1\tA[c]; B[c];\t5\t;r1 ; r2\t10\n"
    )
    (t,) = parse_task_list(p)
    assert t.inputs == [("A[c]", 0.0, 5.0), ("B[c]", 0.0, 5.0)]
    assert [c[0] for c in t.changed] == ["r1", "r2"]


def test_parse_empty_changed_bounds_are_none(tmp_path):
    p = tmp_path / "changed.txt"
    p.write_text("ID\tCHANGED RXN\tCHANGED LB\tCHANGED UB\nT1\tr1\t\t10\nT2\tr1\t-5\t\nT3\tr1\t1\t2\n")
    t1, t2, t3 = parse_task_list(p)
    assert t1.changed == [("r1", None, 10.0)]
    assert t2.changed == [("r1", -5.0, None)]
    assert t3.changed == [("r1", 1.0, 2.0)]


def test_parse_headers_ignore_case_and_whitespace(tmp_path):
    p = tmp_path / "hdr.txt"
    p.write_text(" id \tIn\t in ub \nT1\tA[c]\t7\n")
    (t,) = parse_task_list(p)
    assert t.inputs == [("A[c]", 0.0, 7.0)]


def test_parse_xls_gives_a_clear_error(tmp_path):
    p = tmp_path / "old.xls"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match=r"\.xls.*not supported.*\.xlsx"):
        parse_task_list(p)


def test_empty_changed_bound_leaves_reaction_bound_alone(model):
    """A None bound keeps the reaction's own bound; r1 is [0, 1000]."""
    inputs, outputs = [("A[c]", 0, 1000)], [("B[c]", 1, 1000)]
    keep_lb = Task("keep_lb", inputs=inputs, outputs=outputs, changed=[("r1", None, 1000.0)])
    (res,) = check_tasks(model, [keep_lb])
    assert res.passed
    # UB 0 with the LB left at 0 blocks r1, so B cannot be made.
    ub_zero = Task("ub_zero", inputs=inputs, outputs=outputs, changed=[("r1", None, 0.0)])
    (res2,) = check_tasks(model, [ub_zero])
    assert not res2.passed
    # The model passed in is not modified.
    assert model.reactions.r1.bounds == (0, 1000)


def test_duplicate_name_comp_warns_and_constrains_all():
    m = cobra.Model("dup")
    a1, a2, b = _met("A1_c", "A"), _met("A2_c", "A"), _met("B_c", "B")
    m.add_metabolites([a1, a2, b])
    r1 = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({a1: -1, b: 1})
    r2 = cobra.Reaction("r2", lower_bound=0, upper_bound=1000)
    r2.add_metabolites({a2: -1, b: 1})
    m.add_reactions([r1, r2])
    task = Task("dup", inputs=[("A[c]", 0, 1000)], outputs=[("B[c]", 1, 1000)])
    with pytest.warns(UserWarning, match=r"matches 2 model metabolites.*RAVEN constrains only the first"):
        (res,) = check_tasks(m, [task])
    assert res.passed
