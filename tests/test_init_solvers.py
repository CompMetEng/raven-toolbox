"""Cross-solver smoke tests for the ftINIT MILP path.

The clean-data calibration and robustness studies were run on Gurobi; the tractability
choices (big-M=100, MIP gap, time limits) and the Gurobi-specific param plumbing
(``opt.problem.Params.MIPGap``) only matter if those choices also work on the *other*
MILP backends real users have. These tests assert that each available MILP-capable
optlang interface produces the same reaction-set verdict as Gurobi on the toy models the
unit tests use — so a regression in solver portability fails CI instead of being found
months later on a user's machine.

Solvers tested: every MILP-capable cobra/optlang interface that imports in this env
(Gurobi, HiGHS via ``hybrid``, GLPK). Missing ones are skipped automatically. Genome-scale
behaviour is measured separately by ``scripts/analyze_init_solvers.py`` (manual benchmark).
"""
from __future__ import annotations

import importlib

import cobra
import pytest

from raven_toolbox.init import ftinit, prep_init_model, run_ftinit
from raven_toolbox.tasks import Task, check_tasks

# Detect which MILP-capable optlang interfaces actually work; skip the rest.
# We do a real import (not just find_spec) because optlang ships every backend's
# module file but those that wrap third-party solvers (gurobi, cplex) only import
# cleanly when the underlying solver is installed — find_spec would say "present"
# and then we'd crash at fixture time on CI runners without Gurobi.
_INTERFACES = {"gurobi": "gurobi_interface", "hybrid": "hybrid_interface", "glpk": "glpk_interface"}


def _solver_available(modname: str) -> bool:
    try:
        importlib.import_module(f"optlang.{modname}")
        return True
    except ImportError:
        return False


_AVAILABLE = [name for name, mod in _INTERFACES.items() if _solver_available(mod)]

# Known upstream blocker: ``optlang.hybrid_interface.Configuration.clone()`` rejects
# ``lp_method='primal'``. Marked strict so this flips red when optlang is fixed and
# we should drop the marker. See docs/init_solver_benchmark.md.
_XFAIL = {"hybrid": pytest.mark.xfail(
    reason="optlang hybrid_interface.Configuration rejects lp_method='primal' (upstream)",
    strict=True, raises=ValueError,
)}


def _param(name: str):
    marks = [_XFAIL[name]] if name in _XFAIL else []
    return pytest.param(name, marks=marks, id=name)


@pytest.fixture(params=[_param(n) for n in _AVAILABLE])
def solver(request):
    """One installed MILP solver per parameter value."""
    return request.param


# ----------------------------------------------------------------------- toy fixtures

def _met(mid, comp="c"):
    return cobra.Metabolite(mid, name=mid.split("_")[0], compartment=comp)


def _toy_ftinit_model() -> cobra.Model:
    """Small flux-consistent network for ftINIT: A→B, B→C, parallel A→C (negative-score)."""
    def rxn(rid, lb, ub, mets):
        r = cobra.Reaction(rid, lower_bound=lb, upper_bound=ub)
        r.add_metabolites(mets)
        return r
    m = cobra.Model("ftoy")
    A, B, C = (_met(x) for x in ("A_c", "B_c", "C_c"))
    m.add_metabolites([A, B, C])
    m.add_reactions([rxn("EX_A", -1000, 0, {A: -1}),
                     rxn("EX_C", 0, 1000, {C: -1}),
                     rxn("r1", 0, 1000, {A: -1, B: 1}),
                     rxn("r2", 0, 1000, {B: -1, C: 1}),
                     rxn("rbad", 0, 1000, {A: -1, C: 1})])
    return m


# --------------------------------------------------------------------- tests

def test_run_ftinit_same_verdict(solver):
    """ftINIT MILP picks the same on-set across solvers on a small network."""
    m = _toy_ftinit_model()
    m.solver = solver
    res = run_ftinit(m, {"r1": 1.0, "r2": 1.0, "rbad": -1.0}, allow_excretion=True)
    assert "rbad" not in res.on_reactions
    assert {"r1", "r2"}.issubset(res.on_reactions)


def test_check_tasks_works_per_solver(solver):
    """check_tasks (one slim_optimize per task) works with each solver."""
    m = _toy_ftinit_model()
    m.solver = solver
    task = Task(id="make_c", inputs=[("A[c]", 0.0, 1000.0)], outputs=[("C[c]", 1.0, 1.0)])
    results = check_tasks(m, [task])
    assert results[0].passed


def test_ftinit_pipeline_with_tasks(solver):
    """The full ftinit() pipeline (prep + staged MILP + gap-fill) runs with each solver."""
    m = _toy_ftinit_model()
    m.solver = solver
    task = Task(id="make_c", inputs=[("A[c]", 0.0, 1000.0)], outputs=[("C[c]", 1.0, 1.0)])
    prep = prep_init_model(m, [task])
    out = ftinit(prep, rxn_scores={"r1": 1.0, "r2": 1.0, "rbad": -1.0}, series="1+1")
    # Functional: the target task remains satisfiable in the extracted model.
    assert check_tasks(out, [task])[0].passed


def test_solver_param_plumbing(solver):
    """mip_gap / time_limit reach the solver without raising (graceful per backend)."""
    m = _toy_ftinit_model()
    m.solver = solver
    # Tight time limit + loose gap on a trivial problem; just verify the call returns.
    res = run_ftinit(m, {"r1": 1.0, "rbad": -1.0}, allow_excretion=True,
                     mip_gap=0.05, time_limit=60)
    assert res.objective is not None
