"""Solver settings and support for optlang's scipy (HiGHS) solver interface.

:data:`RAVEN_TOLERANCE` and :func:`set_raven_tolerance` give a model the tolerance that RAVEN
gives every LP it solves.

Which function to call:

* :func:`set_raven_tolerance` before a model is solved, for the RAVEN solver tolerance.
* :func:`optimize_model` instead of ``model.optimize()`` when a model may be infeasible and a
  solution with that status is wanted, not an exception.
* :func:`repair_scipy_problem` once on a ``cobra.Model`` on the scipy solver before reactions are
  removed from it, and :func:`switch_to_scipy` to move a model built on another solver to scipy.
* :func:`fast_solver_while_building` around the creation of a large model, and
  :func:`edit_on_fast_solver` around structural edits of a large model, when the default solver is
  scipy.
* :func:`constraint_shadow_prices` for shadow prices on any solver, including scipy.

The scipy interface of optlang differs from the others in ways that matter for large models;
:func:`repair_scipy_problem` switches the problem to a subclass that works around them:

* **Column map after a removal.** optlang renumbers the variable columns incorrectly after a
  removal when the problem has more variables than constraints, so fluxes are read from the wrong
  columns or raise an ``IndexError``, and a later removal deletes the wrong column. Removals are
  applied lazily by the next ``solver.update()``, so the damage appears in code that removed
  nothing itself. The subclass renumbers correctly and rebuilds the matrix if the map was already
  stale. A model class that calls the repair whenever ``model.solver`` is read is covered however
  variables are removed; for a plain ``cobra.Model`` call ``repair_scipy_problem`` before the first
  removal (a later call restores the map, but values read in between were wrong).
* **Dense coefficient matrix.** optlang copies the dense matrix on every structural change, so
  building or editing a genome-scale model takes hours. The subclass keeps the coefficients in
  dictionaries and hands HiGHS a sparse matrix built once per change (``problem.A`` is a
  ``scipy.sparse.csr_matrix``, or a dense array when there are no rows).
* **Solution after a solve.** The scipy interface discards the solution on every objective
  direction assignment, which ``cobra.Model.optimize`` does after the solve, so ``Reaction.flux``
  finds no values; the subclass discards it only when the direction changes. After a solve without
  optimum the interface returns ``None`` values and raises for the objective, so
  ``cobra.Model.optimize`` fails with a ``TypeError``; the subclass returns NaN instead.
* **Duals.** ``Constraint.dual`` raises, but the ``linprog`` result is kept.
  :func:`constraint_shadow_prices` reads the row marginals from it, so shadow prices equal those of
  the glpk and gurobi interfaces. The stored result is used only after it is shown to be an optimal
  primal and dual solution of the problem as it is now.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterable
from itertools import islice
from typing import Any

import cobra
import numpy as np
from scipy import sparse

try:
    from optlang import scipy_interface as _scipy_interface
    from optlang.util import SolverTolerances
    from scipy.optimize import linprog
except ImportError:  # pragma: no cover - optlang always includes the interface
    _scipy_interface = None

__all__ = [
    "RAVEN_TOLERANCE",
    "set_raven_tolerance",
    "repair_scipy_problem",
    "switch_to_scipy",
    "fast_solver_while_building",
    "edit_on_fast_solver",
    "is_scipy_interface",
    "constraint_shadow_prices",
    "optimize_model",
]

#: Feasibility and optimality tolerance that RAVEN's ``solveLP`` gives the
#: solver (``optimizeProb``). cobrapy leaves the solver's own default, 1e-6 for
#: Gurobi and 1e-7 for GLPK, which widens flux ranges and moves tied optima
#: when a model has little room, for instance a fixed growth rate.
RAVEN_TOLERANCE = 1e-9


def set_raven_tolerance(model: cobra.Model, tolerance: float = RAVEN_TOLERANCE) -> cobra.Model:
    """Give a model the solver tolerance that RAVEN uses.

    The model is changed in place and returned. ``cobra.Model.tolerance`` stays on the model when its
    solver is replaced and when it is copied.

    Parameters
    ----------
    model : cobra.Model
        The model.
    tolerance : float, optional
        Feasibility and optimality tolerance. Default :data:`RAVEN_TOLERANCE` (1e-9).

    Returns
    -------
    cobra.Model
        The model that was given.

    Examples
    --------
    >>> import cobra
    >>> from raven_toolbox.utils.solver import set_raven_tolerance
    >>> model = set_raven_tolerance(cobra.Model("toy"))
    >>> model.tolerance
    1e-09
    """
    model.tolerance = tolerance
    return model


def optimize_model(model, **kwargs):
    """Solve a model and return its solution, also when the LP is infeasible.

    On some solvers ``model.optimize()`` raises for an infeasible LP (Gurobi holds no primal values
    after presolve, optlang's scipy interface returns ``None``; cobra reads them and raises
    ``GurobiError`` or ``TypeError``). When the solver status is not optimal this function returns a
    ``Solution`` with that status, a NaN objective and NaN fluxes. With an optimal status every error is
    raised.

    Parameters
    ----------
    model : cobra.Model
        The model.
    **kwargs
        Passed to ``model.optimize()``.

    Returns
    -------
    cobra.Solution
        The solution. Check its ``status`` before using the fluxes.

    Examples
    --------
    >>> import cobra
    >>> from raven_toolbox.utils.solver import optimize_model
    >>> model = cobra.Model("toy")
    >>> model.add_metabolites([cobra.Metabolite("a_c")])
    >>> model.add_reactions([cobra.Reaction("r")])
    >>> model.reactions.r.reaction = "a_c --> "
    >>> model.reactions.r.lower_bound = 1  # a_c is consumed and nothing produces it
    >>> optimize_model(model).status
    'infeasible'
    """
    try:
        return model.optimize(**kwargs)
    except Exception:
        if model.solver.status == "optimal":
            raise
        import pandas as pd
        from cobra.core import Solution

        return Solution(
            objective_value=float("nan"),
            status=model.solver.status,
            fluxes=pd.Series(float("nan"), index=[r.id for r in model.reactions]),
        )


# Relative tolerance of the optimality certificate, scaled by the size of the quantity checked.
_CERTIFICATE_TOL = 1e-6


if _scipy_interface is not None:

    class _SparseProblem(_scipy_interface.Problem):
        """``scipy_interface.Problem`` with a sparse coefficient matrix.

        Renumbers every variable column after a removal and keeps its solution when the
        direction is set to the value it has.
        """

        def __init__(self):
            super().__init__()
            self._init_sparse()

        # Feasibility and optimality tolerance of HiGHS; None keeps its default (1e-7).
        _feasibility = None
        _optimality = None

        def optimize(self, method="highs", verbosity=False, tolerance=1e-9, **kwargs):
            """``linprog`` on the problem, with the tolerances that were set on the model."""
            c = np.array([self.objective.get(name, 0) for name in self._variables], dtype=float)
            if self.direction == "max":
                c *= -1
            options = {"maxiter": 10000, "disp": verbosity}
            bounds = list(self.bounds.values())

            def solve(options):
                return linprog(c, self.A, self.upper_bounds, bounds=bounds, method=method, options=options, **kwargs)

            tight = dict(options)
            if self._feasibility is not None:
                tight["primal_feasibility_tolerance"] = self._feasibility
            if self._optimality is not None:
                tight["dual_feasibility_tolerance"] = self._optimality
            solution = solve(tight)
            # HiGHS reports numerical difficulties (status 4) on some LPs at a tolerance
            # below its default; those are solved again at the default.
            if solution.status == 4 and tight != options:
                solution = solve(options)
            self._solution = solution
            self._status = solution.status
            if self._status == 0:
                self._var_primals = solution.x
                self._slacks = solution.slack
            else:
                self._var_primals = None
                self._slacks = None
            self._f = solution.fun

        @property
        def status(self):
            """The solver status; a solve that ended with numerical difficulties is ``numeric``."""
            if self._status is None:
                raise RuntimeError("Problem has not been optimized yet")
            return _scipy_interface.SCIPY_STATUS.get(self._status, _scipy_interface.interface.NUMERIC)

        def _init_sparse(self):
            # _rows[row name] = {variable name: coefficient}; _cols[variable name] = {row names}
            self._rows = {}
            self._cols = {name: set() for name in self._variables}
            self._matrix = None
            self._A = None

        @property
        def direction(self):
            return self._direction

        @direction.setter
        def direction(self, value):
            if value != self._direction:
                self._direction = value
                self._reset_solution()

        @property
        def A(self):
            """The linear coefficient matrix: sparse, or a dense array without rows."""
            n, m = len(self._variables), len(self._constraints)
            if m == 0:
                return np.zeros((0, n))
            if self._matrix is None:
                data, rows, cols = [], [], []
                columns = self._variables
                for row_name, row_index in self._constraints.items():
                    for name, coefficient in self._rows[row_name].items():
                        data.append(coefficient)
                        rows.append(row_index)
                        cols.append(columns[name])
                self._matrix = sparse.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=float)
            return self._matrix

        def get_var_primal(self, name):
            """The primal value; NaN after a solve that found no optimum, None before any solve."""
            if self._var_primals is None:
                return None if self._status is None else float("nan")
            return self._var_primals[self._get_var_index(name)]

        @property
        def objective_value(self):
            """The optimum; NaN after a solve that found none."""
            if self._f is None and self._status is not None:
                return float("nan")
            return super().objective_value

        def add_variable(self, name):
            if name in self._variables:
                raise ValueError("A variable named " + name + " already exists.")
            self._variables[name] = len(self._variables)
            self.bounds[name] = (0, None)
            self._cols[name] = set()
            self._matrix = None
            self._reset_solution()

        def add_constraint(self, name, coefficients=None, ub=0):
            coefficients = coefficients or {}
            if name in self._constraints:
                raise ValueError("A constraint named " + name + " already exists.")
            self._constraints[name] = len(self._constraints)
            self.upper_bounds = np.append(self.upper_bounds, ub)
            row = {v: c for v, c in coefficients.items() if c != 0 and v in self._variables}
            self._rows[name] = row
            for v in row:
                self._cols[v].add(name)
            self._matrix = None
            self._reset_solution()

        def remove_variable(self, name):
            index = self._get_var_index(name)
            for row_name in self._cols.pop(name):
                del self._rows[row_name][name]
            del self.bounds[name]
            del self._variables[name]
            self._update_variable_indices(index)
            self._matrix = None
            self._reset_solution()

        def remove_constraint(self, name):
            index = self._get_constraint_index(name)
            self.upper_bounds = np.delete(self.upper_bounds, index)
            for v in self._rows.pop(name):
                self._cols[v].discard(name)
            del self._constraints[name]
            self._update_constraint_indices(index)
            self._matrix = None
            self._reset_solution()

        def change_var_name(self, name, value):
            super().change_var_name(name, value)
            if name != value:
                for row_name in self._cols[name]:
                    self._rows[row_name][value] = self._rows[row_name].pop(name)
                self._cols[value] = self._cols.pop(name)

        def change_constraint_name(self, name, value):
            super().change_constraint_name(name, value)
            if name != value:
                self._rows[value] = self._rows.pop(name)
                for v in self._rows[value]:
                    self._cols[v].discard(name)
                    self._cols[v].add(value)

        def _update_variable_indices(self, start=0):
            i = start
            for name in islice(self._variables, start, None):
                self._variables[name] = i
                i += 1


if _scipy_interface is not None:

    class _SparseModel(_scipy_interface.Model):
        """``scipy_interface.Model`` on a `_SparseProblem` that copies and pickles sparsely.

        optlang pickles a model as JSON and restores it by adding the variables and
        constraints to a fresh dense problem, which takes tens of minutes for a genome-scale
        model. This class restores into a `_SparseProblem`, so ``copy.deepcopy``,
        ``cobra.Model.copy`` and the worker processes of a pool get their copy in seconds.
        """

        # optlang finds its interface module (the scipy one) through this attribute.
        __module__ = _scipy_interface.__name__

        def _initialize_problem(self):
            self.problem = _SparseProblem()

        def _remove_constraints(self, constraints):
            # optlang's scipy interface calls the generic removal inside its loop, once per
            # constraint with the whole list, so removing two or more constraints at once
            # (up to 350) raises ``LookupError: Constraint ... not in solver``.
            for constraint in constraints:
                if constraint.lb is not None:
                    self.problem.remove_constraint(constraint.lower_constraint_name)
                if constraint.ub is not None:
                    self.problem.remove_constraint(constraint.upper_constraint_name)
            _scipy_interface.interface.Model._remove_constraints(self, constraints)

        def _initialize_configuration(self):
            super()._initialize_configuration()
            _install_tolerances(self)

        def __reduce__(self):
            problem = self.problem
            return (_restore_sparse_model, (self.to_json(), problem._feasibility, problem._optimality))

    def _restore_sparse_model(state, feasibility, optimality):
        model = _SparseModel()
        model._init_from_json(state)
        model.problem._feasibility = feasibility
        model.problem._optimality = optimality
        return model

    def _install_tolerances(solver):
        """Let ``solver.configuration.tolerances`` (and so ``cobra.Model.tolerance``) set the
        feasibility and optimality tolerance of HiGHS, which optlang's scipy interface ignores."""

        def setter(attribute):
            return lambda value: setattr(solver.problem, attribute, float(value))

        solver.configuration.tolerances = SolverTolerances(
            {
                "feasibility": (lambda: solver.problem._feasibility or 1e-7, setter("_feasibility")),
                "optimality": (lambda: solver.problem._optimality or 1e-7, setter("_optimality")),
            }
        )


def is_scipy_interface(interface) -> bool:
    """Whether something is optlang's scipy interface.

    Parameters
    ----------
    interface : str or module or object
        A solver name, an optlang interface module, or the ``problem`` attribute of a model's
        solver.

    Returns
    -------
    bool
        True for the name ``"scipy"`` (any case) and for the scipy interface module.
    """
    if _scipy_interface is None:
        return False
    if isinstance(interface, str):
        return interface.lower() == "scipy"
    return interface is _scipy_interface


def _is_scipy_problem(problem) -> bool:
    return _scipy_interface is not None and isinstance(problem, _scipy_interface.Problem)


def _map_is_stale(problem) -> bool:
    return list(problem._variables.values()) != list(range(len(problem._variables)))


def _adopt_dense_matrix(problem) -> None:
    """Fill the dictionaries of a `_SparseProblem` from the dense matrix it was swapped from."""
    dense = problem._A
    names = list(problem._variables)
    row_names = list(problem._constraints)
    problem._init_sparse()
    problem._rows = {name: {} for name in row_names}
    for r, c in zip(*np.nonzero(dense), strict=True):
        problem._rows[row_names[r]][names[c]] = float(dense[r, c])
        problem._cols[names[c]].add(row_names[r])


def _rebuild_from_expressions(problem, solver) -> None:
    """Fill the dictionaries of a `_SparseProblem` from the constraint expressions of ``solver``.

    The expressions are the source the matrix is derived from, so this is correct whatever
    happened to the matrix itself.
    """
    problem._init_sparse()
    problem._rows = {name: {} for name in problem._constraints}
    for i, name in enumerate(problem._variables):
        problem._variables[name] = i
    for constraint in solver.constraints:
        coefficients = None
        for row_name, sign in (
            (constraint.upper_constraint_name, 1.0),
            (constraint.lower_constraint_name, -1.0),
        ):
            if row_name not in problem._rows:
                continue
            if coefficients is None:
                coefficients = constraint.coefficient_dict()
            row = {v: sign * c for v, c in coefficients.items() if c != 0 and v in problem._variables}
            problem._rows[row_name] = row
            for v in row:
                problem._cols[v].add(row_name)


def repair_scipy_problem(model) -> bool:
    """Make the scipy problem of a model correct and fast to edit.

    Returns False and does nothing if the solver is not optlang's scipy interface. Otherwise it moves
    the problem to the subclass described in the module documentation, applies pending changes,
    restores the column map and matrix if a removal had already left them stale, and returns True.
    Once switched, a call only checks the type. Call it before removing variables
    (``remove_reactions``, ``Reaction.delete``, ``remove_cons_vars``) from a plain ``cobra.Model`` on the
    scipy solver.

    Parameters
    ----------
    model : cobra.Model
        The model.

    Returns
    -------
    bool
        True if the model is on the scipy solver and its problem is repaired.

    See Also
    --------
    switch_to_scipy
    """
    solver: Any = getattr(model, "_solver", None)
    problem: Any = getattr(solver, "problem", None)
    if not _is_scipy_problem(problem):
        return False
    if isinstance(problem, _SparseProblem):
        return True
    stale = _map_is_stale(problem)
    problem.__class__ = _SparseProblem
    solver.__class__ = _SparseModel
    _install_tolerances(solver)
    if stale:
        problem._init_sparse()
        problem._rows = {name: {} for name in problem._constraints}
    else:
        _adopt_dense_matrix(problem)
    solver.update()
    if stale:
        _rebuild_from_expressions(problem, solver)
    return True


def switch_to_scipy(model) -> None:
    """Move a model to the scipy solver, fast, with the sparse problem of this module.

    ``model.solver = "scipy"`` clones the problem through optlang, which grows a dense matrix row by
    row. This builds the problem the way ``cobra.Model`` does on creation, so a genome-scale model
    switches in seconds. The objective and its direction are kept. A model whose solver holds anything
    besides one forward and one reverse variable per reaction and one constraint per metabolite is
    switched with optlang's clone and then repaired.

    Parameters
    ----------
    model : cobra.Model
        The model, changed in place.

    See Also
    --------
    repair_scipy_problem, fast_solver_while_building
    """
    if repair_scipy_problem(model):
        return
    solver = model.solver
    plain = len(solver.constraints) == len(model.metabolites) and len(solver.variables) == 2 * len(
        model.reactions
    )
    if _scipy_interface is None or not plain:
        from cobra import Model

        Model.solver.fset(model, "scipy")
        repair_scipy_problem(model)
        return
    from cobra.util.solver import linear_reaction_coefficients
    from optlang.symbolics import Zero

    coefficients = linear_reaction_coefficients(model)
    direction = model.objective.direction
    new = _SparseModel()
    new.objective = _scipy_interface.Objective(Zero)
    model._solver = new
    _populate_sparse(model, new)
    model.objective = coefficients
    model.objective.direction = direction
    if getattr(model, "_tolerance", None) is not None:
        try:
            model.tolerance = model._tolerance
        except Exception:  # the scipy interface has no tolerance settings to receive it
            pass


def _populate_sparse(model, solver) -> None:
    """Give ``solver`` one forward and one reverse variable per reaction of ``model`` and one
    constraint per metabolite, as ``cobra.Model._populate_solver`` does.

    ``_populate_solver`` sets the coefficients of a constraint one at a time, and the scipy
    interface removes and adds the rows of a constraint for each call, which takes hours for a
    genome-scale model. Here every constraint is created with its whole expression and added
    once.
    """
    from optlang.symbolics import Zero, add

    problem = solver.interface
    variables = {}
    for reaction in model.reactions:
        variables[reaction.id] = (problem.Variable(reaction.id), problem.Variable(reaction.reverse_id))
    solver.add([v for pair in variables.values() for v in pair])
    terms: dict[str, list[Any]] = {metabolite.id: [] for metabolite in model.metabolites}
    for reaction in model.reactions:
        forward, reverse = variables[reaction.id]
        for metabolite, coefficient in reaction.metabolites.items():
            terms[metabolite.id].append(coefficient * forward)
            terms[metabolite.id].append(-coefficient * reverse)
    solver.add(
        [
            problem.Constraint(add(*expressions) if expressions else Zero, name=name, lb=0, ub=0)
            for name, expressions in terms.items()
        ]
    )
    for reaction in model.reactions:
        reaction.update_variable_bounds()


@contextlib.contextmanager
def edit_on_fast_solver(model):
    """Run structural edits of a model on glpk when the model is on the scipy solver.

    Removing reactions and metabolites or rewriting many coefficients is quick on glpk and slow on the
    scipy interface. Inside the block a scipy model is on glpk; on exit it is converted back with
    :func:`switch_to_scipy`. The block changes nothing for a model on another solver, inside a
    ``with model:`` context (its rollbacks hold the solver objects) or when glpk is unavailable. The
    optlang solver of the model is a different object inside the block and after it.

    Parameters
    ----------
    model : cobra.Model
        The model.

    Yields
    ------
    bool
        True if the model is on glpk inside the block, False if the block changes nothing.

    Examples
    --------
    ::

        with edit_on_fast_solver(model):
            model.remove_reactions(reactions_to_remove)
    """
    from cobra import Model

    solver: Any = getattr(model, "_solver", None)
    if not _is_scipy_problem(getattr(solver, "problem", None)) or getattr(model, "_contexts", None):
        yield False
        return
    try:
        Model.solver.fset(model, "glpk")
    except Exception:
        yield False
        return
    try:
        yield True
    finally:
        switch_to_scipy(model)


@contextlib.contextmanager
def fast_solver_while_building():
    """Create models inside the block on a solver that builds large models quickly.

    When cobra's default solver is scipy, the block runs with glpk as the default and restores it on
    exit. Models created inside are on glpk; convert them with :func:`switch_to_scipy`. With another
    default solver, or without glpk, the block changes nothing.

    Yields
    ------
    bool
        True if the default solver was changed inside the block.

    Examples
    --------
    ::

        with fast_solver_while_building():
            model = cobra.io.read_sbml_model("model.xml")
        switch_to_scipy(model)
    """
    import cobra

    config = cobra.Configuration()
    previous = config.solver
    switched = False
    if getattr(previous, "__name__", str(previous)).endswith("scipy_interface"):
        try:
            config.solver = "glpk"
            switched = True
        except Exception:
            switched = False
    try:
        yield switched
    finally:
        if switched:
            config.solver = previous


def _objective_vector(problem) -> np.ndarray:
    """Cost vector ``linprog`` minimises: the objective, negated for a maximisation."""
    cost = np.array([problem.objective.get(name, 0.0) for name in problem._variables], dtype=float)
    if problem.direction == "max":
        cost = -cost
    return cost


def _certified_result(problem):
    """The stored ``linprog`` result if it is an optimum of ``problem`` as it is now, else None.

    Checks primal feasibility against the current rows and bounds, stationarity of the cost
    vector against the stored marginals, and equality of the primal and dual objective. A
    result that passes is an optimal primal-dual pair of the current problem, whatever solve
    produced it.
    """
    result = getattr(problem, "_solution", None)
    try:
        x = np.asarray(result.x, dtype=float)
        y = np.asarray(result.ineqlin.marginals, dtype=float)
        z_low = np.asarray(result.lower.marginals, dtype=float)
        z_up = np.asarray(result.upper.marginals, dtype=float)
        fun = float(result.fun)
        matrix = sparse.csr_matrix(problem.A, dtype=float)
        rhs = np.asarray(problem.upper_bounds, dtype=float)
        bounds = list(problem.bounds.values())
    except (AttributeError, TypeError, ValueError):
        return None
    n, m = len(problem._variables), len(problem._constraints)
    if matrix.shape != (m, n) or not (x.shape == z_low.shape == z_up.shape == (n,)) or y.shape != (m,):
        return None
    if len(bounds) != n or not np.all(np.isfinite(x)):
        return None
    low = np.array([-np.inf if b[0] is None else b[0] for b in bounds], dtype=float)
    up = np.array([np.inf if b[1] is None else b[1] for b in bounds], dtype=float)
    cost = _objective_vector(problem)
    tol = _CERTIFICATE_TOL

    def close(a, b, scale):
        return abs(a - b) <= tol * (1.0 + scale)

    if m and np.any(matrix @ x - rhs > tol * (1.0 + np.abs(rhs))):
        return None
    if np.any(x < low - tol * (1.0 + np.abs(low))) or np.any(x > up + tol * (1.0 + np.abs(up))):
        return None
    reduced = cost - (matrix.T @ y if m else 0.0)
    if np.max(np.abs(reduced - (z_low + z_up)), initial=0.0) > tol * (1.0 + np.max(np.abs(cost), initial=0.0)):
        return None
    primal = float(cost @ x)
    dual = float(rhs @ y) if m else 0.0
    dual += float(np.sum(low[np.isfinite(low)] * z_low[np.isfinite(low)]))
    dual += float(np.sum(up[np.isfinite(up)] * z_up[np.isfinite(up)]))
    if not (close(primal, fun, abs(fun)) and close(primal, dual, abs(primal))):
        return None
    return result


def constraint_shadow_prices(model, constraint_ids: Iterable[str]) -> np.ndarray:
    """Shadow prices of named constraints on every solver, as one array.

    The shadow price of a constraint is the change of the optimum, in the direction of the model
    objective, per unit rise of its right-hand side; for a metabolite balance it is the optimum gained
    per unit of the metabolite supplied. It equals ``cobra.Metabolite.shadow_price`` on the glpk, gurobi
    and cplex interfaces; for the scipy interface, which raises, it is computed from the HiGHS row
    marginals of the last solve.

    The model must have been solved. On the scipy interface the stored result is used only if it is an
    optimum of the problem as it is now; otherwise the model is solved once more and the function raises
    if that solve is not optimal.

    Parameters
    ----------
    model : cobra.Model
        A model that has been optimized.
    constraint_ids : iterable of str
        Ids of constraints of the model, for example metabolite ids.

    Returns
    -------
    numpy.ndarray
        One shadow price per id, in the order given.

    Raises
    ------
    RuntimeError
        If the scipy problem has no optimal solution to read the marginals from.
    KeyError
        If an id is not a constraint of the model.

    Examples
    --------
    Growth uses 10 mmol glucose per unit, so one more mmol of glucose gives 0.1 more growth:

    >>> import cobra
    >>> from raven_toolbox.utils.solver import constraint_shadow_prices
    >>> model = cobra.Model("toy")
    >>> model.add_metabolites([cobra.Metabolite("glc_e")])
    >>> model.add_reactions([cobra.Reaction("EX_glc"), cobra.Reaction("growth")])
    >>> model.reactions.EX_glc.reaction = "glc_e <=> "
    >>> model.reactions.growth.reaction = "10 glc_e --> "
    >>> model.reactions.EX_glc.lower_bound = -10
    >>> model.objective = "growth"
    >>> _ = model.optimize()
    >>> abs(round(float(constraint_shadow_prices(model, ["glc_e"])[0]), 3))
    0.1
    """
    ids = list(constraint_ids)
    constraints = model.constraints
    repair_scipy_problem(model)
    problem = getattr(model.solver, "problem", None)
    if _scipy_interface is None or not isinstance(problem, _scipy_interface.Problem):
        return np.array([float(constraints[cid].dual) for cid in ids], dtype=float)

    result = _certified_result(problem)
    if result is None:
        model.slim_optimize()
        result = _certified_result(problem)
        if result is None:
            raise RuntimeError(
                "shadow prices need an optimal solution of the scipy problem; the model has none "
                f"(solver status {model.solver.status!r})"
            )
    marginals = np.asarray(result.ineqlin.marginals, dtype=float)
    rows = problem._constraints
    # The rows hold `A x <= ub` and `-A x <= -lb`. The marginal of a row is d(minimised value)/d(rhs
    # of that row), so d/d(ub) is the upper row's marginal and d/d(lb) is minus the lower row's.
    # A maximisation minimises the negated objective.
    sign = -1.0 if problem.direction == "max" else 1.0
    prices = np.zeros(len(ids), dtype=float)
    for k, cid in enumerate(ids):
        constraint = constraints[cid]
        value = 0.0
        if constraint.ub is not None:
            value += marginals[rows[constraint.upper_constraint_name]]
        if constraint.lb is not None:
            value -= marginals[rows[constraint.lower_constraint_name]]
        prices[k] = sign * value
    return prices
