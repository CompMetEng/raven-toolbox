"""Solver settings that RAVEN applies to every LP it solves."""
from __future__ import annotations

import cobra

#: Feasibility and optimality tolerance that RAVEN's ``solveLP`` gives the
#: solver (``optimizeProb``). cobrapy leaves the solver's own default, 1e-6 for
#: Gurobi and 1e-7 for GLPK, which widens flux ranges and moves tied optima
#: when a model has little room, for instance a fixed growth rate.
RAVEN_TOLERANCE = 1e-9


def set_raven_tolerance(model: cobra.Model, tolerance: float = RAVEN_TOLERANCE) -> cobra.Model:
    """Set the solver tolerance of ``model`` to RAVEN's value.

    ``cobra.Model.tolerance`` stays on the model when its solver is replaced
    and when it is copied. The model is changed in place and returned.
    """
    model.tolerance = tolerance
    return model
