"""Minimum-cardinality flux distribution: port of RAVEN's ``getMinNrFluxes``.

Finds a flux distribution satisfying the model's steady state and bounds that
activates as few reactions as possible (optionally weighted per reaction),
via a big-M MILP. This is a different problem from pFBA
(:func:`cobra.flux_analysis.pfba`), which minimises the sum of flux
magnitudes and not the count of nonzero fluxes. cobrapy has no
built-in equivalent.

Two differences from ``getMinNrFluxes.m``:

- **No irreversible splitting.** The default ``'irrev'`` formulation of
  ``getMinNrFluxes.m`` converts a reversible reaction into two irreversible reactions,
  each with its own binary indicator. cobrapy and optlang need no irreversible model,
  so a reversible reaction here gets one indicator ``y`` and two constraints
  (``v <= M*y`` and ``v >= -M*y``) that bound ``|v|`` directly. At a
  cardinality-optimal solution the two formulations are equivalent: the MATLAB MILP
  would never pay for two indicators on one reaction when zeroing the smaller direction
  and keeping the net flux is strictly cheaper.
- **Big-M from the model's own bounds.** ``getMinNrFluxes.m`` first solves the model's
  existing objective to optimality, pins it, and minimises the sum of fluxes subject to
  that pin, to estimate the big-M scale and a MILP warm start. That preliminary solve
  can fail (and then report "no feasible solution") even when the cardinality MILP,
  which does not reference the objective, is feasible. This port estimates big-M from
  the reactions' finite bounds (``5 * the largest finite |bound|``, at least 1000, the
  same lower limit as in MATLAB), so the estimate does not depend on a solve. No MILP
  warm start is set; a warm start only speeds up a solve that reaches the same optimum.

The MILP of ``getMinNrFluxes.m`` does not reference ``model.c``, the objective; only its
preliminary scale-estimating solve does.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cobra
import numpy as np
import pandas as pd

__all__ = ["MinNrFluxesResult", "get_min_nr_fluxes"]

_ACTIVE_THRESHOLD = 0.5


@dataclass
class MinNrFluxesResult:
    """Outcome of a minimum-cardinality flux search.

    Parameters
    ----------
    fluxes:
        Flux for every reaction in the model (empty if infeasible).
    active:
        Ids from ``to_minimize`` whose indicator was on in the solution.
    status:
        Solver status string (for example ``"optimal"``, ``"infeasible"``).
    """

    fluxes: pd.Series
    active: list[str]
    status: str


def get_min_nr_fluxes(
    model: cobra.Model,
    to_minimize: Sequence[str] | None = None,
    *,
    scores: Sequence[float] | None = None,
    big_m: float | None = None,
) -> MinNrFluxesResult:
    """Find a flux distribution activating as few reactions as possible.

    Parameters
    ----------
    model:
        Model to solve on, subject to its own steady-state constraints and
        reaction bounds as given (the model's ``objective`` plays no role;
        a minimal solution among optimal ones requires constraining the objective
        beforehand, for example pinning it near its optimum).
    to_minimize:
        Reaction ids to give a binary "is this active" indicator. Defaults
        to every reaction in the model.
    scores:
        Per-reaction weights, same length and order as ``to_minimize``.
        Negative scores discourage that reaction from being active (the more
        negative, the stronger); entries of 0 or above are replaced by
        the largest (least negative) negative score present, or by 0 if
        ``scores`` has no negative entries, as RAVEN's ``getMinNrFluxes`` does.
        Defaults to an implicit -1 for every reaction (unweighted cardinality minimisation).
    big_m:
        Override the big-M constant. Defaults to an estimate from the
        model's own bounds (see module docstring).

    Returns
    -------
    MinNrFluxesResult
    """
    ids = list(to_minimize) if to_minimize is not None else [r.id for r in model.reactions]
    n = len(ids)

    if scores is None:
        coeffs = np.ones(n)
    else:
        raw = np.asarray(scores, dtype=float)
        if raw.shape[0] != n:
            raise ValueError(
                f"scores must have the same length as to_minimize ({n}), got {raw.shape[0]}."
            )
        negative = raw[raw < 0]
        clamp_to = float(negative.max()) if negative.size else 0.0
        coeffs = -np.where(raw >= 0, clamp_to, raw)

    if big_m is None:
        finite_bounds = [
            abs(b) for r in model.reactions for b in (r.lower_bound, r.upper_bound)
            if math.isfinite(b)
        ]
        big_m = max(5.0 * max(finite_bounds), 1000.0) if finite_bounds else 1000.0

    with model:
        prob = model.problem
        indicators = []
        extra_cons = []
        obj_terms = []
        for rid, coeff in zip(ids, coeffs, strict=True):
            rxn = model.reactions.get_by_id(rid)
            y = prob.Variable(f"_mnf_{rid}", type="binary")
            indicators.append(y)
            obj_terms.append(coeff * y)
            extra_cons.append(
                prob.Constraint(rxn.flux_expression - big_m * y, ub=0, name=f"_mnf_ub_{rid}")
            )
            extra_cons.append(
                prob.Constraint(rxn.flux_expression + big_m * y, lb=0, name=f"_mnf_lb_{rid}")
            )
        model.add_cons_vars([*indicators, *extra_cons])
        model.objective = prob.Objective(sum(obj_terms), direction="min")

        # slim_optimize only reads the objective value, so it is safe to call on an
        # infeasible model; model.optimize() builds a full Solution by reading every
        # reaction's primal value from the solver, which Gurobi refuses to expose for
        # an infeasible problem (raising GurobiError instead of the empty result this
        # branch is supposed to return) -- so the status must be checked first.
        model.slim_optimize()
        status = model.solver.status

        if status != "optimal":
            return MinNrFluxesResult(fluxes=pd.Series(dtype=float), active=[], status=status)

        solution = model.optimize()
        fluxes = solution.fluxes
        active = [rid for rid, y in zip(ids, indicators, strict=True)
                  if (y.primal or 0.0) > _ACTIVE_THRESHOLD]

    return MinNrFluxesResult(fluxes=fluxes, active=active, status=status)
