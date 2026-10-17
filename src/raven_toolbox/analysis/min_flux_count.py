"""Minimum-cardinality flux distribution — port of RAVEN's ``getMinNrFluxes``.

Finds a flux distribution satisfying the model's steady state and bounds that
activates as *few* reactions as possible (optionally weighted per reaction),
via a big-M MILP. This is a different, harder problem than pFBA
(:func:`cobra.flux_analysis.pfba`), which minimizes the *sum* of flux
magnitudes rather than the *count* of nonzero fluxes — cobrapy has no
built-in equivalent.

Two disclosed simplifications relative to ``getMinNrFluxes.m``, neither of
which changes the MILP actually being solved (see the source read that
motivated them: MATLAB's own MILP never references ``model.c``, the
objective — only its own *preliminary* scale-estimating solve does):

- **No irreversible-splitting.** MATLAB's COBRA-toolbox solvers require an
  irreversible model, so a reversible reaction becomes two split reactions,
  each with its own binary indicator. cobrapy/optlang have no such
  requirement, so a reversible reaction here gets one indicator ``y`` and
  two constraints (``v <= M*y`` and ``v >= -M*y``), bounding ``|v|`` directly.
  At a cardinality-optimal solution the two are equivalent: MATLAB's MILP
  would never pay for two indicators on one reaction when zeroing the
  smaller direction and keeping the net flux is strictly cheaper.
- **Big-M from the model's own bounds, not a preliminary pFBA solve.**
  ``getMinNrFluxes.m`` first solves the model's *existing* objective to
  optimality, pins it, and minimizes the sum of fluxes subject to that pin —
  purely to estimate a safe big-M scale and a MILP warm start. That
  preliminary solve can fail (and therefore report "no feasible solution")
  even when the actual cardinality MILP below it — which never references
  the objective at all — is perfectly feasible on its own. This port
  estimates big-M directly from the reactions' own finite bounds instead
  (``5 * the largest finite |bound|``, floored at 1000, matching MATLAB's
  own floor), so it cannot fail for a reason unrelated to whether a minimal
  solution actually exists. No MILP warm start is set; a warm start only
  speeds up a solve that would otherwise still reach the same optimum.
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
        Solver status string (e.g. ``"optimal"``, ``"infeasible"``).
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
        constrain it yourself first — e.g. pin it near its optimum — if a
        minimal solution *among optimal ones* is wanted).
    to_minimize:
        Reaction ids to give a binary "is this active" indicator. Defaults
        to every reaction in the model.
    scores:
        Per-reaction weights, same length and order as ``to_minimize``.
        Negative scores discourage that reaction from being active (more
        negative = more discouraged); non-negative entries are clamped up to
        the largest (least negative) negative score present, matching
        MATLAB's "positive scores are not possible" rule — or to 0 if
        ``scores`` has no negative entries at all, as RAVEN's
        ``getMinNrFluxes`` does. Defaults to an implicit -1 for every
        reaction (plain unweighted cardinality minimization).
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
