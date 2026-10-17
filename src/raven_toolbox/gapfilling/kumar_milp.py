"""Objective-based gap-filling via a global growth-floor MILP.

Finds the smallest set of template reactions that makes the objective
(biomass) >= *min_growth*, following the database repair of Kumar et al. 2007
(BMC Bioinformatics 8:212, doi:10.1186/1471-2105-8-212). Reactions already in
the draft keep their bounds: a reaction with the wrong direction is a curation
issue and is not changed here.

The universal template should include any exchange or transport reactions
that are desired as repair candidates; they are treated identically to other
database reactions.

This is computationally more expensive than LP-based gap-filling
(:mod:`~raven_toolbox.gapfilling.fast_lp`) because it involves a MILP solve,
but it optimises for objective feasibility rather than mere connectivity.
It is the counterpart of RAVEN's ``gapFillMILP``; ``cobra.flux_analysis.gapfill``
solves a similar problem.
"""
from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import cobra
from cobra.util import linear_reaction_coefficients

from raven_toolbox.manipulation.transfer import add_reactions_from_model

# Integrality tolerance for the MILP, as RAVEN's optimizeProb sets IntFeasTol.
_INTEGRALITY_TOL = 1e-9


@dataclass
class KumarGapFillResult:
    """Outcome of a Kumar 2007 MILP gap-fill.

    Parameters
    ----------
    added_reactions:
        Template reaction IDs selected by the MILP.
    model:
        Draft model with *added_reactions* incorporated.
    exit_status:
        Solver status string (e.g. ``"optimal"``, ``"infeasible"``), or
        ``"not_restored"`` when the solver reported a solution but the
        selected repairs do not bring the objective to *min_growth* in
        *model*. The selection is still returned for inspection.
    """

    added_reactions: list[str] = field(default_factory=list)
    model: cobra.Model = field(default_factory=cobra.Model)
    exit_status: str = "not_solved"


def _as_models(templates: cobra.Model | Iterable[cobra.Model]) -> list[cobra.Model]:
    return [templates] if isinstance(templates, cobra.Model) else list(templates)


def _merge_templates(
    model: cobra.Model, templates: list[cobra.Model]
) -> tuple[cobra.Model, list[str]]:
    working = model.copy()
    template_ids: list[str] = []
    for t in templates:
        new = [r.id for r in t.reactions if r.id not in working.reactions]
        if new:
            added = add_reactions_from_model(working, t, new, genes=False, note=None)
            template_ids += [r.id for r in added]
    return working, template_ids


def fill_gaps_kumar_milp(
    model: cobra.Model,
    templates: cobra.Model | Iterable[cobra.Model],
    *,
    min_growth: float | None = None,
    big_m: float = 1000.0,
    verbose: bool = True,
) -> KumarGapFillResult:
    """Gap-fill to make the model objective (biomass) >= min_growth.

    Solves a global MILP that selects the fewest template reactions that make
    the model's objective reaction feasible at *min_growth*. The draft
    reactions keep their bounds.

    Parameters
    ----------
    model:
        Draft model to gap-fill. Must have a non-zero objective
        (biomass reaction).
    templates:
        Universal reaction database model(s).
    min_growth:
        Minimum required objective value. If ``None``, runs FBA on the
        merged model and uses 10% of the maximum as the floor.
    big_m:
        Big-M constant for coupling constraints. Must exceed any expected
        flux magnitude in the model.
    verbose:
        Print progress messages.

    Returns
    -------
    KumarGapFillResult
    """
    templates = _as_models(templates)

    # ---- Merge draft + templates ----
    working, template_ids = _merge_templates(model, templates)

    if verbose:
        print(
            f"fill_gaps_kumar_milp: merged model has {len(working.reactions)} reactions "
            f"({len(model.reactions)} draft, {len(template_ids)} template)."
        )

    # ---- Determine min_growth ----
    # Find objective reactions
    obj_coeffs = linear_reaction_coefficients(working)
    if not obj_coeffs:
        raise ValueError(
            "fill_gaps_kumar_milp: model has no objective function. "
            "Set a non-zero objective before calling this function."
        )

    if min_growth is None:
        # FBA on merged model (all template reactions available).
        max_val = working.slim_optimize()
        if not math.isfinite(max_val) or max_val <= 0:
            if verbose:
                print(
                    f"fill_gaps_kumar_milp: FBA on the merged model gives {max_val}. "
                    "Cannot establish a growth floor."
                )
            return KumarGapFillResult(model=model.copy(), exit_status="infeasible")
        min_growth = 0.1 * max_val
        if verbose:
            print(
                f"fill_gaps_kumar_milp: setting min_growth = {min_growth:.4g} "
                f"(10% of FBA max {max_val:.4g})."
            )

    # ---- Build MILP ----
    prob = working.problem
    indicators: list = []
    extra_cons: list = []
    obj_terms: list = []

    # Database reaction candidates
    db_indicators: dict[str, Any] = {}
    finite_bounds = [
        abs(b)
        for r in working.reactions
        for b in (r.lower_bound, r.upper_bound)
        if math.isfinite(b)
    ]
    effective_big_m = max(finite_bounds) if finite_bounds else big_m

    for rid in template_ids:
        rxn = working.reactions.get_by_id(rid)
        y = prob.Variable(f"_gf_db_{rid}", type="binary")
        indicators.append(y)
        db_indicators[rid] = y
        obj_terms.append(y)

        ub = rxn.upper_bound if math.isfinite(rxn.upper_bound) else effective_big_m
        lb = rxn.lower_bound if math.isfinite(rxn.lower_bound) else -effective_big_m

        # v_k <= ub * y (zero when y=0)
        extra_cons.append(
            prob.Constraint(
                rxn.flux_expression - ub * y, ub=0, name=f"_gf_dbub_{rid}"
            )
        )
        # v_k >= lb * y (zero when y=0, because lb*0 = 0 >= 0 forces v >= 0 together with c_ub)
        extra_cons.append(
            prob.Constraint(
                rxn.flux_expression - lb * y, lb=0, name=f"_gf_dblb_{rid}"
            )
        )

    working.add_cons_vars(indicators + extra_cons)

    # Growth floor: set minimum lb on objective reactions
    for rxn, coeff in obj_coeffs.items():
        if rxn.id in working.reactions:
            obj_rxn = working.reactions.get_by_id(rxn.id)
            if coeff > 0:
                obj_rxn.lower_bound = max(obj_rxn.lower_bound, min_growth)
            elif coeff < 0:
                obj_rxn.upper_bound = min(obj_rxn.upper_bound, -min_growth)

    # MILP objective: minimise the number of added template reactions
    if obj_terms:
        working.objective = prob.Objective(sum(obj_terms), direction="min")
    else:
        if verbose:
            print("fill_gaps_kumar_milp: no repair candidates found.")
        return KumarGapFillResult(model=model.copy(), exit_status="infeasible")

    # ---- Solve ----
    if verbose:
        print(
            f"fill_gaps_kumar_milp: solving MILP "
            f"({len(db_indicators)} database candidates)..."
        )

    # A fractional indicator within the solver's default integrality tolerance
    # (about 1e-5 on GLPK and Gurobi) times big_m lets a "closed" candidate
    # carry enough flux to meet a small growth floor. RAVEN's optimizeProb uses
    # the same tight tolerance for MILPs.
    with contextlib.suppress(Exception):
        working.solver.configuration.tolerances.integrality = _INTEGRALITY_TOL

    # Solve through optlang: cobra's optimize() reads primal values even when
    # there are none, which Gurobi refuses for an infeasible MILP.
    status = working.solver.optimize()

    if status != "optimal":
        if verbose:
            print(f"fill_gaps_kumar_milp: solver returned '{status}'.")
        return KumarGapFillResult(model=model.copy(), exit_status=status)

    # ---- Extract solution ----
    threshold = 0.5
    added_rxn_ids = [
        rid for rid, y in db_indicators.items() if (y.primal or 0) > threshold
    ]

    if verbose:
        print(
            f"fill_gaps_kumar_milp: added {len(added_rxn_ids)} template reaction(s)."
        )

    # ---- Build output model ----
    filled = model.copy()

    # Add selected template reactions
    remaining = set(added_rxn_ids)
    for t in templates:
        ids_from_t = [r.id for r in t.reactions if r.id in remaining]
        if ids_from_t:
            add_reactions_from_model(filled, t, ids_from_t, genes=False, note=None)
            remaining -= set(ids_from_t)

    # The rounded selection must meet the floor on its own; a solution that
    # relied on fractional indicators does not.
    if not _meets_floor(filled, obj_coeffs, min_growth):
        if verbose:
            print(
                "fill_gaps_kumar_milp: the selected repairs do not reach "
                f"min_growth {min_growth:.4g} in the filled model."
            )
        status = "not_restored"

    return KumarGapFillResult(
        added_reactions=sorted(added_rxn_ids),
        model=filled,
        exit_status=status,
    )


def _meets_floor(model: cobra.Model, obj_coeffs: dict, min_growth: float) -> bool:
    """Whether ``model`` is feasible with the MILP's growth floor applied.

    The floor is the same bound the MILP puts on each objective reaction,
    relaxed by a relative 1e-6 for the LP feasibility tolerance.
    """
    floor = min_growth * (1 - 1e-6)
    with model:
        for rxn, coeff in obj_coeffs.items():
            if rxn.id not in model.reactions:
                return False
            target = model.reactions.get_by_id(rxn.id)
            if coeff > 0:
                if target.upper_bound < floor:
                    return False
                target.lower_bound = max(target.lower_bound, floor)
            elif coeff < 0:
                if target.lower_bound > -floor:
                    return False
                target.upper_bound = min(target.upper_bound, -floor)
        model.slim_optimize(error_value=math.nan)
        return model.solver.status == "optimal"
