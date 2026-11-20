"""Simulate the measured conditions in a model.

This module only does the FBA half of the distance: given a model and one
:class:`~raven_toolbox.experiments.flux_data.FluxData` dataset (either
``experiment_data.flux_data`` or ``experiment_data.max_growth``), simulate every
condition and report growth and exchange fluxes.
:mod:`.distance` turns those raw numbers into an RMSE against the
matching experimental values -- kept separate so the RMSE math is
testable with hand-built inputs and no model or FBA at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    import cobra

    from .flux_data import FluxData


@dataclass
class ConditionSimResult:
    """One condition's simulated outcome.

    ``feasible=False`` (LP infeasible/unbounded, or no biomass flux)
    means ``growth``/``exch_fluxes``/``block_fluxes`` are meaningless;
    :mod:`.distance` applies the fixed infeasibility penalty for that
    row instead of reading them.
    """

    feasible: bool
    growth: float = float("nan")
    exch_fluxes: dict[str, float] = field(default_factory=dict)
    block_fluxes: dict[str, float] = field(default_factory=dict)


def simulate_conditions(
    model: "cobra.Model",
    flux_data: "FluxData",
    *,
    constrain: bool,
    zero_flux_rxns: list[str],
    bio_rxn_id: str,
    make_anaerobic: Optional[Callable[["cobra.Model"], None]] = None,
    change_protein_biomass: Optional[Callable[["cobra.Model", float], None]] = None,
    optimize: Optional[Callable[["cobra.Model"], "cobra.Solution"]] = None,
) -> list[ConditionSimResult]:
    """Simulate every condition in one dataset.

    The steps are:

    1. Every exchange reaction that appears as a *condition name*
       anywhere in ``flux_data.conds`` (matched by name against
       ``flux_data.exch_mets``) is blocked (``lower_bound = 0``) up
       front, across the board -- not just the reaction for the row
       being solved -- so one row's carbon source can't leak flux into
       another row's simulation.
    2. For row ``i``, that row's own carbon-source reaction is
       unblocked: fixed at the measured uptake rate if ``constrain``
       is True (``experiment_data.flux_data``), or fully opened to ``-1000``
       if ``constrain`` is False (``experiment_data.max_growth`` -- "what's
       the *best possible* growth on this carbon source", not "growth
       at this measured rate").
    3. Optional per-condition adjustments run if the corresponding
       callable is supplied: anaerobic switch when the oxygen exchange
       column reads exactly 0, protein-content rescaling when
       ``Ptot`` is given and nonzero.
    4. Solve; record growth, plus (``constrain`` datasets only, since
       ``max_growth`` has no exchange measurements to compare against)
       every non-NaN measured exchange's simulated flux and every
       ``zero_flux_rxns`` entry's simulated flux.

    All mutation happens inside ``with model:``, which cobra reverts
    on exit -- only bounds are touched here, so this is safe to call
    repeatedly against one persistent (per-worker) model instance; a
    copying the model per call is avoided (prohibitively
    expensive on a real-scale model).

    Parameters
    ----------
    model
        The model, with every other constraint already applied.
    flux_data
        The dataset to simulate.
    constrain
        True for flux data (fix uptake at the measured rate), False
        for max-growth data (open uptake fully).
    zero_flux_rxns
        Reaction IDs assumed zero-flux in every condition
        (``experiment_data.zero_flux``); the row's own carbon source is
        skipped even if it's also listed here.
    bio_rxn_id
        The biomass reaction ID.
    make_anaerobic, change_protein_biomass
        Optional organism-specific hooks, with no organism-agnostic default.
        Data with no anaerobic or ``Ptot`` rows never needs them, and
        passing ``None`` for either skips that adjustment instead of
        raising.
    optimize
        Function that solves the model and returns a solution with
        ``status``, ``objective_value`` and ``fluxes``. ``None`` calls
        ``model.optimize()``.

    Returns
    -------
    list[ConditionSimResult]
        One entry per row of ``flux_data.conds``, in order.

    Raises
    ------
    ValueError
        If any ``flux_data.conds`` entry can't be matched by name to
        an exchange metabolite in ``flux_data.exch_mets``.
    """
    if optimize is None:
        optimize = _default_optimize
    cond_to_exch_idx = _match_conditions_to_exch(flux_data)
    all_condition_rxn_ids = sorted(
        {flux_data.exch_rxn_ids[j] for j in set(cond_to_exch_idx)}
    )
    o2_col = _find_col(flux_data.exch_mets, "oxygen")
    measured_cols = [
        j for j in range(len(flux_data.exch_rxn_ids))
    ] if constrain else []

    results: list[ConditionSimResult] = []
    for i, exch_idx in enumerate(cond_to_exch_idx):
        carbon_rxn_id = flux_data.exch_rxn_ids[exch_idx]
        with model:
            for rxn_id in all_condition_rxn_ids:
                model.reactions.get_by_id(rxn_id).lower_bound = 0.0

            carbon_rxn = model.reactions.get_by_id(carbon_rxn_id)
            if constrain:
                carbon_rxn.lower_bound = float(flux_data.exch_fluxes[i, exch_idx])
            else:
                carbon_rxn.lower_bound = -1000.0

            if o2_col is not None and flux_data.exch_fluxes[i, o2_col] == 0.0:
                if make_anaerobic is not None:
                    make_anaerobic(model)

            ptot = float(flux_data.p_tot[i])
            if not (np.isnan(ptot) or ptot == 0.0):
                if change_protein_biomass is not None:
                    change_protein_biomass(model, ptot)

            sol = optimize(model)
            if (
                sol.status != "optimal"
                or sol.objective_value is None
                or np.isnan(sol.objective_value)
            ):
                results.append(ConditionSimResult(feasible=False))
                continue

            growth = float(sol.fluxes.get(bio_rxn_id, 0.0))
            exch_fluxes: dict[str, float] = {}
            block_fluxes: dict[str, float] = {}
            if constrain:
                for j in measured_cols:
                    if np.isnan(flux_data.exch_fluxes[i, j]):
                        continue
                    rxn_id = flux_data.exch_rxn_ids[j]
                    exch_fluxes[rxn_id] = float(sol.fluxes.get(rxn_id, 0.0))
                for rxn_id in zero_flux_rxns:
                    if rxn_id == carbon_rxn_id:
                        continue
                    block_fluxes[rxn_id] = float(sol.fluxes.get(rxn_id, 0.0))
            results.append(
                ConditionSimResult(
                    feasible=True,
                    growth=growth,
                    exch_fluxes=exch_fluxes,
                    block_fluxes=block_fluxes,
                )
            )
    return results


def _default_optimize(model: "cobra.Model") -> "cobra.Solution":
    return model.optimize()


def _match_conditions_to_exch(flux_data: "FluxData") -> list[int]:
    """Column index into ``exch_mets``/``exch_rxn_ids`` for each
    condition, matched by name. Raises ``ValueError`` when a condition
    cannot be matched."""
    met_to_idx: dict[str, int] = {}
    for j, met in enumerate(flux_data.exch_mets):
        met_to_idx.setdefault(met, j)
    missing = sorted({c for c in flux_data.conds if c not in met_to_idx})
    if missing:
        raise ValueError(
            'Carbon source(s) "'
            + "; ".join(missing)
            + '" in the provided fluxData or maxGrowth cannot be matched '
            "by name with an exchange reaction."
        )
    return [met_to_idx[c] for c in flux_data.conds]


def _find_col(exch_mets: list[str], name: str) -> Optional[int]:
    try:
        return exch_mets.index(name)
    except ValueError:
        return None
