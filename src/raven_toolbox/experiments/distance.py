"""The distance between a model and measured growth conditions.

The distance of a condition is the root mean square of the differences between simulated and
measured values (growth rate and exchange fluxes), averaged over the conditions. Fluxes are
multiplied by the number of carbon atoms of the exchanged metabolite
(:func:`compute_carbon_weights`) and the growth rate by the carbon content of biomass
(:data:`BIOMASS_CARBON_EQUIV`), so the distance is in Cmmol/gDW/h.

* In a **flux condition** the growth rate, the measured exchange fluxes and the ``zero_flux``
  reactions (compared with 0) are compared.
* In a **maximum-growth condition** only the growth rate is compared. It is the maximal growth
  rate only if the simulated model has growth as objective;
  :func:`~raven_toolbox.experiments.simulate_conditions` warns if it does not.

:func:`dataset_rmse` scores one data set and :func:`experiment_distance` combines both. A condition
that cannot be solved gets :data:`INFEASIBLE_PENALTY`; a condition with weight 0 is left out. The
functions read only numbers, so the distance can be computed from stored simulations without a
model.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING

import cobra
import numpy as np

from .conditions import ConditionSimResult

if TYPE_CHECKING:
    from .data import ExperimentData
    from .flux_data import FluxData

#: Assumed carbon content of biomass (Cmmol/gDW), which converts a growth rate (1/h) to the
#: Cmmol/gDW/h scale of the carbon-weighted fluxes.
BIOMASS_CARBON_EQUIV = 41.0

#: Error of a condition that cannot be solved: much larger than the error of a solved condition, so
#: a model that cannot grow scores worse than one that grows with a large error.
INFEASIBLE_PENALTY = 99.0


def compute_carbon_weights(
    model: cobra.Model, rxn_ids: Iterable[str], bio_rxn_id: str,
) -> dict[str, float]:
    """Weight of each reaction that puts its flux on a common carbon scale.

    Multiplying an exchange flux by the number of carbon atoms of the exchanged metabolite gives
    Cmmol/gDW/h, so that fluxes of different metabolites can be added in one distance. The biomass
    reaction gets :data:`BIOMASS_CARBON_EQUIV`.

    Parameters
    ----------
    model : cobra.Model
        Model that provides the metabolite formulas.
    rxn_ids : iterable of str
        Reactions to weight: the exchange reactions of the data sets, the zero-flux reactions and the
        biomass reaction.
    bio_rxn_id : str
        Id of the biomass reaction.

    Returns
    -------
    dict of str to float
        Weight per reaction id. The biomass reaction gets 41. Every other reaction gets the number of
        carbon atoms in the formula of its single exchanged metabolite. A reaction with several
        metabolites, a metabolite without a readable formula and a metabolite without carbon get 1, so
        that mismatches in their fluxes still count.

        Formulas are read as cobra reads them: element symbols with an optional integer or decimal
        count (``C6H12O6``, ``C2.5H5``); charge signs and ``*`` are ignored (``C6H12O6*H2O``).
        Parentheses (``C6H12O6(H2O)2``, ``(C5H8)n``) and empty formulas are not readable. The MATLAB
        ``computeCarbonWeights`` gives the same weights.

    Warns
    -----
    UserWarning
        If no metabolite has a formula; every exchange then gets weight 1.

    Raises
    ------
    KeyError
        If a reaction id other than the biomass reaction is not in the model.

    See Also
    --------
    dataset_rmse, experiment_distance : Use the weights.

    Examples
    --------
    >>> import cobra
    >>> from raven_toolbox.experiments import compute_carbon_weights
    >>> model = cobra.Model("toy")
    >>> model.add_metabolites([cobra.Metabolite("glc_e"), cobra.Metabolite("o2_e")])
    >>> model.add_reactions([cobra.Reaction("EX_glc"), cobra.Reaction("EX_o2"),
    ...                      cobra.Reaction("growth")])
    >>> model.reactions.EX_glc.reaction = "glc_e <=> "
    >>> model.reactions.EX_o2.reaction = "o2_e <=> "
    >>> model.reactions.growth.reaction = "10 glc_e --> "
    >>> model.metabolites.glc_e.formula = "C6H12O6"
    >>> model.metabolites.o2_e.formula = "O2"
    >>> compute_carbon_weights(model, ["EX_glc", "EX_o2", "growth"], "growth")
    {'EX_glc': 6.0, 'EX_o2': 1.0, 'growth': 41.0}
    """
    carbon_weights: dict[str, float] = {}
    if not any(m.formula for m in model.metabolites):
        warnings.warn(
            "The model has no metabolite formulas; every exchange "
            "reaction gets the carbon weight 1.",
            UserWarning, stacklevel=2,
        )
    for rxn_id in rxn_ids:
        if rxn_id == bio_rxn_id:
            carbon_weights[rxn_id] = BIOMASS_CARBON_EQUIV
            continue
        rxn = model.reactions.get_by_id(rxn_id)
        mets = list(rxn.metabolites)
        carbon = _carbon_atoms(mets[0].formula) if len(mets) == 1 else None
        value = float(carbon) if carbon is not None else 1.0
        carbon_weights[rxn_id] = value if value != 0.0 else 1.0
    return carbon_weights


def _carbon_atoms(formula: str | None) -> float | None:
    """Number of carbon atoms in ``formula``, or None if it has no formula, cannot be read, or has
    no carbon. cobra warns about a formula it cannot read; here that is an expected case."""
    if not formula:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elements = cobra.Metabolite("formula", formula=formula).elements
    if not elements:
        return None
    return elements.get("C")


def dataset_rmse(
    flux_data: FluxData,
    sims: list[ConditionSimResult],
    *,
    constrain: bool,
    carbon_weights: dict[str, float],
    penalty: float = INFEASIBLE_PENALTY,
) -> tuple[float, np.ndarray]:
    """Distance between the simulation of one data set and its measurements.

    The error of a flux condition (``constrain=True``) is the root mean square of the differences
    between carbon-weighted simulated and measured values: the growth rate, every measured exchange
    flux and every zero-flux reaction (measured as 0). Zero-flux reactions simulated at exactly 0 are
    left out, so that correct predictions do not dilute the error. The error of a maximum-growth
    condition (``constrain=False``) is the absolute difference of the carbon-weighted growth rates. A
    condition that could not be solved gets ``penalty``. The distance is the mean of the condition
    errors, each multiplied by the condition weight if the data has weights (the mean is over the
    conditions, not divided by the sum of the weights). A condition with weight 0 has error NaN and is
    not part of the mean.

    Parameters
    ----------
    flux_data : FluxData
        The data set that was simulated.
    sims : list of ConditionSimResult
        Result of :func:`~raven_toolbox.experiments.simulate_conditions` for that data set. Entries of
        conditions with weight 0 are not read.
    constrain : bool
        The value given to ``simulate_conditions``: True compares growth and exchange fluxes, False
        growth only.
    carbon_weights : dict of str to float
        Weights from :func:`compute_carbon_weights`. A reaction that is not in the dictionary gets
        weight 1.
    penalty : float, optional
        Error of a condition that could not be solved. Default 99 (:data:`INFEASIBLE_PENALTY`), large
        enough that a model that cannot grow scores worse than one that grows with a large error.

    Returns
    -------
    rmse : float
        Weighted mean of the errors of the conditions with weight other than 0 (Cmmol/gDW/h). NaN if
        every condition has weight 0.
    rmse_list : numpy.ndarray
        Error of each condition before weighting, shape ``(n_conds,)``, with the penalty for unsolved
        conditions; NaN for weight 0.

    Raises
    ------
    ValueError
        If ``sims`` and ``flux_data`` have different numbers of conditions.

    See Also
    --------
    simulate_conditions, compute_carbon_weights, experiment_distance

    Examples
    --------
    One maximum-growth condition with measured growth 0.4 1/h and simulated growth 0.5 1/h; the error
    is 0.1 * 41:

    >>> import numpy as np
    >>> from raven_toolbox.experiments import ConditionSimResult, FluxData, dataset_rmse
    >>> data = FluxData(conds=["glc"], p_tot=np.array([np.nan]), gr_rate=np.array([0.4]),
    ...                 exch_fluxes=np.array([[-1000.0]]), exch_mets=["glc"],
    ...                 exch_rxn_ids=["EX_glc"])
    >>> sims = [ConditionSimResult(feasible=True, growth=0.5)]
    >>> rmse, per_condition = dataset_rmse(data, sims, constrain=False, carbon_weights={})
    >>> round(rmse, 3), per_condition.round(3).tolist()
    (4.1, [4.1])
    """
    n = len(flux_data.conds)
    if len(sims) != n:
        raise ValueError(
            f"sims has {len(sims)} entries; expected {n} to match "
            f"flux_data.conds."
        )
    rxn_to_col = {r: j for j, r in enumerate(flux_data.exch_rxn_ids)}

    if flux_data.weight is None:
        active = np.ones(n, dtype=bool)
    else:
        active = np.asarray(flux_data.weight) != 0

    rmse_list = np.empty(n, dtype=float)
    for i, sim in enumerate(sims):
        if not active[i]:
            rmse_list[i] = np.nan
            continue
        if not sim.feasible:
            rmse_list[i] = penalty
            continue

        bio_meas = float(flux_data.gr_rate[i]) * BIOMASS_CARBON_EQUIV
        bio_sim = sim.growth * BIOMASS_CARBON_EQUIV

        if not constrain:
            rmse_list[i] = abs(bio_meas - bio_sim)
            continue

        measured = [bio_meas]
        simulated = [bio_sim]
        for rxn_id, sim_flux in sim.exch_fluxes.items():
            w = carbon_weights.get(rxn_id, 1.0)
            measured.append(w * float(flux_data.exch_fluxes[i, rxn_to_col[rxn_id]]))
            simulated.append(w * sim_flux)
        for rxn_id, sim_flux in sim.block_fluxes.items():
            w = carbon_weights.get(rxn_id, 1.0)
            weighted_sim = w * sim_flux
            if weighted_sim == 0.0:
                # Already-correct zero predictions don't dilute the RMSE
                continue
            measured.append(0.0)
            simulated.append(weighted_sim)

        diff = np.asarray(measured, dtype=float) - np.asarray(simulated, dtype=float)
        rmse_list[i] = float(np.sqrt(np.mean(diff ** 2)))

    if not active.any():
        return float("nan"), rmse_list
    weighted = rmse_list
    if flux_data.weight is not None:
        weighted = rmse_list * flux_data.weight
    return float(np.mean(weighted[active])), rmse_list


def experiment_distance(
    experiment_data: ExperimentData,
    *,
    flux_sims: list[ConditionSimResult] | None,
    max_growth_sims: list[ConditionSimResult] | None,
    carbon_weights: dict[str, float],
    penalty: float = INFEASIBLE_PENALTY,
    max_growth_weight: float = 1.0,
) -> tuple[float, dict[str, np.ndarray]]:
    """Distance between a model and all measured conditions, as one number.

    Combines the distance of the flux conditions (``ExperimentData.flux_data``) and of the
    maximum-growth conditions (``ExperimentData.max_growth``), each from :func:`dataset_rmse`, as
    ``(d_flux + w * d_max_growth) / (1 + w)`` with ``w = max_growth_weight``. A smaller distance is a
    better fit. Use it to compare model versions on the same data or as the objective when fitting
    parameters. Fluxes are compared in Cmmol/gDW/h (:func:`compute_carbon_weights`). A data set that
    is absent, or in which every condition has weight 0, does not contribute.

    The maximum-growth comparison is the maximal growth rate only if the simulated model has growth
    as objective; :func:`~raven_toolbox.experiments.simulate_conditions` warns if it does not.

    Parameters
    ----------
    experiment_data : ExperimentData
        Measured conditions, from :func:`load_experiment_data`.
    flux_sims : list of ConditionSimResult or None
        :func:`~raven_toolbox.experiments.simulate_conditions` result for ``flux_data`` with
        ``constrain=True``; ``None`` if ``flux_data`` is ``None``.
    max_growth_sims : list of ConditionSimResult or None
        Result for ``max_growth`` with ``constrain=False``; ``None`` if ``max_growth`` is ``None``.
    carbon_weights : dict of str to float
        Weights from :func:`compute_carbon_weights`.
    penalty : float, optional
        Error of a condition that could not be solved. Default 99.
    max_growth_weight : float, optional
        Weight of the maximum-growth distance against the flux distance. Default 1 (the average).

    Returns
    -------
    rmse : float
        Combined distance (Cmmol/gDW/h). NaN if no condition has weight other than 0.
    detail : dict of str to numpy.ndarray
        Error of each condition (``rmse_list`` of :func:`dataset_rmse`) under ``"flux_data"`` and
        ``"max_growth"`` for the data sets that were scored.

    Raises
    ------
    ValueError
        If a data set is present but its simulation is ``None``.

    See Also
    --------
    dataset_rmse, simulate_conditions, load_experiment_data

    Examples
    --------
    Maximum-growth conditions only: measured growth 0.4 1/h against simulated 0.5 1/h. See
    :mod:`raven_toolbox.experiments` for a complete example from a model to a distance.

    >>> import numpy as np
    >>> from raven_toolbox.experiments import (
    ...     ConditionSimResult, ExperimentData, FluxData, experiment_distance)
    >>> data = FluxData(conds=["glc"], p_tot=np.array([np.nan]), gr_rate=np.array([0.4]),
    ...                 exch_fluxes=np.array([[-1000.0]]), exch_mets=["glc"],
    ...                 exch_rxn_ids=["EX_glc"])
    >>> distance, detail = experiment_distance(
    ...     ExperimentData(max_growth=data), flux_sims=None,
    ...     max_growth_sims=[ConditionSimResult(feasible=True, growth=0.5)],
    ...     carbon_weights={})
    >>> round(distance, 3)
    4.1
    """
    parts: list[float] = []
    weights: list[float] = []
    detail: dict[str, np.ndarray] = {}

    if experiment_data.flux_data is not None:
        if flux_sims is None:
            raise ValueError("experiment_data.flux_data is set but flux_sims is None.")
        rmse, per_cond = dataset_rmse(
            experiment_data.flux_data, flux_sims,
            constrain=True, carbon_weights=carbon_weights, penalty=penalty,
        )
        if not np.isnan(rmse):
            parts.append(rmse)
            weights.append(1.0)
        detail["flux_data"] = per_cond

    if experiment_data.max_growth is not None:
        if max_growth_sims is None:
            raise ValueError("experiment_data.max_growth is set but max_growth_sims is None.")
        rmse, per_cond = dataset_rmse(
            experiment_data.max_growth, max_growth_sims,
            constrain=False, carbon_weights=carbon_weights, penalty=penalty,
        )
        if not np.isnan(rmse):
            parts.append(rmse)
            weights.append(float(max_growth_weight))
        detail["max_growth"] = per_cond

    if parts:
        rmse = float(np.dot(parts, weights) / np.sum(weights))
    else:
        rmse = float("nan")
    return rmse, detail
