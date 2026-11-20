"""The distance between a model and measured growth conditions.

The distance is the root mean square of the differences between simulated and measured values
of a condition, averaged over the conditions. The measured values are the growth rate and the
exchange fluxes of the condition, and the differences are put on a common carbon scale so that
a flux of a metabolite with many carbon atoms counts more than one with few: every flux is
multiplied by the number of carbon atoms of the metabolite it exchanges
(:func:`compute_carbon_weights`), which gives Cmmol/gDW/h, and the growth rate is multiplied by
the assumed carbon content of biomass (:data:`BIOMASS_CARBON_EQUIV`). The distance therefore has
the unit Cmmol/gDW/h.

Two kinds of conditions are scored, which differ in what is compared:

* In a **flux condition** the uptake of the carbon source is limited to the measured rate, and
  the growth rate and the measured exchange fluxes are compared. Reactions that should carry no
  flux (the ``zero_flux`` list) are compared with 0.
* In a **maximum-growth condition** the uptake of the carbon source is unlimited, and only the
  growth rate is compared.

:func:`dataset_rmse` scores the conditions of one file, and :func:`experiment_distance`
combines the two files into one value. A condition that the model cannot solve gets a fixed
penalty (:data:`INFEASIBLE_PENALTY`), so that it counts as a poor condition and not as a
missing one. A condition with weight 0 is left out.

The simulations that the functions compare with the data come from
:func:`~raven_toolbox.experiments.simulate_conditions`. The functions of this module read only
numbers, so the distance can be computed without a model.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterable

import numpy as np

from .conditions import ConditionSimResult

if TYPE_CHECKING:
    import cobra

    from .data import ExperimentData
    from .flux_data import FluxData

#: Assumed carbon content of biomass, 41 Cmmol/gDW (a carbon mass fraction of about 49%). It
#: converts a growth rate (1/h) into Cmmol/gDW/h, so that the growth rate and the carbon-weighted
#: exchange fluxes of a condition are on the same scale in the distance.
BIOMASS_CARBON_EQUIV = 41.0

#: Error that a condition gets when the model cannot be solved in it. It is much larger than the
#: error of a condition that is solved, so that a model that cannot grow in a condition has a
#: worse distance than a model that grows with a large error. The value 99 is a convention.
INFEASIBLE_PENALTY = 99.0


def compute_carbon_weights(
    model: "cobra.Model", rxn_ids: Iterable[str], bio_rxn_id: str,
) -> dict[str, float]:
    """Weight of each reaction that puts its flux on a common carbon scale.

    Multiplying an exchange flux by the number of carbon atoms of the metabolite that it exchanges
    gives the flux in Cmmol/gDW/h, so that fluxes of different metabolites can be compared and
    added in one distance. The biomass reaction gets :data:`BIOMASS_CARBON_EQUIV`, which converts
    the growth rate to the same scale.

    Parameters
    ----------
    model : cobra.Model
        Model that provides the metabolite formulas.
    rxn_ids : iterable of str
        Ids of the reactions to compute a weight for: the exchange reactions of the data sets, the
        reactions of the zero-flux list and the biomass reaction.
    bio_rxn_id : str
        Id of the biomass reaction.

    Returns
    -------
    dict of str to float
        Weight per reaction id. The biomass reaction gets 41. Every other reaction gets the number
        of carbon atoms in the formula of its single exchanged metabolite. A reaction with more
        than one metabolite, a metabolite without a formula and a metabolite with no carbon atoms
        (O2, water) get the weight 1, so that their flux mismatches still count in the distance.

    Warns
    -----
    UserWarning
        If no metabolite of the model has a formula; every exchange then gets the weight 1.

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
        carbon = mets[0].elements.get("C") if len(mets) == 1 else None
        value = float(carbon) if carbon is not None else 1.0
        carbon_weights[rxn_id] = value if value != 0.0 else 1.0
    return carbon_weights


def dataset_rmse(
    flux_data: "FluxData",
    sims: list[ConditionSimResult],
    *,
    constrain: bool,
    carbon_weights: dict[str, float],
    bio_rxn_id: str,
    penalty: float = INFEASIBLE_PENALTY,
) -> tuple[float, np.ndarray]:
    """Distance between the simulation of one data set and its measurements.

    The error of a condition is the root mean square of the differences between the carbon-weighted
    simulated and measured values. For a flux condition (``constrain=True``) the values are the
    growth rate, every measured exchange flux, and every reaction of the zero-flux list (measured
    as 0); a zero-flux reaction that is simulated at exactly 0 is left out so that correct
    predictions do not dilute the error. For a maximum-growth condition (``constrain=False``) the
    error is the absolute difference of the carbon-weighted growth rates. A condition that the
    model could not solve gets ``penalty``. The distance of the data set is the mean of the errors
    of its conditions, each multiplied by the weight of the condition if the data has weights (the
    mean is over the conditions, not divided by the sum of the weights). A condition with weight 0
    is left out: its error is NaN and it is not part of the mean.

    Parameters
    ----------
    flux_data : FluxData
        The data set that was simulated: ``ExperimentData.flux_data`` or
        ``ExperimentData.max_growth``.
    sims : list of ConditionSimResult
        The simulation of ``flux_data`` from :func:`~raven_toolbox.experiments.simulate_conditions`,
        one entry per condition. The entries of conditions with weight 0 are not read.
    constrain : bool
        Must be the value that was given to ``simulate_conditions``: True compares growth and
        exchange fluxes (flux conditions), False compares growth only (maximum-growth conditions).
    carbon_weights : dict of str to float
        Weights from :func:`compute_carbon_weights`. A reaction that is not in the dictionary gets
        the weight 1.
    bio_rxn_id : str
        Id of the biomass reaction. It is not read; the growth rate comes from ``sims``.
    penalty : float, optional
        Error of a condition that could not be solved. Default 99
        (:data:`INFEASIBLE_PENALTY`).

    Returns
    -------
    rmse : float
        Weighted mean of the errors of the conditions with a weight other than 0, in
        Cmmol/gDW/h. NaN if every condition has weight 0.
    rmse_list : numpy.ndarray
        Error of each condition, shape ``(n_conds,)``, with the penalty for unsolved conditions and
        before the weighting; NaN for a condition with weight 0.

    Raises
    ------
    ValueError
        If ``sims`` has another length than ``flux_data`` has conditions.

    See Also
    --------
    simulate_conditions : Produces ``sims``.
    compute_carbon_weights : Produces ``carbon_weights``.
    experiment_distance : Combines the distances of the flux and the maximum-growth conditions.

    Examples
    --------
    One maximum-growth condition in which the organism grew at 0.4 1/h and the model at 0.5 1/h; the
    error is 0.1 * 41:

    >>> import numpy as np
    >>> from raven_toolbox.experiments import ConditionSimResult, FluxData, dataset_rmse
    >>> data = FluxData(conds=["glc"], p_tot=np.array([np.nan]), gr_rate=np.array([0.4]),
    ...                 exch_fluxes=np.array([[-1000.0]]), exch_mets=["glc"],
    ...                 exch_rxn_ids=["EX_glc"])
    >>> sims = [ConditionSimResult(feasible=True, growth=0.5)]
    >>> rmse, per_condition = dataset_rmse(data, sims, constrain=False, carbon_weights={},
    ...                                    bio_rxn_id="growth")
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
    experiment_data: "ExperimentData",
    *,
    flux_sims: list[ConditionSimResult] | None,
    max_growth_sims: list[ConditionSimResult] | None,
    carbon_weights: dict[str, float],
    bio_rxn_id: str,
    penalty: float = INFEASIBLE_PENALTY,
    max_growth_weight: float = 1.0,
) -> tuple[float, dict[str, np.ndarray]]:
    """Distance between a model and all measured conditions, as one number.

    Combines the distance of the flux conditions (``ExperimentData.flux_data``) and of the
    maximum-growth conditions (``ExperimentData.max_growth``), each from :func:`dataset_rmse`, as
    ``(d_flux + w * d_max_growth) / (1 + w)``, where ``w`` is ``max_growth_weight``. Use it to
    compare model versions with the same data, or as the objective when parameters of a model are
    adjusted to the data; a smaller distance is a better fit. Fluxes are compared in
    Cmmol/gDW/h (see the module documentation).

    A data set that is not in ``experiment_data`` does not contribute, and neither does a data set
    in which every condition has weight 0.

    Parameters
    ----------
    experiment_data : ExperimentData
        The measured conditions, from :func:`load_experiment_data`.
    flux_sims : list of ConditionSimResult or None
        Simulation of ``experiment_data.flux_data`` from
        :func:`~raven_toolbox.experiments.simulate_conditions` with ``constrain=True``. ``None``
        if ``experiment_data.flux_data`` is ``None``.
    max_growth_sims : list of ConditionSimResult or None
        Simulation of ``experiment_data.max_growth`` with ``constrain=False``. ``None`` if
        ``experiment_data.max_growth`` is ``None``.
    carbon_weights : dict of str to float
        Weights from :func:`compute_carbon_weights`.
    bio_rxn_id : str
        Id of the biomass reaction (not read, see :func:`dataset_rmse`).
    penalty : float, optional
        Error of a condition that could not be solved. Default 99.
    max_growth_weight : float, optional
        Weight of the maximum-growth conditions against the flux conditions in the combination.
        Default 1, which averages the two distances. With 2 the maximum-growth conditions count
        twice as much as the flux conditions.

    Returns
    -------
    rmse : float
        The combined distance in Cmmol/gDW/h. NaN if neither data set is present or no condition has
        a weight other than 0.
    detail : dict of str to numpy.ndarray
        Error of each condition (see ``rmse_list`` of :func:`dataset_rmse`), under the keys
        ``"flux_data"`` and ``"max_growth"`` for the data sets that were scored.

    Raises
    ------
    ValueError
        If a data set is present in ``experiment_data`` but its simulation is ``None``.

    See Also
    --------
    dataset_rmse : The distance of one data set.
    simulate_conditions : Produces the simulations.
    load_experiment_data : Produces ``experiment_data``.

    Examples
    --------
    Data with maximum-growth conditions only: the distance is that of the maximum-growth
    conditions, here a growth rate of 0.5 1/h against a measured 0.4 1/h. The package documentation
    of :mod:`raven_toolbox.experiments` has a complete example, from a model to a distance.

    >>> import numpy as np
    >>> from raven_toolbox.experiments import (
    ...     ConditionSimResult, ExperimentData, FluxData, experiment_distance)
    >>> data = FluxData(conds=["glc"], p_tot=np.array([np.nan]), gr_rate=np.array([0.4]),
    ...                 exch_fluxes=np.array([[-1000.0]]), exch_mets=["glc"],
    ...                 exch_rxn_ids=["EX_glc"])
    >>> distance, detail = experiment_distance(
    ...     ExperimentData(max_growth=data), flux_sims=None,
    ...     max_growth_sims=[ConditionSimResult(feasible=True, growth=0.5)],
    ...     carbon_weights={}, bio_rxn_id="growth")
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
            constrain=True, carbon_weights=carbon_weights, bio_rxn_id=bio_rxn_id,
            penalty=penalty,
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
            constrain=False, carbon_weights=carbon_weights, bio_rxn_id=bio_rxn_id,
            penalty=penalty,
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
