"""Simulate measured growth conditions in a model.

:func:`simulate_conditions` sets up each condition of a data set, solves it and records the growth
rate and the fluxes that are compared with the measurements. The comparison is done by
:func:`~raven_toolbox.experiments.dataset_rmse` and
:func:`~raven_toolbox.experiments.experiment_distance`, separately so that the distance can be
computed from stored simulations.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cobra

    from .flux_data import FluxData


@dataclass
class ConditionSimResult:
    """The simulated outcome of one condition, as returned by :func:`simulate_conditions`.

    Attributes
    ----------
    feasible : bool
        False if the model could not be solved to optimality (infeasible or unbounded) or the condition
        was skipped. The other fields have no meaning then, and
        :func:`~raven_toolbox.experiments.dataset_rmse` gives the condition a penalty.
    growth : float
        Flux of the biomass reaction (1/h). NaN if not feasible.
    exch_fluxes : dict of str to float
        Simulated flux (mmol/gDW/h) by reaction id of every exchange measured in the condition. Empty
        for maximum-growth conditions.
    block_fluxes : dict of str to float
        Simulated flux of every ``zero_flux_rxns`` reaction except the carbon source of the condition.
        Empty for maximum-growth conditions.
    skipped : bool
        True for a condition with weight 0, which is not simulated (``feasible`` is False).
    """

    feasible: bool
    growth: float = float("nan")
    exch_fluxes: dict[str, float] = field(default_factory=dict)
    block_fluxes: dict[str, float] = field(default_factory=dict)
    skipped: bool = False


def simulate_conditions(
    model: cobra.Model,
    flux_data: FluxData,
    *,
    constrain: bool,
    zero_flux_rxns: list[str],
    bio_rxn_id: str,
    make_anaerobic: Callable[[cobra.Model], None] | None = None,
    change_protein_biomass: Callable[[cobra.Model, float], None] | None = None,
    optimize: Callable[[cobra.Model], cobra.Solution] | None = None,
) -> list[ConditionSimResult]:
    """Simulate every condition of a data set in a model.

    For each condition the carbon-source uptake is set, the model is solved, and the growth rate and
    the fluxes covered by the measurements are recorded. The model is not changed (every change is
    undone with ``with model:``).

    .. important::
       The model objective must be the maximisation of growth. The function does not set it: each
       condition is solved with the model objective and the growth rate is the flux of ``bio_rxn_id``
       in that solution. Set ``model.objective = bio_rxn_id`` first. Otherwise the growth in a
       maximum-growth condition is not the maximal growth rate. A warning is issued once per call if
       the objective is not the maximisation of ``bio_rxn_id`` alone, also when ``optimize`` is given,
       since the check reads the model.

    ``flux_data`` is either the flux conditions of ``ExperimentData.flux_data`` (``constrain=True``) or
    the maximum-growth conditions of ``ExperimentData.max_growth`` (``constrain=False``). The carbon
    source of a condition is the exchange reaction whose metabolite name (``flux_data.exch_mets``) is
    the condition name. For each condition:

    1. The lower bound of every carbon-source exchange of a condition with weight other than 0 is
       set to 0.
    2. The lower bound of this condition's carbon source is set to the measured flux
       (``constrain=True``) or to ``-1000`` (``constrain=False``).
    3. If the measured oxygen flux is exactly 0 and ``make_anaerobic`` is given, the model is made
       anaerobic. If ``p_tot`` is given and not 0 and ``change_protein_biomass`` is given, the protein
       content is changed.
    4. The model is solved. With ``constrain=True``, the fluxes of the measured exchanges and of
       ``zero_flux_rxns`` (except this carbon source) are recorded.

    A condition with weight 0 is skipped: it is not matched to an exchange reaction or simulated, and
    its result has ``skipped=True`` and ``feasible=False``.

    Parameters
    ----------
    model : cobra.Model
        Model with all other constraints already applied.
    flux_data : FluxData
        The data set to simulate.
    constrain : bool
        True for flux conditions (uptake limited to the measured rate, fluxes recorded), False for
        maximum-growth conditions (uptake unlimited, only growth recorded).
    zero_flux_rxns : list of str
        Ids of reactions that carry no flux in any condition (``ExperimentData.zero_flux``).
    bio_rxn_id : str
        Id of the growth reaction, whose flux is the growth rate.
    make_anaerobic : callable, optional
        ``f(model)`` that changes the model in place to anaerobic growth. No default, since this
        depends on the organism; without it the oxygen flux of the data is ignored.
    change_protein_biomass : callable, optional
        ``f(model, p_tot)`` that changes the model in place to a protein content of ``p_tot`` g/gDW.
        Without it ``p_tot`` is ignored.
    optimize : callable, optional
        ``f(model)`` that solves the model and returns a solution with ``status``, ``objective_value``
        and ``fluxes``, for example :func:`~raven_toolbox.utils.solver.optimize_model`. Default
        ``model.optimize()``.

    Returns
    -------
    list of ConditionSimResult
        One result per condition, in order.

    Raises
    ------
    ValueError
        If a condition with weight other than 0 matches no exchange metabolite, or if an exchange
        reaction of ``flux_data``, a reaction of ``zero_flux_rxns`` or the biomass reaction is not in
        the model. Nothing is solved then.

    See Also
    --------
    load_experiment_data, dataset_rmse, experiment_distance

    Examples
    --------
    >>> import cobra
    >>> import numpy as np
    >>> from raven_toolbox.experiments import FluxData, simulate_conditions
    >>> model = cobra.Model("toy")
    >>> model.add_metabolites([cobra.Metabolite("glc_e")])
    >>> model.add_reactions([cobra.Reaction("EX_glc"), cobra.Reaction("growth")])
    >>> model.reactions.EX_glc.reaction = "glc_e <=> "
    >>> model.reactions.growth.reaction = "10 glc_e --> "
    >>> model.objective = "growth"
    >>> data = FluxData(conds=["glc"], p_tot=np.array([np.nan]), gr_rate=np.array([0.9]),
    ...                 exch_fluxes=np.array([[-10.0]]), exch_mets=["glc"],
    ...                 exch_rxn_ids=["EX_glc"])
    >>> sim = simulate_conditions(model, data, constrain=True, zero_flux_rxns=[],
    ...                           bio_rxn_id="growth")[0]
    >>> sim.feasible, round(sim.growth, 6), round(sim.exch_fluxes["EX_glc"], 6)
    (True, 1.0, -10.0)
    """
    if optimize is None:
        optimize = _default_optimize
    active = _active_conditions(flux_data)
    cond_to_exch_idx = _match_conditions_to_exch(flux_data, active)
    _check_reactions(model, flux_data, zero_flux_rxns, bio_rxn_id)
    _warn_unless_growth_is_the_objective(model, bio_rxn_id)
    all_condition_rxn_ids = sorted(
        {flux_data.exch_rxn_ids[j] for j in set(cond_to_exch_idx.values())}
    )
    o2_col = _find_col(flux_data.exch_mets, "oxygen")
    measured_cols = [
        j for j in range(len(flux_data.exch_rxn_ids))
    ] if constrain else []

    results: list[ConditionSimResult] = []
    for i in range(len(flux_data.conds)):
        if i not in cond_to_exch_idx:
            results.append(ConditionSimResult(feasible=False, skipped=True))
            continue
        exch_idx = cond_to_exch_idx[i]
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


def _default_optimize(model: cobra.Model) -> cobra.Solution:
    return model.optimize()


def _check_reactions(
    model: cobra.Model, flux_data: FluxData, zero_flux_rxns: list[str], bio_rxn_id: str,
) -> None:
    """Raise ``ValueError`` naming the reactions that the simulation needs and the model lacks."""
    needed = {*flux_data.exch_rxn_ids, *zero_flux_rxns, bio_rxn_id}
    in_model = {rxn.id for rxn in model.reactions}
    missing = sorted(needed - in_model)
    if missing:
        raise ValueError(f"The reactions {', '.join(missing)} are not in the model.")


def _warn_unless_growth_is_the_objective(model: cobra.Model, bio_rxn_id: str) -> None:
    """Warn once if the objective of the model is not the maximisation of ``bio_rxn_id`` alone."""
    from cobra.util.solver import linear_reaction_coefficients

    coefficients = linear_reaction_coefficients(model)
    ids = {rxn.id for rxn in coefficients}
    growth_only = ids == {bio_rxn_id} and coefficients[model.reactions.get_by_id(bio_rxn_id)] > 0
    if model.objective_direction == "max" and growth_only:
        return
    warnings.warn(
        f"The objective of the model is not the maximisation of the growth reaction "
        f"{bio_rxn_id}. The growth rate of a condition is the flux of {bio_rxn_id} in a "
        f"solution of the objective that the model has, and that is not the maximal growth "
        f"rate. Set the objective with model.objective = \"{bio_rxn_id}\".",
        UserWarning,
        stacklevel=3,
    )


def _active_conditions(flux_data: FluxData) -> list[int]:
    """Indices of the conditions with a weight other than 0."""
    if flux_data.weight is None:
        return list(range(len(flux_data.conds)))
    return [i for i in range(len(flux_data.conds)) if flux_data.weight[i] != 0]


def _match_conditions_to_exch(
    flux_data: FluxData, active: list[int],
) -> dict[int, int]:
    """Column index into ``exch_mets``/``exch_rxn_ids`` for each active
    condition, matched by name. Raises ``ValueError`` when a condition
    cannot be matched."""
    met_to_idx: dict[str, int] = {}
    for j, met in enumerate(flux_data.exch_mets):
        met_to_idx.setdefault(met, j)
    missing = sorted({flux_data.conds[i] for i in active
                      if flux_data.conds[i] not in met_to_idx})
    if missing:
        raise ValueError(
            'Carbon source(s) "'
            + "; ".join(missing)
            + '" in the provided fluxData or maxGrowth cannot be matched '
            "by name with an exchange reaction."
        )
    return {i: met_to_idx[flux_data.conds[i]] for i in active}


def _find_col(exch_mets: list[str], name: str) -> int | None:
    try:
        return exch_mets.index(name)
    except ValueError:
        return None
