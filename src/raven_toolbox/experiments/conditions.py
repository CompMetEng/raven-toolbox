"""Simulate measured growth conditions in a model.

:func:`simulate_conditions` is the simulation half of the distance between a model and
measurements: it sets up each condition of a data set in the model, solves it and records the
growth rate and the fluxes that are compared with the measurements. The comparison itself is
done by :func:`~raven_toolbox.experiments.dataset_rmse` and
:func:`~raven_toolbox.experiments.experiment_distance`. The two steps are separate so that the
distance can be computed from stored simulations.
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
    """The simulated outcome of one condition, as returned by :func:`simulate_conditions`.

    Attributes
    ----------
    feasible : bool
        False if the model could not be solved to optimality in the condition (infeasible or
        unbounded), or if the condition was skipped. The other fields have no meaning then, and
        :func:`~raven_toolbox.experiments.dataset_rmse` gives the condition a fixed penalty instead
        of reading them.
    growth : float
        Flux of the biomass reaction (1/h). NaN if not feasible.
    exch_fluxes : dict of str to float
        Simulated flux (mmol/gDW/h) of every exchange reaction that was measured in the condition,
        by reaction id. Empty for maximum-growth conditions.
    block_fluxes : dict of str to float
        Simulated flux of every reaction of the ``zero_flux_rxns`` list, except the carbon source
        of the condition, by reaction id. Empty for maximum-growth conditions.
    skipped : bool
        True for a condition with weight 0, which is not simulated. ``feasible`` is False then, and
        the condition does not enter the distance.
    """

    feasible: bool
    growth: float = float("nan")
    exch_fluxes: dict[str, float] = field(default_factory=dict)
    block_fluxes: dict[str, float] = field(default_factory=dict)
    skipped: bool = False


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
    """Simulate every condition of a data set in a model.

    For each condition the model is set to the uptake of the condition's carbon source, solved, and
    the growth rate and the fluxes that the measurements cover are recorded. The model is not
    changed: every change is undone with ``with model:``.

    .. important::
       The model objective must be the maximisation of the growth. The functions do not set the
       objective: each condition is solved with the objective that the model has (in its direction,
       normally maximisation), and the growth rate is the flux of the biomass reaction ``bio_rxn_id``
       in that solution. Set ``model.objective = bio_rxn_id`` before the simulation. With another
       objective the biomass flux is the value in one of the optimal solutions of that objective,
       not the maximal growth rate, and in a maximum-growth condition the distance then compares
       a number that is not the maximal growth rate with the measurement. A data set is either the flux conditions of
    ``ExperimentData.flux_data`` (``constrain=True``) or the maximum-growth conditions of
    ``ExperimentData.max_growth`` (``constrain=False``); see the package documentation.

    The carbon source of a condition is the exchange reaction whose metabolite name
    (``flux_data.exch_mets``) is the name of the condition.

    For each condition:

    1. The lower bound of every exchange reaction that is the carbon source of some condition with a
       weight other than 0 is set to 0, so that the carbon source of one condition cannot take up
       substrate in another.
    2. The lower bound of the carbon source of this condition is set to the measured flux if
       ``constrain`` is True, so that the model can take up at most the measured rate, and to
       ``-1000`` if it is False (unlimited uptake, which gives the best possible growth on the
       carbon source).
    3. If the measured flux of the exchange metabolite named ``oxygen`` is exactly 0 and
       ``make_anaerobic`` is given, the model is made anaerobic. If the protein content
       ``p_tot`` of the condition is given and not 0 and ``change_protein_biomass`` is given,
       the protein content of the model is changed.
    4. The model is solved. If ``constrain`` is True, the simulated flux of every measured
       exchange and of every reaction in ``zero_flux_rxns`` (except the carbon source of the
       condition) is recorded.

    A condition with weight 0 (``flux_data.weight``) is left out: it is not matched to an exchange
    reaction, not simulated, and its result has ``skipped=True`` and ``feasible=False``. Its name
    need not match an exchange metabolite.

    Parameters
    ----------
    model : cobra.Model
        The model, with every other constraint already applied.
    flux_data : FluxData
        The data set to simulate.
    constrain : bool
        True for flux conditions, which limit the uptake to the measured rate and record the
        fluxes; False for maximum-growth conditions, which leave the uptake unlimited and record
        only the growth rate.
    zero_flux_rxns : list of str
        Ids of reactions that carry no flux in any condition (``ExperimentData.zero_flux``). A flux
        through them is counted as an error in the distance.
    bio_rxn_id : str
        Id of the biomass reaction, whose flux in the solution is the growth rate. It should be the
        objective of the model (see above).
    make_anaerobic : callable, optional
        Function ``f(model)`` that changes the model in place to anaerobic growth. How a model is
        made anaerobic depends on the organism, so there is no default; without the function the
        oxygen flux of the data is ignored.
    change_protein_biomass : callable, optional
        Function ``f(model, p_tot)`` that changes the model in place to a protein content of
        ``p_tot`` g/gDW. Without the function ``p_tot`` is ignored.
    optimize : callable, optional
        Function ``f(model)`` that solves the model and returns a solution with ``status``,
        ``objective_value`` and ``fluxes``, for example
        :func:`~raven_toolbox.utils.solver.optimize_model`. Default ``model.optimize()``.

    Returns
    -------
    list of ConditionSimResult
        One result per condition of ``flux_data``, in order.

    Raises
    ------
    ValueError
        If the name of a condition with a weight other than 0 matches no exchange metabolite of
        ``flux_data.exch_mets``, or if an exchange reaction of ``flux_data``, a reaction of
        ``zero_flux_rxns`` or the biomass reaction is not in the model. The message names the
        reactions that are missing, and nothing is solved.

    See Also
    --------
    load_experiment_data : Reads the data sets.
    dataset_rmse : Compares the simulations of one data set with the measurements.
    experiment_distance : Combines the flux and maximum-growth conditions into one distance.

    Examples
    --------
    Growth is 1.0 1/h when 10 mmol/gDW/h glucose can be taken up and 10 mmol glucose give one unit
    of growth:

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


def _default_optimize(model: "cobra.Model") -> "cobra.Solution":
    return model.optimize()


def _check_reactions(
    model: "cobra.Model", flux_data: "FluxData", zero_flux_rxns: list[str], bio_rxn_id: str,
) -> None:
    """Raise ``ValueError`` naming the reactions that the simulation needs and the model lacks."""
    needed = {*flux_data.exch_rxn_ids, *zero_flux_rxns, bio_rxn_id}
    in_model = {rxn.id for rxn in model.reactions}
    missing = sorted(needed - in_model)
    if missing:
        raise ValueError(f"The reactions {', '.join(missing)} are not in the model.")


def _active_conditions(flux_data: "FluxData") -> list[int]:
    """Indices of the conditions with a weight other than 0."""
    if flux_data.weight is None:
        return list(range(len(flux_data.conds)))
    return [i for i in range(len(flux_data.conds)) if flux_data.weight[i] != 0]


def _match_conditions_to_exch(
    flux_data: "FluxData", active: list[int],
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


def _find_col(exch_mets: list[str], name: str) -> Optional[int]:
    try:
        return exch_mets.index(name)
    except ValueError:
        return None
