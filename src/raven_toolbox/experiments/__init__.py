"""Score a model against measured growth rates and exchange fluxes.

The package simulates each measured growth condition in a model and condenses the mismatch with
the measurements into one number, the distance. Use it to compare model versions or as the
objective when fitting model parameters to data.

Two kinds of conditions are scored:

* A **flux condition** (``fluxData.tsv``) limits the carbon-source uptake to its measured rate
  and compares the growth rate and the measured exchange fluxes.
* A **maximum-growth condition** (``maxGrowth.tsv``) leaves the uptake unlimited and compares
  only the maximal growth rate.

Both files have the layout of :func:`load_flux_data`. ``zeroFlux.tsv`` lists reactions that carry
no flux in any condition.

.. important::
   The model objective must be the maximisation of growth. The functions do not set it:
   :func:`simulate_conditions` solves each condition with the model objective and reads the
   growth rate as the flux of the biomass reaction, so in a maximum-growth condition the compared
   value is the maximum of the model objective. :func:`simulate_conditions` warns once per call
   if the objective is not the maximisation of the biomass reaction.

Workflow: :func:`load_experiment_data` reads a folder, :func:`compute_carbon_weights` puts fluxes
on a common carbon scale, :func:`simulate_conditions` simulates one data set, and
:func:`experiment_distance` (or :func:`dataset_rmse` for one data set) returns the distance.
:func:`apply_flux_data_constraints` is independent of the distance: it constrains a model to one
measured condition, for example before flux variability analysis.

Fluxes are in mmol/gDW/h and growth rates in 1/h; negative exchange fluxes are uptake.

Examples
--------
A toy model with one flux condition (glucose uptake 10 mmol/gDW/h, measured growth 0.9 1/h)
while the model grows at 1.0 1/h:

>>> import cobra
>>> import numpy as np
>>> from raven_toolbox.experiments import (
...     ExperimentData, FluxData, compute_carbon_weights, experiment_distance,
...     simulate_conditions)
>>> model = cobra.Model("toy")
>>> model.add_metabolites([cobra.Metabolite("glc_e")])
>>> model.add_reactions([cobra.Reaction("EX_glc"), cobra.Reaction("growth")])
>>> model.reactions.EX_glc.reaction = "glc_e <=> "
>>> model.reactions.growth.reaction = "10 glc_e --> "
>>> model.metabolites.glc_e.formula = "C6H12O6"
>>> model.objective = "growth"
>>> data = FluxData(conds=["glc"], p_tot=np.array([np.nan]), gr_rate=np.array([0.9]),
...                 exch_fluxes=np.array([[-10.0]]), exch_mets=["glc"],
...                 exch_rxn_ids=["EX_glc"])
>>> sims = simulate_conditions(model, data, constrain=True, zero_flux_rxns=[],
...                            bio_rxn_id="growth")
>>> weights = compute_carbon_weights(model, ["EX_glc", "growth"], "growth")
>>> distance, _ = experiment_distance(
...     ExperimentData(flux_data=data), flux_sims=sims, max_growth_sims=None,
...     carbon_weights=weights)
>>> round(distance, 3)
2.899
"""
from .conditions import ConditionSimResult, simulate_conditions
from .constrain import apply_flux_data_constraints
from .data import ExperimentData, load_experiment_data
from .distance import (
    BIOMASS_CARBON_EQUIV,
    INFEASIBLE_PENALTY,
    compute_carbon_weights,
    dataset_rmse,
    experiment_distance,
)
from .flux_data import FluxData, load_flux_data

__all__ = [
    "BIOMASS_CARBON_EQUIV",
    "INFEASIBLE_PENALTY",
    "ConditionSimResult",
    "ExperimentData",
    "FluxData",
    "apply_flux_data_constraints",
    "compute_carbon_weights",
    "dataset_rmse",
    "experiment_distance",
    "load_experiment_data",
    "load_flux_data",
    "simulate_conditions",
]
