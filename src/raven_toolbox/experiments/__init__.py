"""Score a model against measured growth rates and exchange fluxes.

Suppose you measured the growth rate and some exchange fluxes (uptake of substrates, secretion
of by-products) of an organism in one or more growth conditions, and want to know how well a
model reproduces them: to compare model versions, or as the objective when parameters of a
model are adjusted to the data. This package simulates each condition in the model and
condenses the mismatch into a single number, the distance.

Two kinds of conditions are scored:

* A **flux condition** limits the uptake of the carbon source to its measured rate. The growth
  rate and the measured exchange fluxes of the model are compared with the measurements. The
  conditions are stored in ``fluxData.tsv``.
* A **maximum-growth condition** leaves the uptake of the carbon source unlimited. Only the
  maximal growth rate of the model is compared with the measured growth rate. The conditions
  are stored in ``maxGrowth.tsv``.

Both files have the layout described in :func:`load_flux_data`. A third file, ``zeroFlux.tsv``,
lists reactions that carry no flux in any condition.

Workflow:

1. :func:`load_experiment_data` reads the three files of a folder into an
   :class:`ExperimentData`.
2. :func:`compute_carbon_weights` gives each reaction the weight that puts all fluxes on the
   same carbon scale.
3. :func:`simulate_conditions` simulates the conditions of one file in the model.
4. :func:`experiment_distance` compares the simulations of both kinds of conditions with the
   measurements and returns the distance. :func:`dataset_rmse` does the same for one file.

:func:`apply_flux_data_constraints` is independent of the distance: it constrains a model to
one measured condition, for example before a flux variability analysis.

Fluxes are in mmol/gDW/h, growth rates in 1/h. A negative exchange flux is an uptake and a
positive one a secretion.

Examples
--------
A model with one substrate and one growth reaction that consumes 10 mmol glucose per unit of
growth, and one measured flux condition in which glucose is taken up at 10 mmol/gDW/h and the
organism grew at 0.9 1/h, while the model grows at 1.0 1/h:

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
...     carbon_weights=weights, bio_rxn_id="growth")
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
