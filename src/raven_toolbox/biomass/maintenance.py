"""Fit GAM and NGAM to measured substrate uptake and maintenance coefficients.

GAM (the ATP hydrolysed per gram of biomass formed) and NGAM (the ATP hydrolysed per hour
and gram of biomass) are fixed by different data, so they are fitted one after the other:

* NGAM from the maintenance coefficient measured in retentostats, or in chemostats at very
  low growth rates. The substrate uptake that a model needs at zero growth is proportional
  to NGAM, so :func:`ngam_from_maintenance_coefficient` needs one solve.
* GAM from the substrate uptake of chemostats at the measured growth rates, with NGAM held
  at its value (:func:`fit_gam`). The model uptake is computed once on a grid of GAM values
  (:func:`uptake_grid`); the loss, the best GAM and the resampling of studies
  (:func:`bootstrap_gam`) then work on that grid and need no further solves.

The chemostat data come from many studies. The loss weights every study equally
(:func:`study_weights`: a unit counts ``1 / sqrt`` of the number of units of its study, and
units from a less reliable origin count less) and is the Huber loss of the log ratio of model
to measured uptake (:func:`huber_loss`, :func:`log_ratio`), so that one study that
disagrees with the others does not move the result.

:func:`fit_maintenance` runs the steps for one data set. A model that has several regimes,
for example aerobic and anaerobic growth, is fitted once per regime with the model as it is
used in that regime.

The ids of the reactions and the names of the metabolites are given in a
:class:`MaintenanceConfig`. A step that depends on the data, such as setting the protein
content of the biomass of a chemostat, is a function in ``condition_setup``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import cobra
import numpy as np
import pandas as pd

from raven_toolbox.biomass.gam import get_gam, set_gam

__all__ = [
    "DEFAULT_GAMS",
    "HUBER_DELTA",
    "INFEASIBLE_LOG_RATIO",
    "MaintenanceConfig",
    "MaintenanceFit",
    "bootstrap_gam",
    "fit_gam",
    "fit_maintenance",
    "huber_loss",
    "log_ratio",
    "min_uptake",
    "ngam_from_maintenance_coefficient",
    "study_weights",
    "uptake_grid",
]

#: Transition of the Huber loss, as a log ratio: residuals below 0.15 (about 15%) count
#: quadratically, larger ones linearly.
HUBER_DELTA = 0.15
#: Log ratio assigned to a growth rate that the model cannot sustain (about a factor 2).
INFEASIBLE_LOG_RATIO = 0.7
#: GAM values (mmol/gDW) at which the model uptake is computed.
DEFAULT_GAMS = np.arange(25.0, 70.01, 1.25)


@dataclass(frozen=True)
class MaintenanceConfig:
    """Reactions and metabolites that carry GAM and NGAM in a model.

    Attributes
    ----------
    biomass_rxn:
        Biomass pseudoreaction that holds the GAM coefficients.
    ngam_rxn:
        Maintenance reaction whose fixed flux is NGAM.
    growth_rxn:
        Reaction whose flux is the growth rate, fixed to the dilution rate of a chemostat.
        Often the same as ``biomass_rxn``, or a reaction that drains the biomass.
    substrate_exchange:
        Exchange reaction of the limiting substrate. Its uptake is a negative flux.
    cofactor_met_names:
        Names of the metabolites of the GAM term in the biomass pseudoreaction, such as ATP,
        ADP, H2O, H+ and phosphate.
    atp_met_name:
        Name of the ATP metabolite, whose coefficient is the GAM.
    """

    biomass_rxn: str
    ngam_rxn: str
    growth_rxn: str
    substrate_exchange: str
    cofactor_met_names: tuple[str, ...]
    atp_met_name: str = "ATP"


@dataclass
class MaintenanceFit:
    """The values fitted to one data set.

    Attributes
    ----------
    ngam, gam:
        Fitted NGAM and GAM.
    gam_p16, gam_p84:
        16th and 84th percentile of the best GAM when the studies are resampled.
    units:
        Number of data rows.
    studies:
        Number of studies.
    median_error:
        Median relative difference between model and measured uptake at the fitted GAM.
    """

    ngam: float
    gam: float
    gam_p16: float
    gam_p84: float
    units: int
    studies: int
    median_error: float


def huber_loss(residual: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    """Huber loss: ``r**2 / 2`` for ``|r| <= delta``, ``delta * (|r| - delta / 2)`` above."""
    r = np.asarray(residual, dtype=float)
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def log_ratio(
    simulated: np.ndarray, measured: np.ndarray, infeasible: float = INFEASIBLE_LOG_RATIO,
) -> np.ndarray:
    """Natural log of ``simulated / measured``, row-wise broadcast over ``simulated``.

    A value that is not finite, such as the NaN of a growth rate the model cannot sustain,
    gives ``infeasible``.
    """
    with np.errstate(all="ignore"):
        r = np.log(np.asarray(simulated, dtype=float) / np.asarray(measured, dtype=float)[None, :])
    return np.where(np.isfinite(r), r, infeasible)


def study_weights(
    study: Iterable, *, discounted: Iterable[bool] | None = None, discount: float = 0.5,
) -> np.ndarray:
    """Weight of every data row: ``1 / sqrt(rows of its study)``, times ``discount`` where
    ``discounted`` is True (for example values read from a figure)."""
    counts = pd.Series(list(study)).groupby(list(study)).transform("size").to_numpy(dtype=float)
    weight = 1.0 / np.sqrt(counts)
    if discounted is not None:
        weight = weight * np.where(np.asarray(list(discounted), dtype=bool), discount, 1.0)
    return weight


def min_uptake(
    model: cobra.Model,
    growth: float,
    config: MaintenanceConfig,
    *,
    setup: Callable[[cobra.Model], None] | None = None,
) -> float:
    """Smallest substrate uptake (a positive number) at a fixed growth rate.

    The model is changed only inside ``with model:``, so it is as before after the call.
    ``setup`` is called on the model before the solve. The result is NaN when the growth
    rate cannot be reached.
    """
    with model:
        if setup is not None:
            setup(model)
        substrate = model.reactions.get_by_id(config.substrate_exchange)
        substrate.lower_bound = -1000.0
        model.reactions.get_by_id(config.growth_rxn).bounds = (growth, growth)
        model.objective = {substrate: 1.0}  # an uptake is a negative flux
        value = model.slim_optimize(error_value=np.nan)
    return -value


def uptake_grid(
    model: cobra.Model,
    data: pd.DataFrame,
    gams: Sequence[float],
    ngam: float,
    config: MaintenanceConfig,
    *,
    growth_col: str = "growth_rate",
    condition_setup: Callable[[cobra.Model, pd.Series], None] | None = None,
) -> np.ndarray:
    """Model uptake for every GAM in ``gams`` (rows) and data row (columns).

    ``condition_setup(model, row)`` is called on the model for each row, for example to set
    the protein content that was measured in that chemostat.
    """
    out = np.full((len(gams), len(data)), np.nan)
    growth = data[growth_col].to_numpy(dtype=float)
    rows = [data.iloc[j] for j in range(len(data))]
    for i, gam in enumerate(gams):
        with model:
            _set_maintenance(model, float(gam), ngam, config)
            for j, row in enumerate(rows):
                setup = None if condition_setup is None else (lambda m, r=row: condition_setup(m, r))
                out[i, j] = min_uptake(model, growth[j], config, setup=setup)
    return out


def ngam_from_maintenance_coefficient(
    model: cobra.Model,
    ms_substrate: float,
    config: MaintenanceConfig,
    *,
    gam: float | None = None,
) -> float:
    """NGAM at which the model needs ``ms_substrate`` (mmol/gDW/h) at zero growth.

    The uptake at zero growth is proportional to NGAM, so one solve is enough. The ATP yield
    of the substrate comes from the model. ``gam`` is set first when given.
    """
    with model:
        _set_maintenance(model, gam, 1.0, config)
        per_unit = min_uptake(model, 0.0, config)
    return float(ms_substrate / per_unit)


def fit_gam(
    model: cobra.Model,
    data: pd.DataFrame,
    ngam: float,
    config: MaintenanceConfig,
    *,
    gams: Sequence[float] | None = None,
    growth_col: str = "growth_rate",
    uptake_col: str = "uptake",
    weights: np.ndarray | None = None,
    condition_setup: Callable[[cobra.Model, pd.Series], None] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Best GAM for the rows of ``data`` at a given NGAM.

    ``weights`` weight the rows (default: all 1; see :func:`study_weights`). Returns the GAM,
    the GAM grid, and the matrix of log ratios (GAM by row of ``data``), which
    :func:`bootstrap_gam` resamples without new solves.
    """
    gams = np.asarray(DEFAULT_GAMS if gams is None else gams, dtype=float)
    measured = data[uptake_col].to_numpy(dtype=float)
    ratios = log_ratio(
        uptake_grid(model, data, gams, ngam, config, growth_col=growth_col,
                    condition_setup=condition_setup),
        measured,
    )
    w = np.ones(len(data)) if weights is None else np.asarray(weights, dtype=float)
    return _best_gam(gams, ratios, w), gams, ratios


def bootstrap_gam(
    gams: Sequence[float],
    ratios: np.ndarray,
    study: Iterable,
    weights: np.ndarray,
    *,
    n: int = 400,
    seed: int = 0,
    percentiles: tuple[float, float] = (16.0, 84.0),
) -> tuple[float, float]:
    """Percentiles of the best GAM when the studies are resampled with replacement.

    ``study`` gives the study of every column of ``ratios``; a resample draws as many studies
    as there are, and a row counts as often as its study was drawn.
    """
    rng = np.random.default_rng(seed)
    gams = np.asarray(gams, dtype=float)
    labels = np.asarray(list(study))
    names = sorted(set(labels))
    best = []
    for _ in range(n):
        drawn = rng.choice(len(names), len(names))
        count = np.zeros(len(labels))
        for k in drawn:
            count[labels == names[k]] += 1
        w = weights * count
        if w.sum() == 0:
            continue
        best.append(_best_gam(gams, ratios, w))
    lo, hi = np.percentile(best, percentiles)
    return float(lo), float(hi)


def fit_maintenance(
    model: cobra.Model,
    data: pd.DataFrame,
    ms_substrate: float,
    config: MaintenanceConfig,
    *,
    study_col: str = "study",
    discounted: Iterable[bool] | None = None,
    discount: float = 0.5,
    growth_col: str = "growth_rate",
    uptake_col: str = "uptake",
    gams: Sequence[float] | None = None,
    bootstrap: int = 400,
    seed: int = 0,
    condition_setup: Callable[[cobra.Model, pd.Series], None] | None = None,
) -> MaintenanceFit:
    """NGAM from the maintenance coefficient and GAM from the chemostats of ``data``.

    Parameters
    ----------
    model:
        Model as it is used for the growth regime of ``data``. It is not changed.
    data:
        One row per chemostat, with the growth rate (``growth_col``), the measured substrate
        uptake (``uptake_col``) and the study (``study_col``).
    ms_substrate:
        Maintenance coefficient of the substrate, mmol/gDW/h, from retentostats.
    config:
        The reactions and metabolites of GAM and NGAM.
    discounted, discount:
        Rows that count ``discount`` times as much in the weights, for example values read
        from figures.
    gams:
        GAM values of the grid (default :data:`DEFAULT_GAMS`).
    bootstrap, seed:
        Number of resamples of the studies and the seed.
    condition_setup:
        Function ``(model, row)`` that sets the model for the conditions of one chemostat.
    """
    data = data.reset_index(drop=True)
    ngam = ngam_from_maintenance_coefficient(model, ms_substrate, config)
    study = data[study_col].to_numpy()
    weights = study_weights(study, discounted=discounted, discount=discount)
    gam, grid, ratios = fit_gam(
        model, data, ngam, config, gams=gams, growth_col=growth_col,
        uptake_col=uptake_col, weights=weights, condition_setup=condition_setup,
    )
    lo, hi = bootstrap_gam(grid, ratios, study, weights, n=bootstrap, seed=seed)
    best = ratios[int(np.argmin(np.abs(grid - gam)))]
    return MaintenanceFit(
        ngam=ngam, gam=gam, gam_p16=lo, gam_p84=hi,
        units=len(data), studies=len(set(study)),
        median_error=float(np.median(np.abs(np.exp(best) - 1.0))),
    )


def _best_gam(gams: np.ndarray, ratios: np.ndarray, weights: np.ndarray) -> float:
    loss = (huber_loss(ratios) * weights[None, :]).sum(axis=1) / weights.sum()
    return float(gams[int(np.argmin(loss))])


def _set_maintenance(
    model: cobra.Model, gam: float | None, ngam: float, config: MaintenanceConfig,
) -> None:
    """Set GAM (when given) and NGAM in place, keeping the offsets between the GAM terms."""
    if gam is None:
        gam = get_gam(model, biomass_rxn=config.biomass_rxn, atp_met_name=config.atp_met_name)
    set_gam(
        model, gam, biomass_rxn=config.biomass_rxn,
        cofactor_met_names=config.cofactor_met_names,
        ngam_rxn=config.ngam_rxn, ngam_value=ngam,
        keep_offsets_of=config.atp_met_name,
    )
