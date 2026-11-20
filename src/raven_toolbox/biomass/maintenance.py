"""Fit GAM and NGAM to measured substrate uptake and maintenance coefficients.

GAM is the ATP hydrolysed per gram of biomass formed (mmol/gDW), and NGAM the ATP hydrolysed per
hour and gram of biomass (mmol/gDW/h); see :mod:`raven_toolbox.biomass.gam`. Different data fix
them, so they are fitted one after the other:

* NGAM from the maintenance coefficient measured in retentostats, or in chemostats at very low
  growth rates: the substrate that the cell takes up per gram of biomass and hour when it does
  not grow. The substrate uptake that a model needs at zero growth is proportional to NGAM, so
  :func:`ngam_from_maintenance_coefficient` needs one solve.
* GAM from the substrate uptake of chemostats at the measured growth rates, with NGAM held at its
  value (:func:`fit_gam`). The model uptake is computed once on a grid of GAM values
  (:func:`uptake_grid`); the loss, the best GAM and the resampling of studies
  (:func:`bootstrap_gam`) then work on that grid and need no further solves.

The chemostat data come from many studies. The loss weights every study equally
(:func:`study_weights`: a row counts ``1 / sqrt`` of the number of rows of its study, and rows
from a less reliable origin, such as values read from a figure, count less) and is the Huber loss
of the log ratio of model to measured uptake (:func:`huber_loss`, :func:`log_ratio`), so that one
study that disagrees with the others does not move the result. The spread of the result is
estimated by resampling the studies.

Data needed:

* The model, with a biomass pseudoreaction that has the GAM term, an NGAM reaction, a reaction
  whose flux is the growth rate, and the exchange reaction of the limiting substrate.
* The maintenance coefficient of the substrate, mmol/gDW/h.
* A table with one row per chemostat: the growth rate (1/h, the dilution rate), the measured
  substrate uptake (mmol/gDW/h, as a positive number) and a label of the study.

:func:`fit_maintenance` runs the steps for one data set. A model with several regimes, for
example aerobic and anaerobic growth, is fitted once per regime, with the model as it is used in
that regime.

The ids of the reactions and the names of the metabolites are given in a
:class:`MaintenanceConfig`. A step that depends on the data, such as setting the protein content
of the biomass of a chemostat, is a function that is passed as ``condition_setup``.
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
    """The reactions and metabolites that carry GAM and NGAM in a model.

    Attributes
    ----------
    biomass_rxn : str
        Biomass pseudoreaction that holds the GAM coefficients.
    ngam_rxn : str
        Maintenance reaction whose fixed flux is NGAM.
    growth_rxn : str
        Reaction whose flux is the growth rate, which is fixed to the dilution rate of a chemostat.
        Often the same as ``biomass_rxn``, or a reaction that drains the biomass.
    substrate_exchange : str
        Exchange reaction of the limiting substrate. Its uptake is a negative flux.
    cofactor_met_names : tuple of str
        Names of the metabolites of the GAM term in the biomass pseudoreaction, such as ATP, ADP,
        H2O, H+ and phosphate.
    atp_met_name : str
        Name of the ATP metabolite, whose coefficient is GAM. Default ``"ATP"``.
    """

    biomass_rxn: str
    ngam_rxn: str
    growth_rxn: str
    substrate_exchange: str
    cofactor_met_names: tuple[str, ...]
    atp_met_name: str = "ATP"


@dataclass
class MaintenanceFit:
    """The result of :func:`fit_maintenance` for one data set.

    Attributes
    ----------
    ngam : float
        Fitted NGAM, mmol/gDW/h.
    gam : float
        Fitted GAM, mmol/gDW.
    gam_p16, gam_p84 : float
        16th and 84th percentile of the best GAM when the studies are resampled, a measure of how
        well the data fix GAM.
    units : int
        Number of data rows (chemostats).
    studies : int
        Number of studies.
    median_error : float
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
    """Huber loss of a residual: quadratic for small residuals, linear for large ones.

    Equal to ``r**2 / 2`` for ``|r| <= delta`` and ``delta * (|r| - delta / 2)`` above, so that a
    single large residual counts less than with a squared loss.

    Parameters
    ----------
    residual : array_like
        Residuals, here log ratios.
    delta : float, optional
        Residual at which the loss turns linear. Default :data:`HUBER_DELTA` (0.15, about 15%).

    Returns
    -------
    numpy.ndarray
        The loss of each residual.

    Examples
    --------
    >>> import numpy as np
    >>> from raven_toolbox.biomass import huber_loss
    >>> huber_loss(np.array([0.1, 0.3])).round(5).tolist()
    [0.005, 0.03375]
    """
    r = np.asarray(residual, dtype=float)
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def log_ratio(
    simulated: np.ndarray, measured: np.ndarray, infeasible: float = INFEASIBLE_LOG_RATIO,
) -> np.ndarray:
    """Natural logarithm of ``simulated / measured``, the residual of the fit.

    A ratio of 2 gives 0.69 and a ratio of 0.5 gives -0.69, so over- and underestimates count the
    same. A value that is not finite, such as the NaN of a growth rate that the model cannot
    sustain, gives ``infeasible``.

    Parameters
    ----------
    simulated : array_like
        Model uptake, one row per GAM of the grid and one column per data row.
    measured : array_like
        Measured uptake of every data row, shape ``(n_rows,)``.
    infeasible : float, optional
        Value for a ratio that is not finite. Default :data:`INFEASIBLE_LOG_RATIO` (0.7, a factor of
        about 2).

    Returns
    -------
    numpy.ndarray
        The log ratios, with the shape of ``simulated``.

    Examples
    --------
    >>> import numpy as np
    >>> from raven_toolbox.biomass import log_ratio
    >>> log_ratio(np.array([[2.0, np.nan]]), np.array([1.0, 1.0])).round(3).tolist()
    [[0.693, 0.7]]
    """
    with np.errstate(all="ignore"):
        r = np.log(np.asarray(simulated, dtype=float) / np.asarray(measured, dtype=float)[None, :])
    return np.where(np.isfinite(r), r, infeasible)


def study_weights(
    study: Iterable, *, discounted: Iterable[bool] | None = None, discount: float = 0.5,
) -> np.ndarray:
    """Weight of every data row, so that every study counts equally.

    A row gets ``1 / sqrt(n)``, where ``n`` is the number of rows of its study, so a study with many
    rows does not outweigh one with few. Rows that are flagged in ``discounted`` count ``discount``
    times as much, for example values that were read from a figure.

    Parameters
    ----------
    study : iterable
        Study label of every row.
    discounted : iterable of bool, optional
        True for the rows that count less.
    discount : float, optional
        Factor for the discounted rows. Default 0.5.

    Returns
    -------
    numpy.ndarray
        One weight per row.

    Examples
    --------
    >>> from raven_toolbox.biomass import study_weights
    >>> study_weights(["a", "a", "b"]).round(4).tolist()
    [0.7071, 0.7071, 1.0]
    >>> study_weights(["a", "a", "b"], discounted=[False, True, False]).round(4).tolist()
    [0.7071, 0.3536, 1.0]
    """
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
    """Smallest substrate uptake at which the model can grow at a given rate.

    The growth rate is fixed, and the uptake of the substrate is minimised. The model is changed only
    inside ``with model:``, so it is as before after the call.

    Parameters
    ----------
    model : cobra.Model
        The model, with GAM and NGAM set.
    growth : float
        Growth rate to fix, 1/h. 0 gives the maintenance uptake.
    config : MaintenanceConfig
        The growth reaction and the substrate exchange reaction.
    setup : callable, optional
        Function ``f(model)`` that is called on the model before the solve, for example to set the
        protein content of a chemostat.

    Returns
    -------
    float
        The uptake as a positive number, mmol/gDW/h. NaN if the growth rate cannot be reached.
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
    """Model uptake for every value of GAM of a grid and every data row.

    Used by :func:`fit_gam`. Computing the uptake once for the whole grid lets the loss, the best
    GAM and the resampling of studies work without further solves.

    Parameters
    ----------
    model : cobra.Model
        The model. It is as before after the call.
    data : pandas.DataFrame
        One row per chemostat.
    gams : sequence of float
        GAM values of the grid, mmol/gDW.
    ngam : float
        NGAM to hold, mmol/gDW/h.
    config : MaintenanceConfig
        The reactions and metabolites of GAM and NGAM.
    growth_col : str, optional
        Column of ``data`` with the growth rate, 1/h. Default ``"growth_rate"``.
    condition_setup : callable, optional
        Function ``f(model, row)`` that sets the model for one chemostat, for example the protein
        content that was measured in it.

    Returns
    -------
    numpy.ndarray
        Uptake in mmol/gDW/h, shape ``(len(gams), len(data))``; NaN where the growth rate cannot be
        reached.
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
    """NGAM at which the model needs a measured substrate uptake at zero growth.

    At zero growth the substrate is spent on maintenance only, so the uptake that the model needs is
    proportional to NGAM and one solve gives the proportionality. The ATP that the substrate yields
    comes from the model.

    Parameters
    ----------
    model : cobra.Model
        The model. It is as before after the call.
    ms_substrate : float
        Measured maintenance coefficient of the substrate, mmol/gDW/h.
    config : MaintenanceConfig
        The reactions and metabolites of GAM and NGAM.
    gam : float, optional
        GAM to set first, mmol/gDW. Default: the GAM of the model.

    Returns
    -------
    float
        NGAM, mmol ATP/gDW/h.
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
    """Best GAM for the chemostats of a data set, at a given NGAM.

    The model uptake is computed on a grid of GAM values, compared with the measured uptake through
    the Huber loss of the log ratio, and the GAM with the smallest weighted loss is returned.

    Parameters
    ----------
    model : cobra.Model
        The model. It is as before after the call.
    data : pandas.DataFrame
        One row per chemostat, with the growth rate and the measured uptake.
    ngam : float
        NGAM to hold, mmol/gDW/h.
    config : MaintenanceConfig
        The reactions and metabolites of GAM and NGAM.
    gams : sequence of float, optional
        GAM values of the grid, mmol/gDW. Default :data:`DEFAULT_GAMS` (25 to 70 in steps of 1.25).
    growth_col, uptake_col : str, optional
        Columns of ``data`` with the growth rate (1/h) and the measured uptake (mmol/gDW/h, a
        positive number). Defaults ``"growth_rate"`` and ``"uptake"``.
    weights : numpy.ndarray, optional
        Weight of every row, for example from :func:`study_weights`. Default: all 1.
    condition_setup : callable, optional
        Function ``f(model, row)`` that sets the model for one chemostat.

    Returns
    -------
    gam : float
        The best GAM, mmol/gDW.
    gams : numpy.ndarray
        The grid.
    ratios : numpy.ndarray
        Log ratios of model to measured uptake, shape ``(len(gams), len(data))``, which
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
    """Spread of the best GAM when the studies are resampled.

    Studies are drawn with replacement, as many as there are, a row counts as often as its study
    was drawn, and the best GAM is found for each resample from the grid of log ratios of
    :func:`fit_gam`. No further solves are needed. A narrow interval means that the data fix GAM.

    Parameters
    ----------
    gams : sequence of float
        The GAM grid.
    ratios : numpy.ndarray
        Log ratios from :func:`fit_gam`.
    study : iterable
        Study label of every column of ``ratios``.
    weights : numpy.ndarray
        Weight of every column, from :func:`study_weights`.
    n : int, optional
        Number of resamples. Default 400.
    seed : int, optional
        Seed of the random generator. Default 0.
    percentiles : tuple of float, optional
        Percentiles to return. Default (16, 84), about one standard deviation.

    Returns
    -------
    tuple of float
        The two percentiles of the best GAM.
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
    """Fit NGAM from a maintenance coefficient and GAM from the chemostats of one data set.

    Runs :func:`ngam_from_maintenance_coefficient`, :func:`study_weights`, :func:`fit_gam` and
    :func:`bootstrap_gam` in turn; see the module documentation for the method and the data.

    Parameters
    ----------
    model : cobra.Model
        The model as it is used for the growth regime of ``data``. It is not changed.
    data : pandas.DataFrame
        One row per chemostat, with the growth rate (``growth_col``, 1/h), the measured substrate
        uptake (``uptake_col``, mmol/gDW/h, a positive number) and the study (``study_col``).
    ms_substrate : float
        Maintenance coefficient of the substrate, mmol/gDW/h, from retentostats.
    config : MaintenanceConfig
        The reactions and metabolites of GAM and NGAM.
    study_col, growth_col, uptake_col : str, optional
        Names of the columns of ``data``. Defaults ``"study"``, ``"growth_rate"``, ``"uptake"``.
    discounted : iterable of bool, optional
        True for the rows that count ``discount`` times as much in the weights, for example values
        that were read from figures.
    discount : float, optional
        Default 0.5.
    gams : sequence of float, optional
        GAM values of the grid. Default :data:`DEFAULT_GAMS`.
    bootstrap : int, optional
        Number of resamples of the studies. Default 400.
    seed : int, optional
        Seed of the resampling. Default 0.
    condition_setup : callable, optional
        Function ``f(model, row)`` that sets the model for the conditions of one chemostat.

    Returns
    -------
    MaintenanceFit
        NGAM, GAM, the spread of GAM, and the sizes of the data.

    See Also
    --------
    set_gam : Applies the result to a model.

    Examples
    --------
    The reaction ids below are those of yeast-GEM; ``chemostats`` is a table with the columns
    ``study``, ``growth_rate`` and ``uptake``::

        config = MaintenanceConfig(
            biomass_rxn="r_4041", ngam_rxn="r_4046", growth_rxn="r_2111",
            substrate_exchange="r_1714",
            cofactor_met_names=("ATP", "ADP", "H2O", "H+", "phosphate"))
        fit = fit_maintenance(model, chemostats, ms_substrate=0.039, config=config)
        set_gam(model, fit.gam, biomass_rxn=config.biomass_rxn,
                cofactor_met_names=config.cofactor_met_names, ngam_rxn=config.ngam_rxn,
                ngam_value=fit.ngam, keep_offsets_of=config.atp_met_name)
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
