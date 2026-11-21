"""Constrain a model to the measurements of one growth condition.

:func:`apply_flux_data_constraints` takes one condition of a flux data set (growth rate and
exchange fluxes, as read by :func:`~raven_toolbox.experiments.load_flux_data`) and sets them as
bounds on the biomass reaction and the exchange reactions of a model. The constrained model
can then be used for flux variability analysis, sampling or other analyses of that condition.
It is independent of the distance functions of this package.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import cobra

    from .flux_data import FluxData


_LooseStrict = Literal["loose"] | float


def apply_flux_data_constraints(
    model: cobra.Model,
    flux_data: FluxData,
    *,
    bio_rxn_id: str,
    condition: int | str = 0,
    max_min_growth: Literal["max", "min"] = "max",
    loose_strict_flux: _LooseStrict = "loose",
    c_source: str | None = None,
) -> None:
    """Constrain the growth rate and exchange fluxes of a model to one measured condition.

    The bounds of the biomass reaction are set to the measured growth rate, and the bounds of every
    exchange reaction of ``flux_data`` that was measured in the condition are set to the measured
    flux. An exchange that was not measured (NaN) is left as it is. The model is changed in place.

    How tightly an exchange is constrained is set by ``loose_strict_flux``. For a measured uptake of
    -10 and a measured secretion of 5:

    =======================  ===============  ===============
    ``loose_strict_flux``    uptake -10       secretion 5
    =======================  ===============  ===============
    ``"loose"``              (-10, 0)         (0, 5)
    ``10`` (percent)         (-10.5, -9.5)    (4.75, 5.25)
    =======================  ===============  ===============

    A measured flux of ``-1000`` or ``1000`` means unconstrained, whatever ``loose_strict_flux`` is:
    ``-1000`` gives the bounds (-1000, 0), free uptake, and ``1000`` gives (0, 1000), free
    secretion.

    Parameters
    ----------
    model : cobra.Model
        Model that has the exchange reactions of ``flux_data``. Changed in place.
    flux_data : FluxData
        Measurements, for example from :func:`~raven_toolbox.experiments.load_flux_data`.
    bio_rxn_id : str
        Id of the biomass reaction.
    condition : int or str, optional
        The condition to apply, as its index (counted from 0) or its name in ``flux_data.conds``.
        Default 0.
    max_min_growth : {"max", "min"}, optional
        ``"max"`` sets the bounds of the biomass reaction to (0, growth rate), so that the model
        grows at most as fast as measured. ``"min"`` sets them to (growth rate, 1000), so that the
        model grows at least that fast, which suits the minimisation of a resource such as the
        protein pool. Default ``"max"``.
    loose_strict_flux : "loose" or float, optional
        ``"loose"`` gives one bound at 0 and the other at the measured flux, which allows a flux of
        zero. A number ``p`` between 0 and 100 gives the bounds ``flux * (1 -/+ p / 200)``, a band
        of ``p`` percent in total, which forces a flux through the reaction. Default ``"loose"``.
    c_source : str, optional
        Id of the preferred carbon source uptake reaction. Its bounds are set to (0, 0) before the
        flux data is applied, so that the data alone determines the carbon source. ``None`` or
        ``""`` blocks no reaction, and a reaction that is not in the model is ignored.

    Raises
    ------
    ValueError
        If ``max_min_growth`` or ``loose_strict_flux`` is invalid, if ``condition`` is a name that
        is not in ``flux_data.conds``, or if an exchange reaction of ``flux_data`` is not in the
        model.
    IndexError
        If ``condition`` is an index that is out of range.
    KeyError
        If the biomass reaction is not in the model.

    See Also
    --------
    load_flux_data : Reads the measurements.
    simulate_conditions : Simulates all conditions of a data set, without changing the model.

    Examples
    --------
    >>> import cobra
    >>> import numpy as np
    >>> from raven_toolbox.experiments import FluxData, apply_flux_data_constraints
    >>> model = cobra.Model("toy")
    >>> model.add_metabolites([cobra.Metabolite("glc_e")])
    >>> model.add_reactions([cobra.Reaction("EX_glc"), cobra.Reaction("growth")])
    >>> model.reactions.EX_glc.reaction = "glc_e <=> "
    >>> model.reactions.growth.reaction = "10 glc_e --> "
    >>> data = FluxData(conds=["glc"], p_tot=np.array([np.nan]), gr_rate=np.array([0.9]),
    ...                 exch_fluxes=np.array([[-10.0]]), exch_mets=["glc"],
    ...                 exch_rxn_ids=["EX_glc"])
    >>> apply_flux_data_constraints(model, data, bio_rxn_id="growth", loose_strict_flux=10)
    >>> model.reactions.EX_glc.bounds, model.reactions.growth.bounds
    ((-10.5, -9.5), (0.0, 0.9))
    """
    if max_min_growth not in ("max", "min"):
        raise ValueError(
            f"max_min_growth must be 'max' or 'min', got {max_min_growth!r}"
        )
    if not (
        loose_strict_flux == "loose"
        or (isinstance(loose_strict_flux, (int, float))
            and not isinstance(loose_strict_flux, bool)
            and 0 <= loose_strict_flux <= 100)
    ):
        raise ValueError(
            f"loose_strict_flux must be 'loose' or a number in [0, 100]; "
            f"got {loose_strict_flux!r}"
        )

    if isinstance(condition, str):
        if condition not in flux_data.conds:
            raise ValueError(
                f"Condition {condition!r} not found in flux_data.conds "
                f"({flux_data.conds})"
            )
        cond_idx = flux_data.conds.index(condition)
    else:
        if not 0 <= condition < len(flux_data.conds):
            raise IndexError(
                f"condition index {condition} out of range "
                f"(have {len(flux_data.conds)} condition(s))"
            )
        cond_idx = condition

    fluxes = flux_data.exch_fluxes[cond_idx, :]
    cobra_rxn_ids = {r.id for r in model.reactions}
    missing = [r for r in flux_data.exch_rxn_ids if r not in cobra_rxn_ids]
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"{len(missing)} exchange reaction ID(s) from flux_data are not "
            f"present in the model (examples: {preview})"
        )

    if c_source:
        try:
            c_source_rxn = model.reactions.get_by_id(c_source)
            c_source_rxn.lower_bound = 0.0
            c_source_rxn.upper_bound = 0.0
        except KeyError:
            pass  # c_source not in model; silent

    bio_rxn = model.reactions.get_by_id(bio_rxn_id)
    gr = float(flux_data.gr_rate[cond_idx])
    if max_min_growth == "max":
        bio_rxn.lower_bound = 0.0
        bio_rxn.upper_bound = gr
    else:
        bio_rxn.lower_bound = gr
        bio_rxn.upper_bound = 1000.0

    for rxn_id, flux in zip(flux_data.exch_rxn_ids, fluxes, strict=True):
        if np.isnan(flux):
            continue
        rxn = model.reactions.get_by_id(rxn_id)

        if abs(flux) == 1000.0:
            if flux == -1000.0:
                rxn.lower_bound = -1000.0
                rxn.upper_bound = 0.0
            else:
                rxn.lower_bound = 0.0
                rxn.upper_bound = 1000.0
            continue

        if loose_strict_flux == "loose":
            if flux < 0:
                rxn.lower_bound = float(flux)
                rxn.upper_bound = 0.0
            else:
                rxn.lower_bound = 0.0
                rxn.upper_bound = float(flux)
        else:
            pct = float(loose_strict_flux)
            lo = flux * (1 - pct / 200.0)
            hi = flux * (1 + pct / 200.0)
            rxn.lower_bound = float(min(lo, hi))
            rxn.upper_bound = float(max(lo, hi))
