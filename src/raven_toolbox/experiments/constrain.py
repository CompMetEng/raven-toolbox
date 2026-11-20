"""Set model exchange-flux bounds from one condition of a flux dataset.

In a typical chemostat experiment you measure exchange fluxes
(substrate uptake, byproduct secretion, biomass growth rate)
under one or more growth conditions. :func:`apply_flux_data_constraints` takes a
:class:`~raven_toolbox.experiments.flux_data.FluxData`, selects one condition,
and applies those measurements as bounds on the matching
exchange reactions in the model.

The ``loose_strict_flux`` argument controls how tightly: ``"loose"``
gives the model room to wiggle around the measurement; a numeric
percentage clamps it within that band.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import cobra

    from .flux_data import FluxData


_LooseStrict = Union[Literal["loose"], float]


def apply_flux_data_constraints(
    model: "cobra.Model",
    flux_data: "FluxData",
    *,
    bio_rxn_id: str,
    condition: int | str = 0,
    max_min_growth: Literal["max", "min"] = "max",
    loose_strict_flux: _LooseStrict = "loose",
    c_source: Optional[str] = None,
) -> None:
    """Constrain the model's exchange fluxes from a ``FluxData`` measurement.

    Sets the biomass reaction bounds to the measured growth rate and
    each exchange reaction in ``flux_data.exch_rxn_ids`` to its
    measured flux for the chosen condition. Also zeros out the
    preferred carbon source so the data determines the choice.

    A ``+/-1000`` measured flux means unconstrained, regardless of
    ``loose_strict_flux``:
    ``-1000`` -> ``lb=-1000, ub=0`` (free uptake), ``+1000`` ->
    ``lb=0, ub=1000`` (free excretion).

    Parameters
    ----------
    model
        Model with the exchange reactions referenced by
        ``flux_data`` already in place. Mutated in place.
    flux_data
        Flux data, for example from
        :func:`~raven_toolbox.experiments.flux_data.load_flux_data`.
    bio_rxn_id
        Biomass reaction ID.
    condition
        Either the 0-indexed condition number, or the condition name
        (matched against ``flux_data.conds``).
    max_min_growth
        ``"max"`` sets ``bio_rxn_id`` upper bound to the measured
        growth rate (and lower bound to 0); ``"min"`` sets the lower
        bound (suitable when minimising a resource such as the protein
        pool).
    loose_strict_flux
        ``"loose"`` keeps one bound at 0 and the other at the
        measured value (allows zero flux). A numeric percentage
        ``p`` brackets the measured value with hard bounds:
        ``lb = val * (1 - p/200)``, ``ub = val * (1 + p/200)``,
        swapped as needed to keep ``lb <= ub`` when ``val`` is
        negative. ``p=10`` allows 10% total variance (i.e. +/-5%).
    c_source
        Reaction ID of the preferred carbon-source uptake reaction;
        its bounds are set to ``(0, 0)`` before applying the flux
        data so that the data alone determines the carbon source
        used. ``None`` or ``""`` blocks no reaction. Silently ignored
        if the reaction id is not found in the model.

    Raises
    ------
    ValueError
        If ``max_min_growth`` / ``loose_strict_flux`` is invalid,
        ``condition`` (as string) is not in ``flux_data.conds``, or any
        ``flux_data.exch_rxn_ids`` is missing from the model.
    IndexError
        If ``condition`` (as int) is out of range.
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

    for rxn_id, flux in zip(flux_data.exch_rxn_ids, fluxes):
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
