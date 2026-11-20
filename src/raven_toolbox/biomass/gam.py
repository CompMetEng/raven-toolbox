"""Growth-associated maintenance (GAM) and non-growth maintenance (NGAM).

Ports the generic part of yeast-GEM's ``changeGAM.m``. GAM is the ATP hydrolysed per gram
of biomass formed (mmol/gDW), held in the coefficients of the ATP hydrolysis terms of the
biomass pseudoreaction. NGAM is the ATP hydrolysed per hour and gram of biomass
(mmol/gDW/h), held as the fixed flux of the maintenance reaction.
"""
from __future__ import annotations

from collections.abc import Iterable

import cobra


def set_gam(
    model: cobra.Model,
    value: float,
    *,
    biomass_rxn: str,
    cofactor_met_names: Iterable[str],
    ngam_rxn: str | None = None,
    ngam_value: float | None = None,
    keep_offsets_of: str | None = None,
) -> cobra.Model:
    """Set GAM (and optionally NGAM) on a model in place.

    Scales every metabolite participating in the biomass pseudoreaction
    whose ``name`` is in ``cofactor_met_names`` (for example ATP, ADP, H2O,
    H+, phosphate) to ``±value``, keeping the sign of its current
    coefficient.

    With ``keep_offsets_of`` the coefficients keep their differences instead:
    the coefficient of the metabolite of that name becomes ``±value`` and every
    other cofactor changes by the same amount in the direction of its sign. A
    proton coefficient that balances the charge of the other biomass terms then
    keeps its offset from the ATP coefficient.

    If both ``ngam_rxn`` and ``ngam_value`` are given, the NGAM
    reaction's bounds are fixed at ``(ngam_value, ngam_value)``. An empty
    ``ngam_rxn`` counts as not given, and ``ngam_value`` without an ``ngam_rxn`` is an
    error. A given ``ngam_rxn`` must be in the model, also when no ``ngam_value`` is given.

    Parameters
    ----------
    model
        cobra model to mutate.
    value
        New GAM value (mmol ATP / gDW per growth unit).
    biomass_rxn
        Reaction id of the biomass pseudoreaction whose stoichiometry
        carries the GAM coefficients.
    cofactor_met_names
        Iterable of metabolite names (not ids). Every metabolite in
        the biomass pseudoreaction whose name is in this set is
        rescaled. yeast-GEM uses ``{"ATP", "ADP", "H2O", "H+", "phosphate"}``.
    keep_offsets_of
        Name of the metabolite whose coefficient becomes ``±value`` while the
        other cofactors shift by the same amount (see above). It must be one of
        ``cofactor_met_names`` and in the biomass pseudoreaction. ``None``
        sets every cofactor to ``±value``.
    ngam_rxn, ngam_value
        Optional NGAM update. If both are provided, the reaction's
        bounds are set to ``(ngam_value, ngam_value)`` (equality
        constraint).

    Returns
    -------
    cobra.Model
        The modified model (the input object).

    Raises
    ------
    KeyError
        If ``biomass_rxn``, or a given ``ngam_rxn``, is not in the model. Both reactions
        are looked up before anything is changed.
    ValueError
        If ``ngam_value`` is given without an ``ngam_rxn``, or ``keep_offsets_of``
        names no cofactor of the biomass pseudoreaction.
    """
    if ngam_value is not None and not ngam_rxn:
        raise ValueError("ngam_value needs an ngam_rxn: the NGAM reaction to set is not given")
    rxn = model.reactions.get_by_id(biomass_rxn)
    ngam = model.reactions.get_by_id(ngam_rxn) if ngam_rxn else None
    cofactor_set = set(cofactor_met_names)

    targets: dict[cobra.Metabolite, float] = {}
    shift = 0.0
    if keep_offsets_of is not None:
        shift = float(value) - _gam_coefficient(rxn, keep_offsets_of, cofactor_set)
    for met, coef in rxn.metabolites.items():
        if met.name in cofactor_set:
            sign = 1 if coef > 0 else -1
            targets[met] = sign * shift if keep_offsets_of is not None else sign * float(value)
    if targets and keep_offsets_of is not None:
        rxn.add_metabolites(targets, combine=True)  # adds the shift, which a context reverts
    elif targets:
        rxn.add_metabolites(targets, combine=False)  # replaces the coefficient, so it is exactly target

    if ngam is not None and ngam_value is not None:
        ngam.bounds = (float(ngam_value), float(ngam_value))

    return model


def get_gam(model: cobra.Model, *, biomass_rxn: str, atp_met_name: str = "ATP") -> float:
    """GAM of the model: the size of the ATP coefficient of the biomass pseudoreaction.

    Raises
    ------
    KeyError
        If ``biomass_rxn`` is not in the model.
    ValueError
        If the pseudoreaction has no metabolite named ``atp_met_name``.
    """
    rxn = model.reactions.get_by_id(biomass_rxn)
    return _gam_coefficient(rxn, atp_met_name, {atp_met_name})


def _gam_coefficient(rxn: cobra.Reaction, met_name: str, cofactor_names: set[str]) -> float:
    if met_name not in cofactor_names:
        raise ValueError(f"keep_offsets_of {met_name!r} is not one of the cofactor names")
    found = [abs(coef) for met, coef in rxn.metabolites.items() if met.name == met_name]
    if len(found) != 1:
        raise ValueError(
            f"{rxn.id} has {len(found)} metabolites named {met_name!r}; the GAM term needs exactly one"
        )
    return found[0]
