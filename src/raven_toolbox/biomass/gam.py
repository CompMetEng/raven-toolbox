"""Growth-associated maintenance (GAM) and non-growth maintenance (NGAM).

Ports the generic part of yeast-GEM's ``changeGAM.m``.
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
) -> cobra.Model:
    """Set GAM (and optionally NGAM) on a model in place.

    Scales every metabolite participating in the biomass pseudoreaction
    whose ``name`` is in ``cofactor_met_names`` (e.g. ATP, ADP, H2O,
    H+, phosphate) to exactly ``±value``, preserving the sign of its current
    coefficient.

    If both ``ngam_rxn`` and ``ngam_value`` are given, the NGAM
    reaction's bounds are fixed at ``(ngam_value, ngam_value)``. An empty
    ``ngam_rxn`` counts as not given. A given ``ngam_rxn`` must be in the model,
    also when no ``ngam_value`` is given.

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
        Iterable of metabolite *names* (not ids) — every metabolite in
        the biomass pseudoreaction whose name is in this set will be
        rescaled. yeast-GEM uses ``{"ATP", "ADP", "H2O", "H+", "phosphate"}``.
    ngam_rxn, ngam_value
        Optional NGAM update. If both are provided, the reaction's
        bounds are set to ``(ngam_value, ngam_value)`` (equality
        constraint).

    Returns
    -------
    cobra.Model
        The (mutated) model, for chaining.

    Raises
    ------
    KeyError
        If ``biomass_rxn``, or a given ``ngam_rxn``, is not in the model. Both reactions
        are looked up before anything is changed.
    """
    rxn = model.reactions.get_by_id(biomass_rxn)
    ngam = model.reactions.get_by_id(ngam_rxn) if ngam_rxn else None
    cofactor_set = set(cofactor_met_names)

    targets: dict[cobra.Metabolite, float] = {}
    for met, coef in rxn.metabolites.items():
        if met.name in cofactor_set:
            target = (1 if coef > 0 else -1) * float(value)
            targets[met] = target
    if targets:
        rxn.add_metabolites(targets, combine=False)  # replaces the coefficient, so it is exactly target

    if ngam is not None and ngam_value is not None:
        ngam.bounds = (float(ngam_value), float(ngam_value))

    return model
