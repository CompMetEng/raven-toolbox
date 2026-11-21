"""Growth-associated (GAM) and non-growth-associated (NGAM) ATP maintenance of a model.

GAM is held in the coefficients of the ATP-hydrolysis cofactors of the biomass reaction; NGAM is
the fixed flux of a maintenance reaction. :func:`set_gam` sets them, :func:`get_gam` reads GAM, and
:func:`~raven_toolbox.biomass.maintenance.fit_maintenance` derives them from measurements.
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
    """Set GAM, and optionally NGAM, in a model.

    For every metabolite of the biomass reaction whose name is in ``cofactor_met_names``, the
    coefficient becomes ``value``, keeping its sign. With ``keep_offsets_of``, the coefficient of the
    metabolite with that name (for example ATP) becomes ``value`` and every other cofactor shifts by
    the same amount in the direction of its sign, so that offsets such as a proton coefficient that
    balances charge are kept. For ATP, water, ADP and phosphate at 60 and a proton at 57, setting GAM
    to 45 gives 45 and a proton at 42, whereas without ``keep_offsets_of`` the proton becomes 45.

    If both ``ngam_rxn`` and ``ngam_value`` are given, both bounds of the NGAM reaction are set to
    ``ngam_value``. An empty ``ngam_rxn`` counts as not given.

    Parameters
    ----------
    model : cobra.Model
        Model to change in place.
    value : float
        New GAM (mmol ATP/gDW).
    biomass_rxn : str
        Id of the biomass reaction.
    cofactor_met_names : iterable of str
        Names (not ids) of the GAM cofactors, for example ``["ATP", "ADP", "H2O", "H+", "phosphate"]``.
    ngam_rxn : str, optional
        Id of the NGAM reaction.
    ngam_value : float, optional
        NGAM flux (mmol ATP/gDW/h). Needs ``ngam_rxn``.
    keep_offsets_of : str, optional
        Name of the metabolite that gets ``value`` while the other cofactors shift with it. Must be one
        of ``cofactor_met_names`` and occur once in the biomass reaction. Default ``None`` sets every
        cofactor to ``value``.

    Returns
    -------
    cobra.Model
        The model that was given, changed.

    Raises
    ------
    KeyError
        If ``biomass_rxn`` or a given ``ngam_rxn`` is not in the model. Both are checked before the
        model is changed.
    ValueError
        If ``ngam_value`` is given without ``ngam_rxn``, or if ``keep_offsets_of`` is not one of the
        cofactor names or does not name exactly one metabolite of the biomass reaction.

    See Also
    --------
    get_gam, raven_toolbox.biomass.maintenance.fit_maintenance

    Examples
    --------
    >>> import cobra
    >>> from raven_toolbox.biomass import get_gam, set_gam
    >>> names = {"atp": "ATP", "h2o": "H2O", "adp": "ADP", "pi": "phosphate", "h": "H+"}
    >>> mets = {k: cobra.Metabolite(k, name=v) for k, v in names.items()}
    >>> growth = cobra.Reaction("growth")
    >>> growth.add_metabolites({mets["atp"]: -60, mets["h2o"]: -60, mets["adp"]: 60,
    ...                         mets["pi"]: 60, mets["h"]: 57})
    >>> model = cobra.Model("toy")
    >>> model.add_reactions([growth])
    >>> cofactors = list(names.values())
    >>> _ = set_gam(model, 45, biomass_rxn="growth", cofactor_met_names=cofactors,
    ...             keep_offsets_of="ATP")
    >>> {m.id: c for m, c in growth.metabolites.items()}
    {'atp': -45.0, 'h2o': -45.0, 'adp': 45.0, 'pi': 45.0, 'h': 42.0}
    >>> get_gam(model, biomass_rxn="growth")
    45.0
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
    """Get GAM of a model: the coefficient of ATP in the biomass reaction, the value that :func:`set_gam` sets.

    Parameters
    ----------
    model : cobra.Model
        The model.
    biomass_rxn : str
        Id of the biomass reaction.
    atp_met_name : str, optional
        Name (not id) of ATP. Default ``"ATP"``.

    Returns
    -------
    float
        Size of the ATP coefficient (mmol/gDW).

    Raises
    ------
    KeyError
        If ``biomass_rxn`` is not in the model.
    ValueError
        If the biomass reaction does not have exactly one metabolite with the name ``atp_met_name``.

    See Also
    --------
    set_gam
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
