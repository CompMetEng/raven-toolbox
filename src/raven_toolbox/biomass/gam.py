"""Growth-associated (GAM) and non-growth-associated (NGAM) ATP maintenance of a model.

A cell spends ATP on maintenance in two ways:

* GAM is the ATP hydrolysed per gram of biomass formed (mmol/gDW). It scales with growth and
  sits in the coefficients of the ATP hydrolysis terms of the biomass pseudoreaction: ATP and
  water are consumed, and ADP, phosphate and a proton are produced.
* NGAM is the ATP hydrolysed per hour and gram of biomass (mmol/gDW/h) at any growth rate. It
  is the fixed flux of a maintenance reaction.

:func:`set_gam` sets them, :func:`get_gam` reads GAM, and
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

    GAM is held in the coefficients of the cofactors of the ATP hydrolysis (ATP, ADP, water,
    phosphate, proton) in the biomass pseudoreaction. Each metabolite of the pseudoreaction whose
    name is in ``cofactor_met_names`` is set to ``value`` with the sign of its current coefficient.

    With ``keep_offsets_of`` the coefficients keep their differences instead. The coefficient of
    the metabolite of that name (ATP) becomes ``value``, and every other cofactor changes by the
    same amount in the direction of its sign. A proton coefficient that balances the charge of the
    other biomass terms then keeps its offset from the ATP coefficient: for ATP, water, ADP and
    phosphate at 60 and a proton at 57, setting GAM to 45 gives 45 and a proton at 42, whereas
    without ``keep_offsets_of`` the proton becomes 45.

    If both ``ngam_rxn`` and ``ngam_value`` are given, the bounds of the NGAM reaction are both set to
    ``ngam_value``. An empty ``ngam_rxn`` counts as not given.

    Parameters
    ----------
    model : cobra.Model
        Model to change in place.
    value : float
        New GAM, mmol ATP per gDW of biomass formed.
    biomass_rxn : str
        Id of the biomass pseudoreaction whose coefficients hold GAM.
    cofactor_met_names : iterable of str
        Names (not ids) of the metabolites of the GAM term, for example
        ``["ATP", "ADP", "H2O", "H+", "phosphate"]``.
    ngam_rxn : str, optional
        Id of the NGAM reaction.
    ngam_value : float, optional
        NGAM, mmol ATP per gDW and hour, the flux at which the NGAM reaction is fixed. Needs
        ``ngam_rxn``.
    keep_offsets_of : str, optional
        Name of the metabolite whose coefficient becomes ``value`` while the other cofactors shift by
        the same amount (see above). It must be one of ``cofactor_met_names`` and occur once in the
        biomass pseudoreaction. Default ``None`` sets every cofactor to ``value``.

    Returns
    -------
    cobra.Model
        The model that was given, changed.

    Raises
    ------
    KeyError
        If ``biomass_rxn`` or a given ``ngam_rxn`` is not in the model. Both are looked up before the
        model is changed.
    ValueError
        If ``ngam_value`` is given without ``ngam_rxn``, or if ``keep_offsets_of`` is not one of the
        cofactor names or does not name exactly one metabolite of the biomass pseudoreaction.

    See Also
    --------
    get_gam : Reads GAM.
    raven_toolbox.biomass.maintenance.fit_maintenance : Fits GAM and NGAM to measurements.

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
    """Get GAM of a model: the size of the ATP coefficient of the biomass pseudoreaction.

    GAM is the ATP hydrolysed per gram of biomass formed (mmol/gDW), the value that
    :func:`set_gam` sets.

    Parameters
    ----------
    model : cobra.Model
        The model.
    biomass_rxn : str
        Id of the biomass pseudoreaction.
    atp_met_name : str, optional
        Name (not id) of the ATP metabolite. Default ``"ATP"``.

    Returns
    -------
    float
        The size of the ATP coefficient, mmol/gDW.

    Raises
    ------
    KeyError
        If ``biomass_rxn`` is not in the model.
    ValueError
        If the pseudoreaction does not have exactly one metabolite with the name ``atp_met_name``.

    See Also
    --------
    set_gam : Sets GAM.
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
