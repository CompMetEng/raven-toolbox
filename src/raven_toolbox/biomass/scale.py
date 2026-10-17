"""Biomass component summing and rescaling.

Ports the generic core of yeast-GEM's ``sumBioMass.m`` /
``scaleBioMass.m`` / ``rescalePseudoReaction.m`` into a single
parameterised module that other GEMs can configure via
:class:`BiomassConfig`.
"""
from __future__ import annotations

import re

import cobra

from raven_toolbox.biomass.config import BiomassComponent, BiomassConfig

# Elemental atomic weights used to translate formulas into g/mol.
# Mirrors the table baked into yeast-GEM's ``sumBioMass.m``; pseudo-
# element "R" is treated as zero mass (placeholder for residues).
_ELEMENT_MASS_G_PER_MOL: dict[str, float] = {
    "C": 12.01, "H": 1.008, "N": 14.007, "O": 15.999,
    "P": 30.974, "S": 32.06, "R": 0.0,
    "Fe": 55.845, "K": 39.098, "Na": 22.99, "Cl": 35.45,
    "Mn": 54.938, "Zn": 65.38, "Ca": 40.078, "Mg": 24.305, "Cu": 63.546,
}

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]*)(\d*)")


def sum_biomass(model: cobra.Model, config: BiomassConfig) -> dict[str, float]:
    """Mass-fraction (g/gDW) per biomass component plus the total.

    Mirrors yeast-GEM's ``sumBioMass.m``. Returns a dict keyed by each
    :class:`BiomassComponent` name plus ``"total"``. Components whose
    pseudoreaction is missing from the model contribute 0 (logged as a
    warning).
    """
    fractions: dict[str, float] = {}
    total = 0.0
    for comp in config.components:
        f = _component_fraction(model, comp)
        fractions[comp.name] = f
        total += f
    fractions["total"] = total
    return fractions


def rescale_pseudoreaction(
    model: cobra.Model,
    config: BiomassConfig,
    component_name: str,
    factor: float,
) -> None:
    """Multiply the substrate coefficients of one component pseudoreaction
    by ``factor`` and rebalance H+ to preserve charge neutrality.

    Ports ``rescalePseudoReaction.m``. "Substrate" here means any
    metabolite whose ``name`` is not the component's product name; the
    component's own metabolite (e.g. ``protein`` for the protein
    pseudoreaction) is left untouched. After the rescaling the
    coefficient of ``config.proton_met`` is recomputed so the
    reaction's total ionic charge sums to zero.
    """
    comp = config.get(component_name)
    rxn = _find_pseudoreaction(model, comp)
    proton_met = model.metabolites.get_by_id(config.proton_met)

    # Step 1: scale substrate coefficients (everything but the component
    # product). Use a fresh dict to avoid mutating during iteration.
    deltas: dict[cobra.Metabolite, float] = {}
    for met, coef in rxn.metabolites.items():
        if met.name == comp.name:
            continue  # the product — leave alone
        deltas[met] = (factor - 1.0) * coef  # combine=True adds (factor-1)*coef
    if deltas:
        rxn.add_metabolites(deltas, combine=True)

    # Step 2: H+ rebalance. The legacy code first zeroes H+, then sets
    # it to negate the rxn's current ionic charge. With combine=True we
    # express the same with two `_set_coefficient` calls.
    _set_coefficient(rxn, proton_met, 0.0)
    total_charge = sum(
        (m.charge or 0) * coef for m, coef in rxn.metabolites.items()
    )
    _set_coefficient(rxn, proton_met, -total_charge)


def scale_biomass(
    model: cobra.Model,
    config: BiomassConfig,
    component_name: str,
    new_value: float,
    *,
    balance_out: str | None = None,
) -> None:
    """Rescale a component to a new mass fraction, optionally balancing
    out a second component so the biomass total becomes 1 g/gDW.

    Ports ``scaleBioMass.m``. The rescaling factor is derived from
    :func:`sum_biomass` on the *current* model state, so call this with
    an in-place ``model`` mutation, not on a stale snapshot.

    With ``balance_out``, that component is rescaled so the components sum
    to exactly 1 g/gDW, whatever the total was before. It must be large
    enough to absorb the difference: a ``ValueError`` is raised, before the
    model is changed, when its fraction is 0 or would become 0 or negative.
    """
    fractions = sum_biomass(model, config)
    current = fractions[component_name]
    if current == 0:
        raise ValueError(
            f"Cannot scale {component_name!r} to {new_value}: current "
            "fraction is 0 (pseudoreaction missing or empty)."
        )
    if balance_out is not None:
        balance_current = fractions[balance_out]
        if balance_current == 0:
            raise ValueError(
                f"Cannot balance with {balance_out!r}: current fraction is 0 "
                "(pseudoreaction missing or empty)."
            )
        # Fractions are linear in the substrate coefficients, so the total
        # after the first rescaling is known before the model is changed.
        balance_new = balance_current + 1.0 - (fractions["total"] - current + new_value)
        if balance_new <= 0:
            raise ValueError(
                f"Cannot balance {component_name!r} = {new_value:g} with "
                f"{balance_out!r}: it would have to go from {balance_current:.4g} "
                f"to {balance_new:.4g} g/gDW to bring the total to 1."
            )
    factor = new_value / current
    rescale_pseudoreaction(model, config, component_name, factor)

    if balance_out is not None:
        # Recompute X after the first rescaling.
        fractions = sum_biomass(model, config)
        total = fractions["total"]
        balance_current = fractions[balance_out]
        balance_factor = (balance_current + (1.0 - total)) / balance_current
        rescale_pseudoreaction(model, config, balance_out, balance_factor)


# --- internals ---------------------------------------------------------

def _component_fraction(model: cobra.Model, comp: BiomassComponent) -> float:
    """Compute g/gDW of a single component from its pseudoreaction."""
    try:
        rxn = _find_pseudoreaction(model, comp)
    except KeyError:
        return 0.0  # missing pseudoreaction → contributes 0
    substrates = [(m, c) for m, c in rxn.metabolites.items() if c < 0]
    if not substrates:
        return 0.0

    if comp.mass_strategy == "grams":
        # Stoichiometry already in g/gDW; just sum the (negative) coefs
        # and flip sign.
        return -sum(c for _m, c in substrates)

    offset = _mw_offset(comp.mass_strategy)
    mass_g = 0.0
    for met, coef in substrates:
        mw = _formula_mw(met.formula or "")
        if mw == 0:
            raise ValueError(
                f"Metabolite {met.id} ({met.name!r}) has an empty "
                "formula; cannot compute mass for biomass summing."
            )
        mw_corrected = mw + offset
        mass_g += -coef * mw_corrected
    return mass_g / 1000.0  # g/mmol → g/gDW (matching yeast-GEM units)


def _mw_offset(strategy: str) -> float:
    if strategy == "mw":
        return 0.0
    if strategy == "mw_minus_2h":
        # 2 × 1.008 = 2.016 g/mol per substrate (charged tRNAs release
        # two protons on amino acylation).
        return -2.016
    if strategy == "mw_minus_water":
        # H2O = 18.015 g/mol per polymerisation step (RNA / DNA).
        return -18.015
    raise ValueError(f"Unknown mass_strategy: {strategy!r}")


def _formula_mw(formula: str) -> float:
    """Molecular weight in g/mol from a Hill-style chemical formula."""
    if not formula:
        return 0.0
    mw = 0.0
    for element, count_str in _FORMULA_TOKEN_RE.findall(formula):
        if not element:
            continue
        try:
            atomic = _ELEMENT_MASS_G_PER_MOL[element]
        except KeyError as exc:
            raise ValueError(
                f"Unknown element {element!r} in formula {formula!r}; "
                "extend raven_toolbox.biomass.scale._ELEMENT_MASS_G_PER_MOL "
                "if the element is real."
            ) from exc
        count = int(count_str) if count_str else 1
        mw += atomic * count
    return mw


def _find_pseudoreaction(model: cobra.Model, comp: BiomassComponent) -> cobra.Reaction:
    for rxn in model.reactions:
        if rxn.name == comp.pseudoreaction_name:
            return rxn
    raise KeyError(
        f"Component {comp.name!r}: no reaction named "
        f"{comp.pseudoreaction_name!r} in model."
    )


def _set_coefficient(rxn: cobra.Reaction, met: cobra.Metabolite, value: float) -> None:
    """Land on ``S[met, rxn] = value`` via cobra's combine=True API."""
    current = rxn.metabolites.get(met, 0.0)
    delta = float(value) - current
    if delta != 0:
        rxn.add_metabolites({met: delta}, combine=True)
