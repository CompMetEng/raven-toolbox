"""Infer missing metabolite formulas from elemental mass balance.

Port of RAVEN's ``guessComposition``. Where a reaction has exactly one
participant with an unknown formula and every other participant's formula is
known, the elemental composition of the unknown one follows from the mass
balance of the reaction. This is repeated to a fixed point, since resolving
one metabolite can make others resolvable.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass

import cobra

__all__ = ["GuessCompositionResult", "guess_composition"]

_ZERO_TOL = 1e-12
_CONSISTENCY_TOL = 1e-10


@dataclass(frozen=True)
class GuessCompositionResult:
    """Result of :func:`guess_composition`.

    Attributes
    ----------
    guessed
        Metabolite names a formula was assigned to.
    could_not_guess
        Metabolite names that had no formula before the call and still have
        none afterwards: either no resolving reaction was found, or
        different reactions implied inconsistent compositions.
    """

    guessed: list[str]
    could_not_guess: list[str]


def _normalize_atoms(atoms: dict[str, float]) -> dict[str, int | float]:
    """Round whole-number atom counts to int; 4.0 renders as "H4" and not as "H4.0".

    A fractional count, for example 0.5, stays a float.
    """
    normalized: dict[str, int | float] = {}
    for element, count in atoms.items():
        rounded = round(count)
        normalized[element] = rounded if abs(count - rounded) < _ZERO_TOL else count
    return normalized


def _known_elements(met: cobra.Metabolite) -> dict[str, float] | None:
    if not met.formula:
        return None
    try:
        elements = met.elements
    except ValueError:
        return None
    return elements or None


def guess_composition(
    model: cobra.Model, *, print_results: bool = True
) -> GuessCompositionResult:
    """Infer missing metabolite formulas from elemental mass balance.

    Modifies ``model`` in place: resolved metabolites get their ``formula``
    (and ``elements``) set. Metabolites sharing a name (for example the same
    compound in different compartments) are resolved and assigned together,
    since composition does not depend on compartment.

    Parameters
    ----------
    model:
        Model to fill in formulas for.
    print_results:
        Print each resolved metabolite's name and guessed formula.

    Returns
    -------
    GuessCompositionResult

    Notes
    -----
    The result is correct only if the existing formulas and reactions are
    correct. A warning is issued if a metabolite would need inconsistent
    compositions from different reactions, so the function can also check a
    model for such errors. Wrong pre-existing formulas are not detected by a
    successful completion.

    Each pass resolves formulas using a snapshot of what was already known
    at the start of that pass: a metabolite resolved earlier in the same
    pass only helps resolve others starting from the next pass, matching
    the original algorithm's own once-per-pass formula parsing.

    A metabolite instance that does not participate in any reaction
    contributes nothing either way: it does not resolve the name and does not
    block a resolution reached through a different compartment instance.
    """
    rxn_index = {rxn.id: i for i, rxn in enumerate(model.reactions)}
    met_index = {met.id: i for i, met in enumerate(model.metabolites)}

    original_missing = {met.name for met in model.metabolites if not met.formula}
    guessed: list[str] = []

    predicted = True
    while predicted:
        predicted = False
        known_elements = {met.id: _known_elements(met) for met in model.metabolites}
        missing_names = sorted({met.name for met in model.metabolites if not met.formula})

        for name in missing_names:
            instances = sorted(
                (met for met in model.metabolites if met.name == name),
                key=lambda m: met_index[m.id],
            )

            current_comp: dict[str, float] | None = None
            status = "missing"

            for met in instances:
                reactions = sorted(met.reactions, key=lambda r: rxn_index[r.id])
                for rxn in reactions:
                    coeff = rxn.metabolites[met]
                    net: dict[str, float] = defaultdict(float)
                    all_known = True
                    for other, other_coeff in rxn.metabolites.items():
                        if other is met:
                            continue
                        other_elements = known_elements[other.id]
                        if other_elements is None:
                            all_known = False
                            break
                        for element, count in other_elements.items():
                            net[element] += other_coeff * count
                    if not all_known:
                        continue

                    comp = {el: v for el, v in net.items() if abs(v) > _ZERO_TOL}
                    signs = {v > 0 for v in comp.values()}
                    if len(signs) > 1:
                        status = "contradiction"
                        break

                    atoms = {el: abs(v) / abs(coeff) for el, v in comp.items()}
                    if current_comp is None:
                        current_comp = atoms
                    keys = set(current_comp) | set(atoms)
                    if all(
                        abs(current_comp.get(k, 0.0) - atoms.get(k, 0.0)) < _CONSISTENCY_TOL
                        for k in keys
                    ):
                        status = "resolved"
                        current_comp = atoms
                    else:
                        status = "contradiction"
                        break
                if status == "contradiction":
                    break

            if status == "contradiction":
                warnings.warn(
                    f'Could not predict composition for "{name}" due to inconsistencies',
                    stacklevel=2,
                )
            elif status == "resolved":
                assert current_comp is not None
                normalized = _normalize_atoms(current_comp)
                for met in instances:
                    met.elements = normalized
                if print_results:
                    print(f'Predicted composition for "{name}" to be {instances[0].formula}')
                guessed.append(name)
                predicted = True

    could_not_guess = sorted(original_missing - set(guessed))
    return GuessCompositionResult(guessed=guessed, could_not_guess=could_not_guess)
