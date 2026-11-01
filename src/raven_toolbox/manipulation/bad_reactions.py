"""Remove reactions that produce or consume mass from nothing (RAVEN ``removeBadRxns``).

The function probes the model repeatedly with :func:`~raven_toolbox.gapfilling.find_leak_metabolite`
(first production, then consumption). Each time a leak is found, it selects one of the flux-carrying
reactions of the probe solution at random and removes it. Elementally unbalanced reactions
(:func:`~raven_toolbox.utils.get_elemental_balance`) are candidates first; with ``rxn_rules`` 2 or
3, reactions with an undetermined balance are candidates when none is unbalanced, and with
``rxn_rules`` 3 any flux-carrying reaction is. The search for a direction ends when no leak remains.
With ``rxn_rules`` 1 or 2 it also ends, with a warning and while a leak remains, when no unbalanced
candidate (for ``rxn_rules`` 2, no unbalanced or undetermined candidate) is in the probe
solution. The function is intended for a closed model (no open exchange reactions). An example is
a model with two versions of a DNA-polymerisation reaction that use incompatible stoichiometric
conventions: together they produce matter while all exchanges are closed.
"""
from __future__ import annotations

import random
import warnings
from collections.abc import Iterable

import cobra

from raven_toolbox.gapfilling.leak import find_leak_metabolite
from raven_toolbox.utils.balance import get_elemental_balance

__all__ = ["remove_bad_reactions"]

_ACTIVE_THRESHOLD = 1e-8
_DEFAULT_BALANCE_ELEMENTS = ("C", "P", "S", "N", "O")


def remove_bad_reactions(
    model: cobra.Model,
    *,
    rxn_rules: int = 1,
    ignore_mets: Iterable[str] | None = None,
    is_names: bool = False,
    balance_elements: Iterable[str] = _DEFAULT_BALANCE_ELEMENTS,
    ignore_int_bounds: bool = False,
    print_report: bool = False,
) -> list[str]:
    """Remove reactions that let ``model`` produce or consume metabolites from nothing.

    Modifies ``model`` in place.

    Parameters
    ----------
    model:
        Model to clean up. The search expects a model without uptake and excretion; it warns about
        open exchange reactions and continues. Close the model first with
        :func:`~raven_toolbox.manipulation.close_model`.
    rxn_rules:
        Which reactions may be removed (default 1):

        - 1: only reactions that are elementally unbalanced.
        - 2: also reactions whose balance could not be determined (missing
          or unparseable metabolite formulas).
        - 3: any flux-carrying reaction in the leak solution.
    ignore_mets:
        Metabolite ids (or names, with ``is_names=True``) to exclude from
        the leak search; forwarded to
        :func:`~raven_toolbox.gapfilling.find_leak_metabolite`.
    is_names:
        If True, ``ignore_mets`` holds metabolite names; otherwise it holds ids.
    balance_elements:
        Elements to check when deciding whether a reaction is unbalanced
        (default carbon, phosphorus, sulfur, nitrogen, oxygen). A reaction
        imbalanced only in an element outside this set is not treated as
        unbalanced.
    ignore_int_bounds:
        Forwarded to :func:`~raven_toolbox.gapfilling.find_leak_metabolite`.
    print_report:
        Print which metabolite triggered each iteration and which reaction
        was removed for it.

    Returns
    -------
    list[str]
        Ids of the reactions removed, in removal order.

    Raises
    ------
    ValueError
        If ``model`` is not feasible to begin with.

    Notes
    -----
    The ``refModel`` option of RAVEN's ``removeBadRxns`` (checking that a deletion keeps a
    reference model feasible) is not available. RAVEN itself only warns that the feature "is
    currently not supported" and ignores it. When several reactions are candidates, the removed
    one is chosen at random with the :mod:`random` state of the standard library, so the result
    varies between runs unless that state is seeded.
    """
    open_exchanges = [r for r in model.boundary if r.lower_bound != 0 or r.upper_bound != 0]
    if open_exchanges:
        warnings.warn(
            "The model contains open exchange reactions. This is not the intended use of "
            "this function. Consider closing them first.",
            stacklevel=2,
        )

    model.slim_optimize()
    if model.solver.status != "optimal":
        raise ValueError(
            "The model is not feasible. Consider removing lower bounds (such as ATP maintenance)."
        )

    balance_elements = set(balance_elements)
    removed: list[str] = []
    warned_arbitrary = {"produce": False, "consume": False}

    for direction in ("produce", "consume"):
        while True:
            leak = find_leak_metabolite(
                model,
                direction,
                ignore_mets=ignore_mets,
                is_names=is_names,
                allow_excretion=(direction == "produce"),
                ignore_int_bounds=ignore_int_bounds,
            )
            if leak.status != "optimal":
                break

            met_name = model.metabolites.get_by_id(leak.metabolites[0]).name
            verb = "make" if direction == "produce" else "consume"
            if print_report:
                print(f"Can {verb}: {met_name}")

            active = [rid for rid, flux in leak.fluxes.items() if abs(flux) > _ACTIVE_THRESHOLD]
            balances = {b.reaction_id: b for b in get_elemental_balance(model, active)}

            unbalanced = [
                rid for rid in active
                if balances[rid].status == "unbalanced" and balance_elements & balances[rid].imbalance.keys()
            ]
            if unbalanced:
                to_remove = random.choice(unbalanced)
            elif rxn_rules == 1:
                warnings.warn(
                    f'No unbalanced reactions were found in the solution, but the model can '
                    f'still {verb} "{met_name}". Aborting search. Consider setting rxn_rules '
                    f"to 2 or 3 for a more exhaustive search",
                    stacklevel=2,
                )
                break
            else:
                unknown = [rid for rid in active if balances[rid].status == "unknown"]
                if unknown:
                    to_remove = random.choice(unknown)
                elif rxn_rules == 2:
                    warnings.warn(
                        f'No unbalanced or unparsable reactions were found in the solution, but '
                        f'the model can still {verb} "{met_name}". Aborting search. Consider '
                        f"setting rxn_rules to 3 for a more exhaustive search",
                        stacklevel=2,
                    )
                    break
                else:
                    if not warned_arbitrary[direction]:
                        warnings.warn(
                            f'No unbalanced or unparsable reactions were found in the solution, '
                            f'but the model can still {verb} "{met_name}". This indicates some '
                            f"error in the metabolite formulas. Removing random reactions in the "
                            f"solution",
                            stacklevel=2,
                        )
                        warned_arbitrary[direction] = True
                    to_remove = random.choice(active)

            removed.append(to_remove)
            if print_report:
                print(f"\tRemoved: {to_remove}")
            model.remove_reactions([to_remove])

    return removed
