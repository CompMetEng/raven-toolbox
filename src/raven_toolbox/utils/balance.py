"""Check the elemental balance of reactions, distinguishing ``unbalanced`` from
``unknown`` (missing formula).

cobra's ``reaction.check_mass_balance()`` treats a missing formula as empty, so a
reaction with missing data can be reported as unbalanced or as balanced. This module
checks for missing formulas first and returns, for a batch of reactions, a status
per reaction (``balanced``, ``unbalanced`` or ``unknown``) and the element imbalance
as structured data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cobra


@dataclass(frozen=True)
class ElementalBalance:
    """Balance result for one reaction.

    Attributes
    ----------
    reaction_id
        ID of the reaction.
    status
        ``"balanced"``: the elements balance;
        ``"unbalanced"``: they do not (see ``imbalance``);
        ``"unknown"``: the reaction has no metabolites, at least one metabolite has no
        formula, or a formula cobra cannot parse (for example the polymer ``(C5H8)n``),
        so the balance cannot be determined (cobra treats a missing formula as empty
        and raises ``ValueError`` for an unparsable one).
    imbalance
        Element to net coefficient (products minus reactants), only for
        ``"unbalanced"``; empty otherwise. Charge is not included.
    """

    reaction_id: str
    status: str
    imbalance: dict[str, float] = field(default_factory=dict)


def get_elemental_balance(
    model: cobra.Model, reactions=None
) -> list[ElementalBalance]:
    """Check whether reactions are elementally balanced.

    Parameters
    ----------
    reactions
        Reaction IDs or reaction objects to check; default all reactions. Boundary
        reactions exchange mass with the environment and are reported as
        ``unbalanced``.

    Returns
    -------
    list of ElementalBalance
        One entry per checked reaction, in model order.
    """
    if reactions is None:
        rxns = list(model.reactions)
    else:
        if isinstance(reactions, (str, cobra.Reaction)):
            reactions = [reactions]
        rxns = [
            r if isinstance(r, cobra.Reaction) else model.reactions.get_by_id(r)
            for r in reactions
        ]

    results: list[ElementalBalance] = []
    for rxn in rxns:
        if not rxn.metabolites:
            # Treated as "unknown" rather than vacuously "balanced" (``any()`` over an
            # empty list is False and the imbalance dict is empty): balance can't be
            # determined for a reaction without stoichiometry.
            results.append(ElementalBalance(rxn.id, "unknown"))
            continue
        if any(not met.formula for met in rxn.metabolites):
            results.append(ElementalBalance(rxn.id, "unknown"))
            continue
        try:
            balance = rxn.check_mass_balance()
        except ValueError:
            # A formula cobra cannot parse (a parenthesised polymer such as "(C5H8)n" makes
            # Metabolite.elements return None, which check_mass_balance turns into a ValueError).
            # The formula is present but uninterpretable: that is "unknown", not a crash.
            results.append(ElementalBalance(rxn.id, "unknown"))
            continue
        imbalance = {
            element: amount
            for element, amount in balance.items()
            if element != "charge"
        }
        if imbalance:
            results.append(ElementalBalance(rxn.id, "unbalanced", imbalance))
        else:
            results.append(ElementalBalance(rxn.id, "balanced"))
    return results
