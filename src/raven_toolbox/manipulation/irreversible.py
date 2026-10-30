"""Convert reversible reactions to an irreversible (forward + reverse) form.

cobrapy's own ``convert_to_irreversible`` was removed, so this is a genuine
implementation rather than a wrapper.
"""
from __future__ import annotations

import copy

import cobra

from raven_toolbox.utils.gpr import set_gene_rule


def convert_to_irreversible(
    model: cobra.Model, rxns: list[str] | None = None
) -> list[str]:
    """Split every reversible reaction into a forward + reverse pair.
    For each reaction with ``lb < 0``, including exchange (boundary)
    reactions -- MATLAB's ``convertToIrrev`` does not special-case them,
    so neither does this:

    - The original reaction is kept as the forward direction. Its
      lower bound is clamped to 0.
    - A new reaction with the same ID plus a ``_REV`` suffix is added,
      representing the reverse direction. Its stoichiometry is the
      negation of the original, its bounds are ``(0, -original_lb)``,
      and it inherits the name (with " (reversible)" appended), the
      gene-protein rule, the subsystem, the annotations and the notes
      of the original. MATLAB's ``convertToIrrev`` copies the same
      per-reaction fields across (``eccodes``, ``rxnMiriams``,
      ``subSystems``, ``rxnNotes``, ...); dropping them would, among
      other things, leave the reverse reaction without an EC code and
      so without a kcat downstream.
    - If the original reaction's objective coefficient is negative, it
      is moved onto the new reverse reaction (sign-flipped positive)
      and zeroed on the forward reaction, so a reversible reaction
      credited under the objective for *reverse*-direction flux still
      gets that credit after splitting. A non-negative coefficient is
      left on the forward reaction unchanged, and the reverse reaction's
      is left at its default (0) -- matching MATLAB's ``convertToIrrev``.

    Parameters
    ----------
    model
        A cobra.Model, mutated in place.
    rxns
        Reaction IDs eligible for splitting. Reactions outside this set
        are left untouched even if reversible. Defaults to every
        reaction in the model, matching MATLAB's ``convertToIrrev``
        default (its ``rxns`` name-value argument defaults to
        ``model.rxns``).

    Returns
    -------
    list of str
        Sorted IDs of newly added reverse reactions (the ones ending in
        ``_REV``). The forward reactions retain their original IDs.
    """
    eligible = None if rxns is None else set(rxns)

    reverse_rxns_to_add: list[cobra.Reaction] = []
    forward_updates: list[cobra.Reaction] = []
    negative_objective: list[tuple[cobra.Reaction, cobra.Reaction, float]] = []

    for rxn in model.reactions:
        if rxn.lower_bound >= 0:
            continue
        if eligible is not None and rxn.id not in eligible:
            continue

        original_lb = rxn.lower_bound

        rev_rxn = cobra.Reaction(
            id=f"{rxn.id}_REV",
            name=(f"{rxn.name} (reversible)" if rxn.name else f"{rxn.id}_REV"),
        )
        rev_rxn.lower_bound = 0.0
        rev_rxn.upper_bound = -original_lb
        rev_rxn.add_metabolites({m: -c for m, c in rxn.metabolites.items()})
        set_gene_rule(rev_rxn, rxn.gpr)
        rev_rxn.subsystem = rxn.subsystem
        rev_rxn.annotation = copy.deepcopy(rxn.annotation)
        rev_rxn.notes = copy.deepcopy(rxn.notes)

        reverse_rxns_to_add.append(rev_rxn)
        forward_updates.append(rxn)
        if rxn.objective_coefficient < 0:
            negative_objective.append((rxn, rev_rxn, rxn.objective_coefficient))

    for rxn in forward_updates:
        rxn.lower_bound = 0.0

    if reverse_rxns_to_add:
        model.add_reactions(reverse_rxns_to_add)

    for forward, reverse, coefficient in negative_objective:
        forward.objective_coefficient = 0.0
        reverse.objective_coefficient = -coefficient

    return sorted(r.id for r in reverse_rxns_to_add)
