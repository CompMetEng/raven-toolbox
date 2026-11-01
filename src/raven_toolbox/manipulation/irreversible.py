"""Convert reversible reactions to an irreversible (forward and reverse) form."""
from __future__ import annotations

import copy

import cobra

from raven_toolbox.utils.gpr import set_gene_rule


def convert_to_irreversible(
    model: cobra.Model, rxns: list[str] | None = None
) -> list[str]:
    """Split every reversible reaction into a forward and a reverse reaction.

    The split applies to each reaction with ``lb < 0``, including exchange (boundary)
    reactions, as in RAVEN's ``convertToIrrev``:

    - The original reaction is kept as the forward direction. Its
      lower bound is set to 0.
    - A new reaction with the same ID plus a ``_REV`` suffix is added,
      representing the reverse direction. Its stoichiometry is the
      negation of the original and its bounds are ``(0, -original_lb)``.
      It takes the name of the original (with " (reversible)" appended), the
      gene-protein rule, the subsystem, the annotations and the notes
      of the original. RAVEN's ``convertToIrrev`` copies the same
      per-reaction fields (``eccodes``, ``rxnMiriams``,
      ``subSystems``, ``rxnNotes``, ...).
    - If the objective coefficient of the original reaction is negative, it
      is moved to the new reverse reaction (with the sign flipped to positive)
      and set to 0 on the forward reaction, so that a reversible reaction
      credited under the objective for reverse-direction flux keeps that credit after
      splitting. A non-negative coefficient stays on the forward reaction, and the
      objective coefficient of the reverse reaction is 0, as in RAVEN's ``convertToIrrev``.

    Parameters
    ----------
    model
        The ``cobra.Model``, modified in place.
    rxns
        Reaction IDs eligible for splitting. Reactions outside this set
        are left unchanged even if reversible. Defaults to every
        reaction in the model, as the ``rxns`` argument of RAVEN's
        ``convertToIrrev`` defaults to ``model.rxns``.

    Returns
    -------
    list of str
        Sorted IDs of newly added reverse reactions (the ones ending in
        ``_REV``). The forward reactions keep their original IDs.
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
