"""Expand reactions with isozymes into one reaction per isozyme.

Operates on cobra's GPR AST, so the model stays a plain ``cobra.Model`` throughout.

MATLAB-COMPAT: GECKO MATLAB and RAVEN ``expandModel.m`` use string manipulation
on grRules to detect and split isozymes. raven_toolbox uses cobrapy's GPR AST
instead. Output should be equivalent for any well-formed GPR; cases that differ
are likely malformed GPR strings that the AST flags as invalid.
"""
from __future__ import annotations

import ast
import copy

import cobra
from cobra.core.gene import GPR

from raven_toolbox.utils.gpr import parse_gr_rule

__all__ = ["expand_model", "gpr_to_dnf"]


def gpr_to_dnf(gpr: GPR | str | None) -> list[list[str]]:
    """Convert a GPR to disjunctive normal form (list of AND-clauses).

    An empty GPR yields an empty list. A single clause (no OR anywhere)
    yields a list of length 1. OR-of-ANDs yields one sublist per
    disjunct, each containing the gene names ANDed together.

    Handles distributivity: ``g1 and (g2 or g3)`` becomes
    ``[[g1, g2], [g1, g3]]``.

    Parameters
    ----------
    gpr
        A cobra :class:`~cobra.core.gene.GPR`, a grRule string or ``None``. A string is
        read with :func:`raven_toolbox.utils.parse_gr_rule`, the grammar of RAVEN's
        ``parseGrRule``, as RAVEN's ``grRuleToDNF`` reads it.

    Raises
    ------
    ValueError
        If a rule string is malformed in that grammar (RAVEN raises ``RAVEN:badGrRule``).
    """
    if isinstance(gpr, str):
        gpr = parse_gr_rule(gpr)
    if gpr is None or gpr.body is None:
        return []
    return _node_to_dnf(gpr.body)


def _node_to_dnf(node) -> list[list[str]]:
    """Recursive helper. Returns DNF as list of AND-clauses."""
    if isinstance(node, ast.Name):
        return [[node.id]]
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.Or):
            result: list[list[str]] = []
            for child in node.values:
                result.extend(_node_to_dnf(child))
            return result
        if isinstance(node.op, ast.And):
            clauses: list[list[str]] = [[]]
            for child in node.values:
                child_dnf = _node_to_dnf(child)
                new_clauses: list[list[str]] = []
                for existing in clauses:
                    for extra in child_dnf:
                        new_clauses.append(existing + extra)
                clauses = new_clauses
            return clauses
    raise ValueError(f"Unexpected GPR node type: {type(node).__name__}")


def expand_model(model: cobra.Model) -> list[str]:
    """Split reactions with isozymes (OR in GPR) into one reaction per isozyme.
    For each reaction whose GPR contains at least one OR, the reaction
    is removed and replaced by one copy per disjunctive clause. The new
    reactions get ID suffix ``_EXP_1``, ``_EXP_2``, etc. The ``_EXP_1`` copy
    takes the position of the original reaction; the other copies are appended
    in the order of the originals, as RAVEN's ``expandModel`` does. All other
    fields (stoichiometry, bounds, name, subsystem) are copied verbatim;
    only the GPR is simplified to the single AND-clause for that
    isozyme.

    Reactions with no GPR, or with a GPR that has no OR, are left
    untouched.

    Parameters
    ----------
    model
        A cobra.Model, mutated in place.

    Returns
    -------
    list of str
        Sorted IDs of newly added expanded reactions (those with
        ``_EXP_N`` suffixes). The original reactions that were split
        are no longer in the model.
    """
    expansions: list[tuple[cobra.Reaction, list[list[str]]]] = []

    for rxn in model.reactions:
        if not rxn.gene_reaction_rule:
            continue
        clauses = gpr_to_dnf(rxn.gpr)
        if len(clauses) <= 1:
            continue
        expansions.append((rxn, clauses))

    position = {r.id: i for i, r in enumerate(model.reactions)}
    sort_key: dict[str, int] = {}
    next_tail = len(position)
    added_ids: list[str] = []
    for original_rxn, clauses in expansions:
        new_rxns: list[cobra.Reaction] = []
        for i, clause in enumerate(clauses, start=1):
            new_rxn = cobra.Reaction(
                id=f"{original_rxn.id}_EXP_{i}",
                name=original_rxn.name,
            )
            new_rxn.lower_bound = original_rxn.lower_bound
            new_rxn.upper_bound = original_rxn.upper_bound
            new_rxn.add_metabolites(dict(original_rxn.metabolites.items()))
            new_rxn.subsystem = original_rxn.subsystem
            new_rxn.gene_reaction_rule = " and ".join(clause)
            # Propagate per-reaction metadata (notably ec-code / annotations)
            # so downstream functions see the same annotations on expanded
            # reactions as on the original. Deep-copy so siblings are independent.
            new_rxn.annotation = copy.deepcopy(original_rxn.annotation)
            new_rxn.notes = copy.deepcopy(original_rxn.notes)
            new_rxns.append(new_rxn)

        obj_coeff = original_rxn.objective_coefficient
        model.remove_reactions([original_rxn])
        model.add_reactions(new_rxns)
        if obj_coeff:  # keep the original in the objective — sum over its isozyme copies
            for new_rxn in new_rxns:
                new_rxn.objective_coefficient = obj_coeff
        sort_key[new_rxns[0].id] = position[original_rxn.id]
        for new_rxn in new_rxns[1:]:
            sort_key[new_rxn.id] = next_tail
            next_tail += 1
        added_ids.extend(r.id for r in new_rxns)

    if expansions:
        # Unsplit reactions keep their original position; the expansion keys sort
        # the first copy into the slot of its original and the rest after them.
        model.reactions.sort(key=lambda r: sort_key.get(r.id, position.get(r.id, next_tail)))

    return sorted(added_ids)
