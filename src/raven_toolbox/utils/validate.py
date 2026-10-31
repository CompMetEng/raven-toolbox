"""Curation checks for a model.

Curation checks that cobra has no single call for: orphaned objects, empty reactions,
duplicated metabolite ``name + compartment``, empty metabolite names, and reactions with a
nonzero objective coefficient (none or several). :func:`check_model` returns the findings as
structured :class:`ModelIssue` records.
"""
from __future__ import annotations

from dataclasses import dataclass

import cobra


@dataclass(frozen=True)
class ModelIssue:
    """One curation issue found in a model.

    Attributes
    ----------
    category
        Machine-readable kind, for example ``"orphan_metabolite"``, ``"empty_reaction"``,
        ``"orphan_gene"``, ``"duplicate_name_compartment"``,
        ``"empty_metabolite_name"``, ``"objective"``.
    object_id
        ID of the offending object, or ``None`` for model-level issues.
    message
        Human-readable description.
    """

    category: str
    object_id: str | None
    message: str


def check_model(model: cobra.Model) -> list[ModelIssue]:
    """Run curation checks on a model and return the issues found.

    Does not raise; returns a possibly empty list of :class:`ModelIssue`.
    """
    issues: list[ModelIssue] = []

    for met in model.metabolites:
        if not met.reactions:
            issues.append(
                ModelIssue("orphan_metabolite", met.id, f"Metabolite {met.id!r} is not used in any reaction.")
            )
        if not (met.name and str(met.name).strip()):
            issues.append(
                ModelIssue("empty_metabolite_name", met.id, f"Metabolite {met.id!r} has no name.")
            )

    for gene in model.genes:
        if not gene.reactions:
            issues.append(
                ModelIssue("orphan_gene", gene.id, f"Gene {gene.id!r} is not associated with any reaction.")
            )

    for rxn in model.reactions:
        if not rxn.metabolites:
            issues.append(
                ModelIssue("empty_reaction", rxn.id, f"Reaction {rxn.id!r} has no metabolites.")
            )

    by_name_comp: dict[tuple[str, str], list[str]] = {}
    for met in model.metabolites:
        by_name_comp.setdefault((met.name, met.compartment), []).append(met.id)
    for (name, comp), ids in by_name_comp.items():
        if name and len(ids) > 1:
            issues.append(
                ModelIssue(
                    "duplicate_name_compartment",
                    None,
                    f"{len(ids)} metabolites share name {name!r} in compartment {comp!r}: {sorted(ids)}",
                )
            )

    objective_rxns = [r.id for r in model.reactions if r.objective_coefficient != 0]
    if not objective_rxns:
        issues.append(ModelIssue("objective", None, "No reaction has a nonzero objective coefficient."))
    elif len(objective_rxns) > 1:
        issues.append(
            ModelIssue("objective", None, f"Multiple objective reactions: {sorted(objective_rxns)}")
        )

    return issues
