"""Copy reactions (with their metabolites and genes) from another model.

cobra's ``Model.merge`` and ``add_reactions`` match metabolites by id. This module transfers a
chosen set of reactions from a source model into a draft model and matches metabolites by
name and compartment (``name[compartment]``) instead. A compound present in both models under
different ids is reused, and only metabolites without a match are created (with the id, formula,
charge and annotation of the source). New genes are created by cobra when the GPR is set. The
function is meant for use after ``getModelFromHomology``, to copy additional reactions into the
draft.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable

import cobra
from cobra import Metabolite, Reaction

from raven_toolbox.manipulation.add import _new_met_id
from raven_toolbox.utils.gpr import set_gene_rule


def _name_comp(met: Metabolite) -> str:
    return f"{met.name}[{met.compartment}]"


def add_reactions_from_model(
    model: cobra.Model,
    source_model: cobra.Model,
    reactions: str | Iterable[str],
    *,
    genes: bool | str | Iterable[str] = False,
    note: str | None = "Added via add_reactions_from_model()",
    confidence: int | None = None,
) -> list[Reaction]:
    """Copy reactions from ``source_model`` into ``model``.

    Parameters
    ----------
    model
        Draft model to copy into (modified in place).
    source_model
        Model to copy reactions from.
    reactions
        Reaction ID(s) in ``source_model``. Reactions already present in
        ``model`` (by ID) are skipped. A new metabolite keeps the id of the source metabolite
        unless that id is in use, in which case it gets an automatic ID ``m<N>``.
    genes
        ``False`` (default): add reactions without GPRs. ``True``: copy each
        reaction's GPR from the source. A string: use it as the GPR for every
        added reaction. A list: per-reaction GPRs (matching the reactions that
        are added). New genes are created automatically.
    note
        Stored in the ``notes['note']`` of each added reaction (``None`` stores no note).
    confidence
        If given, stored in the ``notes['confidence_score']`` of each added reaction.

    Returns
    -------
    list of cobra.Reaction
        The reactions added, in input order.
    """
    rxn_ids = [reactions] if isinstance(reactions, str) else list(reactions)
    missing = [r for r in rxn_ids if r not in source_model.reactions]
    if missing:
        raise ValueError(f"Reactions not found in the source model: {missing}")

    new_ids = [r for r in rxn_ids if r not in model.reactions]
    if not new_ids:
        raise ValueError("All reactions are already in the model.")
    source_rxns = [source_model.reactions.get_by_id(r) for r in new_ids]

    if genes is False:
        rules = [""] * len(source_rxns)
    elif genes is True:
        rules = [r.gpr for r in source_rxns]
    elif isinstance(genes, str):
        rules = [genes] * len(source_rxns)
    else:
        rules = list(genes)
        if len(rules) != len(source_rxns):
            raise ValueError(
                f"genes list has {len(rules)} rules but {len(source_rxns)} "
                "reactions are being added."
            )

    # Match metabolites by name[comp]; create only the genuinely new ones.
    draft_by_name = {_name_comp(m): m for m in model.metabolites}
    new_mets: list[Metabolite] = []
    pending: set[str] = set()
    # Track ids minted within this batch so two source mets that share an id
    # but differ in name[comp] don't collide when add_metabolites runs.
    pending_ids: set[str] = set()
    for srx in source_rxns:
        for met in srx.metabolites:
            key = _name_comp(met)
            if key in draft_by_name or key in pending:
                continue
            pending.add(key)
            if met.id not in model.metabolites and met.id not in pending_ids:
                new_id = met.id
            else:
                # _new_met_id only knows the model; loop past in-batch hits too.
                new_id = _new_met_id(model, "m")
                while new_id in pending_ids:
                    n = int(new_id[1:]) + 1
                    new_id = f"m{n}"
                    while new_id in model.metabolites:
                        n += 1
                        new_id = f"m{n}"
            pending_ids.add(new_id)
            new_met = Metabolite(
                new_id,
                name=met.name,
                compartment=met.compartment,
                formula=met.formula,
                charge=met.charge,
            )
            new_met.annotation = copy.deepcopy(met.annotation)
            new_met.notes = copy.deepcopy(met.notes)
            new_mets.append(new_met)
            draft_by_name[key] = new_met
    if new_mets:
        model.add_metabolites(new_mets)

    added: list[Reaction] = []
    for srx in source_rxns:
        rxn = Reaction(srx.id, name=srx.name)
        rxn.bounds = srx.bounds
        rxn.subsystem = srx.subsystem
        added.append(rxn)
    model.add_reactions(added)  # one batch — per-reaction adds are super-linear at scale

    for srx, rxn, rule in zip(source_rxns, added, rules, strict=True):
        rxn.add_metabolites(
            {draft_by_name[_name_comp(met)]: coef for met, coef in srx.metabolites.items()}
        )
        if rule:
            set_gene_rule(rxn, rule)
        rxn.annotation = copy.deepcopy(srx.annotation)
        notes = copy.deepcopy(srx.notes)
        if note is not None:
            notes["note"] = note
        if confidence is not None:
            notes["confidence_score"] = confidence
        rxn.notes = notes

    return added
