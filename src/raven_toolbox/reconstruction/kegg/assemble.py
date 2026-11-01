"""Shared assembly of a draft model from a mapping of KOs to genes.

Both KEGG runtime paths assign genes to KOs, from organism annotations
(:func:`get_kegg_model_for_organism`) or from HMM hits (:func:`assign_kos`), and then share
this assembly. It maps KOs to reactions against the
gene-free reference model, OR-joins the genes into each reaction's GPR, keeps
gene-backed reactions (and spontaneous ones when allowed), and applies the ``keep*``
quality filters.
"""
from __future__ import annotations

import cobra
import pandas as pd

from raven_toolbox.utils.gpr import set_gene_rule

_DOMAINS = {"eukaryotes", "prokaryotes"}


def flag_set(rxn_flags: pd.DataFrame | None, column: str) -> set[str]:
    """Reaction ids whose ``column`` flag is truthy.

    A flag is truthy when it is True or when its string form is "true" or "1" in any case.
    """
    if rxn_flags is None or column not in rxn_flags:
        return set()
    mask = rxn_flags[column].map(lambda v: str(v).strip().lower() in ("true", "1"))
    return set(rxn_flags.loc[mask, "reaction"])


def assemble_model_from_ko_genes(
    reference_model: cobra.Model,
    ko_reaction: pd.DataFrame,
    ko_to_genes: dict[str, list[str]],
    *,
    rxn_flags: pd.DataFrame | None = None,
    keep_spontaneous: bool = True,
    keep_undefined_stoich: bool = True,
    keep_incomplete: bool = True,
    keep_general: bool = False,
    model_id: str | None = None,
    model_name: str | None = None,
    note: str | None = None,
    prune_orthology: bool = False,
) -> tuple[cobra.Model, dict[str, list[str]]]:
    """Build a draft model from a ``{ko: [gene, ...]}`` assignment.

    Returns ``(model, gpr_map)`` where ``gpr_map`` is the kept reactions' gene
    lists, so callers can add gene annotations afterwards.

    ``rxn_flags`` is a per-reaction quality-flag table (a ``reaction`` column
    plus any of ``spontaneous``, ``undefined_stoich``, ``incomplete`` and ``general``).
    A reaction flagged ``undefined_stoich``, ``incomplete`` or ``general`` is
    dropped unless its matching ``keep_*`` argument is True; this takes
    precedence over the reaction having genes. ``keep_spontaneous`` (default
    True) additionally keeps otherwise gene-less reactions flagged
    ``spontaneous``.

    ``prune_orthology`` restricts each gene-backed reaction's ``kegg.orthology``
    annotation to the KOs that contributed a gene, as in the HMM branch of RAVEN
    ``getKEGGModelForOrganism``. The organism-annotation path keeps the full
    reference KO list, and the default is False.
    """
    rxn_to_kos: dict[str, set[str]] = {}
    for ko, rid in zip(ko_reaction["ko"], ko_reaction["reaction"], strict=True):
        rxn_to_kos.setdefault(rid, set()).add(ko)

    spontaneous = flag_set(rxn_flags, "spontaneous")
    drop_if = {
        "undefined_stoich": (keep_undefined_stoich, flag_set(rxn_flags, "undefined_stoich")),
        "incomplete": (keep_incomplete, flag_set(rxn_flags, "incomplete")),
        "general": (keep_general, flag_set(rxn_flags, "general")),
    }

    gpr_map: dict[str, list[str]] = {}
    matched_kos: dict[str, set[str]] = {}
    spontaneous_kept: set[str] = set()
    for rxn in reference_model.reactions:
        rid = rxn.id
        # Quality filters first: dropped even if it would have genes.
        if any(not keep_flag and rid in flagged for keep_flag, flagged in drop_if.values()):
            continue
        rxn_kos = rxn_to_kos.get(rid, ())
        genes = sorted({g for ko in rxn_kos for g in ko_to_genes.get(ko, ())})
        if genes:
            gpr_map[rid] = genes
            # KOs that actually contributed a gene, for optional annotation pruning.
            matched_kos[rid] = {ko for ko in rxn_kos if ko_to_genes.get(ko)}
        elif rid in spontaneous and keep_spontaneous:
            spontaneous_kept.add(rid)

    keep = set(gpr_map) | spontaneous_kept
    model = reference_model.copy()
    if model_id is not None:
        model.id = model_id
    if model_name is not None:
        model.name = model_name
    model.remove_reactions(
        [r for r in model.reactions if r.id not in keep], remove_orphans=True
    )
    for rid, genes in gpr_map.items():
        set_gene_rule(model.reactions.get_by_id(rid), " or ".join(genes))
    if prune_orthology:
        # Restrict each gene-backed reaction's kegg.orthology to the KOs that
        # matched a gene, preserving the reference annotation's order. (The
        # order-preserving intersection also avoids the index misalignment in
        # RAVEN getKEGGModelForOrganism's rxnMiriams pruning.)
        for rid, kos in matched_kos.items():
            annotation = model.reactions.get_by_id(rid).annotation
            orthology = annotation.get("kegg.orthology")
            if orthology is None:
                continue
            if isinstance(orthology, list):
                annotation["kegg.orthology"] = [k for k in orthology if k in kos]
            elif orthology not in kos:
                annotation["kegg.orthology"] = []
    if note is not None:
        for rid in keep:
            model.reactions.get_by_id(rid).notes["note"] = note
    return model, gpr_map
