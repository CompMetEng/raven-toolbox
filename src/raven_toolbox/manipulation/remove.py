"""Remove metabolites or genes from a model.

For removing *reactions*, use cobra directly:
``cobra.Model.remove_reactions(reactions, remove_orphans=...)``.

The two functions here delegate the core to cobra and add the cobra-absent behaviour:

* ``remove_metabolites`` — cobra matches metabolites by ID; RAVEN's ``isNames``
  deletes a metabolite in **every compartment at once** by name. That name
  resolution is the *sole* reason this wrapper exists (see the note on it).
* ``remove_genes`` — cobra's ``cobra.manipulation.remove_genes`` already rewrites
  GPRs through the boolean AST (removing one gene of ``A and B`` empties the
  rule, of ``A or B`` keeps the other) — exactly RAVEN's intent, without its
  ``eval``. What it adds is RAVEN's default of **constraining** flux-blocked
  reactions to zero instead of deleting them (``blocked_reactions``) and RAVEN's
  removal of genes that no remaining reaction uses.
"""
from __future__ import annotations

from collections.abc import Iterable

import cobra
from cobra import Gene, Metabolite
from cobra.manipulation import remove_genes as _cobra_remove_genes


def _as_list(obj) -> list:
    if isinstance(obj, (str, Metabolite, Gene)):
        return [obj]
    return list(obj)


def remove_metabolites(
    model: cobra.Model,
    metabolites: str | Metabolite | Iterable,
    *,
    by_name: bool = False,
    destructive: bool = False,
) -> None:
    """Remove metabolites, optionally matching by name across all compartments.

    Parameters
    ----------
    by_name
        If True, ``metabolites`` are metabolite *names*; every metabolite with a
        matching name is removed, regardless of compartment (RAVEN ``isNames``).
        If False, they are IDs/objects, resolved via cobra.
    destructive
        Passed to cobra: if True, also remove every reaction the metabolite
        participates in.

    Note
    ----
    With ``by_name=False`` this is just ``model.remove_metabolites`` — the
    ``by_name`` cross-compartment deletion is the only thing this adds over cobra.
    """
    if by_name:
        wanted = set(_as_list(metabolites))
        targets = [m for m in model.metabolites if m.name in wanted]
    else:
        targets = model.metabolites.get_by_any(_as_list(metabolites))
    if targets:
        model.remove_metabolites(targets, destructive=destructive)


def remove_genes(
    model: cobra.Model,
    genes: str | Gene | Iterable,
    *,
    blocked_reactions: str = "constrain",
    remove_orphans: bool = False,
) -> list[str]:
    """Remove genes and handle reactions left unable to carry flux.

    GPR rewriting (with correct AND/OR semantics) and gene deletion are done by cobra;
    this adds a policy for reactions whose GPR becomes empty (no enzyme left):

    * ``"constrain"`` (default, as RAVEN's ``removeGenes``) — keep them with an empty
      GPR and bounds ``(0, 0)``.
    * ``"remove"`` — delete them.
    * ``"keep"`` — leave them with an empty GPR and unchanged bounds.

    Requested genes that are not in the model are ignored. When none of the requested
    genes is in the model, the model is not changed and ``[]`` is returned. Otherwise,
    besides the requested genes, every gene that no remaining reaction named in its GPR
    before the removal is deleted too, as RAVEN's ``removeGenes`` does: genes that no
    reaction uses (orphan genes, also when not requested) and, with ``"remove"``, genes
    used only by the deleted reactions. A gene whose only reaction is blocked and kept
    (``"constrain"`` or ``"keep"``), such as the other subunit of a complex that lost
    one, stays in ``model.genes``.

    ``remove_orphans`` (only meaningful with ``blocked_reactions="remove"``)
    passes through to cobra: also drop metabolites orphaned by the removal.

    Returns
    -------
    list of str
        IDs of the reactions that became flux-blocked (had a GPR, now empty).
    """
    if blocked_reactions not in ("remove", "constrain", "keep"):
        raise ValueError(
            f"blocked_reactions must be 'remove', 'constrain', or 'keep', "
            f"got {blocked_reactions!r}"
        )

    # Resolve to gene IDs that are actually in the model (RAVEN filters likewise).
    requested = [g.id if isinstance(g, Gene) else g for g in _as_list(genes)]
    present = [gid for gid in requested if gid in model.genes]
    if not present:
        return []
    named_before = {r.id: {g.id for g in r.genes} for r in model.reactions}

    # Reactions touched by these genes that currently have a GPR.
    affected: set[str] = set()
    for gid in present:
        affected.update(r.id for r in model.genes.get_by_id(gid).reactions)
    had_gpr = {rid for rid in affected if model.reactions.get_by_id(rid).gene_reaction_rule}

    # cobra rewrites GPRs (AST) and removes the gene objects; we manage reactions.
    _cobra_remove_genes(model, present, remove_reactions=False)

    blocked = [
        rid for rid in had_gpr if not model.reactions.get_by_id(rid).gene_reaction_rule
    ]

    if blocked_reactions == "remove":
        model.remove_reactions(blocked, remove_orphans=remove_orphans)
    elif blocked_reactions == "constrain":
        for rid in blocked:
            model.reactions.get_by_id(rid).bounds = (0, 0)

    used = set().union(*(named_before[r.id] for r in model.reactions))
    unused = [g.id for g in model.genes if g.id not in used]
    if unused:
        _cobra_remove_genes(model, unused, remove_reactions=False)

    return sorted(blocked)
