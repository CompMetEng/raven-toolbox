"""Remove metabolites or genes from a model.

Reactions are removed with cobra directly:
``cobra.Model.remove_reactions(reactions, remove_orphans=...)``.

The two functions delegate the core to cobra and add behaviour that cobra lacks:

* ``remove_metabolites``: cobra matches metabolites by ID. RAVEN's ``isNames`` deletes a
  metabolite in every compartment at once by name. The name resolution is the only addition to
  cobra.
* ``remove_genes``: ``cobra.manipulation.remove_genes`` rewrites GPRs through the boolean syntax
  tree (removing one gene of ``A and B`` empties the rule, removing one gene of ``A or B`` keeps
  the other), which is the intent of RAVEN's function. The function adds RAVEN's default of
  constraining flux-blocked reactions to zero by default, with deletion as an option
  (``blocked_reactions``), and
  RAVEN's removal of genes that no remaining reaction uses.
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
        If True, ``metabolites`` are metabolite names; every metabolite with a
        matching name is removed, in any compartment (RAVEN ``isNames``).
        If False, they are IDs or objects, resolved by cobra.
    destructive
        Passed to cobra. If True, every reaction that contains the metabolite is
        removed as well.

    Note
    ----
    With ``by_name=False`` the function calls ``model.remove_metabolites``. The
    deletion across compartments by name is the only addition to cobra.
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
    """Remove genes and handle reactions that can no longer carry flux.

    cobra rewrites the GPRs (AND and OR semantics) and deletes the genes; this function adds a
    policy for reactions whose GPR becomes empty (no enzyme left):

    * ``"constrain"`` (default, as RAVEN's ``removeGenes``): keep them with an empty
      GPR and bounds ``(0, 0)``.
    * ``"remove"``: delete them.
    * ``"keep"``: keep them with an empty GPR and unchanged bounds.

    Requested genes that are not in the model are ignored. When none of the requested genes is in
    the model, the model is not changed and ``[]`` is returned. Otherwise, after the requested genes
    are removed, every gene that is not named in the GPR (before the removal) of any remaining
    reaction is deleted too, as in RAVEN's ``removeGenes``. These are genes that no reaction uses
    (also when not requested) and, with ``"remove"``, genes used only by the deleted reactions. A
    gene whose only reaction is blocked and kept (``"constrain"`` or ``"keep"``), such as the other
    subunit of a complex that lost one, stays in ``model.genes``.

    ``remove_orphans`` (relevant with ``blocked_reactions="remove"`` only) is passed to
    cobra's ``remove_reactions``, which also removes the metabolites left without a reaction.

    Returns
    -------
    list of str
        Sorted IDs of the reactions that became flux-blocked (had a GPR, now empty).
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
