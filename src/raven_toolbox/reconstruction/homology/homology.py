"""Build a draft model from template models and homology hits.

Behaviour:

* ``strictness`` selects the hit-filtering policy (bidirectional matching,
  best hits only, or both);
* GPR rewriting operates on the abstract syntax tree of cobra's GPR;
* ``complex_policy`` selects what happens to AND-linked subunits that lack an
  ortholog (``"flag"``, ``"keep"`` or ``"drop"``);
* best-hit selection uses the bitscore;
* the ortholog map is returned as a dict in ``HomologyResult.gene_map``, and the
  ``notes`` of each transferred reaction contain ``homology_source``.
"""
from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass, field

import cobra
import pandas as pd

from raven_toolbox.manipulation.merge import merge_models
from raven_toolbox.reconstruction.homology.hits import validate_hits
from raven_toolbox.utils.gpr import set_gene_rule

# cobra's GPR parser splits a gene id at whitespace, parentheses, "|" and "&".
_GPR_SPLIT = re.compile(r"[\s()|&]")


@dataclass
class HomologyResult:
    """Result of :func:`get_model_from_homology`.

    Attributes
    ----------
    model
        The draft ``cobra.Model``.
    gene_map
        ``{model_id: {template_gene: [new_gene, ...]}}`` ortholog mapping used.
    candidates
        Reactions that transfer at the identity threshold ``review_identity`` but
        not at ``min_identity``, with the best hit supporting each, sorted by
        identity, strongest first. ``None`` unless ``review_identity`` was given.
        These reactions are not in ``model``.
    """

    model: cobra.Model
    gene_map: dict = field(default_factory=dict)
    candidates: pd.DataFrame | None = None


class _Unmapped:
    """A GPR leaf gene with no ortholog in the new organism."""

    __slots__ = ("gene",)

    def __init__(self, gene: str):
        self.gene = gene

    def __eq__(self, other):
        return isinstance(other, _Unmapped) and other.gene == self.gene

    def __hash__(self):
        return hash(("_Unmapped", self.gene))


def _rewrite_node(node, ortho: dict):
    """Rewrite a GPR AST node, substituting template genes by their orthologs.

    Returns a gene id, an ``_Unmapped`` for a template gene without an ortholog, or
    ``(operator, [children])`` for a group. Groups are flattened and free of duplicates,
    so several template genes that map to the same ortholog do not repeat it. An isozyme
    group is represented by its mapped members: unmapped members are left out when at least one
    member is mapped, and when none is mapped the last of them stands for the whole group, so a
    complex keeps one unmapped subunit where all of its isozymes lack an ortholog.
    """
    if isinstance(node, ast.Name):
        new_genes = ortho.get(node.id)
        if new_genes:
            return _group("or", list(new_genes))
        return _Unmapped(node.id)

    children = [_rewrite_node(c, ortho) for c in node.values]
    if isinstance(node.op, ast.Or):
        mapped = [c for c in children if not isinstance(c, _Unmapped)]
        if mapped:
            return _group("or", mapped)
        return children[-1]
    return _group("and", children)


def _contains_unmapped(expr) -> bool:
    if isinstance(expr, _Unmapped):
        return True
    return isinstance(expr, tuple) and any(_contains_unmapped(c) for c in expr[1])


def _apply_policy(expr, policy: str, model_id: str):
    """Resolve the unmapped leaves of a rewritten tree under ``complex_policy``.

    ``flag`` turns each into an ``OLD_<model>_<gene>`` placeholder gene, ``keep`` removes
    them (a group left empty disappears with them) and ``drop`` removes the whole reaction
    as soon as one is left. Returns ``None`` when nothing remains.
    """
    if policy == "drop":
        return None if _contains_unmapped(expr) else expr
    if isinstance(expr, _Unmapped):
        return f"OLD_{model_id}_{expr.gene}" if policy == "flag" else None
    if isinstance(expr, str):
        return expr
    op, children = expr
    return _group(op, [c for c in (_apply_policy(c, policy, model_id) for c in children) if c is not None])


def _prune_genes(node, removed: set):
    """Remove template genes from a GPR AST, as RAVEN's removeGenes does.

    Returns the pruned tree, or ``None`` when the rule cannot be satisfied without them
    (a removed gene in an AND, or every branch of an OR removed).
    """
    if isinstance(node, ast.Name):
        return None if node.id in removed else node
    kids = [_prune_genes(c, removed) for c in node.values]
    if isinstance(node.op, ast.And):
        return None if any(k is None for k in kids) else node
    kept = [k for k in kids if k is not None]
    return ast.BoolOp(op=ast.Or(), values=kept) if kept else None


def _group(op: str, children: list):
    """Flatten same-operator groups, drop duplicates, and unwrap a single child."""
    flat: list = []
    for child in children:
        for part in (child[1] if isinstance(child, tuple) and child[0] == op else [child]):
            if part not in flat:
                flat.append(part)
    if not flat:
        return None
    return flat[0] if len(flat) == 1 else (op, flat)


def _render(expr, parent: str | None = None) -> str:
    """GPR string for a rewritten tree; nested groups of the other operator get parentheses."""
    if isinstance(expr, str):
        return expr
    op, children = expr
    text = f" {op} ".join(_render(c, op) for c in children)
    return f"({text})" if parent is not None and parent != op else text


def _rewrite_gpr(rxn, ortho: dict, policy: str, model_id: str, removed: frozenset = frozenset()):
    """Return the rewritten GPR string, or None if the reaction is unsupported.

    ``removed`` holds template genes taken out of the template before the rewrite (genes a
    preferred model already covers); a reaction that cannot work without them is unsupported.
    """
    if not rxn.gene_reaction_rule:
        return None
    # A reaction is only transferred if at least one of its genes has an ortholog.
    if not any(g.id in ortho or g.id in removed for g in rxn.genes):
        return None
    body = rxn.gpr.body
    if removed:
        body = _prune_genes(body, set(removed))
        if body is None:
            return None
    result = _apply_policy(_rewrite_node(body, ortho), policy, model_id)
    return None if result is None else _render(result)


def _strictness_to_flags(strictness: int) -> tuple[bool, bool]:
    """Map strictness 1, 2 or 3 onto (bidirectional, best_hits_only)."""
    if strictness == 1:
        return True, False
    if strictness == 2:
        return False, False
    if strictness == 3:
        return True, True
    raise ValueError(f"strictness must be 1, 2 or 3, got {strictness}")


def _ortholog_map(
    hits, model_for, model_ids, *, bidirectional, best_hits_only, map_direction,
    model_genes, max_evalue, min_align_len, min_identity,
):
    """Build {model_id: {template_gene: [new_gene, ...]}} from the hits table."""
    h = hits[
        (hits.evalue < max_evalue)
        & (hits.align_len >= min_align_len)
        & (hits.identity >= min_identity)
    ]

    if best_hits_only:
        # Every hit tied for the best bitscore is kept, as in RAVEN.
        best = h.groupby(["from_id", "to_id", "from_gene"])["bitscore"].transform("max")
        h = h[h.bitscore == best]

    # Directional views, normalised to (model_id, new_gene, template_gene).
    fwd = (
        h[h.from_id == model_for][["to_id", "from_gene", "to_gene"]]
        .rename(columns={"to_id": "model_id", "from_gene": "new_gene", "to_gene": "template_gene"})
    )
    rev = (
        h[h.to_id == model_for][["from_id", "from_gene", "to_gene"]]
        .rename(columns={"from_id": "model_id", "from_gene": "template_gene", "to_gene": "new_gene"})
    )
    fwd = fwd[fwd.model_id.isin(model_ids)]
    rev = rev[rev.model_id.isin(model_ids)]

    if bidirectional:
        pairs = fwd.merge(rev, on=["model_id", "new_gene", "template_gene"], how="inner")
    elif map_direction == "new_to_old":
        pairs = fwd
    else:
        pairs = rev
    pairs = pairs[["model_id", "new_gene", "template_gene"]].drop_duplicates()
    if pairs.empty:
        return {}

    # Keep only template genes that actually exist in their model. A list
    # comprehension over the columns avoids DataFrame.apply(axis=1)'s per-row
    # Series construction (model_genes values are sets, so membership is O(1)).
    keep = [
        tg in model_genes.get(mid, ())
        for mid, tg in zip(pairs.model_id, pairs.template_gene, strict=True)
    ]
    pairs = pairs[keep]

    ortho: dict = {}
    for model_id, template_gene, new_gene in zip(pairs.model_id, pairs.template_gene, pairs.new_gene, strict=True):
        ortho.setdefault(model_id, {}).setdefault(template_gene, [])
        if new_gene not in ortho[model_id][template_gene]:
            ortho[model_id][template_gene].append(new_gene)
    for per_model in ortho.values():
        for genes in per_model.values():
            genes.sort()
    return ortho


def _apply_preferred_order(ortho: dict, order: list[str]) -> tuple[dict, dict]:
    """Give each new gene to the first model (in order) that maps it, as RAVEN does.

    A template gene of a later model that maps to any new gene an earlier model already
    maps is removed from that model altogether, together with its other mappings. The
    new genes of every model count as taken, including those of its removed genes.
    Returns the pruned mapping and ``{model_id: {removed template genes}}``.
    """
    taken: set = set()
    pruned: dict = {mid: {} for mid in ortho}
    removed: dict = {mid: set() for mid in ortho}
    for position, model_id in enumerate(order):
        per_model = ortho.get(model_id, {})
        for template_gene, new_genes in per_model.items():
            if position > 0 and taken.intersection(new_genes):
                removed[model_id].add(template_gene)
            else:
                pruned[model_id][template_gene] = new_genes
        taken.update(ng for new_genes in per_model.values() for ng in new_genes)
    return pruned, removed


def get_model_from_homology(
    models,
    hits: pd.DataFrame,
    model_for: str,
    *,
    preferred_order=None,
    strictness: int = 1,
    map_direction: str = "new_to_old",
    complex_policy: str = "flag",
    only_genes_in_models: bool = False,
    max_evalue: float = 1e-30,
    min_align_len: int = 100,
    min_identity: float = 40,
    review_identity: float | None = None,
    keep_gene_free: bool = False,
    preserve_notes: bool = False,
) -> HomologyResult:
    """Build a draft model for ``model_for`` by transferring reactions from templates.

    Parameters that change the result:

    * ``strictness`` (default 1) selects the hit-filtering policy:

      - 1: bidirectional. A template gene and a new-organism gene must have
        an accepted hit in both directions.
      - 2: one-directional. A hit found in a single direction is accepted; the
        direction is selected by ``map_direction``.
      - 3: bidirectional and best hits only. Before matching, each gene keeps
        only its best hits (by bitscore) in each direction, all of them when
        several tie, then the bidirectional rule is applied.

    * ``map_direction`` selects the direction of the one-directional search when
      ``strictness`` is 2: ``"new_to_old"`` (default, hits found
      searching from ``model_for``) or ``"old_to_new"``.
    * ``complex_policy`` decides what happens to an AND-linked subunit with no
      ortholog: ``"flag"`` (default) keeps the reaction with a placeholder
      gene (``OLD_<model>_<gene>``) for later curator review, ``"keep"`` removes that
      subunit, ``"drop"`` drops the whole reaction. An isozyme group counts as a subunit
      without an ortholog only when none of its members has one, and then one placeholder
      stands for it. ``"drop"`` removes the reaction whenever a placeholder would remain,
      also when another isozyme branch is complete.
    * ``only_genes_in_models`` restricts the hits table to hits whose template
      gene appears in the given ``models`` before mapping orthologs, so a gene
      the templates lack cannot be a best hit. It raises ``ValueError`` when no
      hit of a search direction between ``model_for`` and a template passes
      the cutoffs and this filter. As in RAVEN, it changes the mapping only with
      ``strictness`` 3: with 1 and 2 a template gene that no template contains is dropped in
      any case, and only the ``ValueError`` differs.
    * ``keep_gene_free`` (default False) also transfers the template reactions that have no
      GPR (spontaneous, transport, exchange, artificial). By default they are dropped, as in
      RAVEN, because no gene supports them in the new organism. True transfers them,
      which suits a curated reference template whose gene-free reactions belong in every
      derived model.
    * ``preserve_notes`` (default False) keeps each transferred reaction's own ``notes``
      (references, confidence score, free text) and only adds ``homology_source``. By default
      they are replaced by a note that the reaction was included by this function.
    * ``preferred_order`` breaks ties when more than one template maps the
      same new-organism gene: the gene is mapped to the earlier template in the list, and a
      gene of a later template that maps to any gene an earlier template already maps
      is removed from that template altogether, with its other mappings. A reaction of
      the later template that cannot work without the removed gene is not transferred.
      Templates are merged in the order they were given in ``models``, whatever the
      preferred order is, and a template that ``preferred_order`` leaves out contributes
      nothing (default: the order of ``models``). RAVEN's ``preferredOrder`` decides which
      template receives a gene in the same way and merges the models in the order of ``models``.

    Organism ids in ``hits`` are case-sensitive. An id that matches
    ``model_for`` or a template id only when case is ignored raises
    ``ValueError``; hits involving other organisms are ignored. RAVEN raises an error for
    hits from an organism that no template has as ``model.id``, and matches the template ids
    case-insensitively.

    A hit passes when its e-value is below ``max_evalue`` and its alignment length and
    identity are at least ``min_align_len`` and ``min_identity``, as in RAVEN.

    Defaults for the three filters come from a `calibration against KEGG annotations and OMA assignments
    <https://github.com/CompMetEng/raven-docs/blob/main/docs/parameter-tuning/studies/homology-cutoff-calibration.md>`_
    on raven-docs, scored with precision weighted above recall (a wrongly transferred
    reaction is harder to undo than a missing one):

    * ``min_identity`` 40 lies in the 35 to 45 range that the two reference sources favour;
      35 scores higher for two organisms (``yli`` 0.868 against 0.860, ``ani`` 0.814 against
      0.804). It is the filter that limits the result.
    * ``min_align_len`` 100, as RAVEN's ``minLen``. Values at or below 150 measured the
      same; on each of the four organisms tested, the score against KEGG annotations (wrong
      matches counted double) was 0.010 to 0.020 lower at 200 than at 100.
    * ``max_evalue`` makes no difference between 1e-4 and 1e-50, because the other two
      filters exclude every hit that it would exclude; 1e-30 is the RAVEN value.

    ``review_identity`` (for example 25) collects the reactions that the thresholds
    reject. Any reaction that transfers at this lower identity but not at ``min_identity``
    is reported in ``HomologyResult.candidates`` with its strongest supporting hit and is
    not added to the model. A missing reaction can be gap-filled, whereas a wrongly
    transferred reaction is hard to find and remove, so the filters stay strict. It must be
    below ``min_identity``, otherwise ``ValueError`` is raised.
    """
    if isinstance(models, cobra.Model):
        models = [models]
    if complex_policy not in ("flag", "keep", "drop"):
        raise ValueError(f"complex_policy must be flag/keep/drop, got {complex_policy!r}")
    if map_direction not in ("new_to_old", "old_to_new"):
        raise ValueError(f"map_direction must be new_to_old/old_to_new, got {map_direction!r}")
    if review_identity is not None and review_identity >= min_identity:
        raise ValueError(
            f"review_identity ({review_identity}) must be below min_identity "
            f"({min_identity}); it exists to catch what min_identity rejects."
        )
    bidirectional, best_hits_only = _strictness_to_flags(strictness)
    validate_hits(hits)
    if review_identity is not None and hits.attrs.get("source") == "SuppOrth":
        warnings.warn(
            "review_identity has no effect on SuppOrth hits: their identity is a placeholder "
            "of 100, so no reaction is a candidate. Filter on SuppOrth's own identity with "
            "hits_from_supporth(min_identity=...) instead.",
            stacklevel=2,
        )

    model_by_id = {m.id: m for m in models}
    model_ids = list(model_by_id)
    # An organism id that differs from model_for or a template id only in case
    # would silently be read as another organism; RAVEN rejects it too.
    known = {model_for, *model_ids}
    known_lower = {k.lower() for k in known}
    miscased = sorted(
        i for i in set(hits.from_id) | set(hits.to_id)
        if i not in known and str(i).lower() in known_lower
    )
    if miscased:
        raise ValueError(
            f"hits contain organism ids that match model_for ({model_for!r}) or a template "
            f"model id ({model_ids}) only when case is ignored: {miscased}. Ids are case-sensitive."
        )
    model_genes = {mid: {g.id for g in m.genes} for mid, m in model_by_id.items()}
    all_model_genes = set().union(*model_genes.values()) if model_genes else set()

    # Sanity: each template should overlap the hits by >=5% of its genes.
    for mid, genes in model_genes.items():
        in_hits = genes & (set(hits.from_gene) | set(hits.to_gene))
        if genes and len(in_hits) < 0.05 * len(genes):
            warnings.warn(
                f"<5% of genes in template '{mid}' appear in the hits table; "
                "check that the FASTA and model use the same gene identifiers.",
                stacklevel=2,
            )

    if only_genes_in_models:
        # The template-side gene of each hit, as in RAVEN.
        template_gene = hits.to_gene.where(hits.from_id == model_for, hits.from_gene)
        kept = template_gene.isin(all_model_genes)
        # As in RAVEN, a search direction with no hit left after the cutoffs and
        # this filter is an error rather than an empty mapping.
        passes = (
            kept
            & (hits.evalue < max_evalue)
            & (hits.align_len >= min_align_len)
            & (hits.identity >= min_identity)
        )
        directions = hits[["from_id", "to_id"]].drop_duplicates().itertuples(index=False)
        for from_id, to_id in directions:
            if model_for not in (from_id, to_id) or not {from_id, to_id} & set(model_ids):
                continue
            if not passes[(hits.from_id == from_id) & (hits.to_id == to_id)].any():
                raise ValueError(
                    f"No genes in matching from {from_id} to {to_id} are present in the "
                    "corresponding model"
                )
        hits = hits[kept]

    ortho = _ortholog_map(
        hits, model_for, model_ids, bidirectional=bidirectional, best_hits_only=best_hits_only,
        map_direction=map_direction,
        model_genes=model_genes, max_evalue=max_evalue, min_align_len=min_align_len,
        min_identity=min_identity,
    )
    mapped = {ng for per in ortho.values() for ngs in per.values() for ng in ngs}
    unusable = sorted(g for g in mapped if _GPR_SPLIT.search(str(g)))
    if unusable:
        raise ValueError(
            f"{len(unusable)} gene ids of {model_for!r} cannot be written into a GPR, because "
            f"cobra splits them at whitespace, parentheses, '|' or '&': {unusable[:3]}. Map "
            "them to gene ids first, for example with protein_to_gene in hits_from_supporth, "
            "or use the accession alone for UniProt headers such as 'sp|P33284|HXK_KLULA'."
        )

    order = [str(x) for x in preferred_order] if preferred_order else model_ids
    removed: dict = {}
    if preferred_order and len(models) > 1:
        ortho, removed = _apply_preferred_order(ortho, order)

    draft = _transfer(model_by_id, order, ortho, model_for, model_ids, complex_policy,
                      keep_gene_free, preserve_notes, removed)

    candidates = None
    if review_identity is not None:
        # Reactions that only a looser identity would have transferred: reported
        # for review, never added. Rejecting a real ortholog and silently binning
        # the evidence are different things, and only the first is intended.
        loose_ortho = _ortholog_map(
            hits, model_for, model_ids, bidirectional=bidirectional,
            best_hits_only=best_hits_only, map_direction=map_direction,
            model_genes=model_genes, max_evalue=max_evalue,
            min_align_len=min_align_len, min_identity=review_identity,
        )
        loose_removed: dict = {}
        if preferred_order and len(models) > 1:
            loose_ortho, loose_removed = _apply_preferred_order(loose_ortho, order)
        loose = _transfer(
            model_by_id, order, loose_ortho, model_for, model_ids, complex_policy,
            keep_gene_free, preserve_notes, loose_removed,
        )
        extra = {r.id for r in loose.reactions} - {r.id for r in draft.reactions}
        candidates = _candidate_evidence(
            extra, model_by_id, order, loose_ortho, hits,
            max_evalue=max_evalue, min_align_len=min_align_len,
            min_identity=review_identity,
        )

    return HomologyResult(model=draft, gene_map=ortho, candidates=candidates)


def _transfer(model_by_id, order, ortho, model_for, model_ids, complex_policy,
              keep_gene_free=False, preserve_notes=False, removed=None) -> cobra.Model:
    """Assemble the draft from the reactions of each template whose GPRs survive rewriting.

    The templates are merged in the order of ``model_ids`` (the order they were given),
    whatever the preferred order is; reaction ids shared between templates are resolved
    by that merge. Templates left out of ``order`` contribute nothing.
    """
    removed = removed or {}
    transferred = []
    for mid in model_ids:
        model = model_by_id.get(mid)
        if model is None or mid not in order:
            continue
        per_model = ortho.get(mid, {})
        m = model.copy()
        keep: dict[str, str] = {}
        for rxn in m.reactions:
            new_gpr = _rewrite_gpr(rxn, per_model, complex_policy, mid, frozenset(removed.get(mid, ())))
            if new_gpr is None and keep_gene_free and not rxn.gene_reaction_rule:
                new_gpr = ""
            if new_gpr is not None:
                keep[rxn.id] = new_gpr
        m.remove_reactions([r for r in m.reactions if r.id not in keep], remove_orphans=True)
        for rid, gpr in keep.items():
            r = m.reactions.get_by_id(rid)
            set_gene_rule(r, gpr)
            if preserve_notes:
                r.notes = {**r.notes, "homology_source": mid}
            else:
                r.notes = {"note": "Included by get_model_from_homology", "confidence_score": 2,
                           "homology_source": mid}
        if m.reactions:
            transferred.append(m)

    draft = merge_models(transferred, match_by="name") if transferred else cobra.Model()
    draft.id = model_for
    draft.name = "Generated by get_model_from_homology using " + ", ".join(model_ids)
    # The draft is a new model: the template's version, date and description are not its own.
    if draft.notes:
        draft.notes = {k: v for k, v in draft.notes.items() if k not in ("metaData", "version")}

    # Drop genes left without a reaction.
    for g in [g for g in draft.genes if not g.reactions]:
        draft.genes.remove(g)
    return draft


def _candidate_evidence(
    reaction_ids, model_by_id, order, ortho, hits, *,
    max_evalue, min_align_len, min_identity,
) -> pd.DataFrame:
    """One row per candidate reaction, with the hit that limits its identity.

    The reported identity is the lower of the two directions. A bidirectional match has
    to reach the threshold in both directions, so a pair with a forward hit of 44 % and a
    reverse hit of 30 % is rejected on the 30 %, and the 30 % is reported.

    Rows are sorted by identity, highest first.
    """
    usable = hits[
        (hits.evalue < max_evalue)
        & (hits.align_len >= min_align_len)
        & (hits.identity >= min_identity)
    ]
    # Weakest hit per ordered (gene, gene) pair, then the weaker of the two
    # directions: whichever value the threshold actually acted on.
    weakest: dict[tuple[str, str], tuple] = {}
    for row in usable.itertuples():
        key = (row.from_gene, row.to_gene)
        current = weakest.get(key)
        if current is None or row.identity < current[0]:
            weakest[key] = (row.identity, row.align_len, row.evalue, row.bitscore)

    def limiting(template_gene: str, target_gene: str):
        both = [
            weakest.get((template_gene, target_gene)),
            weakest.get((target_gene, template_gene)),
        ]
        found = [x for x in both if x is not None]
        return min(found, key=lambda x: x[0]) if found else None

    rows = []
    for rid in sorted(reaction_ids):
        for mid in order:
            model = model_by_id.get(mid)
            if model is None or rid not in model.reactions:
                continue
            per_model = ortho.get(mid, {})
            support = []
            for gene in (g.id for g in model.reactions.get_by_id(rid).genes):
                for target_gene in per_model.get(gene, ()):
                    evidence = limiting(gene, target_gene)
                    if evidence is not None:
                        support.append((gene, target_gene, *evidence))
            if not support:
                continue
            support.sort(key=lambda s: s[2], reverse=True)  # by identity
            gene, target_gene, identity, align_len, evalue, bitscore = support[0]
            rows.append({
                "reaction": rid, "template_model": mid, "template_gene": gene,
                "target_gene": target_gene, "identity": identity,
                "align_len": align_len, "evalue": evalue, "bitscore": bitscore,
                "n_support": len(support),
            })
            break  # first template in preferred order wins, as for the draft

    frame = pd.DataFrame(rows, columns=[
        "reaction", "template_model", "template_gene", "target_gene",
        "identity", "align_len", "evalue", "bitscore", "n_support",
    ])
    return frame.sort_values("identity", ascending=False).reset_index(drop=True)
