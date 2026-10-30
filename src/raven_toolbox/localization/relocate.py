"""Apply a curator's firm compartment decision and repair the model to keep it consistent.

:func:`relocate_reactions` takes a curated map ``{reaction_id: compartment}`` — the curator asserting
"this reaction belongs *here*" — and produces a new :class:`~raven_toolbox.localization.AssignmentProposal`
that honours those pins **and makes the consequential changes** needed for the new localisation to work:

* **gene consistency** — an enzyme localises to one compartment, so other reactions fully explained by
  the moved genes are co-moved with it (reported, and disableable);
* **transports added** — a moved reaction's metabolites are bridged into/out of the new compartment
  where they now must cross a membrane;
* **transports removed** — transports that only served the reaction's *old* compartment and are now
  orphaned (no reaction touches that species there any more) are dropped;
* **re-certification** — the repaired model is materialised and its growth checked, so a decision that
  breaks functionality is reported rather than silently shipped.

The transport repair is deliberately **surgical**: only species touched by the moved reactions are
reconsidered, so the diff is attributable to the curator's edit and unrelated placements are never
churned (unlike a full re-solve). It reuses :func:`assign_compartments`'s materialisation and the same
star-topology hub, so the result feeds straight back into :func:`apply_assignment`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cobra

from raven_toolbox.localization.assign import (
    AssignmentProposal,
    _base_met,
    _check_unique_keys,
    apply_assignment,
)
from raven_toolbox.localization.curation import _is_impermeant

__all__ = ["RelocationResult", "relocate_reactions"]


@dataclass
class RelocationResult:
    """Outcome of :func:`relocate_reactions`: the repaired proposal plus a log of every change.

    ``moved`` maps every relocated reaction id to ``(from_compartment, to_compartment)`` (the curator's
    pins and any gene-linked co-moves); ``co_moved`` lists just the reactions moved as a gene-consistency
    consequence. ``transports_added``/``transports_removed`` are ``(metabolite_base, compartment)``
    tuples. ``growth_before``/``growth_after`` are the materialised model's objective value before and
    after; ``certified`` is True when the relocated model still meets the growth floor.
    """

    proposal: AssignmentProposal
    moved: dict[str, tuple[str | None, str]] = field(default_factory=dict)
    co_moved: list[str] = field(default_factory=list)
    transports_added: list[tuple[str, str]] = field(default_factory=list)
    transports_removed: list[tuple[str, str]] = field(default_factory=list)
    growth_before: float = 0.0
    growth_after: float = 0.0
    min_growth: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def certified(self) -> bool:
        return self.growth_after > 1e-9 and self.growth_after >= self.min_growth - 1e-9

    def apply(self, model: cobra.Model, *, base_metabolite=None, default_compartment: str = "c",
              universal: cobra.Model | None = None) -> cobra.Model:
        """Materialise the repaired proposal into a new compartmentalised model."""
        return apply_assignment(model, self.proposal, default_compartment=default_compartment,
                                base_metabolite=base_metabolite, universal=universal)

    def summary(self) -> str:
        """Human-readable report of the relocation: reactions moved, transports changed, and the
        growth outcome."""
        lines = [f"relocated {len(self.moved)} reaction(s) "
                 f"({len(self.co_moved)} co-moved for gene consistency)"]
        for rid, (frm, to) in self.moved.items():
            tag = " (gene-linked)" if rid in self.co_moved else ""
            lines.append(f"  {rid}: {frm} -> {to}{tag}")
        if self.transports_added:
            lines.append(f"transports added: {len(self.transports_added)}  "
                         + ", ".join(f"{b}->{c}" for b, c in self.transports_added[:8])
                         + (" ..." if len(self.transports_added) > 8 else ""))
        if self.transports_removed:
            lines.append(f"transports removed: {len(self.transports_removed)}  "
                         + ", ".join(f"{b}@{c}" for b, c in self.transports_removed[:8])
                         + (" ..." if len(self.transports_removed) > 8 else ""))
        lines.append(f"growth {self.growth_before:.4g} -> {self.growth_after:.4g} "
                     f"(floor {self.min_growth:.4g}); certified={self.certified}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def relocate_reactions(
    model: cobra.Model,
    proposal: AssignmentProposal,
    decisions: dict[str, str],
    *,
    base_metabolite: Callable[[cobra.Metabolite], str] | None = None,
    default_compartment: str = "c",
    biomass_reaction: str | None = None,
    min_growth: float | None = None,
    universal: cobra.Model | None = None,
    move_gene_siblings: bool = True,
    prune_orphan_transports: bool = True,
) -> RelocationResult:
    """Firmly place reactions in curator-chosen compartments and repair the model around the decision.

    ``decisions`` maps reaction ids to the compartment the curator wants them in. Returns a
    :class:`RelocationResult` with the repaired proposal and a log of every secondary change
    (gene-linked co-moves, transports added/removed, and the growth certificate). ``model`` and
    ``proposal`` are the draft and its current assignment (as returned by
    :func:`assign_compartments`); pass the same ``base_metabolite``/``universal``.

    ``move_gene_siblings`` (default True) co-moves any *other* reaction whose entire gene set now points
    to a single moved-to compartment — enforcing "one enzyme, one compartment" — and reports them;
    isozyme-shared reactions (genes split across compartments) are left in place. ``prune_orphan_transports``
    removes a transport of species ``b`` at a reaction's old compartment when nothing else uses ``b``
    there any more. The transport repair only touches species of the moved reactions.
    """
    base = base_metabolite if base_metabolite is not None else _base_met
    _check_unique_keys(model.metabolites, base)
    floor = float(min_growth if min_growth is not None else proposal.min_growth)
    warnings: list[str] = []

    # current placement -- the FULL compartment list -- of every reaction that has one
    placed_now: dict[str, list[str]] = {rid: list(cs) for rid, cs in proposal.placements.items() if cs}

    def comps_now(rid: str) -> list[str]:
        """Every compartment the reaction currently occupies: its placement list, or (for a reaction
        pinned in the model, not in the proposal) the compartments of its metabolites."""
        if rid in placed_now:
            return placed_now[rid]
        if rid in model.reactions:
            return sorted({m.compartment for m in model.reactions.get_by_id(rid).metabolites
                           if m.compartment})
        return []

    def _from(rid: str) -> str | None:
        cs = comps_now(rid)
        return "+".join(cs) if cs else None

    def _growth(p: AssignmentProposal) -> float:
        applied = apply_assignment(model, p, default_compartment=default_compartment,
                                   base_metabolite=base, universal=universal)
        if biomass_reaction and biomass_reaction in applied.reactions:
            applied.objective = biomass_reaction
        return float(applied.slim_optimize(error_value=0.0) or 0.0)

    def _copy(p: AssignmentProposal) -> AssignmentProposal:
        return AssignmentProposal(
            placements={rid: list(cs) for rid, cs in p.placements.items()},
            gene_compartments={g: list(cs) for g, cs in p.gene_compartments.items()},
            added_transports=list(p.added_transports), added_reactions=list(p.added_reactions),
            unplaced_reactions=list(p.unplaced_reactions), min_growth=p.min_growth, status=p.status)

    # ---- validate + normalise the curator's decisions ----
    decided: dict[str, str] = {}
    for rid, comp in decisions.items():
        if rid not in model.reactions:
            warnings.append(f"{rid}: not in model; skipped")
            continue
        if model.reactions.get_by_id(rid).boundary:
            warnings.append(f"{rid}: boundary/exchange reaction; skipped")
            continue
        if comps_now(rid) == [comp]:
            continue  # already there, and only there
        decided[rid] = comp

    moved: dict[str, tuple[str | None, str]] = {rid: (_from(rid), c) for rid, c in decided.items()}
    co_moved: list[str] = []

    # ---- gene-consistency: co-move reactions fully explained by the moved genes ----
    if move_gene_siblings and decided:
        gene_target: dict[str, str] = {}
        conflicted: set[str] = set()
        for rid, comp in decided.items():
            for g in model.reactions.get_by_id(rid).genes:
                if g.id in gene_target and gene_target[g.id] != comp:
                    conflicted.add(g.id)
                gene_target[g.id] = comp
        candidates: set[str] = set()
        for gid in gene_target:
            if gid in conflicted or gid not in model.genes:
                continue
            for r in model.genes.get_by_id(gid).reactions:
                candidates.add(r.id)
        for rid in candidates:
            if rid in decided:
                continue
            r = model.reactions.get_by_id(rid)
            if r.boundary or not r.genes:
                continue
            if len(proposal.placements.get(rid, [])) > 1:
                continue  # don't collapse a deliberately dual-localised reaction by gene inference
            # a conflicted gene is ambiguous: inject None so the reaction is not unanimous -> not moved
            targets = {(None if g.id in conflicted else gene_target.get(g.id)) for g in r.genes}
            if len(targets) == 1 and None not in targets:
                to = targets.pop()
                if to is not None and comps_now(rid) != [to]:
                    moved[rid] = (_from(rid), to)
                    co_moved.append(rid)

    if not moved:
        g = _growth(proposal)
        return RelocationResult(proposal=_copy(proposal), growth_before=g, growth_after=g,
                                min_growth=floor, warnings=warnings or ["no reaction relocated"])

    # ---- apply the moves: a curator/gene decision is a firm SINGLE-compartment placement; every other
    # reaction keeps its full (possibly dual-localised) placement list unchanged. ----
    new_place: dict[str, list[str]] = {rid: list(cs) for rid, cs in placed_now.items()}
    for rid, (_frm, to) in moved.items():
        new_place[rid] = [to]

    def comps_new(rid: str) -> list[str]:
        if rid in new_place:
            return new_place[rid]
        if rid in model.reactions:
            return sorted({m.compartment for m in model.reactions.get_by_id(rid).metabolites
                           if m.compartment})
        return []

    # ---- incidence of the moved reactions' species under the new layout (all compartments of every
    # reaction, so dual-localised copies and boundary anchors are both counted) ----
    affected: set[str] = set()
    name_of: dict[str, str] = {}
    for rid in moved:
        for m in model.reactions.get_by_id(rid).metabolites:
            b = base(m)
            affected.add(b)
            name_of.setdefault(b, m.name or m.id)
    incidence: dict[str, set[str]] = {b: set() for b in affected}
    for r in model.reactions:
        cs = comps_new(r.id)
        if not cs:
            continue
        for m in r.metabolites:
            b = base(m)
            if b in incidence:
                incidence[b].update(cs)

    # ---- surgical transport reconciliation (only species of the moved reactions) ----
    current = set(proposal.added_transports)
    adds: set[tuple[str, str]] = set()
    removes: set[tuple[str, str]] = set()
    for rid, (_frm, to) in moved.items():
        left = [c for c in comps_now(rid) if c != to]  # compartments the reaction no longer occupies
        for m in model.reactions.get_by_id(rid).metabolites:
            b = base(m)
            # bridge the species into the new compartment if it also lives elsewhere and has no transport
            if to != default_compartment and len(incidence.get(b, ())) >= 2 and (b, to) not in current:
                adds.add((b, to))
            # drop a transport at each vacated compartment that nothing touches there any more
            if prune_orphan_transports:
                for oc in left:
                    if (oc != default_compartment and (b, oc) in current
                            and oc not in incidence.get(b, set())):
                        removes.add((b, oc))
    new_transports = (current - removes) | adds

    # ---- gene -> compartment map recomputed from the new layout ----
    gene_comps: dict[str, set[str]] = {}
    for rid, cs in new_place.items():
        for c in cs:
            for g in model.reactions.get_by_id(rid).genes:
                gene_comps.setdefault(g.id, set()).add(c)

    new_prop = AssignmentProposal(
        placements={rid: list(cs) for rid, cs in new_place.items()},
        gene_compartments={g: sorted(cs) for g, cs in gene_comps.items()},
        added_transports=sorted(new_transports),
        added_reactions=list(proposal.added_reactions),
        unplaced_reactions=list(proposal.unplaced_reactions),
        min_growth=floor,
        status="curated",
    )

    g_before = _growth(proposal)
    g_after = _growth(new_prop)

    for b, c in sorted(adds):
        if _is_impermeant(b, name_of.get(b, b)):
            warnings.append(f"added transport of impermeant species {name_of.get(b, b)} into {c} "
                            f"(a shuttle, not a free transporter, may be the biological reality)")
    if floor and g_after < floor - 1e-9:
        warnings.append(f"growth {g_after:.4g} is below the floor {floor:.4g} after this relocation; "
                        f"the decision may need a gap-fill or a different compartment")
    elif g_after <= 1e-9:
        warnings.append("the materialised model does not grow after this relocation")

    # NB: growth stored raw (unrounded) so `certified` reflects a genuine tiny-but-positive growth
    return RelocationResult(
        proposal=new_prop, moved=moved, co_moved=co_moved,
        transports_added=sorted(adds), transports_removed=sorted(removes),
        growth_before=g_before, growth_after=g_after, min_growth=floor, warnings=warnings)
