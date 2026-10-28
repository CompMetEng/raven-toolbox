"""Shared data structures and materialisation for compartment assignment.

This module holds the pieces the compartment-assignment method builds on, independent of how
placement is decided:

* :class:`AssignmentProposal` — the outcome of an assignment (per-reaction / per-gene compartments,
  added transports and gap-fill reactions, and a growth certificate);
* :class:`GrowthCondition` — a soft multi-medium growth requirement;
* :func:`apply_assignment` — materialise a proposal into a compartmentalised :class:`cobra.Model`
  (a deep copy): move reactions to their compartments, duplicate them for multi-localisation, add the
  star-topology transports, and pull in any gap-fill reactions.

Placement itself is decided by :func:`raven_toolbox.localization.assign_compartments`, whose acceptance
test is a real FBA on the materialised model — so :func:`apply_assignment` (the materialised model) is
authoritative for functionality. Localisation
evidence is an agnostic ``gene x compartment`` score table
(:class:`raven_toolbox.localization.scores.LocalizationScores`) — any predictor or database.

Built on raven-toolbox + cobra.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import cobra

from raven_toolbox.manipulation.compartments import MERGED_BASE_ID_NOTE

__all__ = ["AssignmentProposal", "GrowthCondition", "apply_assignment"]


# --------------------------------------------------------------------------- helpers
def _base_met(m: cobra.Metabolite) -> str:
    """The default compartment-agnostic key of a metabolite.

    The base id that :func:`~raven_toolbox.manipulation.merge_compartments` stored in
    ``notes["merged_base_id"]``, if there is one; otherwise the id without the trailing
    ``_<compartment>`` suffix of the metabolite's own compartment.
    """
    stored = (m.notes or {}).get(MERGED_BASE_ID_NOTE)
    if isinstance(stored, str) and stored:
        return stored
    if m.compartment and m.id.endswith(f"_{m.compartment}"):
        return m.id[: -(len(m.compartment) + 1)]
    return m.id


def _check_unique_keys(metabolites: Iterable[cobra.Metabolite],
                       base: Callable[[cobra.Metabolite], str]) -> None:
    """Raise :class:`ValueError` when two metabolites in one compartment have the same key, which
    would make them one species."""
    seen: dict[tuple[str, str], str] = {}
    clashes: list[str] = []
    for m in metabolites:
        key = (base(m), m.compartment)
        if key in seen and seen[key] != m.id:
            clashes.append(f"{seen[key]} and {m.id} (key {key[0]!r} in {key[1]!r})")
        else:
            seen.setdefault(key, m.id)
    if clashes:
        raise ValueError("metabolites with the same key in one compartment: "
                         + "; ".join(clashes[:5])
                         + "; pass a base_metabolite that tells them apart")


def _rxn_compartment(rxn: cobra.Reaction) -> str | None:
    comps = {m.compartment for m in rxn.metabolites if m.compartment}
    return next(iter(comps)) if len(comps) == 1 else None


@dataclass(frozen=True)
class GrowthCondition:
    """An extra medium the placement must also support, checked when certifying functionality.

    :func:`assign_compartments` materialises the placement and confirms it reaches
    ``min_growth`` by a real FBA not only on the primary medium but on each ``GrowthCondition`` medium
    too — so a transport or placement has to serve **every** condition that needs it, which keeps
    alternative-carbon-source pathways (respiration, β-oxidation, the glyoxylate cycle, ...) genuinely
    connected rather than stranded in flux-dead islands that single-medium growth cannot see.

    Parameters
    ----------
    name:
        Short label for the condition, used in the per-condition growth report. Distinct per condition.
    medium:
        ``exchange_reaction_id -> max uptake (>= 0)``. Exchanges not listed are closed for uptake,
        so ``medium`` fully specifies the environment (carbon source *and* supplements).
    min_growth:
        Biomass floor this condition must reach on the materialised model.
    """

    name: str
    medium: Mapping[str, float]
    min_growth: float


@dataclass
class AssignmentProposal:
    """Outcome of :func:`assign_compartments`.

    Attributes
    ----------
    placements:
        ``reaction_id -> [compartments]`` the reaction is assigned to (one or more).
    gene_compartments:
        ``gene_id -> [compartments]``.
    added_transports:
        ``(base_metabolite_id, compartment)`` pairs needing a transport to/from
        ``default_compartment``.
    added_reactions:
        ids of gap-fill reactions pulled from the ``universal`` model to keep the
        compartmentalised network functional.
    unplaced_reactions:
        relocate-set reactions with no scored gene support (placed by function only).
    objective:
        placement-master objective (localisation score − transport − multi-penalty − gap-fill cost).
    min_growth:
        the growth floor enforced.
    status:
        solver status (``"optimal"``, ``"infeasible"``, ...) plus ``"certified"`` / ``"uncertified"``
        — the placement's functionality was (or was not) confirmed by a real FBA on the materialised
        model.
    certified:
        ``True`` iff ``apply_assignment`` + FBA reaches the growth floor on **every** medium.
    growths:
        set by :func:`assign_compartments`: the materialised biomass flux per medium
        (``"__primary__"`` plus each :class:`GrowthCondition` name), so the caller can see *how far*
        an uncertified placement fell short without re-running FBA.
    """

    placements: dict[str, list[str]] = field(default_factory=dict)
    gene_compartments: dict[str, list[str]] = field(default_factory=dict)
    added_transports: list[tuple[str, str]] = field(default_factory=list)
    added_reactions: list[str] = field(default_factory=list)
    unplaced_reactions: list[str] = field(default_factory=list)
    objective: float = 0.0
    min_growth: float = 0.0
    status: str = "not_solved"
    certified: bool = False
    growths: dict[str, float] = field(default_factory=dict)


def apply_assignment(
    model: cobra.Model, proposal: AssignmentProposal, *, default_compartment: str = "c",
    base_metabolite: Callable[[cobra.Metabolite], str] | None = None,
    universal: cobra.Model | None = None,
) -> cobra.Model:
    """Build the compartmentalised model from a proposal (deep copy; original untouched).

    ``default_compartment`` is the hub compartment star-topology transports route through. Pass the
    same ``base_metabolite`` and ``universal`` used in :func:`assign_compartments` so the same
    compartment-agnostic keying is used (existing per-compartment metabolites are reused rather than
    duplicated) and the chosen gap-fill reactions are added. The default key is the base id that
    :func:`~raven_toolbox.manipulation.merge_compartments` stored on a merged metabolite, otherwise
    the id without the suffix of the metabolite's own compartment. :class:`ValueError` when two
    metabolites in one compartment of ``model`` have the same key.
    """
    out = model.copy()
    base = base_metabolite if base_metabolite is not None else _base_met
    _check_unique_keys(out.metabolites, base)

    # (base key, compartment) -> existing metabolite, so a reaction moved into a compartment
    # reuses the species already there instead of creating a parallel duplicate.
    index: dict[tuple[str, str], cobra.Metabolite] = {}
    for m in out.metabolites:
        index.setdefault((base(m), m.compartment), m)

    def resolve(template: cobra.Metabolite, compartment: str) -> cobra.Metabolite:
        key = (base(template), compartment)
        met = index.get(key)
        if met is None:
            new_id = f"{template.id}__{compartment}"
            if new_id in out.metabolites:
                met = out.metabolites.get_by_id(new_id)
            else:
                met = cobra.Metabolite(new_id, name=template.name, compartment=compartment,
                                       formula=template.formula, charge=template.charge)
                if MERGED_BASE_ID_NOTE in (template.notes or {}):
                    met.notes[MERGED_BASE_ID_NOTE] = template.notes[MERGED_BASE_ID_NOTE]
                out.add_metabolites([met])
            index[key] = met
        return met

    for rid, comps in proposal.placements.items():
        if not comps:
            continue
        rxn = out.reactions.get_by_id(rid)
        _move_reaction(rxn, comps[0], resolve)
        for extra in comps[1:]:
            _duplicate_reaction(out, rxn, extra, resolve)
    for n, (b, c) in enumerate(proposal.added_transports):
        _add_transport(out, b, c, default_compartment, index, resolve, n)
    if universal is not None:
        for rid in proposal.added_reactions:
            _add_universal_reaction(out, universal.reactions.get_by_id(rid), resolve)
    return out


def _add_universal_reaction(model, urxn, resolve):
    if urxn.id in model.reactions:
        return
    new = cobra.Reaction(urxn.id, name=urxn.name,
                         lower_bound=urxn.lower_bound, upper_bound=urxn.upper_bound)
    model.add_reactions([new])
    # Resolve each metabolite through the same (base id, compartment) lookup `resolve` gives relocated
    # reactions, rather than matching the candidate's id verbatim. A relocated reaction materialises its
    # non-default-compartment species under a *generated* id (`A_c__m`), never the universal database's
    # own `A_m`; keying on the id would therefore miss the existing node and create a second, parallel
    # copy, leaving the gap-fill reaction a disconnected island. Keying on (base, compartment) lands on
    # the species already there.
    new.add_metabolites({resolve(m, m.compartment): coeff for m, coeff in urxn.metabolites.items()})
    new.gene_reaction_rule = urxn.gene_reaction_rule


def _move_reaction(rxn, compartment, resolve):
    old = list(rxn.metabolites.items())
    if all(m.compartment == compartment for m, _ in old):
        return
    rxn.subtract_metabolites(dict(old))
    rxn.add_metabolites({resolve(m, compartment): coeff for m, coeff in old})


def _duplicate_reaction(model, rxn, compartment, resolve):
    new_id = f"{rxn.id}_{compartment}"
    if new_id in model.reactions:
        return
    dup = cobra.Reaction(new_id, name=rxn.name,
                         lower_bound=rxn.lower_bound, upper_bound=rxn.upper_bound)
    model.add_reactions([dup])
    dup.add_metabolites({resolve(m, compartment): coeff for m, coeff in rxn.metabolites.items()})
    dup.gene_reaction_rule = rxn.gene_reaction_rule


def _add_transport(model, base_key, compartment, default_compartment, index, resolve, n):
    src = index.get((base_key, default_compartment))
    if src is None:
        # No default-compartment representative of this base yet — create the star hub from any
        # existing per-compartment copy. Without this, a pool split across two *non-default*
        # compartments (e.g. an organelle-local intermediate in ``m`` and ``p`` but never ``c``)
        # would have both its transports silently dropped and stay severed, so the built model would
        # disagree with the requested transport set.
        template = next((m for (b, _c), m in index.items() if b == base_key), None)
        if template is None:
            return
        src = resolve(template, default_compartment)
    tr_id = f"tr_{n}_{compartment}"
    if tr_id in model.reactions:
        return
    dest = resolve(src, compartment)
    tr = cobra.Reaction(tr_id, name=f"transport {src.name} ({default_compartment}<->{compartment})",
                        lower_bound=-1000, upper_bound=1000)
    tr.add_metabolites({src: -1.0, dest: 1.0})
    model.add_reactions([tr])
