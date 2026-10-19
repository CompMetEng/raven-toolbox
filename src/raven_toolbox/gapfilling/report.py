"""Gap analysis summary — port of RAVEN's ``gapReport``.

Runs a battery of connectivity checks over a model: which reactions cannot
carry flux (at all, and even if net production of every metabolite were
allowed), how the model's metabolites split into isolated sub-networks, which
metabolites cannot be net-produced and the minimal set of metabolites that
would need an external source to fix that, which metabolites can be produced
or consumed for free once every existing boundary reaction is closed, and
(optionally) how much of the remaining damage a set of template models could
repair.

Two of ``gapReport.m``'s own helpers — ``checkProduction`` and
``getAllSubGraphs`` — have no separate Python port: their only real caller in
RAVEN is ``gapReport`` itself (their ledger rows note this), so their logic
is folded in here as private helpers rather than exposed as reusable
functions with no second user. A third, ``canExchange``, is reimplemented the
same way for the same reason, on top of the same probe-reaction technique
:func:`~raven_toolbox.gapfilling.find_leak_metabolite` already uses.

The mass-balancing section (``can_produce_without_input`` /
``can_consume_without_output``) temporarily closes every one of the model's
own boundary reactions, as ``gapReport.m`` does with the reactions
``getExchangeRxns`` returns.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import cobra
from cobra.flux_analysis import find_blocked_reactions

from raven_toolbox.gapfilling.fill import GapFillResult, connect_blocked_reactions

__all__ = ["GapReportResult", "MinToConnect", "gap_report", "print_gap_report"]

_FLUX_THRESHOLD = 1e-5
_PROBE_PREFIX = "_gap_report_probe_"
_EXCRETE_PREFIX = "_gap_report_excrete_"


@dataclass
class MinToConnect:
    """One entry of the minimal set of metabolites to connect.

    ``connects`` counts every not-produced metabolite that giving this one a
    free source would, directly or transitively, also make producible. That
    can include the metabolite itself, but only if the rest of the network
    can actually drain it once it has a source (e.g. some other reaction
    consumes it on net) — a metabolite with no other route at all never
    connects itself.
    """

    metabolite: str
    connects: int


@dataclass
class GapReportResult:
    """Outcome of a gap analysis.

    Parameters
    ----------
    no_flux_reactions:
        Reactions that cannot carry flux under the model's own constraints.
    no_flux_reactions_relaxed:
        Reactions that still cannot carry flux even if every metabolite is
        allowed unconstrained net production.
    subgraphs:
        Metabolite ids in each isolated sub-network (two metabolites are
        connected if they share a reaction), largest first.
    not_produced_metabolites:
        Metabolites that cannot have net production.
    needed_for_production:
        Metabolite id → ids of every not-produced metabolite that a free
        source of it would, directly or transitively, also make producible
        (see :class:`MinToConnect` on when that includes the metabolite
        itself).
    min_to_connect:
        A minimal set of metabolites to give a free source, chosen greedily
        (most newly-connected metabolites first), that together account for
        every entry in ``not_produced_metabolites``.
    can_produce_without_input:
        Metabolites that can be produced with every boundary reaction closed.
    can_consume_without_output:
        Metabolites that can be consumed with every boundary reaction closed.
    gap_fill:
        Result of gap-filling against ``template_models``, or None if none
        were given.
    """

    no_flux_reactions: list[str]
    no_flux_reactions_relaxed: list[str]
    subgraphs: list[list[str]]
    not_produced_metabolites: list[str]
    needed_for_production: dict[str, list[str]]
    min_to_connect: list[MinToConnect]
    can_produce_without_input: list[str]
    can_consume_without_output: list[str]
    gap_fill: GapFillResult | None = None


def _blocked_ids(model: cobra.Model, reactions: list[cobra.Reaction] | None = None) -> set[str]:
    return set(find_blocked_reactions(model, reaction_list=reactions, zero_cutoff=_FLUX_THRESHOLD))


def _relaxed_copy(model: cobra.Model) -> cobra.Model:
    """A copy where every metabolite may freely accumulate (net production allowed)."""
    relaxed = model.copy()
    for met in relaxed.metabolites:
        met.constraint.ub = None
    return relaxed


def _subgraphs(model: cobra.Model) -> list[list[str]]:
    """Connected components of the metabolite graph, largest first.

    Ties are broken by the index (in ``model.metabolites``) of each
    component's first-discovered member, matching ``getAllSubGraphs.m``'s own
    sequential-assignment order before its final size sort.
    """
    adjacency: dict[str, set[str]] = {m.id: set() for m in model.metabolites}
    for rxn in model.reactions:
        ids = [m.id for m in rxn.metabolites]
        for a in ids:
            adjacency[a].update(i for i in ids if i != a)

    assigned: set[str] = set()
    groups: list[list[str]] = []
    for met in model.metabolites:
        if met.id in assigned:
            continue
        component: list[str] = []
        stack = [met.id]
        assigned.add(met.id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in assigned:
                    assigned.add(neighbor)
                    stack.append(neighbor)
        groups.append(component)

    groups.sort(key=len, reverse=True)
    return groups


def _can_exchange(model: cobra.Model, direction: Literal["produce", "consume"]) -> list[str]:
    """Which metabolites can be freely produced/consumed via a temporary exchange.

    Reimplements ``canExchange.m`` (``addExchangeRxns`` + ``haveFlux``); not
    exposed publicly since cobrapy already covers the same need directly
    (``analyse_topology`` / ``cobra.flux_analysis.find_blocked_reactions``)
    for any caller other than this one.
    """
    working = model.copy()
    coeff = -1.0 if direction == "produce" else 1.0
    probes: dict[str, cobra.Reaction] = {}
    for met in model.metabolites:
        target = working.metabolites.get_by_id(met.id)
        probe = cobra.Reaction(f"{_PROBE_PREFIX}{met.id}", lower_bound=0.0, upper_bound=1000.0)
        probe.add_metabolites({target: coeff})
        probes[met.id] = probe
    working.add_reactions(list(probes.values()))

    blocked = _blocked_ids(working, list(probes.values()))
    return [met_id for met_id, probe in probes.items() if probe.id not in blocked]


def _reachable(start: set[str], edges: dict[str, set[str]], remaining: set[str]) -> set[str]:
    """Nodes reachable from ``start`` via ``edges``, restricted to ``remaining``.

    Deliberately does *not* seed the search with the metabolite whose row
    ``start`` came from: whether it reaches itself depends entirely on
    whether ``edges`` says so (see :class:`MinToConnect`).
    """
    reach: set[str] = set()
    queue = list(start & remaining)
    while queue:
        node = queue.pop()
        if node in reach:
            continue
        reach.add(node)
        queue.extend(edges[node] & remaining)
    return reach


def _greedy_min_to_connect(
    not_produced: list[str], needed_for_production: dict[str, list[str]]
) -> list[MinToConnect]:
    remaining = set(not_produced)
    edges = {mid: set(needed_for_production[mid]) for mid in not_produced}
    result: list[MinToConnect] = []

    while remaining:
        best_id: str | None = None
        best_reach: set[str] = set()
        for mid in not_produced:
            if mid not in remaining:
                continue
            reach = _reachable(edges[mid], edges, remaining)
            if len(reach) > len(best_reach):
                best_reach, best_id = reach, mid
        if best_id is None or not best_reach:
            break
        result.append(MinToConnect(metabolite=best_id, connects=len(best_reach)))
        remaining -= best_reach

    return result


def _check_production(
    model: cobra.Model, excretion_compartments: set[str]
) -> tuple[list[str], dict[str, list[str]], list[MinToConnect]]:
    """Reimplements ``checkProduction.m`` with ``checkNeededForProduction=True``.

    Not exposed publicly (see the module docstring): its only real caller is
    ``gapReport`` itself.
    """
    allowed = [m for m in model.metabolites if m.compartment in excretion_compartments]

    working = model.copy()
    excrete: dict[str, cobra.Reaction] = {}
    for met in allowed:
        target = working.metabolites.get_by_id(met.id)
        rxn = cobra.Reaction(f"{_EXCRETE_PREFIX}{met.id}", lower_bound=0.0, upper_bound=1000.0)
        rxn.add_metabolites({target: -1.0})
        excrete[met.id] = rxn
    working.add_reactions(list(excrete.values()))

    blocked = _blocked_ids(working, list(excrete.values()))
    not_produced = [met.id for met in allowed if excrete[met.id].id in blocked]
    not_produced_rxns = [excrete[mid] for mid in not_produced]

    # For each not-produced metabolite, flip its own excretion reaction into an
    # uptake (the same reaction, not a parallel one -- checkProduction.m negates
    # the S-matrix column in place) and see which not-produced metabolites'
    # excretion reactions that unblocks. A metabolite with no other reaction of
    # its own does *not* unlock itself this way: flipped to pure uptake, nothing
    # is left to make it carry flux, unlike a parallel uptake alongside the
    # still-present excretion reaction, which would trivially loop.
    needed_for_production: dict[str, list[str]] = {}
    for mid in not_produced:
        target = working.metabolites.get_by_id(mid)
        with working:
            excrete[mid].add_metabolites({target: 2.0})  # -1 (excrete) -> +1 (uptake)
            blocked_now = _blocked_ids(working, not_produced_rxns)
        needed_for_production[mid] = [other for other in not_produced if excrete[other].id not in blocked_now]

    min_to_connect = _greedy_min_to_connect(not_produced, needed_for_production)
    return not_produced, needed_for_production, min_to_connect


def _require_feasible(model: cobra.Model, condition: str) -> None:
    """Raise a ValueError naming the forced-flux reactions if ``model`` is infeasible."""
    if not math.isnan(model.slim_optimize(error_value=float("nan"))):
        return
    forced = [r.id for r in model.reactions if r.lower_bound > 0 or r.upper_bound < 0]
    hint = f" Reactions with a forced flux: {', '.join(forced)}." if forced else ""
    raise ValueError(
        f"The model is infeasible {condition}, so gap_report cannot analyse it.{hint} "
        "Set the bounds of ATP maintenance and any other forced flux to zero first."
    )


def gap_report(
    model: cobra.Model,
    *,
    template_models: cobra.Model | Iterable[cobra.Model] | None = None,
) -> GapReportResult:
    """Run a gap analysis on ``model``.

    Parameters
    ----------
    model:
        Model to analyse. Never modified.
    template_models:
        Model(s) to gap-fill blocked reactions from
        (:func:`~raven_toolbox.gapfilling.connect_blocked_reactions`). If
        omitted, ``result.gap_fill`` is None and that step is skipped.

    Returns
    -------
    GapReportResult

    Raises
    ------
    ValueError
        If ``model`` is infeasible under its own bounds, or once its boundary
        reactions are closed for ``can_produce_without_input`` and
        ``can_consume_without_output``. The usual cause is a reaction with a
        forced flux, such as ATP maintenance with a positive lower bound; set
        its bounds to zero before calling ``gap_report``.
    """
    _require_feasible(model, "under its own bounds")
    no_flux_reactions = [r.id for r in model.reactions if r.id in _blocked_ids(model)]
    no_flux_reactions_relaxed = [r.id for r in model.reactions if r.id in _blocked_ids(_relaxed_copy(model))]

    subgraphs = _subgraphs(model)

    not_produced, needed_for_production, min_to_connect = _check_production(
        model, set(model.compartments)
    )

    closed = model.copy()
    for rxn in closed.boundary:
        rxn.bounds = (0.0, 0.0)
    _require_feasible(closed, "once its boundary reactions are closed")
    can_produce_without_input = _can_exchange(closed, "produce")
    can_consume_without_output = _can_exchange(closed, "consume")

    gap_fill = None
    if template_models is not None:
        gap_fill = connect_blocked_reactions(model, template_models)

    return GapReportResult(
        no_flux_reactions=no_flux_reactions,
        no_flux_reactions_relaxed=no_flux_reactions_relaxed,
        subgraphs=subgraphs,
        not_produced_metabolites=not_produced,
        needed_for_production=needed_for_production,
        min_to_connect=min_to_connect,
        can_produce_without_input=can_produce_without_input,
        can_consume_without_output=can_consume_without_output,
        gap_fill=gap_fill,
    )


def print_gap_report(
    model: cobra.Model,
    *,
    print_fn=print,
    **kwargs,
) -> GapReportResult:
    """Print :func:`gap_report`'s result and return it.

    A simpler, Pythonic rendering of ``gapReport.m``'s console report — not a
    literal reproduction of its text; the structured result is what's
    compared for parity.
    """
    result = gap_report(model, **kwargs)

    print_fn(f"Gap analysis for {model.id} - {model.name}\n")
    print_fn("***Overview")
    print_fn(
        f"{len(result.no_flux_reactions)} out of {len(model.reactions)} reactions cannot carry flux "
        f"({len(result.no_flux_reactions_relaxed)} if net production of all metabolites is allowed)"
    )

    print_fn("\n***Isolated subnetworks")
    print_fn(f"A total of {len(result.subgraphs)} isolated sub-networks are present in the model")
    for i, group in enumerate(result.subgraphs, start=1):
        print_fn(f"\t{i}. {len(group)} metabolites")

    print_fn("\n***Metabolite connectivity")
    print_fn(
        f"To enable net production of all metabolites, a total of {len(result.min_to_connect)} "
        f"metabolites must be connected"
    )
    print_fn("Top 10 metabolites to connect:")
    for i, entry in enumerate(result.min_to_connect[:10], start=1):
        print_fn(f"\t{i}. {entry.metabolite}")

    print_fn("\n***Mass balancing")
    print_fn(
        f"{len(result.can_consume_without_output)} metabolites could be consumed without any outputs\n"
        f"{len(result.can_produce_without_input)} metabolites could be produced without any inputs"
    )

    if result.gap_fill is not None:
        print_fn("\n***Automated gap-filling")
        print_fn(
            f"{len(result.gap_fill.newly_connected)} unconnected reactions can be connected by including "
            f"{len(result.gap_fill.added_reactions)} template reactions"
        )

    return result
