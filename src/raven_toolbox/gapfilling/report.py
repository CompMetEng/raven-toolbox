"""Gap analysis summary, a port of RAVEN's ``gapReport``.

Runs connectivity checks over a model: which reactions cannot carry flux (under the
model's constraints, and when net production of every metabolite is allowed), how the
metabolites split into isolated sub-networks, which metabolites cannot be net-produced
and the minimal set of metabolites that need an external source to fix that, and which
metabolites can be produced or consumed for free once every existing boundary reaction
is closed. Optionally, it reports how many blocked reactions a set of template models
can connect.

The helpers ``checkProduction`` and ``getAllSubGraphs`` of ``gapReport.m`` are private
functions of this module, because ``gapReport`` is their only caller in RAVEN. The helper
``canExchange`` is reimplemented as a private function in the same way, with the
probe-reaction technique that :func:`~raven_toolbox.gapfilling.find_leak_metabolite`
uses.

The mass-balancing section (``can_produce_without_input`` and
``can_consume_without_output``) temporarily closes every boundary reaction of the model,
as ``gapReport.m`` does with the reactions that ``getExchangeRxns`` returns.
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

    ``connects`` counts the not-produced metabolites that giving this one a free
    source makes producible, directly or transitively, and that no earlier entry of
    the set has connected. The count includes the metabolite itself only when the
    rest of the network can drain it once it has a source (for example, another
    reaction consumes it on net). A metabolite with no other reaction does not
    connect itself.
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
        Reactions that cannot carry flux even if every metabolite is allowed
        unconstrained net production.
    subgraphs:
        Metabolite ids in each isolated sub-network (two metabolites are
        connected if they share a reaction), largest first.
    not_produced_metabolites:
        Metabolites that cannot have net production.
    needed_for_production:
        Metabolite id to the ids of the not-produced metabolites that a free
        source of it makes producible, directly or transitively (see
        :class:`MinToConnect` on when the list includes the metabolite itself).
    min_to_connect:
        Metabolites to give a free source, chosen greedily (most newly connected
        metabolites first). The selection stops when no remaining metabolite
        connects any remaining metabolite (a metabolite whose reach is only itself is
        still selected), so a not-produced metabolite that no selected metabolite
        connects is not covered.
    can_produce_without_input:
        Metabolites that can be produced with every boundary reaction closed.
    can_consume_without_output:
        Metabolites that can be consumed with every boundary reaction closed.
    gap_fill:
        Result of gap-filling against ``template_models``, or None if no template
        models were given.
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
    return set(
        find_blocked_reactions(
            model, reaction_list=reactions, zero_cutoff=_FLUX_THRESHOLD, processes=1
        )
    )


def _relaxed_copy(model: cobra.Model) -> cobra.Model:
    """A copy in which every metabolite may accumulate (net production allowed)."""
    relaxed = model.copy()
    for met in relaxed.metabolites:
        met.constraint.ub = None
    return relaxed


def _subgraphs(model: cobra.Model) -> list[list[str]]:
    """Connected components of the metabolite graph, largest first.

    Components of equal size are ordered by the index (in ``model.metabolites``) of
    their first-discovered member, which is the sequential-assignment order of
    ``getAllSubGraphs.m`` before its final size sort.
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
    """Ids of the metabolites that can be freely produced or consumed via a temporary exchange.

    Reimplements ``canExchange.m`` (``addExchangeRxns`` and ``haveFlux``).
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

    The search does not start from the metabolite whose row ``start`` came from; that
    metabolite is in the result only when ``edges`` leads back to it (see
    :class:`MinToConnect`).
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
    """Reimplements ``checkProduction.m`` with ``checkNeededForProduction=True``."""
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
    """Raise a ValueError if ``model`` is infeasible.

    The message lists the reactions with a forced flux.
    """
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
        Model to analyse. The function does not modify it.
    template_models:
        Model(s) to gap-fill blocked reactions from
        (:func:`~raven_toolbox.gapfilling.connect_blocked_reactions`). If
        omitted, the gap-filling step is skipped and ``result.gap_fill`` is None.

    Returns
    -------
    GapReportResult

    Raises
    ------
    ValueError
        If ``model`` is infeasible under its own bounds, or once its boundary
        reactions are closed for ``can_produce_without_input`` and
        ``can_consume_without_output``. The message lists the reactions with a
        forced flux (lower bound above zero or upper bound below zero), such as
        ATP maintenance with a positive lower bound. Setting the bounds of the forced
        reactions to zero before the call removes the cause.
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
    """Print the result of :func:`gap_report` and return it.

    The text is a Python rendering of the console report of ``gapReport.m`` and
    differs from its wording. Parity with RAVEN is compared on the structured result.

    Parameters
    ----------
    model : cobra.Model
        Model to analyse, passed to :func:`gap_report`.
    print_fn : callable
        Function that receives each output string of the report as one argument; a string can
        contain line breaks. Default :func:`print`.
    **kwargs : dict
        Keyword arguments forwarded to :func:`gap_report` (for example ``template_models``).

    Returns
    -------
    GapReportResult
        The result of :func:`gap_report`.
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
