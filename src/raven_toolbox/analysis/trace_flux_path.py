"""Highest-flux-fraction path between two reactions: port of RAVEN's
``traceFluxPath``.

Traces how flux flows from one reaction to another through the network: at
each metabolite junction, flux splits proportionally among every reaction
that net-consumes it, and a best-first (greedy) search finds the path with
the highest cumulative fraction of the source reaction's own flux. By
default, common currency and cofactor metabolites (ATP, NAD, CoA, H2O, ...) are
excluded so the path follows material (carbon-skeleton) flow.

The search is forward-only: it follows metabolites that ``from_rxn``
produces (given the sign of its flux) to reactions that consume
them, so ``from_rxn`` and ``to_rxn`` in parallel branches, or connected only
in reverse, yield no path.
"""
from __future__ import annotations

import heapq
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import cobra
import pandas as pd

__all__ = ["TraceFluxPathResult", "trace_flux_path", "print_flux_path"]

_DEFAULT_CUTOFF = 1e-8
_DEFAULT_MAX_HOPS = 30

_COMPARTMENT_SUFFIX = re.compile(r"[\[(][^\])]*[\])]$|_[a-zA-Z]\d*$")

# traceFluxPath.m lists each entry in both upper- and lower-case because its
# own match is case-insensitive anyway, making the duplicates redundant; this
# keeps one canonical (lowercase) spelling per compound instead.
_CURRENCY_METABOLITES = frozenset({
    "atp", "adp", "amp", "datp", "dadp", "damp",
    "gtp", "gdp", "gmp", "dgtp", "dgdp", "dgmp",
    "ctp", "cdp", "cmp", "dctp", "dcdp", "dcmp",
    "utp", "udp", "ump", "dutp", "dudp", "dump",
    "ttp", "tdp", "tmp", "dttp", "dtdp", "dtmp",
    "nad", "nadh", "nadp", "nadph", "nad+", "nadp+",
    "fad", "fadh2", "fmn", "fmnh2",
    "coa", "coenzyme a",
    "h2o", "water", "h+", "proton", "oh-", "hydroxide", "h", "oh1",
    "phosphate", "orthophosphate", "pyrophosphate", "diphosphate", "pi", "ppi", "pp",
    "co2", "carbon dioxide", "bicarbonate", "hco3-", "hco3",
    "o2", "oxygen", "nh3", "ammonia", "nh4+", "ammonium", "nitrogen", "nh4",
    "sulfate", "sulfite", "sulfide", "thiosulfate",
    "thioredoxin", "thioredoxin-sh", "thioredoxin-s2", "trdrd", "trdox",
    "ferredoxin", "ferredoxin reduced", "ferredoxin oxidized", "fdred", "fdox",
    "ubiquinol", "ubiquinone", "menaquinol", "menaquinone", "q8h2", "q8", "mqn8", "mql8",
    "lipoamide", "dihydrolipoamide",
    "s-adenosyl-l-methionine", "s-adenosylhomocysteine", "sam", "sah", "amet", "ahcys",
    "5-methyltetrahydrofolate", "tetrahydrofolate", "dihydrofolate", "thf", "dhf", "mlthf", "methf",
})


@dataclass
class TraceFluxPathResult:
    """Outcome of a :func:`trace_flux_path` search.

    Parameters
    ----------
    reactions:
        Reaction ids on the best path, including ``from_rxn`` and ``to_rxn``.
        Empty if no path was found.
    metabolites:
        Metabolite id at each junction; ``len(metabolites) == len(reactions) - 1``.
    cumulative_fraction:
        Fraction of ``from_rxn``'s flux that passes through this exact path
        (0 = no path found, 1 = all of it reaches ``to_rxn`` directly).
    """

    reactions: list[str]
    metabolites: list[str]
    cumulative_fraction: float


def _flux(fluxes: pd.Series | Mapping[str, float], rxn_id: str) -> float:
    try:
        return float(fluxes[rxn_id])
    except KeyError:
        raise ValueError(f"fluxes has no entry for reaction {rxn_id!r}") from None


def _is_currency(met: cobra.Metabolite, exclude: frozenset[str]) -> bool:
    candidates = {_COMPARTMENT_SUFFIX.sub("", met.id).strip().lower()}
    if met.name:
        candidates.add(met.name.strip().lower())
    return not candidates.isdisjoint(exclude)


def trace_flux_path(
    model: cobra.Model,
    fluxes: pd.Series | Mapping[str, float],
    from_rxn: str,
    to_rxn: str,
    *,
    cutoff: float = _DEFAULT_CUTOFF,
    max_hops: int = _DEFAULT_MAX_HOPS,
    trace_material: bool = True,
    carbon_only: bool = False,
    exclude_mets: Iterable[str] | None = None,
) -> TraceFluxPathResult:
    """Find the highest-flux-fraction path from ``from_rxn`` to ``to_rxn``.

    Parameters
    ----------
    model:
        Model both reactions belong to.
    fluxes:
        Flux for every reaction the search might visit, indexed by reaction
        id (a full-model flux vector, for example from ``model.optimize().fluxes``,
        always qualifies).
    from_rxn, to_rxn:
        Source and target reaction ids.
    cutoff:
        A reaction counts as consuming a metabolite when its stoichiometric coefficient
        times its flux is below ``-cutoff``.
    max_hops:
        Longer paths are pruned.
    trace_material:
        Exclude common currency and cofactor metabolites (ATP, ADP, NAD, NADH,
        H2O, CoA, Pi, CO2, ...) so the path follows material flow.
    carbon_only:
        Additionally require each intermediate metabolite to contain at
        least one carbon atom (via ``Metabolite.elements``, which parses the
        formula, so for example a calcium ion is not counted as carbon-bearing). A
        metabolite with no formula counts as having no carbon.
    exclude_mets:
        Extra metabolite names or ids (compartment suffix stripped) to
        exclude, on top of the currency list.

    Returns
    -------
    TraceFluxPathResult

    Raises
    ------
    ValueError
        If ``from_rxn`` or ``to_rxn`` is not in the model, or ``fluxes`` has
        no entry for a reaction the search needs.
    """
    if from_rxn not in model.reactions:
        raise ValueError(f"Reaction {from_rxn!r} not found in the model.")
    if to_rxn not in model.reactions:
        raise ValueError(f"Reaction {to_rxn!r} not found in the model.")

    if from_rxn == to_rxn:
        return TraceFluxPathResult(reactions=[from_rxn], metabolites=[], cumulative_fraction=1.0)

    exclude: frozenset[str] = frozenset(m.strip().lower() for m in (exclude_mets or ()))
    if trace_material:
        exclude = exclude | _CURRENCY_METABOLITES

    rxn_order = {r.id: i for i, r in enumerate(model.reactions)}
    met_order = {m.id: i for i, m in enumerate(model.metabolites)}

    def ordered_mets(rxn: cobra.Reaction) -> list[cobra.Metabolite]:
        return sorted(rxn.metabolites, key=lambda m: met_order[m.id])

    def ordered_consumers(met: cobra.Metabolite) -> list[cobra.Reaction]:
        return sorted(met.reactions, key=lambda r: rxn_order[r.id])

    best_at_node = {r.id: 0.0 for r in model.reactions}
    best_at_node[from_rxn] = 1.0

    # Best-first by frac (descending). A min-heap keyed on -frac with a tie-breaking
    # counter reproduces the same pop order (highest frac first, FIFO among ties) as
    # the previous sorted-insert list, in O(log n) per push/pop instead of O(n).
    _counter = 0
    heap: list[tuple[float, int, dict]] = [
        (-1.0, _counter, {"rxn": from_rxn, "frac": 1.0, "rpath": [from_rxn], "mpath": []})
    ]
    best_frac = 0.0
    best_rpath: list[str] = []
    best_mpath: list[str] = []

    while heap:
        _, _, entry = heapq.heappop(heap)
        cur, frac, rpath, mpath = entry["rxn"], entry["frac"], entry["rpath"], entry["mpath"]

        if cur == to_rxn:
            if frac > best_frac:
                best_frac, best_rpath, best_mpath = frac, rpath, mpath
            continue

        if len(rpath) > max_hops:
            continue

        cur_reaction = model.reactions.get_by_id(cur)
        cur_flux = _flux(fluxes, cur)

        for met in ordered_mets(cur_reaction):
            coeff = cur_reaction.metabolites[met]
            if coeff * cur_flux <= 0:
                continue  # consumed or zero-contribution, not net-produced here

            if exclude and _is_currency(met, exclude):
                continue
            if carbon_only and (met.elements or {}).get("C", 0) == 0:
                continue

            consumers = []
            for consumer in ordered_consumers(met):
                if consumer.id in rpath:
                    continue  # no revisiting
                consumer_coeff = consumer.metabolites[met]
                consumer_flux = _flux(fluxes, consumer.id)
                if consumer_coeff * consumer_flux < -cutoff:
                    consumers.append((consumer, consumer_coeff, consumer_flux))
            if not consumers:
                continue

            total_consumed = sum(abs(c * f) for _, c, f in consumers)

            for consumer, c, f in consumers:
                new_frac = frac * (abs(c * f) / total_consumed)

                if new_frac > best_at_node[consumer.id] or consumer.id == to_rxn:
                    best_at_node[consumer.id] = max(best_at_node[consumer.id], new_frac)
                    new_entry = {
                        "rxn": consumer.id,
                        "frac": new_frac,
                        "rpath": [*rpath, consumer.id],
                        "mpath": [*mpath, met.id],
                    }
                    _counter += 1
                    heapq.heappush(heap, (-new_frac, _counter, new_entry))

    return TraceFluxPathResult(
        reactions=best_rpath,
        metabolites=best_mpath,
        cumulative_fraction=best_frac,
    )


def print_flux_path(
    model: cobra.Model,
    fluxes: pd.Series | Mapping[str, float],
    from_rxn: str,
    to_rxn: str,
    *,
    print_fn=print,
    **kwargs,
) -> TraceFluxPathResult:
    """Print :func:`trace_flux_path`'s result and return it.

    The output is a simplified rendering of ``traceFluxPath.m``'s console
    diagram. Parity with RAVEN is defined on the structured result.

    Parameters
    ----------
    model : cobra.Model
        Model both reactions belong to.
    fluxes : pd.Series or Mapping[str, float]
        Flux by reaction id, as for :func:`trace_flux_path`.
    from_rxn : str
        Source reaction id.
    to_rxn : str
        Target reaction id.
    print_fn : callable
        Function called with each output line. The default is the builtin ``print``.
    **kwargs : dict
        Keyword arguments forwarded to :func:`trace_flux_path`: ``cutoff``, ``max_hops``,
        ``trace_material``, ``carbon_only`` and ``exclude_mets``.
    """
    result = trace_flux_path(model, fluxes, from_rxn, to_rxn, **kwargs)

    if not result.reactions:
        print_fn(f"No forward flux path found from {from_rxn} to {to_rxn}.")
        return result

    parts = [result.reactions[0]]
    for i, met_id in enumerate(result.metabolites):
        parts.append(f"--[{met_id}]--> {result.reactions[i + 1]}")
    print_fn(" ".join(parts))
    print_fn(f"Cumulative fraction: {result.cumulative_fraction:.2%}")
    return result
