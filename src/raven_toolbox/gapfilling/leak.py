"""Stoichiometric leak detection — port of RAVEN's ``findLeakMetabolite``.

Tests whether a model can produce (excrete) or consume (take up) any
metabolite for free — a sign of a stoichiometric leak, futile cycle, or
unconstrained exchange somewhere in the network. This is the unified
replacement for RAVEN's ``makeSomething`` (``'produce'``) and
``consumeSomething`` (``'consume'``).

Works by grafting a throwaway exchange reaction onto every metabolite (an
"any metabolite" probe), forcing at least one unit of combined flux through
them via a synthetic metabolite, and minimizing the L1 norm of the resulting
flux distribution (equivalent to :func:`cobra.flux_analysis.pfba` against a
zero objective). If more than one metabolite ends up in the minimal
solution, a second pass tries every implicated metabolite alone to see if it
can carry the probe on its own; some metabolites can only leak in pairs (or
larger groups), in which case all of them are reported together.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import cobra
import pandas as pd

from raven_toolbox.analysis.min_flux_count import get_min_nr_fluxes

__all__ = ["LeakMetaboliteResult", "find_leak_metabolite"]

_ACTIVE_THRESHOLD = 0.1
_PROBE_PREFIX = "_leak_probe_"
_FAKE_MET_ID = "_leak_fake"
_FAKE_EXCHANGE_ID = "_leak_fake_exchange"


@dataclass
class LeakMetaboliteResult:
    """Outcome of a leak search.

    Parameters
    ----------
    fluxes:
        Flux for every reaction in the original model (empty if no leak was
        found).
    metabolites:
        Ids of the metabolite(s) implicated. Usually one; more than one
        means those metabolites could only be exchanged together.
    status:
        ``"optimal"`` if a leak was found, ``"infeasible"`` otherwise.
    """

    fluxes: pd.Series
    metabolites: list[str]
    status: str


def _resolve_ignored(model: cobra.Model, ignore_mets: Iterable[str] | None, is_names: bool) -> set[str]:
    if not ignore_mets:
        return set()
    values = set(ignore_mets)
    if not is_names:
        return values
    return {m.id for m in model.metabolites if m.name in values}


def _l1_objective(model: cobra.Model):
    return model.problem.Objective(
        sum(r.forward_variable + r.reverse_variable for r in model.reactions),
        direction="min",
    )


def _build_probed_model(
    model: cobra.Model,
    direction: Literal["produce", "consume"],
    ignored: set[str],
    allow_excretion: bool,
    ignore_int_bounds: bool,
) -> tuple[cobra.Model, dict[str, cobra.Reaction]]:
    aug = model.copy()

    if ignore_int_bounds:
        for rxn in aug.reactions:
            if not rxn.boundary:
                rxn.bounds = (-1000.0, 1000.0)

    produce = direction == "produce"
    exch_coeff = -1.0 if produce else 1.0
    fake_row_coeff = 1.0 if produce else -1.0
    fake_exch_coeff = -1.0 if produce else 1.0

    if produce and allow_excretion:
        # Relax every metabolite's own mass balance to allow free net
        # production (the lower bound, still 0, keeps free net consumption
        # disallowed): matches findLeakMetabolite.m's model.b upper-bound
        # relaxation, applied only for 'produce'.
        for met in aug.metabolites:
            met.constraint.ub = None

    fake_met = cobra.Metabolite(_FAKE_MET_ID)
    aug.add_metabolites([fake_met])

    probes: dict[str, cobra.Reaction] = {}
    for met in model.metabolites:
        if met.id in ignored:
            continue
        target = aug.metabolites.get_by_id(met.id)
        probe = cobra.Reaction(f"{_PROBE_PREFIX}{met.id}", lower_bound=0.0, upper_bound=float("inf"))
        probe.add_metabolites({target: exch_coeff, fake_met: fake_row_coeff})
        probes[met.id] = probe
    aug.add_reactions(list(probes.values()))

    fake_exchange = cobra.Reaction(_FAKE_EXCHANGE_ID, lower_bound=1.0, upper_bound=float("inf"))
    fake_exchange.add_metabolites({fake_met: fake_exch_coeff})
    aug.add_reactions([fake_exchange])

    return aug, probes


def find_leak_metabolite(
    model: cobra.Model,
    direction: Literal["produce", "consume"],
    *,
    ignore_mets: Iterable[str] | None = None,
    is_names: bool = False,
    min_nr_fluxes: bool = False,
    allow_excretion: bool = True,
    ignore_int_bounds: bool = False,
) -> LeakMetaboliteResult:
    """Find a metabolite that can be freely produced or consumed.

    Parameters
    ----------
    model:
        Model to test. Never modified; the search runs on an internal copy.
        Its boundary reactions are used as they are: close every exchange,
        sink and demand reaction first (``for r in model.boundary: r.bounds =
        (0, 0)`` on a copy), or an open uptake or excretion counts as a leak.
        Positive lower bounds (such as on a maintenance reaction) still
        apply and can make the search infeasible; the result then has status
        ``"infeasible"``, which does not show that the model is leak-free.
    direction:
        ``"produce"`` to look for freely excreted metabolites, or
        ``"consume"`` to look for freely consumed metabolites.
    ignore_mets:
        Metabolite ids (or names, with ``is_names=True``) to exclude from
        the search. Unlike ``findLeakMetabolite.m``, this only accepts
        ids/names, not a logical or index vector — there is no positional
        model-array equivalent to index into in cobrapy.
    is_names:
        If True, ``ignore_mets`` holds metabolite names rather than ids;
        every metabolite sharing one of those names, in any compartment, is
        excluded.
    min_nr_fluxes:
        Solve for the minimum *number* of active fluxes
        (:func:`~raven_toolbox.analysis.get_min_nr_fluxes`) instead of the
        minimum sum. Slower, but can be used if the sum gives too many
        active reactions to interpret.
    allow_excretion:
        Allow every metabolite to freely accumulate. Only used when
        ``direction`` is ``"produce"`` (default True).
    ignore_int_bounds:
        Relax every non-boundary reaction to ``[-1000, 1000]`` before
        searching, to find leaks that are only possible once internal
        bounds (including reversibility) are ignored.

    Returns
    -------
    LeakMetaboliteResult
    """
    if direction not in ("produce", "consume"):
        raise ValueError("direction must be 'produce' or 'consume'")

    ignored = _resolve_ignored(model, ignore_mets, is_names)
    aug, probes = _build_probed_model(model, direction, ignored, allow_excretion, ignore_int_bounds)
    original_ids = [r.id for r in model.reactions]

    aug.objective = _l1_objective(aug)
    solution = aug.optimize()
    if aug.solver.status != "optimal":
        return LeakMetaboliteResult(fluxes=pd.Series(dtype=float), metabolites=[], status="infeasible")

    candidates = [
        met.id for met in model.metabolites
        if met.id in probes and abs(solution.fluxes[probes[met.id].id]) > _ACTIVE_THRESHOLD
    ]

    if len(candidates) > 1:
        winner = None
        for candidate in candidates:
            with aug:
                for other in candidates:
                    if other != candidate:
                        probes[other].bounds = (0.0, 0.0)
                aug.objective = aug.problem.Objective(0, direction="min")
                aug.optimize()
                if aug.solver.status == "optimal":
                    winner = candidate
                    break
        if winner is not None:
            for other in candidates:
                if other != winner:
                    probes[other].bounds = (0.0, 0.0)

    if min_nr_fluxes:
        mnf = get_min_nr_fluxes(aug, [r.id for r in aug.reactions])
        if mnf.status != "optimal":
            return LeakMetaboliteResult(fluxes=pd.Series(dtype=float), metabolites=[], status="infeasible")
        final_fluxes = mnf.fluxes
    else:
        aug.objective = _l1_objective(aug)
        final_solution = aug.optimize()
        if aug.solver.status != "optimal":
            return LeakMetaboliteResult(fluxes=pd.Series(dtype=float), metabolites=[], status="infeasible")
        final_fluxes = final_solution.fluxes

    metabolites = [
        met.id for met in model.metabolites
        if met.id in probes and abs(final_fluxes[probes[met.id].id]) > _ACTIVE_THRESHOLD
    ]
    return LeakMetaboliteResult(fluxes=final_fluxes[original_ids], metabolites=metabolites, status="optimal")
