"""Reduce a model by removing/merging reactions that cannot carry flux.

Individual reduction modes (each removes a specific class of reaction, in place):

* ``remove_zero_interval_reactions`` — reactions locked at zero flux (``lb == ub == 0``);
  RAVEN ``simplifyModel`` ``deleteZeroInterval``.
* ``remove_dead_end_reactions`` — reactions touching a *topological* dead-end metabolite
  (one that, given reaction directions, can only be produced or only consumed, or
  participates in a single reaction); RAVEN ``deleteInaccessible``. Iterates to a fixpoint
  and prunes orphaned metabolites.
* ``remove_no_flux_reactions`` — reactions that cannot carry flux under the model's
  bounds (exchange reactions opened by default), found by FVA (both min and max flux
  zero); RAVEN ``deleteMinMax``. Catches blocked reactions that are not topological
  dead-ends.
* ``remove_duplicate_reactions`` — all-but-one of each set of reactions with identical
  stoichiometry and reversibility (bounds merged to the widest); RAVEN ``deleteDuplicates`` (detection-only: ``find_duplicate_reactions``).
* ``constrain_reversible_reactions`` — does not remove reactions; tightens bounds, making
  reversible reactions that can only carry flux one way irreversible (via FVA); RAVEN
  ``constrainReversible``.
* ``group_linear_reactions`` — lossy fold of single-producer/single-consumer chains into
  one reaction (drops gene rules); RAVEN ``mergeLinear``.

``simplify_model`` composes the above via RAVEN's ``simplifyModel`` boolean-flag interface,
applying the selected modes in RAVEN's order. ``remove_dead_end_reactions`` and
``remove_no_flux_reactions`` are complementary: the first is a cheap topological sweep, the
second an exact FVA-based sweep that also removes flux-blocked reactions the first misses.
"""
from __future__ import annotations

import ast
import math
import re
import warnings
from collections.abc import Iterable

import cobra
from cobra.flux_analysis import find_blocked_reactions, flux_variability_analysis

from raven_toolbox.manipulation.irreversible import convert_to_irreversible
from raven_toolbox.utils.sort import reversible_orientation

_EXP_SUFFIX = re.compile(r"_EXP_\d+$")


def _prune_orphan_metabolites(model: cobra.Model) -> list[str]:
    orphans = [m for m in model.metabolites if not m.reactions]
    if orphans:
        model.remove_metabolites(orphans)
    return [m.id for m in orphans]


def _can_produce_and_consume(met) -> tuple[bool, bool]:
    """Whether the network can both produce and consume ``met``, by direction alone.

    Purely topological, matching RAVEN ``deleteInaccessible``: a reaction's forward
    direction always counts (regardless of its bounds), and its reverse direction
    counts exactly when it is reversible (``lower_bound < 0``). This deliberately
    ignores ``upper_bound`` and does not check whether a reaction can actually carry
    flux — that bounds-aware check is ``remove_no_flux_reactions``'s job (RAVEN
    ``deleteMinMax``, via FVA). A reaction locked at zero flux (``lb == ub == 0``)
    therefore still counts as a producer/consumer here, as in RAVEN.
    """
    produce = consume = False
    for rxn in met.reactions:
        coef = rxn.get_coefficient(met)
        reversible = rxn.lower_bound < 0
        if coef > 0:
            produce = True
            consume |= reversible
        elif coef < 0:
            consume = True
            produce |= reversible
    return produce, consume


def remove_dead_end_reactions(
    model: cobra.Model, *, reserved: Iterable[str] | None = None
) -> tuple[list[str], list[str]]:
    """Iteratively remove dead-end reactions and metabolites.

    A metabolite is a dead end if it participates in only one reaction, or if (accounting for
    reaction directionality) it can only be produced or only consumed — such metabolites cannot
    carry steady-state flux, so the reactions touching them are removed. Repeats until stable.
    ``reserved`` reaction ids are never removed, even if they touch a dead-end metabolite.

    Returns ``(removed_reaction_ids, removed_metabolite_ids)``.
    """
    reserved = set(reserved or [])
    removed_rxns: list[str] = []
    removed_mets: list[str] = []

    def is_dead(m: cobra.Metabolite) -> bool:
        return len(m.reactions) <= 1 or not all(_can_produce_and_consume(m))

    removed_mets += _prune_orphan_metabolites(model)
    # Worklist of metabolites to (re)examine. Only a metabolite touched by a
    # reaction removed in the previous pass can have newly become a dead end, so
    # re-checking every metabolite each pass is unnecessary once the first pass
    # has run.
    pending: set[cobra.Metabolite] = set(model.metabolites)
    while pending:
        dead = [m for m in pending if m in model.metabolites and is_dead(m)]
        if not dead:
            break
        rxns = {r for m in dead for r in m.reactions}
        to_delete = [r for r in rxns if r.id not in reserved]
        if not to_delete:
            break
        touched = {m for r in to_delete for m in r.metabolites}
        removed_rxns += [r.id for r in to_delete]
        model.remove_reactions(to_delete)
        removed_mets += _prune_orphan_metabolites(model)
        pending = {m for m in touched if m in model.metabolites}
    return removed_rxns, removed_mets


def remove_zero_interval_reactions(
    model: cobra.Model, *, reserved: Iterable[str] | None = None
) -> list[str]:
    """Remove reactions locked at zero flux (``lb == ub == 0``), pruning orphans.

    RAVEN ``simplifyModel``'s ``deleteZeroInterval`` mode: such reactions can never
    carry flux, so they only enlarge downstream problems. Genes used only by the
    removed reactions are removed with them, and every metabolite left without a
    reaction is removed, including ones that were orphans before. ``reserved`` reaction
    ids are never removed. Modifies in place; returns the removed reaction ids.
    """
    reserved = set(reserved or [])
    zero = [
        r for r in model.reactions
        if r.lower_bound == 0 and r.upper_bound == 0 and r.id not in reserved
    ]
    if zero:
        model.remove_reactions(zero, remove_orphans=True)
    _prune_orphan_metabolites(model)
    return [r.id for r in zero]


def remove_no_flux_reactions(
    model: cobra.Model,
    *,
    open_exchanges: bool = True,
    reserved: Iterable[str] | None = None,
) -> list[str]:
    """Remove reactions that cannot carry flux under the tested bounds (RAVEN ``deleteMinMax``).

    Runs FVA and drops every reaction whose minimum and maximum flux are both zero.
    With ``open_exchanges`` (default) the reactions in ``model.exchanges`` are first
    opened to at least ``(-1000, 1000)``; sink and demand reactions and all internal
    bounds keep their current values, so a reaction blocked only by one of those is
    still removed. With ``open_exchanges=False`` the current bounds are used, as RAVEN's
    ``deleteMinMax`` does, which can remove more. A reaction that is blocked under these
    bounds stays blocked under any tighter ones, so removing it before task discovery /
    the merge is safe as long as no later step relaxes a sink, demand or internal
    bound. ``reserved`` reaction ids are never removed. Genes used only by the removed
    reactions are removed with them, and every metabolite left without a reaction is
    removed. Modifies in place; returns the removed reaction ids.
    """
    reserved = set(reserved or [])
    blocked = [
        rid for rid in find_blocked_reactions(model, open_exchanges=open_exchanges)
        if rid not in reserved
    ]
    if blocked:
        model.remove_reactions(blocked, remove_orphans=True)
    _prune_orphan_metabolites(model)
    return blocked


def _signature(rxn, met_index: dict[str, int]):
    """Stoichiometry (reversible reactions written in the ``sort_model`` direction) and
    reversibility, as RAVEN ``contractModel`` compares reactions."""
    sign = reversible_orientation(rxn, met_index)
    mets = frozenset((m.id, sign * c) for m, c in rxn.metabolites.items())
    return (mets, rxn.reversibility)


def _stoich_signature(rxn, *, ignore_direction: bool) -> frozenset:
    """Signature considering stoichiometry only (used by find_duplicate_reactions).

    When ``ignore_direction`` is True, ``A → B`` and ``B → A`` (the same
    reaction with all coefficients negated) share a signature. Both
    orientations are accumulated and the lexicographically smaller one
    wins so the dict-key lookup is direction-symmetric.
    """
    forward = frozenset((m.id, c) for m, c in rxn.metabolites.items())
    if not ignore_direction:
        return forward
    backward = frozenset((m.id, -c) for m, c in rxn.metabolites.items())
    return min(forward, backward, key=lambda s: sorted(s))


def find_duplicate_reactions(
    model: cobra.Model,
    *,
    ignore_direction: bool = True,
) -> list[list[cobra.Reaction]]:
    """Return groups of reactions that share identical stoichiometry.

    Detection-only counterpart to :func:`remove_duplicate_reactions`.
    Bounds, objective coefficients, GPRs and annotations are ignored —
    only stoichiometry is compared, mirroring the legacy yeast-GEM
    ``findDuplicatedRxns`` and matching the typical curation use case
    (find reactions that *could* be merged).

    Parameters
    ----------
    ignore_direction
        When True (default), ``A → B`` and ``B → A`` are treated as
        duplicates (yeast-GEM's convention). Set ``False`` to require
        identical orientation.

    Returns
    -------
    A list of duplicate groups. Each group is itself a list with
    ≥ 2 reactions sharing the same stoichiometry. Reactions that have
    no duplicate are omitted.
    """
    groups: dict[frozenset, list[cobra.Reaction]] = {}
    for rxn in model.reactions:
        sig = _stoich_signature(rxn, ignore_direction=ignore_direction)
        groups.setdefault(sig, []).append(rxn)
    return [g for g in groups.values() if len(g) >= 2]


def remove_duplicate_reactions(
    model: cobra.Model,
    *,
    reserved: Iterable[str] | None = None,
    restrict_to_metabolites: Iterable[cobra.Metabolite] | None = None,
) -> list[str]:
    """Remove all-but-one of each set of duplicate reactions.

    Reactions are duplicates when they have identical stoichiometry and the same
    reversibility, as in RAVEN ``contractModel``: reversible reactions are compared in the
    direction ``sort_model(sort_reversible=True)`` gives them, so ``A <=> B`` and ``B <=> A``
    are duplicates. Bounds and objective coefficients may differ; the survivor takes the
    lowest lower bound, the highest upper bound and the highest objective coefficient of its
    group, with a warning where they differ (the values are taken as stored, not mirrored for
    a reaction written in the opposite direction). One of each set is kept (reserved
    reactions are never removed). Returns the removed reaction IDs.

    The survivor is the first-encountered reaction in ``model.reactions`` order
    (matching RAVEN's ``contractModel``). Its ``gene_reaction_rule`` becomes the
    union of every duplicate's top-level OR-clauses, deduplicated, rather than
    just its own — an isozyme relationship recorded on a *different* duplicate
    would otherwise be silently dropped, and ``contractModel`` merges this way.
    If every reaction in the group shares an ``_EXP_<digits>`` suffix
    (:func:`~raven_toolbox.manipulation.expand.expand_model`'s naming), the model
    is assumed to have gone through ``expand_model``, and the suffix is stripped
    from the survivor's id — so an expand-then-remove-duplicates round trip
    returns the original reaction id, not an arbitrarily-numbered expansion copy.

    Parameters
    ----------
    restrict_to_metabolites:
        If given, only reactions touching at least one of these metabolites
        are checked for duplicates; every other reaction is left alone even
        if it is already a duplicate of another (matching ``contractModel``'s
        ``mets`` argument, used internally by ``replace_metabolite``).
    """
    reserved = set(reserved or [])
    touching = (
        {r for m in restrict_to_metabolites for r in m.reactions}
        if restrict_to_metabolites is not None else None
    )
    met_index = {m.id: i for i, m in enumerate(model.metabolites)}
    groups: dict = {}
    for i, rxn in enumerate(model.reactions):
        key = _signature(rxn, met_index) if touching is None or rxn in touching else (object(), i)
        groups.setdefault(key, []).append(rxn)

    removed: list[str] = []
    for rxns in groups.values():
        if len(rxns) <= 1:
            continue
        keep = rxns[0]
        to_remove = [r for r in rxns if r is not keep and r.id not in reserved]
        if not to_remove:
            continue

        _merge_bounds_and_objective(keep, rxns)
        _merge_gene_reaction_rules(keep, rxns)
        if all(_EXP_SUFFIX.search(r.id) for r in rxns):
            keep.id = _EXP_SUFFIX.sub("", keep.id)

        removed += [r.id for r in to_remove]
        model.remove_reactions(to_remove)
    return removed


def _merge_bounds_and_objective(keep: cobra.Reaction, group: list[cobra.Reaction]) -> None:
    """Give ``keep`` the widest bounds and the largest objective coefficient of ``group``."""
    lower = min(r.lower_bound for r in group)
    upper = max(r.upper_bound for r in group)
    objective = max(r.objective_coefficient for r in group)
    if len({r.lower_bound for r in group}) > 1:
        warnings.warn(f"Duplicates of reaction {keep.id} have different lower bounds. "
                      "Uses the lowest.", stacklevel=3)
    if len({r.upper_bound for r in group}) > 1:
        warnings.warn(f"Duplicates of reaction {keep.id} have different upper bounds. "
                      "Uses the highest.", stacklevel=3)
    if len({r.objective_coefficient for r in group}) > 1:
        warnings.warn(f"Duplicates of reaction {keep.id} have different objective coefficients. "
                      "Uses the largest.", stacklevel=3)
    keep.bounds = (lower, upper)
    keep.objective_coefficient = objective


def _top_level_or_clauses(gpr: cobra.core.gene.GPR) -> list[str]:
    """The top-level OR-clauses of a GPR, as strings, via cobra's own AST.

    A clause containing "and" is parenthesised for readability when rejoined
    with " or " — not required for correctness (Python's own operator
    precedence already makes "A and B or C" parse as "(A and B) or C"), but
    matches how a human, and RAVEN's own ``contractModel``, would write it.
    """
    body = gpr.body
    if body is None:
        return []
    is_or = isinstance(body, ast.BoolOp) and isinstance(body.op, ast.Or)
    clauses = body.values if is_or else [body]
    out = []
    for clause in clauses:
        text = ast.unparse(clause)
        if isinstance(clause, ast.BoolOp) and isinstance(clause.op, ast.And):
            text = f"({text})"
        out.append(text)
    return out


def _merge_gene_reaction_rules(keep: cobra.Reaction, group: list[cobra.Reaction]) -> None:
    """Set ``keep``'s GPR to the union of every reaction in ``group``'s OR-clauses."""
    clauses: list[str] = []
    seen: set[str] = set()
    for rxn in group:
        for clause in _top_level_or_clauses(rxn.gpr):
            if clause not in seen:
                seen.add(clause)
                clauses.append(clause)
    if clauses:
        keep.gene_reaction_rule = clauses[0] if len(clauses) == 1 else " or ".join(clauses)


def constrain_reversible_reactions(
    model: cobra.Model, *, eps: float = 1e-10
) -> list[str]:
    """Constrain reversible reactions that can only carry flux one way.

    Runs FVA on
    each reversible reaction; if it can only carry forward flux its lower bound
    is set to 0, and if it can only carry reverse flux it is flipped to a forward
    reaction (stoichiometry, bounds, and objective negated). Returns the changed
    reaction IDs.

    Matches RAVEN ``simplifyModel``'s ``constrainReversible``: it classifies a bound as
    zero at ``|flux| < 1e-10`` and runs its FVA (``getAllowedBounds`` → ``solveLP``) at
    ``FeasibilityTol = 1e-9``. We set the same feasibility tolerance on the solver so the
    FVA min/max are trustworthy at that 1e-10 threshold (at Gurobi's looser 1e-6 default a
    reaction with tiny one-way flux is mis-classified, changing its reversibility and, in
    turn, how ``group_linear_reactions`` merges it).
    """
    revs = [r for r in model.reactions if r.lower_bound < 0 < r.upper_bound]
    if not revs:
        return []
    try:  # Gurobi-specific; match RAVEN solveLP's precision. Harmless on other backends.
        model.solver.problem.Params.FeasibilityTol = 1e-9
        model.solver.problem.Params.OptimalityTol = 1e-9
    except Exception:  # noqa: BLE001
        pass
    # Infeasible models surface as either OptimizationError (Gurobi/HiGHS) or
    # NaN-filled ranges (some optlang backends silently). Catch both and raise
    # a single clear error — an unguarded ``abs(NaN) < eps`` comparison silently
    # evaluates to False, letting bogus "all reactions truly reversible"
    # decisions sneak through.
    try:
        fva = flux_variability_analysis(
            model, reaction_list=revs, fraction_of_optimum=0.0
        )
    except Exception as exc:  # noqa: BLE001 - solver-family agnostic
        raise RuntimeError(
            "constrain_reversible_reactions: FVA failed — the model is likely "
            "infeasible at fraction_of_optimum=0. Fix the infeasibility first "
            "(often a missing exchange or an over-constrained essential). "
            f"({exc})"
        ) from exc
    if fva[["minimum", "maximum"]].isna().any().any():
        raise RuntimeError(
            "constrain_reversible_reactions: FVA returned NaN ranges — the "
            "model is infeasible at fraction_of_optimum=0. Fix the infeasibility "
            "first (often a missing exchange or an over-constrained essential)."
        )

    changed: list[str] = []
    for rxn in revs:
        lo = fva.at[rxn.id, "minimum"]
        hi = fva.at[rxn.id, "maximum"]
        # Guard against ±inf ranges (unbounded objective): treat them as truly
        # reversible rather than "zero" by the abs(·) < eps check.
        if math.isinf(lo) or math.isinf(hi):
            continue
        min_zero, max_zero = abs(lo) < eps, abs(hi) < eps
        if min_zero == max_zero:  # both ~0 (blocked) or both nonzero (truly reversible)
            continue
        if max_zero:  # only reverse flux → flip to a forward reaction
            old_lb = rxn.lower_bound
            rxn.add_metabolites({m: -2 * c for m, c in rxn.metabolites.items()})
            rxn.bounds = (0.0, -old_lb)
            rxn.objective_coefficient = -rxn.objective_coefficient
        else:  # only forward flux
            rxn.lower_bound = 0.0
        changed.append(rxn.id)
    return changed


def group_linear_reactions(
    model: cobra.Model, *, reserved: Iterable[str] | None = None
) -> None:
    """Merge linear (single-producer, single-consumer) reaction chains.

    **Lossy**: gene-reaction associations are discarded (RAVEN does the same), since merged
    reactions have no meaningful combined GPR. The model is first made irreversible, then any
    metabolite that is produced by exactly one reaction and consumed by exactly one reaction is
    eliminated by merging the two reactions. Metabolites are visited in model order and the
    reaction with the lower index survives a merge, as in RAVEN's ``groupLinear``, so the
    survivor of a chain is its first reaction. ``reserved`` reaction ids are never merged
    away. Mutates in place.
    """
    reserved = set(reserved or [])

    # Lossy: drop all gene information.
    for rxn in model.reactions:
        rxn.gene_reaction_rule = ""
    for gene in list(model.genes):
        model.genes.remove(gene)

    convert_to_irreversible(model)

    # Worklist of metabolites to (re)consider for merging. Each metabolite
    # participating in a merge can expose new linear chains in its neighbours,
    # so touched mets are re-enqueued rather than restarting a full scan of
    # all metabolites after every merge.
    met_pos = {m: i for i, m in enumerate(model.metabolites)}
    rxn_pos = {r: i for i, r in enumerate(model.reactions)}
    pending: list = list(reversed(model.metabolites))  # popped in model order
    seen_in_pass: set = set()
    while pending:
        met = pending.pop()
        if met not in model.metabolites:  # removed in a previous merge
            continue
        rxns = list(met.reactions)
        if len(rxns) != 2 or any(r.id in reserved for r in rxns):
            continue
        r1, r2 = sorted(rxns, key=rxn_pos.__getitem__)
        c1, c2 = r1.get_coefficient(met), r2.get_coefficient(met)
        if (c1 > 0) == (c2 > 0):  # need one producer and one consumer
            continue
        ratio = abs(c1 / c2)
        new_lb = max(r1.lower_bound, r2.lower_bound / ratio)
        new_ub = min(r1.upper_bound, r2.upper_bound / ratio)
        new_obj = r1.objective_coefficient + r2.objective_coefficient * ratio
        # Re-enqueue every metabolite touched by either side — the merge can
        # turn neighbours into single-producer/consumer chains in turn.
        touched = {m for m in r1.metabolites} | {m for m in r2.metabolites}
        # Merge r2*ratio into r1; the shared metabolite cancels and is dropped.
        merged = {m: c * ratio for m, c in r2.metabolites.items()}
        merged[met] = -c1  # cancels exactly, without floating-point residue
        r1.add_metabolites(merged)
        model.remove_reactions([r2])
        r1.bounds = (new_lb, new_ub)
        r1.objective_coefficient = new_obj
        seen_in_pass.clear()
        for m in sorted(touched, key=met_pos.__getitem__, reverse=True):
            if m in model.metabolites and id(m) not in seen_in_pass:
                seen_in_pass.add(id(m))
                pending.append(m)
    # One terminal cleanup pass (cheap; only what remains).
    empty = [r for r in model.reactions if not r.metabolites]
    if empty:
        model.remove_reactions(empty)
    _prune_orphan_metabolites(model)


def simplify_model(
    model: cobra.Model,
    *,
    delete_zero_interval: bool = False,
    delete_dead_end: bool = False,
    delete_no_flux: bool = False,
    delete_duplicates: bool = False,
    group_linear: bool = False,
    constrain_reversible: bool = False,
    reserved: Iterable[str] | None = None,
    open_exchanges: bool = True,
) -> None:
    """Reduce a model by the selected simplification modes (RAVEN ``simplifyModel``).

    A single entry point mirroring RAVEN's ``simplifyModel`` boolean-flag interface, so a
    caller composes a simplification the same way RAVEN does (e.g. ftINIT's first
    simplification = ``delete_zero_interval + delete_dead_end + delete_no_flux``; its
    second = ``constrain_reversible``). Modes run in RAVEN's order: duplicates, zero-interval,
    topological dead-end, no-flux (FVA ``deleteMinMax``), constrain-reversible, then linear
    grouping, so reversible reactions that carry flux in one direction only are made
    irreversible before they are grouped. Each mode delegates to the dedicated function in
    this module. All operate in place. ``reserved`` reaction ids are never removed by the
    zero-interval, dead-end, no-flux, duplicate and group-linear modes; ``open_exchanges``
    is forwarded to the no-flux pass.

    RAVEN's ``deleteUnconstrained`` (boundary-metabolite removal) has no analogue here:
    cobra models use explicit boundary reactions rather than an ``unconstrained`` field.
    """
    if delete_duplicates:
        remove_duplicate_reactions(model, reserved=reserved)
    if delete_zero_interval:
        remove_zero_interval_reactions(model, reserved=reserved)
    if delete_dead_end:
        remove_dead_end_reactions(model, reserved=reserved)
    if delete_no_flux:
        remove_no_flux_reactions(model, open_exchanges=open_exchanges, reserved=reserved)
    if constrain_reversible:
        constrain_reversible_reactions(model)
    if group_linear:
        group_linear_reactions(model, reserved=reserved)
