"""Compartment assignment: flux-free score placement and materialised-FBA certification.

:func:`assign_compartments` places reactions into subcellular compartments from soft
``gene x compartment`` localisation scores and keeps the result functional. The placement
optimisation contains no flux model: with no flux variable and no growth constraint, a
tolerance-rounded gating binary has no flux to leak, so the placement cannot be certified by a
flux that the materialised model does not carry. The method has four steps:

1. a **flux-free placement master** that maximises the localisation score with mono-localisation;
2. a **structural confinement repair**: reactions sharing a non-transportable metabolite are
   co-located (or pinned to a fixed reaction's compartment), and transportable pools that a
   placement splits get star-topology transports;
3. **materialised-FBA certification**: a ``cobra`` FBA on the model that :func:`apply_assignment`
   builds, over the primary medium and every :class:`GrowthCondition`, confirms the placement is
   functional. The certificate is the acceptance test;
4. **feedback on failure**: optional gap-fill from a ``universal`` model, then (for a placement
   gap) a tightening of the placement, and a new solve until the placement is certified or the
   round budget is used.

The returned :class:`AssignmentProposal` has ``certified=True`` if and only if the materialised
model reaches the growth floor on every medium. Otherwise ``certified=False``, and the biomass flux
of each medium is in :attr:`AssignmentProposal.growths`.

See the `compartment-assignment redesign
<https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/localization/localization_redesign.md>`_
on raven-gecko-parity for the design rationale.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import cobra
from cobra.util import linear_reaction_coefficients
from optlang.symbolics import Real, add, mul

from raven_toolbox.localization.assign import (
    AssignmentProposal,
    GrowthCondition,
    _base_met,
    _check_unique_keys,
    _rxn_compartment,
    apply_assignment,
)
from raven_toolbox.localization.scores import LocalizationScores

__all__ = ["assign_compartments"]

_PRIMARY = "__primary__"


# --------------------------------------------------------------------------- shared scope
@dataclass
class _Scope:
    """Movable and pinned reactions, gene coupling and metabolite keying used by
    :func:`assign_compartments`."""

    biomass_reaction: str
    min_growth: float
    movable: list
    movable_ids: set
    pinned: list
    pinned_comp: dict
    base: Callable
    transp: set
    score_df: object
    genes_in_scope: set
    unplaced: list
    gene_rxns: dict
    touches: dict


def _prepare_scope(model, scores, reactions_to_relocate, *, transportable, base_metabolite,
                   biomass_reaction, min_growth) -> _Scope:
    """Resolve the biomass reaction, growth floor, split into movable and pinned reactions, genes
    and metabolite keying.
    """
    if biomass_reaction is None:
        obj = linear_reaction_coefficients(model)
        if not obj:
            raise ValueError("model has no objective; pass biomass_reaction=...")
        biomass_reaction = max(obj, key=lambda r: obj[r]).id
    if min_growth is None:
        with model:
            model.objective = biomass_reaction
            base_opt = model.slim_optimize(error_value=0.0)
        if not base_opt or base_opt <= 0:
            raise ValueError(
                "the draft model does not grow; cannot set a growth floor (pass min_growth=...)."
            )
        min_growth = 0.1 * base_opt

    to_relocate = set(reactions_to_relocate)
    movable = []
    for rid in sorted(to_relocate):
        r = model.reactions.get_by_id(rid)
        if r.boundary or _rxn_compartment(r) is None or r.id == biomass_reaction:
            continue
        movable.append(r)
    movable_ids = {r.id for r in movable}
    pinned = [r for r in model.reactions if r.id not in movable_ids]
    pinned_comp = {r.id: _rxn_compartment(r) for r in pinned}

    base = base_metabolite if base_metabolite is not None else _base_met
    movable_base = {base(m) for r in movable for m in r.metabolites}
    transp = set(movable_base) if transportable is None else set(transportable) & movable_base

    score_df = scores.df
    genes_in_scope: set[str] = set()
    unplaced: list[str] = []
    for r in movable:
        gs = [g.id for g in r.genes]
        scored = [g for g in gs if g in score_df.index]
        if gs and not scored:
            unplaced.append(r.id)
        genes_in_scope.update(scored)
    gene_rxns: dict[str, list[str]] = {g: [] for g in genes_in_scope}
    for r in movable:
        for g in {gg.id for gg in r.genes} & genes_in_scope:
            gene_rxns[g].append(r.id)

    touches = _touch_map(movable + pinned, base)
    return _Scope(biomass_reaction=biomass_reaction, min_growth=min_growth, movable=movable,
                  movable_ids=movable_ids, pinned=pinned, pinned_comp=pinned_comp, base=base,
                  transp=transp, score_df=score_df, genes_in_scope=genes_in_scope,
                  unplaced=unplaced, gene_rxns=gene_rxns, touches=touches)


# --------------------------------------------------------------------------- master
def _pin_deterministic(prob, opt) -> None:
    """Pin solver parameters so a degenerate placement objective gives the same solution on every
    run and on every machine. In a benchmark of the yeast master (identical model, one parameter
    varied at a time), three settings changed which co-optimal placement Gurobi returned: thread
    count, seed and presolve level. The MIP gap changed the result only when loosened above ~1e-2,
    which also gives non-optimal solutions, and no tolerance changed it. The function sets one
    thread, a fixed seed, presolve level 2 (the optimizeProb default of the MATLAB port), a MIP gap
    of 0 (exact optimum) and an integrality tolerance of 1e-9. Only the Gurobi backend is
    configured; other backends keep their defaults."""
    if "gurobi" not in getattr(prob, "__name__", ""):
        return
    try:
        gp_model = opt.problem
        gp_model.update()
        gp_model.setParam("Threads", 1)
        gp_model.setParam("Seed", 0)
        gp_model.setParam("Presolve", 2)
        gp_model.setParam("MIPGap", 0.0)
        gp_model.setParam("IntFeasTol", 1e-9)
    except Exception:  # noqa: BLE001 — reproducibility hint, never fatal
        pass


def _score(score_df, g: str, c: str) -> float:
    if c not in score_df.columns or g not in score_df.index:
        return 0.0
    v = score_df.at[g, c]
    return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


# Tie-break weight for the placement master's second pass: small enough never to override a real
# per-reaction score difference (DeepLoc scores are ~[0, 1] summed over a reaction's genes), just enough
# to send a genes-free or score-tied reaction to the default compartment deterministically.
_DEFAULT_COMPARTMENT_PRIOR = 1e-3


def _solve_placement_master(
    model, movable, genes_in_scope, gene_rxns, score_df, compartments, default_compartment, *,
    multi_compartment_penalty, forced, colocation_groups, time_limit,
):
    """Flux-free score-maximising placement MILP (mono-localisation).

    Returns ``(status, placements, gene_compartments)``. ``forced`` pins ``x[r, c] = 1`` (align a
    reaction to a fixed compartment); ``colocation_groups`` tie sets of movable reactions to a
    shared compartment (``x[a, c] = x[b, c]``), and the score objective picks which one. The master
    has no flux variable and no growth floor.

    The problem is solved in two lexicographic passes. The primary objective places genes by score
    and penalises spread, which leaves the per-reaction placement ``x`` a free co-optimum. The
    second pass fixes the gene layout and, for each reaction, maximises the summed score of the
    reaction's genes for the compartment chosen in ``x``, plus a prior of 1e-3 for
    ``default_compartment``. The reaction placement is then deterministic.
    """
    model.solver  # noqa: B018 — initialise the solver so model.problem is usable
    prob = model.problem
    opt = prob.Model()
    # Build the MILP in a canonical order so the solver is handed the same
    # problem every run and across the MATLAB port: movable is already sorted by
    # id, but genes_in_scope is a set whose iteration order is randomised per
    # process (string hash randomisation), which would otherwise reorder the
    # variables/constraints and let a degenerate objective pick a different
    # co-optimal placement each run. Iterate it sorted everywhere.
    genes_sorted = sorted(genes_in_scope)
    x = {(r.id, c): prob.Variable(f"x_{r.id}_{c}", type="binary")
         for r in movable for c in compartments}
    y = {(g, c): prob.Variable(f"y_{g}_{c}", type="binary")
         for g in genes_sorted for c in compartments}
    cons: list = []

    for r in movable:
        cons.append(prob.Constraint(add([x[r.id, c] for c in compartments]),
                                    lb=1.0, ub=1.0, name=f"place_{r.id}"))
        for g in sorted({gg.id for gg in r.genes} & genes_in_scope):
            for c in compartments:
                cons.append(prob.Constraint(x[r.id, c] - y[g, c], ub=0.0,
                                            name=f"couple_{r.id}_{g}_{c}"))
    for g in genes_sorted:
        cons.append(prob.Constraint(add([y[g, c] for c in compartments]),
                                    lb=1.0, name=f"gene1_{g}"))
        for c in compartments:
            expr = add([x[rid, c] for rid in gene_rxns[g]]) if gene_rxns[g] else Real(0.0)
            cons.append(prob.Constraint(y[g, c] - expr, ub=0.0, name=f"has_{g}_{c}"))

    for rid, c_force in forced.items():
        if (rid, c_force) in x:
            cons.append(prob.Constraint(x[rid, c_force], lb=1.0, ub=1.0, name=f"force_{rid}"))
    for gi, group in enumerate(colocation_groups):
        members = [rid for rid in group if any((rid, c) in x for c in compartments)]
        for c in compartments:
            for a, b in zip(members, members[1:], strict=False):
                cons.append(prob.Constraint(x[a, c] - x[b, c], lb=0.0, ub=0.0,
                                            name=f"colo_{gi}_{a}_{b}_{c}"))

    opt.add(list(x.values()) + list(y.values()) + cons)

    # Primary objective: place each gene in the compartment(s) its DeepLoc score favours, penalising
    # spread across compartments. This fixes the gene layout but leaves the *reaction* placement x a free
    # co-optimum (the objective never mentions x), which a deterministic solver resolves to an arbitrary
    # vertex -- yeast-GEM reaction agreement 52.8%. A lexicographically-lower second pass then chooses x.
    primary = []
    for g in genes_sorted:
        for c in compartments:
            s = _score(score_df, g, c)
            if s:
                primary.append(mul([Real(s), y[g, c]]))
    if multi_compartment_penalty:
        for v in y.values():
            primary.append(mul([Real(-multi_compartment_penalty), v]))
    opt.objective = prob.Objective(add(primary) if primary else Real(0.0), direction="max")
    if time_limit is not None:
        opt.configuration.timeout = int(time_limit)
    _pin_deterministic(prob, opt)
    opt.optimize()
    if opt.status not in ("optimal", "feasible", "suboptimal", "time_limit"):
        return opt.status, {}, {}

    # Lexicographic second pass. Fix the gene layout to the primary optimum -- fixing the *solution* (each
    # y binary), not the objective value, so there is no near-optimal tolerance to tune -- then place each
    # reaction in the compartment its own enzymes are predicted to occupy: reward x[r, c] by the summed
    # gene score of r's genes for c, with a small prior for `default_compartment` so genes-free and
    # score-tied reactions fall there deterministically rather than to an arbitrary co-optimal vertex.
    # The gene layout (agreement, multi-compartment consolidation) is untouched; only reaction placement,
    # which was free, is now meaningful.
    try:  # read every primal before touching any bound -- the first bound change discards the solution
        y_star = {k: 1.0 if (v.primal or 0.0) >= 0.5 else 0.0 for k, v in y.items()}
    except (AttributeError, ValueError):
        return "no_incumbent", {}, {}
    for k, v in y.items():
        v.lb = v.ub = y_star[k]
    secondary = []
    for r in movable:
        r_genes = sorted({gg.id for gg in r.genes} & genes_in_scope)
        for c in compartments:
            w = sum(_score(score_df, g, c) for g in r_genes)
            if c == default_compartment:
                w += _DEFAULT_COMPARTMENT_PRIOR
            if w:
                secondary.append(mul([Real(w), x[r.id, c]]))
    opt.objective = prob.Objective(add(secondary) if secondary else Real(0.0), direction="max")
    _pin_deterministic(prob, opt)
    opt.optimize()
    if opt.status not in ("optimal", "feasible", "suboptimal", "time_limit"):
        return opt.status, {}, {}
    try:
        placements = {r.id: [c for c in compartments if (x[r.id, c].primal or 0.0) >= 0.5]
                      for r in movable}
        gene_comps = {g: [c for c in compartments if (y[g, c].primal or 0.0) >= 0.5]
                      for g in genes_in_scope}
    except (AttributeError, ValueError):
        return "no_incumbent", {}, {}
    return opt.status, placements, gene_comps


# --------------------------------------------------------------------------- repair
def _placed_compartment(rid: str, placements: dict[str, list[str]], pinned_comp: dict[str, str]):
    """The single compartment a reaction occupies (mono placement or a pinned
    single-compartment)."""
    if rid in placements:
        cs = placements[rid]
        return cs[0] if len(cs) == 1 else None
    return pinned_comp.get(rid)


def _touch_map(reactions, base):
    """``reaction_id -> {base_metabolite}`` for the reactions given."""
    return {r.id: {base(m) for m in r.metabolites} for r in reactions}


def _diagnose_confinement(
    movable, pinned_comp, placements, touches, transp, compartments, protected=frozenset(),
):
    """Find non-transportable metabolites that a placement splits across compartments.

    Returns ``(forced, groups, relaxed)``: ``forced`` pins movable reactions to a pinned reaction's
    compartment; ``groups`` co-locate all-movable reaction sets sharing a confined metabolite;
    ``relaxed`` lists confined metabolites that must be transported. A confined metabolite
    touched in one compartment only is not listed.

    ``protected`` are reactions pinned by the growth-failure feedback
    (:func:`_diagnose_growth_gap`). Confinement does not overwrite their placement, because
    overwriting would let the two diagnosers move a reaction between two compartments until the
    round budget is used. When a confined metabolite is shared with a protected reaction, it is
    relaxed (transported) instead; the protected placement stays and the pool is reconnected.
    """
    movable_ids = {r.id for r in movable}
    # confined base -> {compartment: [rids]}
    used: dict[str, dict[str, list[str]]] = {}
    for rid, bases in touches.items():
        comp = _placed_compartment(rid, placements, pinned_comp)
        if comp is None:  # multi-compartment reaction (e.g. an existing transport) — bridges pools
            continue
        for b in bases:
            if b in transp:
                continue  # transportable — handled by a transport, not confinement
            used.setdefault(b, {}).setdefault(comp, []).append(rid)

    forced: dict[str, str] = {}
    groups: list[list[str]] = []
    relaxed: set[str] = set()
    for b, by_comp in used.items():
        if len(by_comp) <= 1:
            continue  # not split
        touching = [rid for rids in by_comp.values() for rid in rids]
        pinned_comps = {comp for comp, rids in by_comp.items()
                        if any(rid not in movable_ids for rid in rids)}
        movers = [rid for rid in touching if rid in movable_ids]
        if len(pinned_comps) > 1 or any(rid in protected for rid in touching):
            relaxed.add(b)  # cannot co-locate (pinned into >=2 comps, or shares a protected pin)
        elif pinned_comps:
            target = next(iter(pinned_comps))
            for rid in movers:
                forced[rid] = target
        elif movers:
            groups.append(sorted(set(movers)))
    return forced, groups, relaxed


def _split_transports(placements, pinned_comp, touches, transp, default_compartment, relaxed):
    """Star-topology transports for transportable pools that a placement splits across compartments.

    A base metabolite touched by single-compartment reactions in more than one compartment needs a
    transport into each non-default compartment (through ``default_compartment``). ``relaxed``
    confined metabolites (which could not be co-located) are transported too.
    """
    used: dict[str, set[str]] = {}
    for rid, bases in touches.items():
        comp = _placed_compartment(rid, placements, pinned_comp)
        if comp is None:
            continue
        for b in bases:
            if b in transp or b in relaxed:
                used.setdefault(b, set()).add(comp)
    out: list[tuple[str, str]] = []
    for b, comps in used.items():
        if len(comps) <= 1:
            continue
        for c in sorted(comps):
            if c != default_compartment:
                out.append((b, c))
    return out


# --------------------------------------------------------------------------- certification
def _apply_medium(applied: cobra.Model, medium: Mapping[str, float] | None) -> None:
    """Close all uptake, then open the listed exchanges, so ``medium`` fully specifies the
    environment.

    ``None`` leaves the model's own medium unchanged (the primary condition). The bound changes are
    reverted when the call is made inside a ``with applied:`` block.
    """
    if medium is None:
        return
    for r in applied.reactions:
        if r.boundary and r.lower_bound < 0:
            r.lower_bound = 0.0
    for rid, up in medium.items():
        if rid in applied.reactions:
            applied.reactions.get_by_id(rid).lower_bound = -abs(float(up))


def _grow_on(applied: cobra.Model, biomass_id: str, medium: Mapping[str, float] | None) -> float:
    with applied:
        _apply_medium(applied, medium)
        applied.objective = biomass_id
        return applied.slim_optimize(error_value=0.0) or 0.0


def _certify(applied, biomass_id, min_growth, conditions) -> tuple[bool, dict[str, float]]:
    """Materialised-FBA certification: biomass flux per medium, and whether every floor is met."""
    growths = {_PRIMARY: _grow_on(applied, biomass_id, None)}
    ok = growths[_PRIMARY] >= min_growth - 1e-9
    for gc in conditions:
        g = _grow_on(applied, biomass_id, gc.medium)
        growths[gc.name] = g
        ok = ok and g >= gc.min_growth - 1e-9
    return ok, growths


def _usable_transports(applied, proposal, biomass_id, min_growth, conditions, default_compartment):
    """Drop transports that can carry no flux in any certification medium. Removing a blocked
    reaction does not change the FBA of that medium. Returns the pruned transport list."""
    from cobra.flux_analysis import find_blocked_reactions

    tr_of = {}  # transport reaction id -> (base, compartment)
    for n, (b, c) in enumerate(proposal.added_transports):
        tr_of[f"tr_{n}_{c}"] = (b, c)
    keep: set[tuple[str, str]] = set()
    media = [None] + [gc.medium for gc in conditions]
    for medium in media:
        with applied:
            _apply_medium(applied, medium)
            applied.objective = biomass_id
            blocked_now = set(find_blocked_reactions(applied, processes=1))
            unblocked = set(applied.reactions.list_attr("id")) - blocked_now
        for trid, bc in tr_of.items():
            if trid in unblocked:
                keep.add(bc)
    return [bc for bc in proposal.added_transports if bc in keep]


def _minimize_transports(model, proposal, sc, conditions, default_compartment, universal):
    """Prune the transport and gap-fill set to the reactions that carry flux in a **parsimonious
    FBA** solution, then re-certify.

    A reaction with zero flux in a feasible solution can be removed without changing that solution,
    so the pruned model still grows. pFBA (minimise total flux at maximum biomass) keeps the subset
    that carries flux (yeast-GEM: ~1260 -> ~280 transports) in one solve. An exact minimum-count
    subset would need a MILP, and pFBA is used instead. Returns the input proposal if pFBA raises an
    error or if the pruned set misses the growth floor on any medium (a transport that a growth
    condition needs may carry no flux on the primary medium).
    """
    from cobra.flux_analysis import pfba

    applied = apply_assignment(model, proposal, default_compartment=default_compartment,
                               base_metabolite=sc.base, universal=universal)
    applied.objective = sc.biomass_reaction
    tr_ids = {f"tr_{n}_{c}": (b, c)  # transport reaction id -> (base, compartment)
              for n, (b, c) in enumerate(proposal.added_transports)}
    try:
        sol = pfba(applied)
    except Exception:  # noqa: BLE001 — pFBA infeasible / backend issue: keep the input set
        return proposal
    minimized = AssignmentProposal(
        placements=proposal.placements, gene_compartments=proposal.gene_compartments,
        added_transports=[bc for rid, bc in tr_ids.items() if abs(sol.fluxes.get(rid, 0.0)) > 1e-7],
        added_reactions=[rid for rid in proposal.added_reactions
                         if abs(sol.fluxes.get(rid, 0.0)) > 1e-7],
        unplaced_reactions=proposal.unplaced_reactions, min_growth=proposal.min_growth)
    applied_min = apply_assignment(model, minimized, default_compartment=default_compartment,
                                   base_metabolite=sc.base, universal=universal)
    ok, growths = _certify(applied_min, sc.biomass_reaction, proposal.min_growth, conditions)
    if not ok:
        return proposal  # a shared transport a condition needed was pruned — keep the safe set
    minimized.growths = growths
    minimized.certified = True
    minimized.status = "certified"
    return minimized


def _enrich_multilocalization(model, proposal, scores, sc, conditions, default_compartment,
                              universal, threshold, eps, loopless):
    """Add FVA-validated multi-compartment placements to a certified mono proposal.

    For each placed reaction, a second compartment is proposed if one of its genes has a DeepLoc
    score ``>= threshold`` there (see the `multi-localisation design
    <https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/localization/multi_localization_design.md>`_).
    Every candidate is materialised as a duplicate reaction, and a **loopless FVA** keeps only the
    duplicates that can carry flux ``>= eps`` in a solution that meets the growth floor. A duplicate
    that carries no flux is dropped. Returns the enriched proposal, or the input if no duplicate
    survives, if the FVA fails or if the enriched model misses the growth floor on a medium.
    """
    from cobra.flux_analysis import flux_variability_analysis

    score_df = scores.df
    compartments = sorted(set(model.compartments) | set(scores.compartments))
    cand: dict[str, set[str]] = {}
    for rid, comps in proposal.placements.items():
        if not comps or rid not in model.reactions:
            continue
        placed = set(comps)
        genes = [g.id for g in model.reactions.get_by_id(rid).genes if g.id in score_df.index]
        for c2 in compartments:
            if c2 in placed:
                continue
            if max((_score(score_df, g, c2) for g in genes), default=0.0) >= threshold:
                cand.setdefault(rid, set()).add(c2)
    if not cand:
        return proposal

    trial = AssignmentProposal(
        placements={rid: list(comps) + sorted(cand.get(rid, ()))
                    for rid, comps in proposal.placements.items()},
        gene_compartments=proposal.gene_compartments, added_transports=proposal.added_transports,
        added_reactions=proposal.added_reactions, unplaced_reactions=proposal.unplaced_reactions,
        min_growth=proposal.min_growth)
    applied = apply_assignment(model, trial, default_compartment=default_compartment,
                               base_metabolite=sc.base, universal=universal)
    dup_ids = {f"{rid}_{c2}": (rid, c2) for rid, cs in cand.items() for c2 in cs}
    present = [d for d in dup_ids if d in applied.reactions]
    if not present:
        return proposal

    # FVA over just the candidate duplicates, in solutions that still meet the growth floor.
    applied.reactions.get_by_id(sc.biomass_reaction).lower_bound = float(proposal.min_growth)
    try:
        # processes=1: run FVA in-process. Its default multiprocessing spawns workers that re-import
        # the caller's module, which crashes on Windows unless the caller guards with
        # ``if __name__ == "__main__"`` — a library must not impose that on its callers.
        fva = flux_variability_analysis(applied, reaction_list=present,
                                        loopless="cycleFreeFlux" if loopless else None,
                                        fraction_of_optimum=0.0, processes=1)
    except Exception:  # noqa: BLE001 — FVA backend issue: keep the mono proposal
        return proposal
    accepted: dict[str, list[str]] = {}
    for did in present:
        if max(abs(fva.at[did, "minimum"]), abs(fva.at[did, "maximum"])) >= eps:
            rid, c2 = dup_ids[did]
            accepted.setdefault(rid, []).append(c2)
    if not accepted:
        return proposal

    placements = {rid: list(comps) + sorted(accepted.get(rid, []))
                  for rid, comps in proposal.placements.items()}
    gene_comps: dict[str, set[str]] = {}
    for rid, comps in placements.items():
        if rid in model.reactions:
            for g in model.reactions.get_by_id(rid).genes:
                gene_comps.setdefault(g.id, set()).update(comps)
    enriched = AssignmentProposal(
        placements=placements, gene_compartments={g: sorted(cs) for g, cs in gene_comps.items()},
        added_transports=proposal.added_transports, added_reactions=proposal.added_reactions,
        unplaced_reactions=proposal.unplaced_reactions, min_growth=proposal.min_growth)
    app2 = apply_assignment(model, enriched, default_compartment=default_compartment,
                            base_metabolite=sc.base, universal=universal)
    ok, growths = _certify(app2, sc.biomass_reaction, proposal.min_growth, conditions)
    if not ok:
        return proposal  # duplicates only add capability, so this is a safety net, not expected
    enriched.growths = growths
    enriched.certified = True
    enriched.status = "certified"
    return enriched


# --------------------------------------------------------------------------- entry point
def assign_compartments(
    model: cobra.Model,
    scores: LocalizationScores,
    reactions_to_relocate: Iterable[str],
    *,
    default_compartment: str = "c",
    transport_cost: float | Mapping[str, float] = 0.5,
    multi_compartment_penalty: float = 0.5,
    transportable: Iterable[str] | None = None,
    base_metabolite: Callable[[cobra.Metabolite], str] | None = None,
    universal: cobra.Model | None = None,
    biomass_reaction: str | None = None,
    min_growth: float | None = None,
    growth_conditions: Iterable[GrowthCondition] | None = None,
    max_rounds: int = 8,
    time_limit: float | None = None,
    prune_transports: bool = True,
    minimize_transports: bool = False,
    multi_localize: bool = False,
    multi_localize_threshold: float = 0.7,
    multi_localize_eps: float = 1e-6,
    multi_localize_loopless: bool = True,
) -> AssignmentProposal:
    """Assign ``reactions_to_relocate`` to compartments and certify functionality with an FBA.

    A flux-free score MILP decides the placement. A repair step adds transports (and gap-fill
    reactions when ``universal`` is given), and the result is certified by an FBA on the applied
    model. When no tightening is available or the round budget ends, the best proposal is
    returned uncertified (see the module docstring and the
    `compartment-assignment redesign
    <https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/localization/localization_redesign.md>`_
    on raven-gecko-parity).

    Parameters
    ----------
    model : cobra.Model
        The draft model. It is not modified. Raises :class:`ValueError` when two metabolites in one
        compartment have the same ``base_metabolite`` key, and when ``min_growth`` is ``None``
        and the model does not grow.
    scores : LocalizationScores
        Gene x compartment localisation scores. A reaction whose genes all have no row in the
        scores is listed in ``unplaced_reactions`` of the result.
    reactions_to_relocate : iterable of str
        Ids of the reactions that the placement may move. Boundary reactions, the biomass reaction
        and reactions whose metabolites do not lie in exactly one compartment (existing
        multi-compartment transports) are skipped. All other reactions keep their compartment.
    default_compartment : str
        Hub compartment of the star-topology transports, and the compartment that receives gene-free
        and score-tied reactions (default ``"c"``). Raises :class:`ValueError` when it is neither a
        compartment of ``model`` nor of ``scores``.
    transport_cost : float or mapping of str to float
        Accepted and not used (default ``0.5``): the placement master is flux-free and score-only,
        and ``minimize_transports`` prunes transports by flux (pFBA), not by cost.
    multi_compartment_penalty : float
        Weight (default ``0.5``) subtracted in the primary placement pass for each gene-compartment
        assignment. A higher value places a gene in fewer compartments. A value of ``0`` applies
        no penalty.
    transportable : iterable of str or None
        Base ids of the metabolites eligible for a cross-compartment transport. ``None`` (default)
        allows every metabolite touched by a relocatable reaction. A narrower set forbids transports
        for the other metabolites, and the reactions that share them are co-located by the
        confinement repair.
    base_metabolite : callable or None
        Maps a metabolite to its compartment-agnostic key (default ``None``). The default key is
        the base id that :func:`~raven_toolbox.manipulation.merge_compartments` stored on a merged
        metabolite, otherwise the id without the suffix of the metabolite's own compartment.
    universal : cobra.Model or None
        Reaction database for gap-fill (default ``None``). When the primary medium falls short of
        the growth floor, the repair adds reactions from this model. When ``None``, no gap-fill is
        done. The metabolite ids of ``universal`` have to match those of ``model``.
    biomass_reaction : str or None
        Id of the reaction whose flux is the growth measure (default ``None``). When ``None``, the
        reaction with the largest objective coefficient is used. Raises :class:`ValueError` when
        the model has no objective.
    min_growth : float or None
        Biomass flux that the materialised model has to reach on the primary medium (default
        ``None``). When ``None``, the floor is 10 % of the maximum flux of ``biomass_reaction`` in
        ``model``.
        Raises :class:`ValueError` when ``biomass_reaction`` cannot carry flux in ``model``.
    growth_conditions : iterable of GrowthCondition or None
        Extra media on which the materialised model has to reach the floor of the condition
        (default ``None``, the primary medium only). The growth of each condition is reported in
        ``growths`` of the result.
    max_rounds : int
        Maximum number of placement rounds (confinement repair and growth-failure feedback; default
        ``8``). When every metabolite is transportable, the confinement repair adds no rounds and
        the common genome-scale case certifies in one round.
    time_limit : float or None
        Solver time limit in seconds for each placement MILP pass, truncated to an integer
        (default ``None``, no limit). A solution found at the limit is used.
    prune_transports : bool
        If True (default), drop transports that are blocked in every certification medium.
    minimize_transports : bool
        If True, prune the transport and gap-fill set after certifying to the reactions that carry
        flux in a parsimonious FBA solution (:func:`_minimize_transports`; yeast-GEM: ~1260 ->
        ~280 transports). The pruned set is certified again against every medium, and the
        un-minimised set is kept if the pruning removed a transport that a growth condition needs.
        Default False; the option adds one pFBA and one certification.
    multi_localize : bool
        If True, add multi-compartment placements after certifying
        (:func:`_enrich_multilocalization`): for each reaction, a second compartment is proposed if
        its gene has a DeepLoc score ``>= multi_localize_threshold`` there, all candidates are
        materialised, and one loopless FVA keeps those that can carry a flux of at least
        ``multi_localize_eps`` in a biomass-supporting solution. The placement master stays
        mono-localised, and the FVA removes duplicates that carry no flux. Default False; the option
        adds one FVA.

        The gate (DeepLoc score above the threshold and flux in the FVA) removes duplicates that
        carry no flux, and does not test whether the enzyme is dual-localised. Many reactions can
        carry some flux in a second compartment without being dual, so many reactions are
        multi-placed (in the yeast-GEM run, about 700 of 2300 reactions at the default threshold).
        The gate recovers the DeepLoc-visible dual enzymes (for example HTS1, VAS1, GLR1) and
        misses enzymes that are dual by post-translational retro-translocation (FUM1, ACO1), whose
        second compartment carries no DeepLoc signal. A candidate source based on function (a
        metabolite required in the second compartment) is not implemented. The result is a lower
        bound on multi-localisation, not a precise dual-localisation call.
    multi_localize_threshold : float
        DeepLoc score that a gene needs in a second compartment for that placement to be proposed
        (default ``0.7``). A higher value gives fewer duplicates with higher confidence and misses
        duals (yeast-GEM: 0.9 drops HTS1, VAS1 and GLR1; 0.7 keeps them).
    multi_localize_eps : float
        Flux magnitude that a proposed duplicate must reach in the FVA to be kept (default
        ``1e-6``).
    multi_localize_loopless : bool
        If True (default), use loopless FVA, so a duplicate that is active only in a futile cycle is
        not kept. False gives a faster check that permits cycles.

    Returns
    -------
    AssignmentProposal
        ``certified`` is set if and only if the materialised model reached the growth floor on every
        medium, and ``growths`` gives the biomass flux per medium (``"__primary__"`` and each
        condition name). ``status`` is ``"certified"`` or ``"uncertified"`` (or a solver status if
        the master failed).
    """
    conditions = list(growth_conditions or ())
    compartments = sorted(set(model.compartments) | set(scores.compartments))
    if default_compartment not in compartments:
        raise ValueError(f"default_compartment={default_compartment!r} not in {compartments}")

    _check_unique_keys(model.metabolites, base_metabolite or _base_met)
    sc = _prepare_scope(model, scores, reactions_to_relocate, transportable=transportable,
                        base_metabolite=base_metabolite, biomass_reaction=biomass_reaction,
                        min_growth=min_growth)
    biomass_reaction, min_growth = sc.biomass_reaction, sc.min_growth
    movable, movable_ids, pinned_comp = sc.movable, sc.movable_ids, sc.pinned_comp
    base, transp, score_df = sc.base, sc.transp, sc.score_df
    genes_in_scope, unplaced = sc.genes_in_scope, sc.unplaced
    gene_rxns, touches = sc.gene_rxns, sc.touches

    # ---- place -> repair -> certify (-> tighten) loop -------------------------
    forced: dict[str, str] = {}
    groups: list[list[str]] = []
    gap_pinned: set[str] = set()  # reactions pinned by growth-failure feedback (confinement defers)
    seen: set = set()             # placement configurations already tried (cycle backstop)
    best = AssignmentProposal(min_growth=min_growth, status="uncertified",
                              unplaced_reactions=unplaced)

    for _round in range(max_rounds):
        # Break if this exact (forced, groups) configuration has already been tried: the master is
        # deterministic, so a repeat means the diagnosers are cycling — return the best result so far
        # rather than burning the remaining budget.
        signature = (frozenset(forced.items()), frozenset(frozenset(g) for g in groups))
        if signature in seen:
            return best
        seen.add(signature)

        status, placements, gene_comps = _solve_placement_master(
            model, movable, genes_in_scope, gene_rxns, score_df, compartments, default_compartment,
            multi_compartment_penalty=multi_compartment_penalty,
            forced=forced, colocation_groups=groups, time_limit=time_limit)
        if not placements:
            best.status = status if status not in ("optimal", "feasible") else "no_incumbent"
            return best

        # structural confinement fixpoint: re-solve until no confined metabolite is split. Growth-gap
        # pins are protected so confinement relaxes (transports) rather than overwriting them.
        new_forced, new_groups, relaxed = _diagnose_confinement(
            movable, pinned_comp, placements, touches, transp, compartments, protected=gap_pinned)
        added_forced = {k: v for k, v in new_forced.items()
                        if forced.get(k) != v and k not in gap_pinned}
        group_keys = {tuple(g) for g in groups}
        added_groups = [g for g in new_groups if tuple(g) not in group_keys]
        if added_forced or added_groups:
            forced.update(added_forced)
            groups.extend(added_groups)
            continue

        transports = _split_transports(placements, pinned_comp, touches, transp,
                                       default_compartment, relaxed)
        proposal = AssignmentProposal(
            placements={rid: cs for rid, cs in placements.items() if cs},
            gene_compartments={g: cs for g, cs in gene_comps.items() if cs},
            added_transports=transports, unplaced_reactions=unplaced,
            min_growth=min_growth, status="uncertified")

        applied = apply_assignment(model, proposal, default_compartment=default_compartment,
                                   base_metabolite=base, universal=universal)
        ok, growths = _certify(applied, biomass_reaction, min_growth, conditions)

        # feedback: gap-fill from the universal model if the primary medium fell short
        if not ok and universal is not None and growths[_PRIMARY] < min_growth - 1e-9:
            added = _gapfill(applied, universal, biomass_reaction, min_growth)
            if added:
                proposal.added_reactions = added
                applied = apply_assignment(model, proposal,
                                           default_compartment=default_compartment,
                                           base_metabolite=base, universal=universal)
                ok, growths = _certify(applied, biomass_reaction, min_growth, conditions)

        proposal.growths = growths
        if ok:
            if prune_transports and transports:
                proposal.added_transports = _usable_transports(
                    applied, proposal, biomass_reaction, min_growth, conditions,
                    default_compartment)
            proposal.certified = True
            proposal.status = "certified"
            if multi_localize:
                proposal = _enrich_multilocalization(
                    model, proposal, scores, sc, conditions, default_compartment, universal,
                    multi_localize_threshold, multi_localize_eps, multi_localize_loopless)
            if minimize_transports and (proposal.added_transports or proposal.added_reactions):
                proposal = _minimize_transports(model, proposal, sc, conditions,
                                                default_compartment, universal)
            return proposal

        # keep the best partial result (largest primary growth) to return if the budget runs out
        if growths[_PRIMARY] > best.growths.get(_PRIMARY, -1.0):
            best = proposal

        # placement-gap feedback: pin the sole producer of a stranded biomass precursor
        fb = _diagnose_growth_gap(applied, proposal, biomass_reaction, movable_ids,
                                  pinned_comp, touches, base, compartments)
        if fb and forced.get(fb[0]) != fb[1]:
            forced[fb[0]] = fb[1]
            gap_pinned.add(fb[0])  # protect from confinement so the two diagnosers cannot oscillate
            continue
        return best  # no tightening available — honest uncertified result

    return best


def _gapfill(applied, universal, biomass_reaction, min_growth) -> list[str]:
    """The ``universal`` reactions whose addition restores the primary growth floor, from a
    flux-based fill.

    On a working copy, every ``universal`` candidate not already present is added in one batch, and
    the function checks that the floor is reachable. It holds biomass at the floor, runs pFBA
    (maximise biomass, then minimise total flux) and returns the added reactions that carry flux,
    sorted. The set is flux-parsimonious and not guaranteed to have the minimum number of reactions.
    The caller re-certifies the result with FBA.

    The fill is an LP (pFBA) and not cobra's indicator MILP: at genome scale
    ``cobra.flux_analysis.gapfill`` failed to find a valid fill in most cases even when the
    restoring reaction was present in the universal model, and on single-reaction knockout recovery
    the LP restores every case where that MILP restores under half.

    **Namespace.** Candidates are matched to the model by metabolite id, so the universal model must
    share the draft's metabolite namespace. A candidate whose metabolites do not resolve becomes a
    dead end that cannot carry flux and is left out. A warning is issued when more than half of the
    candidates share no metabolite id with the model.
    """
    if biomass_reaction not in applied.reactions:
        return []
    floor = max(min_growth, 1e-4)
    try:
        from cobra.flux_analysis import pfba
        # Work on a copy: no context-manager rollback (whose failure on some optlang backends would
        # otherwise discard a valid result), and the caller rebuilds `applied` from the proposal anyway.
        work = applied.copy()
        candidates = [u for u in universal.reactions if u.id not in work.reactions]
        if not candidates:
            return []
        fresh = [cobra.Reaction(u.id, name=u.name, lower_bound=u.lower_bound, upper_bound=u.upper_bound)
                 for u in candidates]
        work.add_reactions(fresh)  # one batch — per-reaction adds are super-linear at scale
        fully_unresolved = 0
        for nr, urxn in zip(fresh, candidates, strict=True):
            stoich, n_unres = {}, 0
            for met, coeff in urxn.metabolites.items():
                if met.id in work.metabolites:
                    stoich[work.metabolites.get_by_id(met.id)] = coeff
                else:
                    n_unres += 1
                    stoich[cobra.Metabolite(met.id, name=met.name, formula=met.formula,
                                            charge=met.charge, compartment=met.compartment)] = coeff
            nr.add_metabolites(stoich)
            fully_unresolved += n_unres == len(urxn.metabolites) and len(urxn.metabolites) > 0
        if fully_unresolved > 0.5 * len(candidates):
            warnings.warn(
                f"gap-fill: {fully_unresolved}/{len(candidates)} universal candidates share no metabolite "
                "id with the model — a likely namespace mismatch; gap-fill will find little or nothing.",
                stacklevel=2)

        work.objective = biomass_reaction
        if (work.slim_optimize(error_value=0.0) or 0.0) < floor - 1e-9:
            return []  # unfixable even with every candidate present
        work.reactions.get_by_id(biomass_reaction).lower_bound = floor
        fluxes = pfba(work).fluxes
        return sorted(nr.id for nr in fresh if abs(fluxes.get(nr.id, 0.0)) > 1e-9)
    except Exception:  # noqa: BLE001 — infeasible / backend quirk: report none added
        return []


def _diagnose_growth_gap(applied, proposal, biomass_reaction, movable_ids, pinned_comp,
                         touches, base, compartments) -> tuple[str, str] | None:
    """Select a blocked movable reaction that touches a biomass precursor and return its pin.

    Returns ``(reaction_id, compartment)``, where ``compartment`` is the compartment of the biomass
    reaction, or ``None``. The reaction is the first reaction in ``proposal.placements`` that is
    movable, is not a gap-fill reaction, is blocked in the materialised model and touches a base
    metabolite that the biomass reaction consumes. ``None`` is also returned when the biomass
    reaction is absent or does not lie in a single compartment. Other gaps are left to ``universal``
    gap-fill.
    """
    from cobra.flux_analysis import find_blocked_reactions

    if biomass_reaction not in applied.reactions:
        return None
    bio = applied.reactions.get_by_id(biomass_reaction)
    bio_comp = _rxn_compartment(applied.reactions.get_by_id(biomass_reaction))
    blocked = set(find_blocked_reactions(applied, processes=1))
    precursors = {base(m) for m, coeff in bio.metabolites.items() if coeff < 0}
    for rid in proposal.placements:
        if rid not in movable_ids or rid in proposal.added_reactions:
            continue
        if rid in blocked and precursors & touches.get(rid, set()) and bio_comp:
            return rid, bio_comp
    return None
