"""BFS metabolite-producibility pre-screening (topological analysis).

Identifies which metabolites cannot be produced from the medium (seed
metabolites) using the reactions in a draft model. No solver is required:
the computation is a pure graph traversal. It is inspired by the topological
analysis used in Meneco (Prigent et al. 2017, PLoS Comput Biol).

Typical use: run :func:`analyse_topology` before a solver-based gap-fill to
(a) identify which metabolites are unreachable and (b) prune the candidate
reaction pool to those relevant to each gap, reducing the solver burden.

Metabolite identifiers in *templates* must match those in *model* to be
matched; if templates use a different namespace, align them first using
:func:`~raven_toolbox.manipulation.transfer.add_reactions_from_model`.
"""
from __future__ import annotations

import warnings
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field

import cobra


@dataclass
class TopologicalAnalysisResult:
    """Outcome of a topological gap analysis.

    Parameters
    ----------
    reachable_metabolites:
        Set of metabolite IDs that can be produced from seeds using
        the draft reactions.
    blocked_metabolites:
        Subset of *targets* that could not be reached from seeds.
    candidate_reactions:
        Mapping of blocked metabolite ID → list of template reaction IDs
        that produce that metabolite directly (matching by metabolite id).
    pruning_fraction:
        Fraction of template reactions that are NOT candidates for any
        blocked metabolite. Higher values mean the template was pruned more.
    """

    reachable_metabolites: set[str] = field(default_factory=set)
    blocked_metabolites: set[str] = field(default_factory=set)
    candidate_reactions: dict[str, list[str]] = field(default_factory=dict)
    pruning_fraction: float = 0.0


def _as_models(templates: cobra.Model | Iterable[cobra.Model]) -> list[cobra.Model]:
    return [templates] if isinstance(templates, cobra.Model) else list(templates)


def _compute_scope(model: cobra.Model, seed_ids: set[str]) -> set[str]:
    """Fixed-point BFS: find all metabolites reachable from *seed_ids*.

    Uses a countdown approach: a reaction fires (forward) when all its
    substrates are reachable; reversible reactions (lb < 0) can also fire in
    reverse when all their products are reachable.

    Complexity: O(|metabolites| + |reactions|).
    """
    # Build adjacency structures
    met_to_fwd_sub_rxns: dict[str, list[str]] = defaultdict(list)  # met → rxns where met is substrate
    met_to_prod_rxns: dict[str, list[str]] = defaultdict(list)      # met → rxns where met is product
    rxn_to_substrates: dict[str, list[str]] = defaultdict(list)     # rxn → substrate met ids
    rxn_to_products: dict[str, list[str]] = defaultdict(list)       # rxn → product met ids
    sub_count_fwd: dict[str, int] = {}  # remaining unreachable substrates (forward)
    sub_count_rev: dict[str, int] = {}  # remaining unreachable products   (reverse)
    rxn_is_rev: dict[str, bool] = {}

    for rxn in model.reactions:
        subs = [m.id for m, c in rxn.metabolites.items() if c < 0]
        prods = [m.id for m, c in rxn.metabolites.items() if c > 0]
        sub_count_fwd[rxn.id] = len(subs)
        sub_count_rev[rxn.id] = len(prods)
        rxn_is_rev[rxn.id] = rxn.lower_bound < 0
        for mid in subs:
            met_to_fwd_sub_rxns[mid].append(rxn.id)
            rxn_to_substrates[rxn.id].append(mid)
        for mid in prods:
            met_to_prod_rxns[mid].append(rxn.id)
            rxn_to_products[rxn.id].append(mid)

    reachable: set[str] = set(seed_ids) & {m.id for m in model.metabolites}
    fired_fwd: set[str] = set()
    fired_rev: set[str] = set()
    queue: deque[str] = deque(reachable)

    # Reactions with zero substrates fire immediately in forward direction
    for rxn_id, cnt in sub_count_fwd.items():
        if cnt == 0 and rxn_id not in fired_fwd:
            fired_fwd.add(rxn_id)
            for p in rxn_to_products[rxn_id]:
                if p not in reachable:
                    reachable.add(p)
                    queue.append(p)

    while queue:
        mid = queue.popleft()

        # Forward: mid is a substrate of some reactions
        for rxn_id in met_to_fwd_sub_rxns.get(mid, []):
            sub_count_fwd[rxn_id] -= 1
            if sub_count_fwd[rxn_id] == 0 and rxn_id not in fired_fwd:
                fired_fwd.add(rxn_id)
                for p in rxn_to_products[rxn_id]:
                    if p not in reachable:
                        reachable.add(p)
                        queue.append(p)

        # Reverse: mid is a product, so in the reverse direction it is a substrate
        for rxn_id in met_to_prod_rxns.get(mid, []):
            if not rxn_is_rev[rxn_id]:
                continue
            sub_count_rev[rxn_id] -= 1
            if sub_count_rev[rxn_id] == 0 and rxn_id not in fired_rev:
                fired_rev.add(rxn_id)
                for s in rxn_to_substrates[rxn_id]:
                    if s not in reachable:
                        reachable.add(s)
                        queue.append(s)

    return reachable


def analyse_topology(
    model: cobra.Model,
    templates: cobra.Model | Iterable[cobra.Model],
    *,
    seeds: Iterable[str] | None = None,
    targets: Iterable[str] | None = None,
    verbose: bool = True,
) -> TopologicalAnalysisResult:
    """BFS metabolite-producibility analysis.

    Starting from seed metabolites (metabolites available in the medium),
    identifies which metabolites in *model* cannot be produced using any
    sequence of draft reactions. For each unreachable target metabolite,
    lists candidate template reactions that produce it directly.

    Parameters
    ----------
    model:
        Draft model to analyse.
    templates:
        Universal database model(s) containing candidate reactions.
        Metabolite identifiers must match those in *model*.
    seeds:
        Metabolite IDs available from the medium. Default: the metabolites of
        every exchange reaction whose bounds let it supply them, as RAVEN's
        ``gapFillTopological`` picks them: a reaction with only substrates
        (``met -->``) and ``lower_bound < 0``, or one with only products
        (``--> met``) and ``upper_bound > 0``. An exchange reaction may have
        more than one metabolite. A :class:`UserWarning` is issued when there
        is none.
    targets:
        Metabolite IDs that should be reachable. Default: every metabolite of
        the first objective reaction (in model order), substrates and products,
        as in RAVEN's ``gapFillTopological``.
    verbose:
        Print a summary of the results.

    Returns
    -------
    TopologicalAnalysisResult

    Raises
    ------
    ValueError
        If *targets* is not given and the model has no objective reaction.
    """
    templates = _as_models(templates)

    # ---- Seeds ----
    if seeds is None:
        seed_ids: set[str] = set()
        for rxn in model.reactions:
            coeffs = list(rxn.metabolites.values())
            no_products = all(c < 0 for c in coeffs)
            no_substrates = all(c > 0 for c in coeffs)
            if (no_products and rxn.lower_bound < 0) or (no_substrates and rxn.upper_bound > 0):
                seed_ids.update(m.id for m in rxn.metabolites)
        if not seed_ids:
            warnings.warn(
                "analyse_topology: no uptake exchange reactions found. "
                "Provide seeds manually via the 'seeds' argument.",
                stacklevel=2,
            )
    else:
        seed_ids = set(seeds)

    # ---- Targets ----
    all_model_met_ids = {m.id for m in model.metabolites}
    if targets is None:
        objective = [r for r in model.reactions if r.objective_coefficient != 0]
        if not objective:
            raise ValueError(
                "analyse_topology: no objective function defined. "
                "Provide targets manually via the 'targets' argument."
            )
        target_ids = {m.id for m, c in objective[0].metabolites.items() if c != 0}
    else:
        target_ids = set(targets) & all_model_met_ids

    # ---- BFS scope computation ----
    reachable = _compute_scope(model, seed_ids)

    blocked = target_ids - reachable

    if verbose:
        print(
            f"analyse_topology: {len(reachable)}/{len(all_model_met_ids)} metabolites "
            f"reachable; {len(blocked)} blocked target metabolite(s)."
        )

    # ---- Find candidate template reactions for each blocked metabolite ----
    # Build a pool of all template reactions (unique by id)
    all_template_rxns: dict[str, cobra.Reaction] = {}
    for t in templates:
        for rxn in t.reactions:
            if rxn.id not in all_template_rxns:
                all_template_rxns[rxn.id] = rxn

    total_template = len(all_template_rxns)

    # Reverse index built once (O(template size)): met id -> template reactions that
    # produce it, instead of rescanning every template reaction per blocked metabolite.
    producer_index: dict[str, list[str]] = defaultdict(list)
    for rxn_id, rxn in all_template_rxns.items():
        for m, c in rxn.metabolites.items():
            # Forward production (c > 0) or reverse production (c < 0, reversible)
            if c > 0 or (c < 0 and rxn.lower_bound < 0):
                producer_index[m.id].append(rxn_id)

    candidate_rxns: dict[str, list[str]] = {}
    all_candidates: set[str] = set()

    for met_id in blocked:
        cands = producer_index.get(met_id, [])
        candidate_rxns[met_id] = cands
        all_candidates.update(cands)

    pruning_fraction = 1.0 - len(all_candidates) / max(total_template, 1)

    if verbose:
        print(
            f"analyse_topology: {total_template - len(all_candidates)}/{total_template} "
            f"template reactions pruned ({pruning_fraction:.0%})."
        )

    return TopologicalAnalysisResult(
        reachable_metabolites=reachable,
        blocked_metabolites=blocked,
        candidate_reactions=candidate_rxns,
        pruning_fraction=pruning_fraction,
    )
