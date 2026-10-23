"""ftINIT preprocessing — once-per-template work shared by every sample on a model.

ftINIT does all omics-independent work once: classify reactions into the categories
the staged MILP may *ignore* (leave in, never remove), discover task-essential
reactions, linearly merge, and scale. The result (:class:`PrepData`) is reused across
every sample.

:func:`classify_reactions` is the reaction taxonomy: exchange, GPR-less
import / simple / advanced transport, spontaneous, GPR-less extracellular, custom, and
"any without a GPR". The staged schedule (:func:`raven_toolbox.init.get_init_steps`) selects
which categories to keep out of each MILP step via an 8-bit pattern.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import cobra

from raven_toolbox.init.merge import merge_linear
from raven_toolbox.manipulation.simplify import simplify_model
from raven_toolbox.tasks import Task, find_task_essential_reactions


@dataclass
class ReactionMasks:
    """Reaction-category id sets (RAVEN's ``toIgnore*``), in 8-bit-pattern order.

    ``ignored(pattern)`` returns the union of the categories whose bit is set — the
    reactions held out of (left untouched by) that MILP step.
    """

    exchange: set[str] = field(default_factory=set)            # b1
    import_rxns: set[str] = field(default_factory=set)         # b2
    simple_transport: set[str] = field(default_factory=set)    # b3
    advanced_transport: set[str] = field(default_factory=set)  # b4
    spontaneous: set[str] = field(default_factory=set)         # b5
    extracellular: set[str] = field(default_factory=set)       # b6 (no-GPR, all mets in ext comp)
    custom: set[str] = field(default_factory=set)              # b7
    no_gpr: set[str] = field(default_factory=set)              # b8

    def _ordered(self) -> list[set[str]]:
        return [self.exchange, self.import_rxns, self.simple_transport,
                self.advanced_transport, self.spontaneous, self.extracellular,
                self.custom, self.no_gpr]

    def ignored(self, pattern: Iterable[int]) -> set[str]:
        out: set[str] = set()
        for bit, group in zip(pattern, self._ordered(), strict=True):
            if bit:
                out |= group
        return out


def _is_exchange_reaction(rxn: cobra.Reaction) -> bool:
    """One-sided reaction: all metabolites are on the same side (RAVEN ``getExchangeRxns``).

    RAVEN's ``getExchangeRxns`` (default ``'all'``) flags every reaction with no products
    *or* no substrates — i.e. whose non-zero stoichiometry is all one sign. That is broader
    than cobra's ``Reaction.boundary`` (single metabolite only): it also catches
    multi-metabolite sink / pool reactions, which ftINIT force-keeps.
    """
    coeffs = list(rxn.metabolites.values())
    if not coeffs:
        return False
    return not (any(c > 0 for c in coeffs) and any(c < 0 for c in coeffs))


def _is_advanced_transport(rxn: cobra.Reaction) -> bool:
    """Even number (>2) of mets pairing up by name across compartments with canceling stoich."""
    mets = list(rxn.metabolites.items())
    if len(mets) <= 2 or len(mets) % 2 != 0:
        return False
    remaining = [(m.name, m.compartment, c) for m, c in mets]
    while remaining:
        name, comp, coeff = remaining[0]
        matches = [i for i in range(1, len(remaining)) if remaining[i][0] == name]
        if len(matches) != 1:
            return False
        j = matches[0]
        if coeff + remaining[j][2] != 0 or comp == remaining[j][1]:
            return False
        remaining = [r for k, r in enumerate(remaining) if k not in (0, j)]
    return True


def classify_reactions(
    model: cobra.Model,
    *,
    ext_comp: str = "e",
    spontaneous: Iterable[str] = (),
    custom: Iterable[str] = (),
) -> ReactionMasks:
    """Classify reactions into the ftINIT ``toIgnore`` categories (``prepINITModel``).

    ``ext_comp`` is the extracellular compartment. ``spontaneous``/``custom`` are
    reaction-id lists. A reaction is "GPR-less" when its gene rule is empty.
    """
    spont, cust = set(spontaneous), set(custom)
    masks = ReactionMasks(
        exchange={r.id for r in model.reactions if _is_exchange_reaction(r)},
        spontaneous={r.id for r in model.reactions if r.id in spont},
        custom={r.id for r in model.reactions if r.id in cust},
        no_gpr={r.id for r in model.reactions if not r.gene_reaction_rule.strip()},
    )
    for rxn in model.reactions:
        if rxn.gene_reaction_rule.strip():
            continue  # transport categories are GPR-less only
        mets = list(rxn.metabolites)
        if len(mets) == 2:
            (m1, m2) = mets
            if m1.compartment != m2.compartment and m1.name == m2.name:
                if ext_comp in (m1.compartment, m2.compartment):
                    masks.import_rxns.add(rxn.id)
                else:
                    masks.simple_transport.add(rxn.id)
        elif _is_advanced_transport(rxn):
            masks.advanced_transport.add(rxn.id)
        if len(mets) > 1 and all(m.compartment == ext_comp for m in mets):
            masks.extracellular.add(rxn.id)
    return masks


@dataclass
class PrepData:
    """One-time ftINIT preprocessing of a template model (RAVEN ``prepData``).

    Built once per template, reused across samples. ``min_model`` is the merged model
    the MILP runs on; ``orig_rxn_ids``/``group_ids`` map its reactions back to the
    ``ref_model`` (the reference after the first simplification; the orientation of the
    essential reactions and the tightening of reversibility are undone on it after the merge,
    so its bounds and stoichiometry are the template's). ``essential_rxns`` are in
    **merged** ids and pre-oriented irreversibly (so the MILP forces flux *forward*).
    ``masks`` is on ``ref_model`` (= original) ids.
    """

    ref_model: cobra.Model
    min_model: cobra.Model
    orig_rxn_ids: list[str]
    group_ids: list[int]
    reversed_rxns: list[bool]
    masks: ReactionMasks
    essential_rxns: set[str] = field(default_factory=set)
    essential_mets_for_tasks: set[str] = field(default_factory=set)
    tasks: list[Task] = field(default_factory=list)

    @property
    def group_of(self) -> dict[str, int]:
        return dict(zip(self.orig_rxn_ids, self.group_ids, strict=True))


def rescale_for_init(model: cobra.Model, max_stoich_diff: float = 25.0) -> None:
    """Compress each reaction's stoichiometric dynamic range.

    Large spreads in stoichiometric coefficients (e.g. a biomass/pool reaction with
    coefficients from 1e-3 to 1e2) force correspondingly extreme flux magnitudes, so no
    single MILP big-M fits all reactions. RAVEN, per reaction: caps every ``|coeff|`` at
    ``max_stoich_diff × min|coeff|`` (keeping signs), then scales the whole reaction so its
    mean ``|coeff|`` is 1. Bounds are not changed. Modifies ``model`` in
    place; only the merged MILP model is scaled (the final output maps back to the
    unscaled ``ref_model`` by reaction id, so reaction *selection* is unaffected).
    """
    for rxn in model.reactions:
        items = list(rxn.metabolites.items())
        if not items:
            continue
        cap = max_stoich_diff * min(abs(c) for _, c in items)
        capped = {m: ((cap if c > 0 else -cap) if abs(c) > cap else c) for m, c in items}
        total = sum(abs(c) for c in capped.values())
        scale = (len(capped) / total) if total else 1.0
        rxn.add_metabolites({m: c * scale for m, c in capped.items()}, combine=False)


def _restore_reactions(model: cobra.Model, saved: dict) -> None:
    """Put back the bounds, stoichiometry and objective coefficient saved per reaction id."""
    for rid, (lb, ub, stoich, objective) in saved.items():
        rxn = model.reactions.get_by_id(rid)
        if dict(rxn.metabolites) != stoich:
            rxn.add_metabolites(stoich, combine=False)
        if (rxn.lower_bound, rxn.upper_bound) != (lb, ub):
            rxn.bounds = (lb, ub)
        if rxn.objective_coefficient != objective:
            rxn.objective_coefficient = objective


def _orient_forward(rxn: cobra.Reaction, direction: int) -> None:
    """Make ``rxn`` carry flux only in its forced direction (irreversible forward).

    Matches RAVEN ``prepINITModel``: reverse the reactions whose forced direction is
    negative (``reverseRxns``), then set ``lb = 0`` for every essential — unconditionally,
    even where ``lb`` was already positive (RAVEN ``minModel1.lb(selInd) = 0``). The MILP
    re-forces essential flux later, so relaxing ``lb`` to 0 here is intended.
    """
    if direction < 0:  # flip so the forced (reverse) direction becomes forward
        rxn.add_metabolites({m: -2 * c for m, c in rxn.metabolites.items()})
        rxn.bounds = (-rxn.upper_bound, -rxn.lower_bound)
    rxn.lower_bound = 0.0


def prep_init_model(
    template: cobra.Model,
    tasks: Iterable[Task] | None = None,
    *,
    ext_comp: str = "e",
    spontaneous: Iterable[str] = (),
    custom: Iterable[str] = (),
    essential_cache_path=None,
    simplify: bool = True,
    scale: bool = True,
    verbose: bool = False,
    processes: int = 1,
) -> PrepData:
    """Build :class:`PrepData` from a template model — the once-per-template work shared
    by every ftINIT sample on this model.

    With ``tasks``, discovers the task-essential reactions (kept regardless of score),
    orients them irreversibly in their required direction, and drops tasks that are
    infeasible. Then classifies reactions into the omics-independent categories, linearly
    merges, and (unless ``scale=False``) rescales the merged model's stoichiometry
    (:func:`rescale_for_init`) so a single MILP big-M is valid across all reactions —
    without this, genome-scale ftINIT is infeasible / intractable. ``simplify=False``
    skips RAVEN's dead-end/FVA-blocked removal and reversibility-tightening passes,
    leaving a larger merged model that may reach a different extraction optimum.

    ``essential_cache_path`` makes the (slow, genome-scale) essential-reaction discovery
    **resumable** across interruptions — see :func:`find_task_essential_reactions`.

    Task-essential discovery runs with ``close_boundaries=True``, matching RAVEN's
    ``prepINITModel`` (which calls ``closeModel`` first), rather than
    :func:`find_task_essential_reactions`'s own additive default — see the comment at the
    call site for why the two callers differ.

    ``verbose`` prints one line per phase (matching ``prepINITModel``'s numbered steps in
    spirit) plus per-task progress during essential-reaction discovery; silent by default.

    ``processes`` (default 1) parallelises task-essential discovery across tasks — see
    :func:`find_task_essential_reactions`. It does not affect the two ``simplify_model``
    passes below; those parallelise via cobra's own ``cobra.Configuration().processes``
    (set that separately if you want them faster too).
    """
    ref_model = template.copy()

    if simplify:
        if verbose:
            print("[prep] first simplification (drop dead-end / zero-flux reactions)",
                  flush=True)
        # RAVEN prepINITModel "first simplification":
        # simplifyModel(model, deleteUnconstrained, ~deleteDuplicates, deleteZeroInterval,
        # deleteInaccessible, deleteMinMax). Drop reactions that cannot carry steady-state
        # flux — zero-interval, topological dead-ends, then FVA-blocked (deleteMinMax).
        # Without the FVA-blocked and zero-interval passes the MILP runs on a ~10% larger
        # model and reaches a different (bypass-heavy) optimum.
        simplify_model(ref_model, delete_zero_interval=True, delete_dead_end=True,
                       delete_no_flux=True)

    essential_pre: dict[str, int] = {}
    task_mets: set[str] = set()
    kept_tasks: list[Task] = []
    if tasks is not None:
        tasks = list(tasks)
        # close_boundaries=True, against find_task_essential_reactions' own default:
        # RAVEN's prepINITModel closes the model before this step —
        #     bModel = closeModel(cModel);
        #     [~, essentialRxnMat, ...] = checkTasks(bModel, [], true, false, true, taskStruct);
        # (prepINITModel.m:81-82). closeModel gives every exchange an explicit boundary
        # metabolite, and checkTasks then zeroes *every* metabolite's balance
        # (checkTasks.m:63, `model.b=zeros(numel(model.mets),2)`) before re-opening only
        # the metabolites a task declares as inputs/outputs — so for this step the task
        # file *is* the whole boundary, not an addition to the model's own open exchanges.
        # The additive default is right for a standalone check_tasks on a model with no
        # boundary metabolites (RAVEN warns and leaves exchanges open there), but it is
        # wrong here: on Human-GEM, whose 1660 exchanges all ship open, it collapses the
        # task-essential set from ~206 reactions to 1, silently removing the task
        # constraint from every ftINIT extraction.
        if verbose:
            print(f"[prep] finding task-essential reactions ({len(tasks)} task(s))", flush=True)
        ess = find_task_essential_reactions(ref_model, tasks, close_boundaries=True,
                                            cache_path=essential_cache_path, verbose=verbose,
                                            processes=processes)
        essential_pre = ess.reactions
        task_mets = ess.task_metabolites
        kept_tasks = [t for t in tasks if t.id not in ess.failed_tasks]

    # Orienting essentials and tightening reversibility change the bounds, the stoichiometry
    # direction and the objective coefficient of these reactions only. They are restored on
    # ref_model after the merge, so that the extracted model keeps the bounds of the template.
    changeable = [r for r in ref_model.reactions if r.id in essential_pre or r.lower_bound < 0]
    saved = {
        r.id: (r.lower_bound, r.upper_bound, dict(r.metabolites), r.objective_coefficient)
        for r in changeable
    }

    # Orient essentials irreversibly (forced direction → forward) before merging, so
    # the merge keeps them forward and the MILP forces them with a simple lower bound.
    for rid, direction in essential_pre.items():
        _orient_forward(ref_model.reactions.get_by_id(rid), direction)

    if simplify:
        if verbose:
            print("[prep] second simplification (FVA + tighten reversibility)", flush=True)
        # RAVEN prepINITModel "second simplification":
        # simplifyModel(minModel1, ..., constrainReversible=true). FVA every reversible
        # reaction and make one-way-only ones irreversible *before* the linear merge. This
        # is what lets many more reactions merge (RAVEN: 5533 → 3734 reversible rxns), so
        # skipping it leaves a larger merged model and a different MILP optimum.
        simplify_model(ref_model, constrain_reversible=True)

    if verbose:
        print("[prep] classifying reactions and merging linear chains", flush=True)
    masks = classify_reactions(ref_model, ext_comp=ext_comp,
                               spontaneous=spontaneous, custom=custom)

    min_model, orig_ids, group_ids, reversed_rxns = merge_linear(ref_model)
    _restore_reactions(ref_model, saved)
    if scale:  # compress stoichiometric dynamic range so the MILP big-M fits all reactions
        if verbose:
            print("[prep] rescaling merged model", flush=True)
        rescale_for_init(min_model)
        for rxn in min_model.reactions:  # prepINITModel: bounds back to the standard ±1000
            if rxn.upper_bound > 0:
                rxn.upper_bound = 1000.0
            if rxn.lower_bound < 0:
                rxn.lower_bound = -1000.0
    if verbose:
        print(f"[prep] done: {len(min_model.reactions)} reactions in min_model, "
              f"{len(essential_pre)} essential (pre-merge)", flush=True)
    group_of = dict(zip(orig_ids, group_ids, strict=True))

    # Map essentials to the merged model: the survivor of each group containing an
    # essential (or the reaction itself if unmerged). All are forward after orientation.
    # An essential that merged into a group which collapsed away (e.g. a trivial
    # source→sink chain) has no survivor and imposes no constraint — skip it.
    survivor_by_group = {group_of[r.id]: r.id for r in min_model.reactions if group_of[r.id]}
    essential_merged: set[str] = set()
    for rid in essential_pre:
        gid = group_of[rid]
        if gid == 0:
            essential_merged.add(rid)
        elif gid in survivor_by_group:
            essential_merged.add(survivor_by_group[gid])

    return PrepData(
        ref_model=ref_model,
        min_model=min_model,
        orig_rxn_ids=orig_ids,
        group_ids=group_ids,
        reversed_rxns=reversed_rxns,
        masks=masks,
        essential_rxns=essential_merged,
        essential_mets_for_tasks=task_mets,
        tasks=kept_tasks,
    )
