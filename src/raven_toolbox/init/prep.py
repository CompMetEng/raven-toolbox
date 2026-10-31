"""ftINIT preprocessing: the once-per-template work shared by every sample on a model.

ftINIT does all omics-independent work once: it classifies reactions into the categories
the staged MILP may ignore (leave in, never remove), discovers task-essential
reactions, merges linear chains and scales the stoichiometry. The result
(:class:`PrepData`) is reused across every sample.

:func:`classify_reactions` is the reaction taxonomy: exchange, GPR-less import, simple
transport and advanced transport, spontaneous, GPR-less extracellular, custom, and any
reaction without a GPR. The staged schedule (:func:`raven_toolbox.init.get_init_steps`) uses
an 8-bit pattern to select which categories each MILP step holds out.
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

    ``ignored(pattern)`` returns the union of the categories whose bit is set, the
    reactions held out of that MILP step (they get score 0 and stay in the model).
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
    """Whether all metabolites of ``rxn`` are on one side (RAVEN ``getExchangeRxns``).

    RAVEN's ``getExchangeRxns`` (default ``'all'``) flags every reaction with no products
    or no substrates, that is, whose non-zero stoichiometry has one sign. cobra's
    ``Reaction.boundary`` covers a single metabolite only; this test also covers
    multi-metabolite sink and pool reactions, which ftINIT keeps.
    """
    coeffs = list(rxn.metabolites.values())
    if not coeffs:
        return False
    return not (any(c > 0 for c in coeffs) and any(c < 0 for c in coeffs))


def _is_advanced_transport(rxn: cobra.Reaction) -> bool:
    """Whether ``rxn`` has an even number (more than 2) of metabolites that pair up by name.

    The pairs lie in different compartments and have cancelling stoichiometry.
    """
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

    ``ext_comp`` is the extracellular compartment. ``spontaneous`` and ``custom`` are
    iterables of reaction ids. A reaction is "GPR-less" when its gene rule is empty.
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
    merged ids and pre-oriented irreversibly (so the MILP forces flux forward).
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

    A large spread of stoichiometric coefficients (for example a biomass or pool reaction with
    coefficients from 1e-3 to 1e2) forces correspondingly extreme flux magnitudes, so no
    single MILP big-M fits all reactions. As RAVEN, for each reaction the function caps every
    ``|coeff|`` at ``max_stoich_diff × min|coeff|`` (keeping signs), then scales the whole
    reaction so that its mean ``|coeff|`` is 1. Bounds are not changed. The function modifies
    ``model`` in place; only the merged MILP model is scaled, and the final output maps back
    to the unscaled ``ref_model`` by reaction id.
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


def _equation_metabolites(equation: str) -> list[str]:
    """Metabolite names of a task equation, as RAVEN ``parseRxnEqu`` reads them.

    The equation is split at `` <=> ``, `` => `` and `` + ``; empty pieces are dropped, and a
    leading number followed by a space is taken as the coefficient and removed.
    """
    text = equation.replace(" <=> ", "\u20ac").replace(" => ", "\u20ac").replace(" + ", "\u20ac").strip()
    names: list[str] = []
    for candidate in text.split("\u20ac"):
        if not candidate:
            continue
        space = candidate.find(" ")
        name = candidate
        if space >= 0 and _is_number(candidate[:space]):
            name = candidate[space + 1:]
        names.append(name.strip())
    return names


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _task_metabolite_names(tasks: Iterable[Task]) -> set[str]:
    """``name[compartment]`` strings named by the inputs, outputs and equations of ``tasks``."""
    names: set[str] = set()
    for task in tasks:
        names.update(token for token, _, _ in task.inputs)
        names.update(token for token, _, _ in task.outputs)
        for equation, _, _ in task.equations:
            names.update(_equation_metabolites(equation))
    return names


def _orient_forward(rxn: cobra.Reaction, direction: int) -> None:
    """Make ``rxn`` carry flux only in its forced direction (irreversible forward).

    Matches RAVEN ``prepINITModel``: reverse the reactions whose forced direction is
    negative (``reverseRxns``), then set ``lb = 0`` for every essential reaction,
    unconditionally, even where ``lb`` was already positive (RAVEN
    ``minModel1.lb(selInd) = 0``). The MILP forces the essential flux again later.
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
    """Build :class:`PrepData` from a template model, the work done once per template.

    The result is shared by every ftINIT sample on this model.

    With ``tasks``, discovers the task-essential reactions (kept regardless of score),
    orients them irreversibly in their required direction, and drops tasks that are
    infeasible. Then classifies reactions into the omics-independent categories, linearly
    merges, and (unless ``scale=False``) rescales the merged model's stoichiometry
    (:func:`rescale_for_init`) so that a single MILP big-M is valid across all reactions.
    In the INIT parameter calibration study on raven-docs (Human-GEM, one cell line), the
    full ftINIT pipeline without scaling was feasible at ``big_m`` 1000 and took 157 s
    against 83 s with scaling, and it was infeasible at ``big_m`` 100. ``simplify=False``
    skips RAVEN's dead-end and FVA-blocked removal and its reversibility-tightening passes,
    leaving a larger merged model that may reach a different extraction optimum.

    ``essential_cache_path`` makes the essential-reaction discovery resumable across
    interruptions: the accumulated results are written to this path; see
    :func:`find_task_essential_reactions`.

    Task-essential discovery runs with ``close_boundaries=True``, as RAVEN's
    ``prepINITModel`` (which calls ``closeModel`` first), and the default of
    :func:`find_task_essential_reactions` is additive. With ``close_boundaries=True`` the
    task file defines the whole boundary of the model during the discovery.

    ``verbose`` prints one line per phase plus per-task progress during essential-reaction
    discovery (default False).

    ``processes`` (default 1) parallelises task-essential discovery across tasks; see
    :func:`find_task_essential_reactions`. The two ``simplify_model`` passes are parallelised
    by cobra's own ``cobra.Configuration().processes``.

    The first simplification removes reactions only; every metabolite stays in the reference model
    while the tasks are checked, so a task can name any metabolite of the template, including
    ones whose names differ only in letter case. Afterwards
    the metabolites without a reaction are removed, except the ones that tasks name (inputs,
    outputs and equations, matched by exact ``name[compartment]``, failed tasks included; when
    several metabolites share a ``name[compartment]``, only the first in model order is kept).
    RAVEN ``prepINITModel`` does the same.

    With the first simplification, every gene left without a reaction is removed from the
    reference model afterwards (RAVEN ``prepINITModel`` removes unused genes there), including
    genes that had no reaction in the template. Task ids may repeat (for example as category
    codes) and may be empty; a task that cannot be performed is removed by its position, so a
    task with the same id that can be performed stays. Raises :class:`ValueError` when a task
    id is not text.
    """
    if tasks is not None:
        tasks = list(tasks)
        bad = [t.id for t in tasks if not isinstance(t.id, str)]
        if bad:
            raise ValueError(f"Task ids must be text, got {bad[0]!r} of type {type(bad[0]).__name__}")
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
        # prepINITModel removes the reactions but keeps every metabolite through the task check;
        # it removes the unused metabolites afterwards, except those that tasks name (below).
        template_order = {m.id: i for i, m in enumerate(ref_model.metabolites)}
        before = list(ref_model.metabolites)
        simplify_model(ref_model, delete_zero_interval=True, delete_dead_end=True,
                       delete_no_flux=True)
        lost = [m for m in before if m.id not in ref_model.metabolites]
        if lost and tasks is not None:
            restored = [cobra.Metabolite(m.id, formula=m.formula, name=m.name, charge=m.charge,
                                         compartment=m.compartment) for m in lost]
            for new, old in zip(restored, lost, strict=True):
                new.annotation = dict(old.annotation)
                new.notes = dict(old.notes)
            ref_model.add_metabolites(restored)
            ref_model.metabolites.sort(key=lambda m: template_order[m.id])
        # prepINITModel removes the reactions with removeReactions(model, rxns, false, true),
        # whose last argument removes every gene that no reaction uses.
        for gene in [g for g in ref_model.genes if not g.reactions]:
            ref_model.genes.remove(gene)

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
        failed_positions = set(ess.failed_indices)
        kept_tasks = [t for i, t in enumerate(tasks) if i not in failed_positions]

        if simplify:
            # prepINITModel: removeMets(cModel, setdiff(unusedMets, essentialMetsForTasks)), where
            # essentialMetsForTasks holds, for each name[compartment] that a task names (failed
            # tasks included), the first metabolite of the model with exactly that name.
            task_met_names = _task_metabolite_names(tasks)
            named_first: dict[str, cobra.Metabolite] = {}
            for met in ref_model.metabolites:
                key = f"{met.name}[{met.compartment}]"
                if key in task_met_names:
                    named_first.setdefault(key, met)
            keep_ids = {met.id for met in named_first.values()}
            ref_model.remove_metabolites(
                [m for m in ref_model.metabolites if not m.reactions and m.id not in keep_ids]
            )

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
