"""Compartment manipulation: merge all compartments, or copy reactions to a new compartment.

The functions port RAVEN's ``mergeCompartments`` and ``copyToComps``. Both work independently of
:func:`raven_toolbox.localization.predict_localization`. ``merge_compartments`` flattens a
multi-compartment model for a simplified analysis (for example checking whether the network can make
a metabolite when no compartment topology is involved); ``copy_to_compartment`` creates the
compartment copy needed to construct dual-localised pathways. cobra has no equivalents.
"""
from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Iterable

import cobra

from raven_toolbox.utils.gpr import set_gene_rule

# Compartments produced by merge_compartments (RAVEN uses 's' for "system").
_MERGED_COMPARTMENT = "s"

#: Key in ``Metabolite.notes`` where :func:`merge_compartments` stores the base id of a merged
#: metabolite, read by the default metabolite key of :mod:`raven_toolbox.localization`.
MERGED_BASE_ID_NOTE = "merged_base_id"


def merge_compartments(
    model: cobra.Model,
    *,
    merged_id: str = _MERGED_COMPARTMENT,
    merged_name: str = "system",
    base_metabolite: Callable[[cobra.Metabolite], str] | None = None,
    drop_single_metabolite_reactions: bool = False,
    deduplicate_reactions: bool = True,
) -> tuple[cobra.Model, list[str], list[str]]:
    """Merge every metabolite of ``model`` into one ``merged_id`` compartment.

    Returns ``(merged_model, deleted_single_met_reactions, deduplicated_reactions)``. The input
    model is not modified. Use cases:

    * Check whether the network can produce or consume a metabolite at all, without
      compartment topology.
    * Simplify a model for visualisation or for an analysis that ignores compartments.
    * Prepare a localisation from scratch (call :func:`merge_compartments`, then
      :func:`raven_toolbox.localization.predict_localization` or
      :func:`raven_toolbox.localization.assign_compartments` with the full reaction list).

    ``base_metabolite`` maps a metabolite to the compartment-independent key that decides which
    metabolites are the same species (two compartment copies with one key become one metabolite).
    The default removes a trailing ``_<compartment>`` suffix from the id (``glc__D_c`` and
    ``glc__D_e`` both give ``glc__D``). A model that uses a different id per compartment for the
    same species, for example the ``s_####`` ids of yeast-GEM, needs ``base_metabolite=lambda m:
    m.name`` (or an annotation key) to merge the compartments; the same callable is passed to
    :func:`~raven_toolbox.localization.assign_compartments`. The key decides the grouping only. The
    merged metabolite keeps the id, name, formula, charge, annotation and notes of one of its copies
    and never gets an id built from the key. The copy is the one already in ``merged_id`` if there
    is one, otherwise the copy whose id sorts first (``ACCOA_c`` before ``ACCOA_m``); among several
    copies in ``merged_id``, the one whose id sorts first. RAVEN's ``mergeCompartments`` applies the
    same rule. The id can therefore name a compartment that no longer exists. A name-based key such
    as ``"1-acyl-sn-glycerol 3-phosphate (16:0)"`` merges the copies, and the merged id is
    still the id of one of them, so the characters of the key do not enter the id.

    Metabolites that share a key become one metabolite in the merged compartment, and their
    stoichiometric contributions are summed per reaction. A reaction whose metabolites all cancel
    (``A[c] → A[m]`` becomes ``A → A``) is always deleted. With
    ``drop_single_metabolite_reactions=True`` (RAVEN's ``deleteRxnsWithOneMet``; default ``False``
    in both) a reaction that the merge reduces to one metabolite is deleted as well. A reaction with
    a single metabolite before the merge is an exchange (``=> glucose``) and is always kept, as in
    RAVEN, because dropping these reactions would remove the medium and the biomass reaction of
    the model. With ``deduplicate_reactions=True`` (default), reactions that are identical after the
    merge (same stoichiometry and same reversibility) are reduced to the first one. The objective is
    carried over to the returned model. ``deleted_single_met_reactions`` lists the reactions whose
    metabolites all cancelled and, with ``drop_single_metabolite_reactions=True``, those reduced to
    one metabolite. RAVEN's ``deletedRxns`` lists only the latter, so it is empty with
    ``deleteRxnsWithOneMet`` off.

    Each merged metabolite stores its base id in ``notes["merged_base_id"]``
    (:data:`MERGED_BASE_ID_NOTE`): the id of the kept copy without the suffix of the compartment it
    came from (``ACCOA`` for ``ACCOA_c``). The default metabolite key of
    :func:`~raven_toolbox.localization.assign_compartments`,
    :func:`~raven_toolbox.localization.apply_assignment`,
    :func:`~raven_toolbox.localization.relocate_reactions` and
    :func:`~raven_toolbox.localization.curation_priority` reads it, so the merged ``ACCOA_c`` in
    compartment ``s`` and ``ACCOA_m`` of a universal model are the same species there. When two merged
    metabolites would get the same base id (a ``base_metabolite`` that keeps ``X_c`` and ``X_m``
    apart), both store their own id, so they remain distinct species. The note is kept by
    ``model.copy()``, cobra's JSON and SBML files and :func:`~raven_toolbox.io.write_yaml_model`
    with :func:`~raven_toolbox.io.read_yaml_model`. RAVEN's ``readYAMLmodel`` shows it in
    ``metNotes``, for example ``merged_base_id: ACCOA note: free text note`` for a metabolite
    that also has the free-text note ``free text note``; no RAVEN function uses it. A round trip
    through RAVEN YAML (``readYAMLmodel`` and ``writeYAMLmodel``) or through a .mat file does not
    preserve the note: RAVEN joins the key and the free text into one string, which
    :func:`~raven_toolbox.io.read_yaml_model` reads back as the free-text note, and
    :func:`~raven_toolbox.io.write_matlab_model` and :func:`~raven_toolbox.io.read_matlab_model`
    carry no metabolite notes. The localisation functions then key the metabolite by its id
    without the suffix of its own compartment, so a merged ``ACCOA_c`` in compartment ``s`` gets the key
    ``ACCOA_c``, and no error is raised.
    """
    out = model.copy()
    key = base_metabolite if base_metabolite is not None else _base_id

    # 1. For each metabolite, derive its compartment-agnostic key. Two mets in different
    #    compartments sharing the key collapse to one.
    new_to_old: dict[str, list[cobra.Metabolite]] = {}
    for m in list(out.metabolites):
        base = key(m)
        new_to_old.setdefault(base, []).append(m)

    # 2. Build the merged metabolites and rewrite reactions. Each merged metabolite is a copy of one
    #    member, chosen as the docstring states. Groups are disjoint and model ids are unique, so the
    #    chosen ids are unique too.
    protos = {}
    for base, mets in new_to_old.items():
        in_target = [m for m in mets if m.compartment == merged_id]
        protos[base] = min(in_target or mets, key=lambda m: m.id)
    base_id_count: dict[str, int] = {}
    for proto in protos.values():
        base_id_count[_base_id(proto)] = base_id_count.get(_base_id(proto), 0) + 1
    canonical: dict[str, cobra.Metabolite] = {}
    for base, proto in protos.items():
        mid = proto.id
        new_met = cobra.Metabolite(mid, name=proto.name, compartment=merged_id,
                                    formula=proto.formula, charge=proto.charge)
        new_met.notes = copy.deepcopy(proto.notes or {})
        new_met.annotation = copy.deepcopy(proto.annotation or {})
        stored = _base_id(proto)
        new_met.notes[MERGED_BASE_ID_NOTE] = stored if base_id_count[stored] == 1 else mid
        new_met._bound = proto._bound
        canonical[base] = new_met

    # Rewrite all reactions: replace each metabolite with its canonical, summing
    # coefficients where multiple original mets collapse to one.
    rewritten: dict[str, dict[str, float]] = {}
    for r in list(out.reactions):
        new_stoich: dict[cobra.Metabolite, float] = {}
        for m, coeff in list(r.metabolites.items()):
            canon = canonical[key(m)]
            new_stoich[canon] = new_stoich.get(canon, 0.0) + coeff
        # Drop zero net coefficients (substrate + product of the same base met cancel).
        new_stoich = {m: c for m, c in new_stoich.items() if c != 0.0}
        rewritten[r.id] = {m.id: c for m, c in new_stoich.items()}

    # Now build a fresh model with the canonical mets + rewritten reactions; the
    # cobra in-place rewrite would require careful constraint surgery, so a clean
    # rebuild is simpler and less error-prone.
    merged = cobra.Model(out.id or "merged")
    merged.solver = out.problem  # else cobra's global default, not the model's
    merged.compartments = {merged_id: merged_name}
    merged.add_metabolites(list(canonical.values()))
    deleted_single: list[str] = []
    deduplicated: list[str] = []
    seen_signatures: dict[tuple, str] = {}
    keep_reactions: list[cobra.Reaction] = []
    for r in out.reactions:
        stoich = rewritten[r.id]
        # Only reactions that *became* trivial through merging are dropped. One
        # that already had a single metabolite is an exchange/boundary reaction
        # (``=> glucose``), and deleting those silently strips the model of its
        # medium and its biomass reaction. RAVEN reserves them the same way.
        was_single = len(r.metabolites) <= 1
        if drop_single_metabolite_reactions and len(stoich) <= 1 and not was_single:
            deleted_single.append(r.id)
            continue
        if not stoich:  # everything cancelled
            deleted_single.append(r.id)
            continue
        sig = (frozenset(stoich.items()), bool(r.lower_bound < 0), bool(r.upper_bound > 0))
        if deduplicate_reactions and sig in seen_signatures:
            deduplicated.append(r.id)
            continue
        seen_signatures[sig] = r.id
        new_r = cobra.Reaction(r.id, name=r.name, lower_bound=r.lower_bound,
                                upper_bound=r.upper_bound)
        new_r.add_metabolites({merged.metabolites.get_by_id(mid): c for mid, c in stoich.items()})
        set_gene_rule(new_r, r.gpr)
        if r.subsystem:
            new_r.subsystem = r.subsystem
        new_r.notes = dict(r.notes or {})
        keep_reactions.append(new_r)
    merged.add_reactions(keep_reactions)

    # The merged model is built from scratch, so the objective has to be carried
    # over explicitly -- without this the caller gets a model that optimises to
    # 0.0 with no indication why.
    surviving = {r.id for r in keep_reactions}
    objective = {
        merged.reactions.get_by_id(r.id): r.objective_coefficient
        for r in out.reactions
        if r.objective_coefficient and r.id in surviving
    }
    if objective:
        merged.objective = objective
        merged.objective_direction = out.objective_direction
    elif any(r.objective_coefficient for r in out.reactions):
        warnings.warn(
            "merge_compartments: every reaction carrying an objective coefficient was "
            "removed by the merge; the returned model has no objective.",
            stacklevel=2,
        )
    return merged, deleted_single, deduplicated


def copy_to_compartment(
    model: cobra.Model,
    reactions: Iterable[str],
    target_compartment: str,
    *,
    target_compartment_name: str | None = None,
    delete_original: bool = False,
    id_suffix: str | None = None,
) -> tuple[cobra.Model, list[str], list[str]]:
    """Copy a set of reactions into ``target_compartment`` (RAVEN's ``copyToComps``).

    Returns ``(model_copy, new_reaction_ids, new_metabolite_ids)``. The input model is not modified.
    Use cases:

    * Build a dual-localised pathway (for example duplicate glycolysis into a peroxisome).
    * Copy a curated subsystem into an additional compartment as a draft to refine.
    * Prepare the input for a flux comparison between alternative compartmentalisations.

    Each copied reaction gets the id ``"<orig_id>_<id_suffix>"`` (default
    ``id_suffix=target_compartment``). Each metabolite of the reaction maps to its copy in
    ``target_compartment``, which is created with the same suffix convention when it does not exist;
    a metabolite that is already in ``target_compartment`` is used unchanged. A reaction whose copy
    id is already in the model is skipped. ``delete_original=True`` removes the original reactions
    after copying (the reactions are moved); their metabolites remain in the model.
    """
    out = model.copy()
    suffix = id_suffix if id_suffix is not None else target_compartment
    if target_compartment not in out.compartments:
        out.compartments = {**out.compartments,
                             target_compartment: target_compartment_name or target_compartment}

    preexisting_met_ids = {x.id for x in out.metabolites}
    new_rxn_ids: list[str] = []
    new_reactions: list[cobra.Reaction] = []
    to_delete: list[str] = []
    seen_new_ids: set[str] = set()
    for rid in list(reactions):
        if rid not in out.reactions:
            raise ValueError(f"reaction {rid!r} not in model")
        src = out.reactions.get_by_id(rid)
        new_id = f"{rid}_{suffix}"
        if new_id in out.reactions or new_id in seen_new_ids:
            continue  # already copied; idempotent
        seen_new_ids.add(new_id)
        new_stoich: dict[cobra.Metabolite, float] = {}
        for m, coeff in src.metabolites.items():
            target_met = _met_in_compartment(out, m, target_compartment, suffix=suffix)
            new_stoich[target_met] = coeff
        new_r = cobra.Reaction(new_id, name=src.name,
                                lower_bound=src.lower_bound, upper_bound=src.upper_bound)
        new_r.add_metabolites(new_stoich)
        set_gene_rule(new_r, src.gpr)
        if src.subsystem:
            new_r.subsystem = src.subsystem
        new_r.notes = dict(src.notes or {})
        new_reactions.append(new_r)
        new_rxn_ids.append(new_id)
        if delete_original:
            to_delete.append(src.id)

    out.add_reactions(new_reactions)  # one batch — per-reaction adds are super-linear at scale
    if to_delete:
        out.remove_reactions(to_delete, remove_orphans=False)

    new_met_ids = [m.id for m in out.metabolites if m.id not in preexisting_met_ids]
    return out, new_rxn_ids, new_met_ids


# ----------------------------------------------------------------- helpers

def _base_id(m: cobra.Metabolite) -> str:
    """Strip the trailing ``_<compartment>`` suffix from a metabolite id (if present)."""
    if m.compartment and m.id.endswith(f"_{m.compartment}"):
        return m.id[: -(len(m.compartment) + 1)]
    return m.id


def _met_in_compartment(model: cobra.Model, source: cobra.Metabolite,
                        compartment: str, *, suffix: str | None = None) -> cobra.Metabolite:
    """Return the copy of ``source`` in ``compartment``, creating it when it does not exist.

    The new metabolite id is ``"<base>_<suffix>"`` (default ``suffix=compartment``).
    An existing metabolite with that id is returned unchanged.
    """
    if source.compartment == compartment:
        return source
    base = _base_id(source)
    new_id = f"{base}_{suffix if suffix is not None else compartment}"
    if new_id in model.metabolites:
        return model.metabolites.get_by_id(new_id)
    new_met = cobra.Metabolite(new_id, name=source.name, compartment=compartment,
                                formula=source.formula, charge=source.charge)
    new_met.notes = dict(source.notes or {})
    model.add_metabolites([new_met])
    return new_met
