"""Merge several models into one.

cobra's ``Model.merge`` is pairwise and matches everything strictly by id; this
merges **N** models and unifies metabolites by **name[compartment]** (so the same
compound under different ids in two models becomes one), while adding **all**
reactions without de-duplication
(a reaction whose ID already exists is renamed ``id_<sourceid>``). Genes are
unified by ID. Like RAVEN, every metabolite and gene of every model is kept, also
when no reaction uses it; the objective of every model is carried over; when two
models name the same compartment differently the first model's name is kept; new
metabolites and genes are added in the order of their source model. Provenance (which source model each object came from) is recorded
in ``notes['origin']``.

The bulk of RAVEN's function is struct field-padding and manual S-matrix
assembly, none of which is needed on ``cobra.Model``.
"""
from __future__ import annotations

import copy
import warnings
from collections.abc import Iterable

import cobra
from cobra import Gene, Metabolite, Model, Reaction

from raven_toolbox.utils.gpr import set_gene_rule


def _unique_id(existing, base: str, suffix: str) -> str:
    """Return base, or base_suffix (then base_suffix_2, ...) if it collides."""
    if base not in existing:
        return base
    candidate = f"{base}_{suffix}"
    n = 2
    while candidate in existing:
        candidate = f"{base}_{suffix}_{n}"
        n += 1
    return candidate


def merge_models(
    models: Iterable[cobra.Model],
    *,
    match_by: str = "name",
    track_origin: bool = True,
) -> cobra.Model:
    """Merge models into a single new model.

    Parameters
    ----------
    models
        The models to merge (two or more). A single model is returned as a copy.
    match_by
        How metabolites are unified across models: ``"name"`` (default) treats
        metabolites with the same *name and compartment* as identical (IDs
        ignored); ``"id"`` matches by metabolite ID.
    track_origin
        If True (default), record the source model's ``id`` in each reaction's,
        metabolite's, and gene's ``notes['origin']``.

    Returns
    -------
    cobra.Model
        A new merged model (``id="MERGED"``). Reactions are **not** de-duplicated
        — matching RAVEN, every reaction from every model is kept, with ID
        collisions renamed ``id_<sourceid>``.
    """
    models = list(models)
    if not models:
        raise ValueError("merge_models requires at least one model.")
    if match_by not in ("name", "id"):
        raise ValueError(f"match_by must be 'name' or 'id', got {match_by!r}")
    if len(models) == 1:
        return models[0].copy()

    merged = Model("MERGED")
    merged.objective_direction = models[0].objective_direction
    comp_names: dict[str, str] = {}
    objective: dict[Reaction, float] = {}
    met_lookup: dict = {}  # name/comp or id key -> merged Metabolite

    def met_key(met: Metabolite):
        return (met.name, met.compartment) if match_by == "name" else met.id

    def ensure_metabolite(src: Metabolite, origin: str) -> Metabolite:
        key = met_key(src)
        if key in met_lookup:
            existing = met_lookup[key]
            # Two source models can map to the same name[comp] (or id) with
            # different formula/charge; silently picking the first-seen would
            # quietly corrupt mass balance. Warn so the caller sees the conflict.
            if src.formula and existing.formula and src.formula != existing.formula:
                warnings.warn(
                    f"merge_models: metabolite {existing.id!r} (from earlier model) "
                    f"and {src.id!r} (from {origin!r}) share key {key!r} but "
                    f"have different formulas ({existing.formula!r} vs {src.formula!r}); "
                    "keeping the first.",
                    stacklevel=3,
                )
            if (
                existing.charge is not None
                and src.charge is not None
                and existing.charge != src.charge
            ):
                warnings.warn(
                    f"merge_models: metabolite {existing.id!r} (from earlier model) "
                    f"and {src.id!r} (from {origin!r}) share key {key!r} but "
                    f"have different charges ({existing.charge} vs {src.charge}); "
                    "keeping the first.",
                    stacklevel=3,
                )
            return existing
        new_id = _unique_id(merged.metabolites, src.id, origin)
        new_met = Metabolite(
            new_id, name=src.name, compartment=src.compartment,
            formula=src.formula, charge=src.charge,
        )
        new_met.annotation = copy.deepcopy(src.annotation)
        new_met.notes = copy.deepcopy(src.notes)
        if track_origin:
            new_met.notes.setdefault("origin", origin)
        merged.add_metabolites([new_met])
        met_lookup[key] = new_met
        return new_met

    for model in models:
        origin = model.id or "model"
        # model.compartments lists only compartments that hold metabolites; RAVEN keeps all.
        for comp, name in {**model.compartments, **model._compartments}.items():
            comp_names.setdefault(comp, name)    # the first model's name for a compartment wins
        genes_before = {g.id for g in merged.genes}

        # Every metabolite and gene of the model, in the model's order, before the reactions
        # that use them: unused ones are kept, and the order does not depend on the reactions.
        for met in model.metabolites:
            ensure_metabolite(met, origin)
        for gene in model.genes:
            if gene.id not in merged.genes:
                new_gene = Gene(gene.id, name=gene.name)
                new_gene.annotation = copy.deepcopy(gene.annotation)
                new_gene.notes = copy.deepcopy(gene.notes)
                merged.genes.append(new_gene)
                new_gene._model = merged

        new_rxns: list[Reaction] = []
        for rxn in model.reactions:
            new_id = _unique_id(merged.reactions, rxn.id, origin)
            new_rxn = Reaction(new_id, name=rxn.name)
            new_rxn.bounds = rxn.bounds
            new_rxn.subsystem = rxn.subsystem
            new_rxns.append(new_rxn)
        merged.add_reactions(new_rxns)  # one batch — per-reaction adds are super-linear at scale

        for rxn, new_rxn in zip(model.reactions, new_rxns, strict=True):
            new_rxn.add_metabolites(
                {ensure_metabolite(m, origin): coef for m, coef in rxn.metabolites.items()}
            )
            if rxn.gene_reaction_rule:
                set_gene_rule(new_rxn, rxn.gpr)
            new_rxn.annotation = copy.deepcopy(rxn.annotation)
            new_rxn.notes = copy.deepcopy(rxn.notes)
            if track_origin:
                new_rxn.notes.setdefault("origin", origin)
            if rxn.objective_coefficient:
                objective[new_rxn] = rxn.objective_coefficient

        if track_origin:
            for gene in merged.genes:
                if gene.id not in genes_before:
                    gene.notes.setdefault("origin", origin)

    merged._compartments.update(comp_names)
    if objective:
        merged.objective = objective
    return merged
