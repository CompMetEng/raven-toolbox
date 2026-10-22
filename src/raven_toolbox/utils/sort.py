"""Sort a model's identifiers alphabetically (useful for deterministic,
diff-friendly output) and standardise reaction directions and ordering.

cobra's ``DictList.sort`` reorders one list (and rebuilds its lookup index), but
there is no single "sort the whole model" call; this provides it.
"""
from __future__ import annotations

import cobra
import numpy as np

from raven_toolbox.utils.parse import subsystem_to_str

#: Subsystems with more reactions than this are skipped by the reaction-order search.
MAX_SUBSYSTEM_REACTIONS = 250


def sort_identifiers(model: cobra.Model) -> cobra.Model:
    """Sort reactions, metabolites and genes alphabetically by ID, in place.

    Returns the same (mutated) model for convenience. Compartments are a plain
    dict and are emitted sorted by writers as needed.
    """
    model.reactions.sort(key=lambda r: r.id)
    model.metabolites.sort(key=lambda m: m.id)
    model.genes.sort(key=lambda g: g.id)
    return model


def _subsystem_entries(rxn: cobra.Reaction) -> list[str]:
    return [e for e in subsystem_to_str(rxn.subsystem).split(";") if e]


def _subsystem_groups(model: cobra.Model) -> dict[str, list[int]]:
    """Map each subsystem name (sorted) to the indices of the reactions that list it."""
    groups: dict[str, list[int]] = {}
    for i, rxn in enumerate(model.reactions):
        for name in set(_subsystem_entries(rxn)):
            groups.setdefault(name, []).append(i)
    return {name: groups[name] for name in sorted(groups)}


def _stoichiometry_columns(model: cobra.Model, indices: list[int]) -> np.ndarray:
    """Dense (involved metabolites x reactions) stoichiometry of the given reactions."""
    met_index = {m.id: i for i, m in enumerate(model.metabolites)}
    columns = [{met_index[m.id]: c for m, c in model.reactions[i].metabolites.items() if c != 0} for i in indices]
    rows = sorted({r for column in columns for r in column})
    row_of = {r: k for k, r in enumerate(rows)}
    sub = np.zeros((len(rows), len(indices)))
    for j, column in enumerate(columns):
        for r, c in column.items():
            sub[row_of[r], j] = c
    return sub


def _order_score(sub: np.ndarray) -> int:
    """Number of metabolites whose first producing reaction precedes the first consuming one.

    Two columns are appended so that every metabolite is produced and consumed at the end.
    """
    ones = np.ones((sub.shape[0], 1))
    s = np.hstack([sub, ones, -ones])
    first_produced = (s > 0).argmax(axis=1)
    first_consumed = (s < 0).argmax(axis=1)
    return int((first_produced < first_consumed).sum())


def subsystem_order_scores(model: cobra.Model) -> dict[str, int]:
    """Order score of every subsystem with 2 to 250 reactions, in the current reaction order.

    The score is the number of metabolites of the subsystem that are produced by a reaction
    before the first reaction that consumes them (reaction directions as stored). It is the
    quantity that ``sort_model(..., sort_reaction_order=True)`` maximises.
    """
    scores = {}
    for name, indices in _subsystem_groups(model).items():
        if 2 <= len(indices) <= MAX_SUBSYSTEM_REACTIONS:
            scores[name] = _order_score(_stoichiometry_columns(model, indices))
    return scores


def _flip(rxn: cobra.Reaction) -> None:
    for met, coef in list(rxn.metabolites.items()):
        rxn.add_metabolites({met: -coef}, combine=False)


def _sort_reversible(model: cobra.Model) -> None:
    met_index = {m.id: i for i, m in enumerate(model.metabolites)}
    for rxn in model.reactions:
        if not rxn.reversibility:
            continue
        mets = sorted((m for m, c in rxn.metabolites.items() if c != 0), key=lambda m: met_index[m.id])
        if not mets:
            continue
        first = sorted(mets, key=lambda m: f"{m.name}{m.compartment}")[0]
        if rxn.metabolites[first] > 0:
            _flip(rxn)


def _sort_subsystem_order(model: cobra.Model, rng: np.random.Generator) -> None:
    for name in _subsystem_groups(model):
        indices = _subsystem_groups(model).get(name, [])
        n = len(indices)
        if n < 2 or n > MAX_SUBSYSTEM_REACTIONS:
            continue
        rxns = [model.reactions[i] for i in indices]
        rev_local = [j for j, r in enumerate(rxns) if r.reversibility]
        sub = _stoichiometry_columns(model, indices)

        order = np.arange(n)
        old_rev = np.ones(len(rev_local))
        old_score = -np.inf
        counter = 0
        first_iter = True
        while True:
            counter += 1
            if counter == 100 * n:
                break
            new_order = order.copy()
            rev = old_rev.copy()
            if not first_iter:
                y = rng.permutation(n)[:2]
                new_order[y[0]] = order[y[1]]
                new_order[y[1]] = order[y[0]]
                # With a 50% chance, one reversible reaction is also reversed.
                if rng.random() > 0.5 and len(rev) > 1:
                    k = rng.integers(len(rev))
                    rev[k] = -rev[k]
            first_iter = False

            temp = sub.copy()
            for k, j in enumerate(rev_local):
                if rev[k] == -1:
                    temp[:, j] = -temp[:, j]
            score = _order_score(temp[:, new_order])
            if score >= old_score:
                if score > old_score:
                    counter = 0
                old_score = score
                old_rev = rev
                order = new_order

        for k, j in enumerate(rev_local):
            if old_rev[k] == -1:
                _flip(rxns[j])
        slots = sorted(indices)
        placed = {rxns[order[k]].id: slots[k] for k in range(n)}
        position = {r.id: i for i, r in enumerate(model.reactions)}
        position.update(placed)
        model.reactions.sort(key=lambda r: position[r.id])


def sort_model(
    model: cobra.Model,
    *,
    sort_reversible: bool = False,
    sort_met_name: bool = False,
    sort_reaction_order: bool = False,
    seed: int | None = None,
) -> cobra.Model:
    """Standardise the direction and order of reactions and metabolites, in place.

    Mirrors RAVEN's ``sortModel``. Returns the same (mutated) model.

    Parameters
    ----------
    model
        The model to sort.
    sort_reversible
        Reverse the stoichiometry of each reversible reaction whose metabolite that is first
        in sorted order of ``name + compartment`` is a product, so that it becomes a reactant.
        Bounds are not changed. Ignored when ``sort_reaction_order`` is true.
    sort_met_name
        Order the metabolites of the model by ``name[compartment]``.
    sort_reaction_order
        Reorder the reactions of each subsystem with 2 to 250 reactions, and reverse reversible
        reactions, so that as many metabolites as possible are produced before they are
        consumed (see ``subsystem_order_scores``). A reaction that lists several subsystems
        takes part in each of them in turn. Reactions without a subsystem are not moved.
    seed
        Seed of the random generator of the reaction-order search; ``None`` draws fresh entropy.

    Notes
    -----
    The reaction-order search is the random pair-swap search of ``sortModel.m``, with the same
    score and stop rule. It draws from a numpy generator instead of MATLAB's, so the same seed
    gives a different permutation in the two implementations; their scores can be compared.
    RAVEN positions the reactions of a later subsystem with indices found before the first
    subsystem was reordered; here the positions are found again after each subsystem, which
    gives the same result when no reaction lists several subsystems.
    """
    if sort_met_name:
        model.metabolites.sort(key=lambda m: f"{m.name}[{m.compartment}]")
    if sort_reversible and not sort_reaction_order:
        _sort_reversible(model)
    if sort_reaction_order:
        _sort_subsystem_order(model, np.random.default_rng(seed))
    return model
