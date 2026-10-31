"""Close the exchange reactions of a model for metabolic-task checking (RAVEN ``closeModel``)."""
from __future__ import annotations

import cobra


def _is_boundary_reaction(rxn: cobra.Reaction) -> bool:
    """Whether ``rxn`` has metabolites on only one side (the rule of ``getExchangeRxns``)."""
    coeffs = list(rxn.metabolites.values())
    if not coeffs:
        return False
    return all(c >= 0 for c in coeffs) or all(c <= 0 for c in coeffs)


def close_model(model: cobra.Model) -> cobra.Model:
    """Return a copy of ``model`` with the exchange reactions closed.

    Every reaction with metabolites on only one side (no substrates or no products: an
    exchange, sink or demand reaction) gets the bounds (0, 0) and can no longer carry flux.
    RAVEN's ``closeModel`` closes the same reactions by balancing each one with a boundary
    metabolite. After closing, the inputs and outputs of a metabolic task are set by the task
    constraints only, as ``checkTasks`` assumes, and no open exchange reactions remain.

    For a reaction with a single metabolite, the set equals cobra's ``Reaction.boundary``; the
    set also includes a reaction that consumes or produces several metabolites without a
    balancing side.

    The input model is not modified.
    """
    out = model.copy()
    close_model_in_place(out)
    return out


def close_model_in_place(model: cobra.Model) -> list[str]:
    """Close the boundary reactions of ``model`` in place; return their ids.

    In-place variant of :func:`close_model` for callers that already hold a working copy.
    """
    closed: list[str] = []
    for rxn in model.reactions:
        if _is_boundary_reaction(rxn):
            rxn.bounds = (0.0, 0.0)
            closed.append(rxn.id)
    return closed
