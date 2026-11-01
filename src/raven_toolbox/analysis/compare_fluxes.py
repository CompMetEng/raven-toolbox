"""Flux differences between two conditions: port of RAVEN's ``compareFluxes``.

Compares two flux distributions for the same model and reports every reaction
whose flux changed by more than ``cutoff``, largest change first, labelling the
reactions that were turned on, turned off or reversed direction. The comparison can be
restricted to the reactions touching a given list of metabolite names.

The result is a DataFrame plus id lists; RAVEN's printed table is not reproduced.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import cobra
import numpy as np
import pandas as pd

__all__ = ["CompareFluxesResult", "compare_fluxes"]

#: Column order of :attr:`CompareFluxesResult.changed`.
_CHANGED_COLUMNS = ["reaction", "name", "flux1", "flux2", "abs_delta", "rel_change", "type"]


@dataclass
class CompareFluxesResult:
    """Outcome of a :func:`compare_fluxes` comparison.

    ``changed`` holds one row per reaction whose ``|flux2 - flux1|`` exceeded the
    cutoff, sorted by ``abs_delta`` descending, with columns ``reaction``,
    ``name``, ``flux1``, ``flux2``, ``abs_delta``, ``rel_change`` and ``type``.
    ``rel_change`` is ``(flux2 - flux1) / |flux1|``, and is ``NaN`` when
    ``|flux1|`` is below the cutoff, because the reaction has no baseline. ``type`` is
    ``"on"``, ``"off"``, ``"flip"`` or ``""``.

    ``turned_on``, ``turned_off`` and ``flipped`` are reaction-id lists in the
    model's own order, not the sorted order of ``changed``. ``missing_metabolites`` lists
    the names of ``metabolite_list`` that match no metabolite, and ``n_considered`` is the
    number of reactions compared.
    """

    changed: pd.DataFrame
    turned_on: list[str] = field(default_factory=list)
    turned_off: list[str] = field(default_factory=list)
    flipped: list[str] = field(default_factory=list)
    missing_metabolites: list[str] = field(default_factory=list)
    n_considered: int = 0


def _flux_array(
    model: cobra.Model,
    fluxes: pd.Series | Mapping[str, float],
    label: str,
) -> np.ndarray:
    """Reaction fluxes as an array in the model's reaction order."""
    missing = [r.id for r in model.reactions if r.id not in fluxes]
    if missing:
        shown = ", ".join(missing[:10])
        more = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
        raise ValueError(f"{label} has no entry for {shown}{more}")
    return np.array([float(fluxes[r.id]) for r in model.reactions], dtype=float)


def compare_fluxes(
    model: cobra.Model,
    reference: pd.Series | Mapping[str, float],
    comparison: pd.Series | Mapping[str, float],
    *,
    cutoff: float = 1e-8,
    metabolite_list: Sequence[str] | None = None,
) -> CompareFluxesResult:
    """Find the reactions whose flux differs between two conditions.

    Parameters
    ----------
    model:
        The model both flux vectors are for.
    reference:
        Flux for the reference condition (for example wild type), indexed by reaction id.
        Every reaction in the model must be present.
    comparison:
        Flux for the condition being compared against it (for example a mutant).
    cutoff:
        A reaction is active when ``|flux| > cutoff``, and changed when
        ``|flux2 - flux1| > cutoff``. Both comparisons are strict.
    metabolite_list:
        Metabolite names, compared case-insensitively against
        ``Metabolite.name``. If given, only reactions touching at least one of
        them are compared; every other reaction is omitted from the result.
        A name matching no metabolite is reported in
        :attr:`CompareFluxesResult.missing_metabolites` and does not raise.

    Returns
    -------
    CompareFluxesResult
    """
    f1 = _flux_array(model, reference, "reference")
    f2 = _flux_array(model, comparison, "comparison")
    rxn_ids = [r.id for r in model.reactions]
    rxn_names = [r.name for r in model.reactions]

    active1 = np.abs(f1) > cutoff
    active2 = np.abs(f2) > cutoff
    delta = f2 - f1

    in_list = np.ones(len(rxn_ids), dtype=bool)
    missing: list[str] = []
    if metabolite_list:  # an empty list means "no filter", as in RAVEN
        in_list = np.zeros(len(rxn_ids), dtype=bool)
        index = {rid: i for i, rid in enumerate(rxn_ids)}
        for name in metabolite_list:
            mets = [m for m in model.metabolites if (m.name or "").lower() == str(name).lower()]
            if not mets:
                missing.append(str(name))
                continue
            for met in mets:
                for rxn in met.reactions:
                    in_list[index[rxn.id]] = True
        # Excluded reactions are forced to "unchanged" so they fall out of every
        # output field, rather than being filtered afterwards.
        active1 = active1 & in_list
        active2 = active2 & in_list
        delta = np.where(in_list, delta, 0.0)

    on_mask = ~active1 & active2
    off_mask = active1 & ~active2
    flip_mask = active1 & active2 & (np.sign(f1) != np.sign(f2))

    changed_mask = np.abs(delta) > cutoff
    # Stable sort by descending |delta|, so reactions with an equal change keep
    # the model's own relative order (MATLAB's sort is stable; numpy's default
    # is not).
    idx = np.flatnonzero(changed_mask)
    idx = idx[np.argsort(-np.abs(delta[idx]), kind="stable")]

    denom = np.abs(f1[idx])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_change = np.where(denom < cutoff, np.nan, delta[idx] / denom)

    # The three masks are disjoint by construction, so assignment order is free.
    types = np.full(len(idx), "", dtype=object)
    types[on_mask[idx]] = "on"
    types[off_mask[idx]] = "off"
    types[flip_mask[idx]] = "flip"

    changed = pd.DataFrame(
        {
            "reaction": [rxn_ids[i] for i in idx],
            "name": [rxn_names[i] for i in idx],
            "flux1": f1[idx],
            "flux2": f2[idx],
            "abs_delta": np.abs(delta[idx]),
            "rel_change": rel_change,
            "type": list(types),
        },
        columns=_CHANGED_COLUMNS,
    )

    return CompareFluxesResult(
        changed=changed,
        turned_on=[rxn_ids[i] for i in np.flatnonzero(on_mask)],
        turned_off=[rxn_ids[i] for i in np.flatnonzero(off_mask)],
        flipped=[rxn_ids[i] for i in np.flatnonzero(flip_mask)],
        missing_metabolites=missing,
        n_considered=int(in_list.sum()),
    )
