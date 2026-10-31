"""N-model structural and functional comparison.

Compare two or more models (for example context-specific models extracted from the same
template) on their reactions, metabolites, genes, EC-codes, metabolite names, reaction
equations (with and without compartments), subsystems, and (optionally) which metabolic
tasks they perform. Returns :class:`pandas.DataFrame`\\ s that can be passed to plotting
libraries such as seaborn and scikit-learn (heatmaps, tSNE and MDS). The function does
not plot.

All matrices use the union of ids or values across the input models as the row index.
Missing entries are ``0`` or ``False``, never ``NaN``.

The ``ec_codes``, ``metabolite_names``, ``equations`` and ``equations_no_compartments``
matrices compare models by content, not by identifier, which applies when two models of
the same organism do not share an id convention. ``compare_models`` covers the comparisons
of RAVEN's ``compareMultipleModels``.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import cast

import cobra
import pandas as pd

from raven_toolbox.tasks import Task, check_tasks
from raven_toolbox.utils.parse import subsystem_to_str


@dataclass
class ModelComparison:
    """Tabular result of :func:`compare_models`.

    All matrices (``reactions``, ``metabolites``, ``genes``, ``ec_codes``,
    ``metabolite_names``, ``equations``, ``equations_no_compartments`` and
    ``subsystems``) are indexed by id or value with one column per model. The
    presence matrices are 0/1; ``subsystems`` is the per-model reaction count
    per subsystem. ``similarity`` is the model × model Jaccard on the reaction set
    (1 = identical, 0 = disjoint). ``tasks`` is the per-model task pass-or-fail matrix
    (``None`` unless task checking was requested); ``failed_tasks`` maps each model id to
    the failure messages for the tasks it did not pass.

    ``equations`` and ``equations_no_compartments`` key each reaction by a canonical string
    built from its metabolites and stoichiometry: sides are sorted, and for a reversible
    reaction the side whose first sorted entry is lexicographically smaller is written as
    the reactant side. Two reactions that differ only in metabolite order or, for
    reversible reactions, in direction compare equal. ``equations`` keeps compartments in each
    metabolite id; ``equations_no_compartments`` strips them.
    """

    model_ids: list[str]
    reactions: pd.DataFrame
    metabolites: pd.DataFrame
    genes: pd.DataFrame
    ec_codes: pd.DataFrame
    metabolite_names: pd.DataFrame
    equations: pd.DataFrame
    equations_no_compartments: pd.DataFrame
    subsystems: pd.DataFrame
    similarity: pd.DataFrame
    tasks: pd.DataFrame | None = None  # filled iff tasks were supplied
    failed_tasks: dict[str, list[str]] = field(default_factory=dict)


def _presence_matrix(items_per_model: list[list[str]], model_ids: list[str]) -> pd.DataFrame:
    """Build a 0/1 DataFrame: union of items as index × one column per model."""
    ordered: list[str] = []
    seen: set[str] = set()
    for items in items_per_model:
        for it in items:
            if it not in seen:
                seen.add(it)
                ordered.append(it)
    df = pd.DataFrame(0, index=ordered, columns=model_ids, dtype="int8")
    for mid, items in zip(model_ids, items_per_model, strict=True):
        if items:  # avoid empty-list edge case
            df.loc[list(set(items) & seen), mid] = 1
    return df


def _ec_codes(model: cobra.Model) -> list[str]:
    """Every EC-code annotation value across the model's reactions, flattened.

    cobra stores ``ec-code`` as a bare string or a list per reaction; both are flattened
    to individual codes here, so (unlike RAVEN's ``eccodes``, one semicolon-joined string
    per reaction) two reactions that share only one of several EC codes still overlap.
    """
    codes: list[str] = []
    for r in model.reactions:
        v = r.annotation.get("ec-code")
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            codes.extend(str(x) for x in v)
        else:
            codes.append(str(v))
    return codes


def _metabolite_names(model: cobra.Model) -> list[str]:
    """Every metabolite's ``name`` (may repeat across compartments; may be empty)."""
    return [m.name for m in model.metabolites]


def _met_key(met: cobra.Metabolite, *, use_compartments: bool) -> str:
    """Metabolite identity for equation-building: the id, or the id with its
    ``_<compartment>`` suffix stripped when ``use_compartments`` is False."""
    if use_compartments or not met.compartment:
        return met.id
    suffix = f"_{met.compartment}"
    return met.id[: -len(suffix)] if met.id.endswith(suffix) else met.id


def _equation_side(pairs: list[tuple[str, float]]) -> list[str]:
    return sorted(f"{abs(c):g} {k}" if abs(c) != 1 else k for k, c in pairs)


def _canonical_equation(rxn: cobra.Reaction, *, use_compartments: bool) -> str:
    """A reaction-identity string independent of metabolite order and, for reversible
    reactions, of which arbitrary direction it was written in.

    This approximates RAVEN's ``constructEquations(sortRevRxns=True,
    sortMetNames=True, useComps=use_compartments)`` and does not reproduce its output
    byte for byte. The direction is canonicalised only for reversible reactions;
    irreversible reactions keep the stored stoichiometry sign.
    """
    left_pairs = [(_met_key(m, use_compartments=use_compartments), c)
                  for m, c in rxn.metabolites.items() if c < 0]
    right_pairs = [(_met_key(m, use_compartments=use_compartments), c)
                   for m, c in rxn.metabolites.items() if c > 0]
    left, right = _equation_side(left_pairs), _equation_side(right_pairs)
    if rxn.reversibility and right and (not left or right[0] < left[0]):
        left, right = right, left
    arrow = "<=>" if rxn.reversibility else "=>"
    return f"{' + '.join(left)} {arrow} {' + '.join(right)}"


def _subsystem_counts(model: cobra.Model) -> dict[str, int]:
    """{subsystem_name: reaction_count}. Reactions with empty subsystem fall under '(none)'."""
    counts: dict[str, int] = {}
    for r in model.reactions:
        # cobra stores subsystem as a string; RAVEN sometimes uses cell-of-cells.
        # Coerce to the same ;-joined string used everywhere else (no data lost).
        sub = subsystem_to_str(r.subsystem).strip() or "(none)"
        counts[sub] = counts.get(sub, 0) + 1
    return counts


def _jaccard_matrix(presence: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Jaccard similarity from a 0/1 presence matrix (rows = items, cols = models)."""
    arr = presence.values.astype(bool)
    out = pd.DataFrame(0.0, index=presence.columns, columns=presence.columns)
    for i, a in enumerate(presence.columns):
        ai = arr[:, i]
        for j, b in enumerate(presence.columns):
            bj = arr[:, j]
            inter = int((ai & bj).sum())
            union = int((ai | bj).sum())
            out.loc[a, b] = inter / union if union else 1.0
    return out


def compare_models(
    models: Iterable[cobra.Model],
    *,
    tasks: str | PathLike[str] | Iterable[Task] | None = None,
) -> ModelComparison:
    """Compare N cobra models on their reactions, metabolites, genes, ec_codes,
    metabolite_names, equations, equations_no_compartments and subsystems (and tasks, if
    provided).

    Models are identified by ``model.id``. A missing or blank id becomes ``model_<i>``
    and a repeated id gets the suffix ``__<n>`` (``__2`` for the second occurrence).

    Parameters
    ----------
    models : Iterable[cobra.Model]
        The models to compare. At least two are required, otherwise :class:`ValueError` is
        raised. The models are not modified.
    tasks : str or os.PathLike or Iterable[Task] or None
        Metabolic tasks, passed to :func:`raven_toolbox.tasks.check_tasks` for each model. It
        is a file path (``str`` or ``os.PathLike``) or a parsed task list. When omitted,
        ``ModelComparison.tasks`` is ``None``.
    """
    models_list = list(models)
    if len(models_list) < 2:
        raise ValueError(f"compare_models needs ≥2 models; got {len(models_list)}")

    # Unique, stable model ids.
    model_ids: list[str] = []
    seen: set[str] = set()
    for i, m in enumerate(models_list):
        mid = (m.id or "").strip() or f"model_{i}"
        base, n = mid, 2
        while mid in seen:
            mid, n = f"{base}__{n}", n + 1
        seen.add(mid)
        model_ids.append(mid)

    reactions = _presence_matrix([[r.id for r in m.reactions] for m in models_list], model_ids)
    metabolites = _presence_matrix([[x.id for x in m.metabolites] for m in models_list], model_ids)
    genes = _presence_matrix([[g.id for g in m.genes] for m in models_list], model_ids)
    ec_codes = _presence_matrix([_ec_codes(m) for m in models_list], model_ids)
    metabolite_names = _presence_matrix([_metabolite_names(m) for m in models_list], model_ids)
    equations = _presence_matrix(
        [[_canonical_equation(r, use_compartments=True) for r in m.reactions] for m in models_list],
        model_ids,
    )
    equations_no_compartments = _presence_matrix(
        [[_canonical_equation(r, use_compartments=False) for r in m.reactions] for m in models_list],
        model_ids,
    )

    # Subsystems: union of names, per-model reaction counts.
    sub_counts = [_subsystem_counts(m) for m in models_list]
    sub_ids = sorted({s for c in sub_counts for s in c})
    subsystems = pd.DataFrame(0, index=sub_ids, columns=model_ids, dtype="int32")
    for mid, c in zip(model_ids, sub_counts, strict=True):
        for s, n in c.items():
            subsystems.at[s, mid] = n

    similarity = _jaccard_matrix(reactions)

    task_df: pd.DataFrame | None = None
    failed: dict[str, list[str]] = {}
    if tasks is not None:
        # raven_toolbox.tasks.check_tasks accepts a path or an iterable of Task; preserve task
        # ids for the index. Capture the list once so all models test the same set.
        from raven_toolbox.tasks.tasklist import parse_task_list
        task_list = (parse_task_list(cast("str | Path", tasks))
                     if isinstance(tasks, (str, bytes)) or hasattr(tasks, "__fspath__")
                     else list(tasks))
        task_ids = [t.id for t in task_list]
        task_df = pd.DataFrame(False, index=task_ids, columns=model_ids, dtype=bool)
        for col, (mid, m) in enumerate(zip(model_ids, models_list, strict=True)):
            results = check_tasks(m, task_list)
            for row, r in enumerate(results):  # by position: task ids may repeat
                task_df.iat[row, col] = bool(r.passed)
                if not r.passed and r.error:
                    failed.setdefault(mid, []).append(f"{r.id}: {r.error}")

    return ModelComparison(model_ids=model_ids, reactions=reactions, metabolites=metabolites,
                           genes=genes, ec_codes=ec_codes, metabolite_names=metabolite_names,
                           equations=equations, equations_no_compartments=equations_no_compartments,
                           subsystems=subsystems, similarity=similarity,
                           tasks=task_df, failed_tasks=failed)
