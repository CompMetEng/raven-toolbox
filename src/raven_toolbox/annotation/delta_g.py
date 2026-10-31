"""Persist scalar annotations such as the Gibbs free energy as a companion CSV.

The SBML standard has no slot for thermodynamic ΔG values, so
projects like yeast-GEM keep them in a paired CSV. This module
provides the generic load and save round trip (the file format is a two
column table ``<id>, <value>``) and exposes the per-call schema so
existing projects can configure their conventions.

The functions operate on cobra ``.notes`` (string round-tripped through
SBML), keeping any other notes untouched.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import cobra
import pandas as pd


def load_delta_g_csv(
    entities: Iterable,
    path: str | Path,
    *,
    id_column: str = "Var1",
    value_column: str = "Var2",
    note_key: str = "deltaG",
    verbose: bool = False,
) -> int:
    """Record ``note_key`` on each entity from a CSV of ``id → value``.

    Every matched value is recorded as the string of the value read from the CSV,
    including yeast-GEM's own "no measurement" placeholder (``10000000.0``,
    on 777 of its 4102 reaction rows). This function does not interpret
    CSV values, matching RAVEN's own ``loadDeltaGCSV``; a caller that treats
    a particular value as absent filters it after loading.

    Parameters
    ----------
    entities
        Cobra entities (``model.metabolites`` or ``model.reactions``).
    path
        Path to the CSV.
    id_column, value_column
        Column labels in the CSV. Defaults match the yeast-GEM
        convention (``Var1`` and ``Var2`` from MATLAB's ``array2table``).
    note_key
        Key under which the value is stored on ``entity.notes``.
        Default ``"deltaG"``.
    verbose
        Print a summary of unmatched entity ids.

    Returns
    -------
    int
        The number of entities that were recorded (that is, matched the CSV and
        carried a value).
    """
    df = pd.read_csv(path)
    if id_column not in df.columns or value_column not in df.columns:
        raise ValueError(
            f"{path} columns {list(df.columns)} do not contain "
            f"both {id_column!r} and {value_column!r}"
        )
    lookup = dict(zip(df[id_column], df[value_column], strict=True))

    stamped = 0
    missing: list[str] = []
    for entity in entities:
        value = lookup.get(entity.id)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            missing.append(entity.id)
            continue
        entity.notes[note_key] = str(value)
        stamped += 1

    if verbose and missing:
        print(
            f"{len(missing)} entity ids were not in {path} "
            f"(e.g. {missing[:3]}); left untouched."
        )
    return stamped


def save_delta_g_csv(
    entities: Iterable,
    path: str | Path,
    *,
    id_column: str = "Var1",
    value_column: str = "Var2",
    note_key: str = "deltaG",
) -> int:
    """Dump ``entity.notes[note_key]`` for each entity to a CSV.

    Entities without ``note_key`` set, or with a string value that cannot be converted to a
    float, get ``NaN`` written, which keeps one row per entity in entity order (mirrors MATLAB's
    ``array2table([ids, values])`` behaviour). A value of another type that ``float`` rejects
    (for example a list) raises :class:`TypeError`.

    Returns
    -------
    int
        The number of rows written (the number of entities).
    """
    rows: list[tuple[str, float]] = []
    for entity in entities:
        raw = entity.notes.get(note_key)
        if raw is None:
            value: float = math.nan
        else:
            try:
                value = float(raw)
            except ValueError:
                value = math.nan
        rows.append((entity.id, value))
    pd.DataFrame(rows, columns=[id_column, value_column]).to_csv(path, index=False)
    return len(rows)


__all__ = ["load_delta_g_csv", "save_delta_g_csv"]
_ = cobra  # silence "imported but unused" — kept for type-checker/IDE context
