"""Container for the measured growth rates and exchange fluxes of several
conditions, and the parser of the ``fluxData.tsv`` format.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# `<met_name> (<rxn_id>)` -> capture met_name and rxn_id.


@dataclass
class FluxData:
    """Per-condition flux measurements with per-reaction exchange data.

    Attributes
    ----------
    conds
        Names of the experimental conditions / samples.
    p_tot
        Total protein content per condition, g/gDCW. Shape
        ``(n_conds,)``.
    gr_rate
        Measured growth rate per condition, 1/h. Shape ``(n_conds,)``.
    exch_fluxes
        Exchange flux per (condition, reaction), mmol/gDCW/h. Shape
        ``(n_conds, n_rxns)``. NaN means "no measurement"; ``+/-1000``
        means unconstrained (used by
        ``apply_flux_data_constraints``).
    exch_mets
        Metabolite names matching the columns of ``exch_fluxes``.
    exch_rxn_ids
        Exchange reaction IDs in the model, matching the columns of
        ``exch_fluxes`` and ``exch_mets``.
    weight
        Optional per-condition weights of the distance to the
        measurements. A weight of 0 leaves the condition out of the
        simulation and of the distance. ``None`` if the source file did not
        have the ``weight`` column.
    source
        Optional per-condition free-text describing where the data
        came from. ``None`` if the source file did not have a
        ``source`` column.
    """

    conds: list[str] = field(default_factory=list)
    p_tot: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    gr_rate: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    exch_fluxes: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    exch_mets: list[str] = field(default_factory=list)
    exch_rxn_ids: list[str] = field(default_factory=list)
    weight: Optional[np.ndarray] = None
    source: Optional[list[str]] = None


def load_flux_data(path: str | Path) -> FluxData:
    """Parse a ``fluxData.tsv`` into a ``FluxData``.

    File format (tab-delimited, one header row):

        condition  Ptot  grRate  <met1> (<rxn1>)  <met2> (<rxn2>)  ...
        c1         0.5   0.4     -10.0            5.0              ...
        c2         0.6   0.5     -8.5             4.2              ...

    Optional extra columns (in any position):

    * ``weight`` (float): weight of the condition in the distance to the measurements; 0
      leaves the condition out of the simulation and of the distance.
    * ``source`` (str): free-text describing the data origin.

    An exchange column header has the form ``<met> (<rxn>)``, split at
    the parenthesis that matches the final closing one, so a reaction id
    may contain parentheses (``X (EX_a(e))`` gives ``X`` and
    ``EX_a(e)``).

    Both columns are dropped from the matrix and surfaced on the
    returned ``FluxData`` as ``weight`` and ``source``.

    Parameters
    ----------
    path
        Path to the TSV file.

    Returns
    -------
    FluxData

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file has no header row, no exchange columns, or a row
        with a different number of cells than the header.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"fluxData file not found: {path}")

    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(line.rstrip("\n").split("\t"))

    if not rows:
        raise ValueError(f"{path} is empty.")
    if len(rows[0]) < 3:
        raise ValueError(
            f"{path}: header has only {len(rows[0])} column(s); "
            f"expected at least 'condition Ptot grRate'."
        )

    header = rows[0]
    data = rows[1:]
    for line_no, r in enumerate(data, start=2):
        if len(r) != len(header):
            raise ValueError(
                f"{path}: row {line_no} has {len(r)} cell(s), the header "
                f"has {len(header)}."
            )

    # Pull out optional columns first; collect indices to drop.
    optional_indices: list[int] = []
    if _index_of(header, "evotuneRMSEweight") is not None:
        raise ValueError(
            f"{path}: the column 'evotuneRMSEweight' is now called 'weight'."
        )
    weight_idx = _index_of(header, "weight")
    source_idx = _index_of(header, "source")

    weights: Optional[np.ndarray] = None
    source_values: Optional[list[str]] = None

    if weight_idx is not None:
        weights = np.array(
            [_parse_float(r[weight_idx]) for r in data],
            dtype=float,
        )
        optional_indices.append(weight_idx)
    if source_idx is not None:
        source_values = [r[source_idx] for r in data]
        optional_indices.append(source_idx)

    # Build the kept-column index list (preserving order).
    kept = [j for j in range(len(header)) if j not in optional_indices]
    kept_header = [header[j] for j in kept]

    # First three columns are condition / Ptot / grRate. Remaining are exchanges.
    if len(kept_header) < 4:
        raise ValueError(
            f"{path}: no exchange-flux columns found after the first three."
        )

    exch_headers = kept_header[3:]
    exch_mets, exch_rxn_ids = _split_exch_headers(exch_headers)

    conds = [r[kept[0]] for r in data]
    p_tot = np.array([_parse_float(r[kept[1]]) for r in data], dtype=float)
    gr_rate = np.array([_parse_float(r[kept[2]]) for r in data], dtype=float)
    exch_fluxes = np.full((len(data), len(exch_headers)), np.nan, dtype=float)
    for i, r in enumerate(data):
        for ci, j in enumerate(kept[3:]):
            exch_fluxes[i, ci] = _parse_float(r[j])

    return FluxData(
        conds=conds,
        p_tot=p_tot,
        gr_rate=gr_rate,
        exch_fluxes=exch_fluxes,
        exch_mets=exch_mets,
        exch_rxn_ids=exch_rxn_ids,
        weight=weights,
        source=source_values,
    )


def _index_of(header: list[str], name: str) -> Optional[int]:
    """Return the index of `name` in `header`, or None if absent."""
    for j, h in enumerate(header):
        if h.strip() == name:
            return j
    return None


def _parse_float(s: str) -> float:
    """Parse a numeric cell; empty / non-numeric -> NaN."""
    s = s.strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _split_header(header: str) -> Optional[tuple[str, str]]:
    """Split ``'<met> (<rxn>)'`` at the parenthesis that matches the final
    closing one; return None for a header of another form."""
    if not header.endswith(")"):
        return None
    depth = 0
    for k in range(len(header) - 1, -1, -1):
        if header[k] == ")":
            depth += 1
        elif header[k] == "(":
            depth -= 1
            if depth == 0:
                rxn = header[k + 1:-1].strip()
                if not rxn:
                    return None
                return header[:k].strip(), rxn
    return None


def _split_exch_headers(headers: list[str]) -> tuple[list[str], list[str]]:
    """Split each header `'<met> (<rxn>)'` into (met, rxn) pairs.

    Headers without parentheses pass through with the rxn_id equal
    to the full header (the
    capture-group falls through unchanged).
    """
    mets: list[str] = []
    rxns: list[str] = []
    no_paren: list[str] = []
    for h in headers:
        parts = _split_header(h.strip())
        if parts is not None:
            mets.append(parts[0])
            rxns.append(parts[1])
        else:
            mets.append(h.strip())
            rxns.append(h.strip())
            no_paren.append(h.strip())
    if no_paren:
        # The reaction id is taken from the header, so a column without the
        # `<met> (<rxn>)` form will be matched against a reaction whose id is
        # the metabolite name -- almost certainly wrong. Surface it.
        logger.warning(
            "load_flux_data: %d exchange column header(s) lack the "
            "'<met> (<rxn>)' form; their reaction id falls back to the full "
            "header text: %s",
            len(no_paren), no_paren[:5],
        )
    return mets, rxns
