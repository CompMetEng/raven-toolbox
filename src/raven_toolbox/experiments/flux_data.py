"""Measured growth rates and exchange fluxes of several conditions, and the parser of their file.

A flux data file has one row per growth condition (for example a carbon source or a chemostat
dilution rate). :func:`load_flux_data` reads it into a :class:`FluxData`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# `<met_name> (<rxn_id>)` -> capture met_name and rxn_id.


@dataclass
class FluxData:
    """Measured growth rates and exchange fluxes of several conditions.

    One row per condition; the exchange columns are the exchanges measured in at least one condition.
    Fluxes are in mmol/gDW/h, negative for uptake. Create it with :func:`load_flux_data` or directly
    from arrays.

    Attributes
    ----------
    conds : list of str
        Condition names. The carbon source of a condition is the exchange metabolite (``exch_mets``)
        with that name.
    p_tot : numpy.ndarray
        Total protein content per condition (g/gDW), shape ``(n_conds,)``. NaN if not measured.
    gr_rate : numpy.ndarray
        Growth rate per condition (1/h), shape ``(n_conds,)``.
    exch_fluxes : numpy.ndarray
        Exchange fluxes, conditions in rows and exchange reactions in columns, shape
        ``(n_conds, n_rxns)``. NaN if not measured; ``-1000`` and ``1000`` mean unconstrained uptake
        and secretion.
    exch_mets : list of str
        Metabolite name of each column of ``exch_fluxes``.
    exch_rxn_ids : list of str
        Exchange reaction id of each column of ``exch_fluxes``.
    weight : numpy.ndarray or None
        Weight of each condition in the distance, shape ``(n_conds,)``. 0 leaves the condition out of
        the simulation and the distance. ``None`` if the file has no ``weight`` column (all weights 1).
    source : list of str or None
        Origin of each condition. ``None`` if the file has no ``source`` column.
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
    weight: np.ndarray | None = None
    source: list[str] | None = None


def load_flux_data(path: str | Path) -> FluxData:
    r"""Read measured growth rates and exchange fluxes of several conditions from a file.

    The file is tab-delimited with one row per condition. The first three columns are the condition
    name, the total protein content ``Ptot`` (g/gDW) and the growth rate ``grRate`` (1/h). Every
    further column is an exchange flux (mmol/gDW/h, negative for uptake) headed
    ``<metabolite name> (<reaction id>)``::

        Condition   Ptot   grRate   glucose (EX_glc)   ethanol (EX_etoh)   weight   source
        glucose     0.40   0.40     -10.0              2.5                 1        study A
        ethanol     0.35   0.12     NaN                -8.0                1        study B

    An empty or non-numeric cell is NaN (not measured). A flux of ``-1000`` or ``1000`` means
    unconstrained uptake or secretion. The optional columns ``weight`` and ``source`` may stand in any
    position. ``weight`` is the weight of the condition in the distance; 0 leaves the condition out of
    the simulation and the distance (see :func:`~raven_toolbox.experiments.simulate_conditions`).
    ``source`` is free text.

    The header is split at the parenthesis matching the final closing one, so reaction ids may contain
    parentheses (``X (EX_a(e))`` gives ``X`` and ``EX_a(e)``).

    Parameters
    ----------
    path : str or pathlib.Path
        Path of the file.

    Returns
    -------
    FluxData
        The measurements, one row per condition.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is empty, has fewer than three header columns or no exchange columns, or has a row
        with another number of cells than the header.

    See Also
    --------
    load_experiment_data, simulate_conditions, apply_flux_data_constraints

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from raven_toolbox.experiments import load_flux_data
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> _ = (folder / "fluxData.tsv").write_text(
    ...     "Condition\tPtot\tgrRate\tglucose (EX_glc)\tweight\n"
    ...     "glucose\t0.4\t0.4\t-10\t1\n")
    >>> data = load_flux_data(folder / "fluxData.tsv")
    >>> data.conds, data.exch_rxn_ids, float(data.exch_fluxes[0, 0])
    (['glucose'], ['EX_glc'], -10.0)
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
    weight_idx = _index_of(header, "weight")
    source_idx = _index_of(header, "source")

    weights: np.ndarray | None = None
    source_values: list[str] | None = None

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


def _index_of(header: list[str], name: str) -> int | None:
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


def _split_header(header: str) -> tuple[str, str] | None:
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
