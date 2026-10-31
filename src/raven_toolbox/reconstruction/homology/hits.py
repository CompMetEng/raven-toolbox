"""Homology hits table, the data structure shared by the homology functions.

The hits are one ``pandas.DataFrame`` of bidirectional hits, one row per hit.
It is the input of :func:`get_model_from_homology` and the output of the BLAST and
DIAMOND wrappers.

Columns (``HIT_COLUMNS``):
``from_id, to_id`` (organism or model ids), ``from_gene, to_gene`` (the matched
genes; ``from_gene`` is in ``from_id``), and the hit metrics
``evalue, identity, align_len, bitscore, ppos``.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

HIT_COLUMNS = [
    "from_id", "to_id", "from_gene", "to_gene",
    "evalue", "identity", "align_len", "bitscore", "ppos",
]


def make_ortholog_hits(
    ortholog_pairs: Iterable[tuple[str, str]],
    source_model_id: str,
    target_id: str,
) -> pd.DataFrame:
    """Build a bidirectional hits table from a predefined ortholog list.

    Each ``(source_gene, target_gene)`` pair is emitted in both directions with
    fixed metrics (evalue 0, identity 100, align_len 1000, bitscore 1000, ppos 100).
    A known ortholog mapping can thus be passed to :func:`get_model_from_homology`
    without a BLAST run.

    Parameters
    ----------
    ortholog_pairs
        Iterable of ``(source_gene, target_gene)``. The source gene belongs to the
        template model, the target gene to the organism being built.
    source_model_id
        ID of the template model the source genes belong to.
    target_id
        ID of the organism to build a model for (``model_for``).
    """
    pairs = [(str(s), str(t)) for s, t in ortholog_pairs]
    if not pairs:
        raise ValueError("ortholog_pairs is empty.")

    rows = []
    for source_gene, target_gene in pairs:
        rows.append((source_model_id, target_id, source_gene, target_gene, 0.0, 100.0, 1000, 1000.0, 100.0))
        rows.append((target_id, source_model_id, target_gene, source_gene, 0.0, 100.0, 1000, 1000.0, 100.0))
    return pd.DataFrame(rows, columns=HIT_COLUMNS)


def validate_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """Check that a hits DataFrame has the required columns and return it unchanged."""
    missing = [c for c in HIT_COLUMNS if c not in hits.columns]
    if missing:
        raise ValueError(f"hits is missing required columns: {missing}")
    return hits
