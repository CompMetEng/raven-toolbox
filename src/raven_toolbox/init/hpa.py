"""Human Protein Atlas (HPA) parsers + gene-scoring adapters.

HPA publishes two datasets per release: a **proteomics** table (``normal_tissue.tsv``)
with per-tissue / per-cell-type *categorical* expression levels (High/Medium/Low/Not
detected) plus reliability flags, and an **RNA-seq** table (``rna_tissue_consensus.tsv``
/ ``rna_tissue_gtex.tsv``) with per-tissue *TPM* values. Both are returned as tidy
:class:`pandas.DataFrame`\\ s; the scoring adapters delegate the GPR walk to
:func:`raven_toolbox.init.score.score_reactions_from_genes` so there is one source of truth
for reaction scoring.

Pipeline (typical (f)tINIT entry):

.. code-block:: python

    hpa = parse_hpa("normal_tissue.tsv")
    gene_scores = hpa_gene_scores(hpa, tissue="liver", celltype="hepatocytes")
    rxn_scores  = score_reactions_from_genes(model, gene_scores)
    # → ftinit(prep, rxn_scores=rxn_scores, gene_scores=gene_scores, ...)

or for RNA-seq:

.. code-block:: python

    rna = parse_hpa_rna("rna_tissue_consensus.tsv")
    gene_scores = rna_gene_scores(rna, tissue="liver")   # ref = per-gene cross-tissue mean
    rxn_scores  = score_reactions_from_genes(model, gene_scores)
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from raven_toolbox.init.score import gene_scores_from_expression

# RAVEN's hpaLevelScores defaults (scoreModel.m). HPA reports either antibody-staining
# levels (Strong/Moderate/Weak/Negative) or "APE" classes (High/Medium/Low/None/Not detected /
# Ascending/Descending/...); the categories of RAVEN's table are mapped here. Unknown levels
# (e.g. "Mixed", "N/A") fall through to NaN and are dropped during scoring.
HPA_LEVEL_SCORES: dict[str, float] = {
    "High": 20.0, "Medium": 15.0, "Low": 10.0, "None": -8.0, "Not detected": -8.0,
    "Strong": 20.0, "Moderate": 15.0, "Weak": 10.0, "Negative": -8.0,
}

_HPA_HEADERS = ("Gene", "Gene name", "Tissue", "Cell type", "Level", "Reliability")
_HPA_RNA_HEADERS = ("Gene", "Gene name", "Tissue")  # extra TPM columns follow


@dataclass
class HPAData:
    """Tidy HPA proteomics data: one row per (gene, tissue, cell type).

    :attr:`df` columns: ``gene_id``, ``gene_name``, ``tissue``, ``celltype``, ``level``,
    ``reliability``. ``level`` is the categorical string from HPA; map it to numbers via
    :func:`hpa_gene_scores` (or pass a custom ``level_scores``).
    """

    df: pd.DataFrame

    def tissues(self) -> list[str]:
        return sorted(self.df["tissue"].unique())

    def celltypes(self, tissue: str) -> list[str]:
        return sorted(self.df.loc[self.df["tissue"] == tissue, "celltype"].unique())


@dataclass
class HPARnaData:
    """Tidy HPA RNA-seq data: one row per (gene, tissue) with TPM.

    :attr:`df` columns: ``gene_id``, ``gene_name``, ``tissue``, ``tpm``.
    """

    df: pd.DataFrame

    def tissues(self) -> list[str]:
        return sorted(self.df["tissue"].unique())

    def expression(self, tissue: str) -> dict[str, float]:
        """{gene_id: TPM} for ``tissue``. Use this directly with
        :func:`raven_toolbox.init.score.gene_scores_from_expression`."""
        sub = self.df.loc[self.df["tissue"] == tissue, ["gene_id", "tpm"]]
        return dict(zip(sub["gene_id"], sub["tpm"], strict=True))


def parse_hpa(path: str | Path) -> HPAData:
    """Parse an HPA proteomics dump (``normal_tissue.tsv``; version ≥17 format).

    Expected columns (any reasonable delimiter; HPA ships tab-separated):
    ``Gene  Gene name  Tissue  Cell type  Level  Reliability``. Returns an
    :class:`HPAData` with one row per (gene, tissue, cell type).
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, na_filter=False)
    _check_headers(df, _HPA_HEADERS, path)
    df = df.rename(columns={
        "Gene": "gene_id", "Gene name": "gene_name", "Tissue": "tissue",
        "Cell type": "celltype", "Level": "level", "Reliability": "reliability",
    })[["gene_id", "gene_name", "tissue", "celltype", "level", "reliability"]]
    return HPAData(df.reset_index(drop=True))


def parse_hpa_rna(path: str | Path) -> HPARnaData:
    """Parse an HPA RNA-seq dump, as RAVEN's ``parseHPArna``.

    One row per gene and tissue. The tissue column is ``Tissue`` (v19) or ``Sample`` (v18); the
    value column is ``TPM`` (v19) or ``Value`` (v18); ``Gene`` and ``Gene name`` are required.
    A file without a tissue or value column (for example one TPM column per tissue) raises
    ``ValueError``. When a gene and tissue occur more than once, the last row is kept and a
    warning is issued. Values that are not numbers are dropped.
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, na_filter=False)
    for col in ("Gene", "Gene name"):
        if col not in df.columns:
            raise ValueError(f'{path}: could not find the column "{col}" (got {list(df.columns)})')
    tissue_col = "Tissue" if "Tissue" in df.columns else "Sample" if "Sample" in df.columns else None
    if tissue_col is None:
        raise ValueError(f'{path}: could not find a tissue column ("Tissue" or "Sample"; '
                         f"got {list(df.columns)})")
    value_col = "TPM" if "TPM" in df.columns else "Value" if "Value" in df.columns else None
    if value_col is None:
        raise ValueError(f'{path}: could not find an expression value column ("TPM" or "Value"; '
                         f"got {list(df.columns)})")
    df = df.rename(columns={"Gene": "gene_id", "Gene name": "gene_name",
                            tissue_col: "tissue", value_col: "tpm"})
    df = df[["gene_id", "gene_name", "tissue", "tpm"]]
    repeated = df.duplicated(["gene_id", "tissue"], keep="last")
    if repeated.any():
        warnings.warn(f"{path}: {int(repeated.sum())} rows repeat a gene and tissue; "
                      "the last row of each is kept", stacklevel=2)
        df = df[~repeated]
    df = df.assign(tpm=pd.to_numeric(df["tpm"], errors="coerce"))
    df = df.dropna(subset=["tpm"]).reset_index(drop=True)
    return HPARnaData(df)


def hpa_gene_scores(
    hpa: HPAData,
    tissue: str,
    celltype: str | None = None,
    *,
    level_scores: Mapping[str, float] | None = None,
    multiple_celltype: str = "best",
) -> dict[str, float]:
    """Numeric gene scores from HPA levels for one ``tissue`` (optionally one ``celltype``).

    Maps HPA's categorical levels to numbers via ``level_scores`` (default
    :data:`HPA_LEVEL_SCORES`). Genes absent from the tissue, or whose level is not in the
    score table, are omitted from the output (downstream
    :func:`score_reactions_from_genes` will then fall back to ``no_gene_score`` for any
    reaction whose genes are all absent).

    When several cell types per tissue carry the gene, ``multiple_celltype`` chooses
    between ``"best"`` (max score, RAVEN default) and ``"average"`` (mean across cell types).
    """
    if multiple_celltype not in ("best", "average"):
        raise ValueError(f"multiple_celltype must be 'best' or 'average'; got {multiple_celltype!r}")
    scores_table = dict(level_scores) if level_scores is not None else HPA_LEVEL_SCORES

    sub = hpa.df.loc[hpa.df["tissue"] == tissue].copy()
    if celltype is not None:
        sub = sub.loc[sub["celltype"] == celltype]
    sub["score"] = sub["level"].map(scores_table)
    sub = sub.dropna(subset=["score"])  # unknown HPA levels drop out (omitted, not -inf)
    if sub.empty:
        return {}
    agg = {"best": "max", "average": "mean"}[multiple_celltype]
    return sub.groupby("gene_id")["score"].agg(agg).to_dict()


def rna_gene_scores(
    rna: HPARnaData,
    tissue: str,
    *,
    reference: Mapping[str, float] | float | None = None,
    factor: float = 5.0,
    max_score: float = 10.0,
    min_score: float = -5.0,
) -> dict[str, float]:
    """Numeric gene scores from HPA RNA-seq TPM for one ``tissue``.

    Thin wrapper over :func:`raven_toolbox.init.score.gene_scores_from_expression` (the same
    ``5·ln(TPM/reference)``-clamped scoring used elsewhere): selects the tissue, derives
    a reference if none is given (per-gene mean TPM across all tissues — RAVEN's default
    for ``arrayData.threshold``), and returns ``{gene_id: score}``.

    ``factor`` scales how fast the score moves away from 0 as TPM diverges from
    ``reference``; ``max_score``/``min_score`` clamp the result to that range.
    """
    if tissue not in set(rna.df["tissue"]):
        raise ValueError(f"tissue {tissue!r} not in dataset (tissues: {rna.tissues()})")
    if reference is None:
        reference = rna.df.groupby("gene_id")["tpm"].mean().to_dict()
    return gene_scores_from_expression(rna.expression(tissue), reference,
                                       factor=factor, max_score=max_score, min_score=min_score)


def _check_headers(df: pd.DataFrame, expected: tuple[str, ...], path: str | Path) -> None:
    missing = [h for h in expected if h not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing HPA columns {missing} (got {list(df.columns)})")
