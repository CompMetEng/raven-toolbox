"""Human Protein Atlas (HPA) parsers and gene-scoring adapters.

HPA publishes two datasets per release: a proteomics table (``normal_tissue.tsv``)
with per-tissue and per-cell-type categorical expression levels (High, Medium, Low, Not
detected) plus reliability flags, and an RNA-seq table (``rna_tissue_consensus.tsv``
or ``rna_tissue_gtex.tsv``) with per-tissue TPM values. The parsers return
:class:`HPAData` and :class:`HPARnaData`, which hold the data as a tidy
:class:`pandas.DataFrame` in ``df``. The scoring adapters return gene scores, and
:func:`raven_toolbox.init.score.score_reactions_from_genes` converts them to reaction
scores.

Pipeline for proteomics data:

.. code-block:: python

    hpa = parse_hpa("normal_tissue.tsv")
    gene_scores = hpa_gene_scores(hpa, tissue="liver", celltype="hepatocytes")
    rxn_scores  = score_reactions_from_genes(model, gene_scores)
    # → ftinit(prep, rxn_scores=rxn_scores, gene_scores=gene_scores, ...)

and for RNA-seq data:

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
# Ascending/Descending/...); the categories of RAVEN's table are mapped here. A level of the
# data that the table lacks (e.g. "Mixed", "N/A") raises ValueError in hpa_gene_scores.
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
    ``reliability``. ``level`` is the categorical string from HPA; :func:`hpa_gene_scores`
    maps it to numbers (a custom ``level_scores`` table can be passed to it).
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
        """Mapping of ``gene_id`` to TPM for ``tissue``, as input of
        :func:`raven_toolbox.init.score.gene_scores_from_expression`.
        """
        sub = self.df.loc[self.df["tissue"] == tissue, ["gene_id", "tpm"]]
        return dict(zip(sub["gene_id"], sub["tpm"], strict=True))


def parse_hpa(path: str | Path) -> HPAData:
    """Parse an HPA proteomics dump (``normal_tissue.tsv``; version 17 or later format).

    The delimiter is detected automatically (HPA ships tab-separated files). The expected
    columns are ``Gene``, ``Gene name``, ``Tissue``, ``Cell type``, ``Level`` and
    ``Reliability``; a missing column raises :class:`ValueError`. Returns an
    :class:`HPAData` with one row per (gene, tissue, cell type).

    Parameters
    ----------
    path : str or Path
        Path of the HPA proteomics file, read with :func:`pandas.read_csv`.
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
    warning is issued. Rows whose value is not a number are dropped.

    Parameters
    ----------
    path : str or Path
        Path of the HPA RNA-seq file, read with :func:`pandas.read_csv`.
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


def level_score_table(level_scores: Mapping[str, float]) -> dict[str, float]:
    """Level scores keyed by upper-cased name.

    The first of the names that differ only in case is used.
    """
    table: dict[str, float] = {}
    for name, value in level_scores.items():
        table.setdefault(str(name).upper(), value)
    return table


def check_levels(df: pd.DataFrame, table: Mapping[str, float]) -> None:
    """Raise :class:`ValueError` for a level in ``df`` that the upper-cased ``table`` lacks.

    All rows count, whichever tissue is scored, as RAVEN's ``scoreModel`` checks every level of
    ``hpaData.levels``.
    """
    unknown = sorted({str(level) or "<blank>" for level in df["level"]
                      if str(level).upper() not in table})
    if unknown:
        raise ValueError("There are expression level categories that do not match the level scores: "
                         + ", ".join(unknown))


def hpa_gene_scores(
    hpa: HPAData,
    tissue: str,
    celltype: str | None = None,
    *,
    level_scores: Mapping[str, float] | None = None,
    multiple_celltype: str = "best",
) -> dict[str, float]:
    """Numeric gene scores from HPA levels for one ``tissue`` (optionally one ``celltype``).

    Maps the categorical HPA levels to numbers via ``level_scores`` (default
    :data:`HPA_LEVEL_SCORES`). As in RAVEN's ``scoreModel``, level names, tissue and cell type
    are compared case-insensitively, and when the table has several names that differ only in
    case, the first one is used. Genes absent from the tissue are omitted from the output;
    :func:`score_reactions_from_genes` then assigns ``no_gene_score`` to a reaction whose genes
    are all absent. A level that occurs anywhere in ``hpa`` and is not in the score table raises
    :class:`ValueError`, whichever tissue is scored. A tissue or cell type that ``hpa`` does not
    contain gives an empty result. RAVEN raises only when neither the HPA nor the RNA data
    contain it; that check needs both data sets and is not made here.

    Parameters
    ----------
    hpa : HPAData
        Parsed HPA proteomics data, as returned by :func:`parse_hpa`.
    tissue : str
        Tissue to score, compared case-insensitively with the ``tissue`` column of ``hpa``.
    celltype : str or None
        Cell type to score, compared case-insensitively. The default ``None`` and an empty
        string use all cell types of the tissue.
    level_scores : Mapping[str, float] or None
        Keyword-only; default ``None``, which uses :data:`HPA_LEVEL_SCORES`; an empty mapping
        does too, as RAVEN's ``hpaLevelScores``. Score of each HPA level name.
    multiple_celltype : str
        Keyword-only; default ``"best"``. Selects how the scores of several cell types of a
        tissue that carry the gene are combined: ``"best"`` is the maximum score (RAVEN
        default) and ``"average"`` is the mean over the cell types. Any other value raises
        :class:`ValueError`.
    """
    if multiple_celltype not in ("best", "average"):
        raise ValueError(f"multiple_celltype must be 'best' or 'average'; got {multiple_celltype!r}")
    scores_table = level_score_table(level_scores or HPA_LEVEL_SCORES)
    check_levels(hpa.df, scores_table)

    sub = hpa.df.loc[hpa.df["tissue"].astype(str).str.upper() == tissue.upper()].copy()
    if celltype:
        sub = sub.loc[sub["celltype"].astype(str).str.upper() == celltype.upper()]
    sub["score"] = sub["level"].map(lambda level: scores_table[str(level).upper()]).astype(float)
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

    Wrapper around :func:`raven_toolbox.init.score.gene_scores_from_expression` (the same
    ``5·ln(TPM/reference)`` scoring with clamping that is used elsewhere). The function selects
    the tissue, derives a reference if none is given (the per-gene mean TPM over all tissues,
    RAVEN's default for ``arrayData.threshold``), and returns ``{gene_id: score}``. As in RAVEN's
    ``scoreModel``, the tissue is compared case-insensitively; when ``rna`` has several names
    that differ only in case, the highest TPM of a gene over those names is used (RAVEN's
    default ``multipleCellScoring='best'``). A tissue that is not in ``rna`` raises
    :class:`ValueError`.

    Parameters
    ----------
    rna : HPARnaData
        Parsed HPA RNA-seq data, as returned by :func:`parse_hpa_rna`.
    tissue : str
        Tissue to score, compared case-insensitively with the ``tissue`` column of ``rna``.
    reference : Mapping[str, float] or float or None
        Keyword-only. Per-gene reference TPM, or one threshold for all genes. The default
        ``None`` uses the mean TPM of each gene over all tissues.
    factor : float
        Keyword-only; default 5.0. Multiplies ``ln(TPM/reference)``.
    max_score : float
        Keyword-only; default 10.0. Upper limit of the gene score.
    min_score : float
        Keyword-only; default -5.0. Lower limit of the gene score.
    """
    matches = [t for t in dict.fromkeys(rna.df["tissue"]) if str(t).upper() == tissue.upper()]
    if not matches:
        raise ValueError(f"tissue {tissue!r} not in dataset (tissues: {rna.tissues()})")
    if reference is None:
        reference = rna.df.groupby("gene_id")["tpm"].mean().to_dict()
    sub = rna.df.loc[rna.df["tissue"].isin(matches), ["gene_id", "tpm"]]
    expression = sub.groupby("gene_id")["tpm"].max().to_dict()
    return gene_scores_from_expression(expression, reference,
                                       factor=factor, max_score=max_score, min_score=min_score)


def _check_headers(df: pd.DataFrame, expected: tuple[str, ...], path: str | Path) -> None:
    missing = [h for h in expected if h not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing HPA columns {missing} (got {list(df.columns)})")
