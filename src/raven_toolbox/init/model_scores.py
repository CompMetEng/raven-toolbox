"""Reaction and gene scores for ftINIT from HPA and gene expression data, as RAVEN's ``scoreModel``.

The inputs are the ones RAVEN's ``ftINIT`` takes: an optional :class:`~raven_toolbox.init.hpa.HPAData`
(categorical protein levels) and an optional :class:`TranscrData` (expression levels per gene
and tissue or cell type), a tissue and optionally a cell type. Protein levels are mapped to
numbers with a level table, expression to ``5·ln(level/threshold)``; both are then combined per
gene (an HPA score replaces an expression score of the same gene) or per reaction
(``data_precedence="reaction"``) and walked through each GPR
by :func:`~raven_toolbox.init.score.score_reactions_from_genes`.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import cobra
import numpy as np

from raven_toolbox.init.hpa import HPAData, HPARnaData, check_levels, level_score_table
from raven_toolbox.init.score import gene_scores_from_expression, score_reactions_from_genes

#: RAVEN's ``hpaLevelScores`` (compared case-insensitively).
RAVEN_HPA_LEVEL_SCORES: dict[str, float] = {
    "High": 20.0, "Medium": 15.0, "Low": 10.0, "None": -8.0,
    "Strong": 20.0, "Moderate": 15.0, "Weak": 10.0, "Negative": -8.0, "Not detected": -8.0,
}


@dataclass
class TranscrData:
    """Gene expression, as RAVEN's ``transcrData``.

    ``levels`` is genes x samples (NaN where nothing was measured); ``tissues`` and ``celltypes``
    name each column. ``threshold`` is a single value or one value per gene above which a gene
    counts as expressed; without it the mean over the measured samples is used.
    """

    genes: list[str]
    tissues: list[str]
    celltypes: list[str]
    levels: np.ndarray
    threshold: float | Sequence[float] | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_rna(cls, rna: HPARnaData, threshold: float | Sequence[float] | None = None) -> TranscrData:
        """Wrap tidy HPA RNA-seq data (one TPM per gene and tissue; no cell types)."""
        wide = rna.df.pivot_table(index="gene_id", columns="tissue", values="tpm", aggfunc="mean")
        return cls(list(wide.index), list(wide.columns), [""] * wide.shape[1],
                   wide.to_numpy(dtype=float), threshold)


def _match(names: Sequence[str], wanted: str) -> np.ndarray:
    return np.array([str(n).upper() == wanted.upper() for n in names], dtype=bool)


def _array_gene_scores(
    data: TranscrData, model_genes: set[str], tissue: str, celltype: str | None,
    multiple_cell_scoring: str,
) -> dict[str, float]:
    levels = np.asarray(data.levels, dtype=float).reshape(len(data.genes), len(data.tissues))
    cols = _match(data.tissues, tissue)
    if celltype:
        cols &= _match(data.celltypes, celltype)
    threshold = data.threshold
    if threshold is not None and np.ndim(threshold) == 0:
        threshold = np.full(len(data.genes), np.asarray(threshold, dtype=float))
    keep = np.array([g in model_genes for g in data.genes], dtype=bool)
    keep &= ~np.all(np.isnan(levels[:, cols]), axis=1) if cols.any() else False
    zeroed = np.where(np.isnan(levels), 0.0, levels)
    n_measured = (~np.isnan(levels)).sum(axis=1)
    n_selected = (~np.isnan(levels[:, cols])).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        average = (zeroed.sum(axis=1) / n_measured) if threshold is None else np.asarray(threshold, float)
        if multiple_cell_scoring == "max":
            current = zeroed[:, cols].max(axis=1) if cols.any() else np.zeros(len(data.genes))
        else:
            current = zeroed[:, cols].sum(axis=1) / n_selected
    current_d = {g: float(c) for g, c, k in zip(data.genes, current, keep, strict=False) if k}
    average_d = {g: float(a) for g, a, k in zip(data.genes, average, keep, strict=False) if k}
    return gene_scores_from_expression(
        {g: v for g, v in current_d.items() if not math.isnan(v)}, average_d)


def _hpa_gene_scores(
    hpa: HPAData, model_genes: set[str], tissue: str, celltype: str | None,
    multiple_cell_scoring: str, level_scores: Mapping[str, float],
) -> dict[str, float]:
    df = hpa.df
    sel = df["tissue"].str.upper() == tissue.upper()
    if celltype:
        sel &= df["celltype"].str.upper() == celltype.upper()
    sub = df.loc[sel & df["gene_id"].isin(model_genes)]
    table = level_score_table(level_scores)
    check_levels(df, table)
    scored = sub.assign(score=[table[str(lv).upper()] for lv in sub["level"]])
    agg = {"max": "max", "average": "mean"}[multiple_cell_scoring]
    return scored.groupby("gene_id")["score"].agg(agg).to_dict()


def score_for_init(
    model: cobra.Model,
    hpa: HPAData | None,
    transcr: TranscrData | HPARnaData | None,
    tissue: str,
    celltype: str | None = None,
    *,
    no_gene_score: float = -2.0,
    isozyme_scoring: str = "max",
    complex_scoring: str = "min",
    multiple_cell_scoring: str = "max",
    hpa_level_scores: Mapping[str, float] | None = None,
    data_precedence: str = "gene",
) -> tuple[dict[str, float], dict[str, float]]:
    """Reaction scores and gene scores for ``tissue`` (``{reaction id: score}``, ``{gene id: score}``).

    Mirrors RAVEN's ``scoreModel``. ``data_precedence="gene"`` lets an HPA score replace an
    expression score of the same gene; ``"reaction"`` scores a reaction from HPA alone if any of
    its genes has an HPA score and from expression otherwise. Genes without data are absent
    from the gene scores. Raises ``ValueError`` for missing data, unknown option values, a
    tissue or cell type found in neither dataset, or expression with fewer than two tissues
    and no threshold.

    Parameters
    ----------
    model : cobra.Model
        Model whose reactions are scored. Only genes of the model are scored.
    hpa : HPAData or None
        Parsed HPA proteomics data. ``None`` scores from ``transcr`` alone. At least one of
        ``hpa`` and ``transcr`` must be given.
    transcr : TranscrData or HPARnaData or None
        Expression data. An :class:`~raven_toolbox.init.hpa.HPARnaData` is converted with
        :meth:`TranscrData.from_rna` without a threshold. ``None`` scores from ``hpa``
        alone.
    tissue : str
        Tissue to score, compared case-insensitively. It must occur in ``hpa`` or ``transcr``.
    celltype : str or None
        Cell type to score, compared case-insensitively. The default ``None`` and an empty
        string use all cell types of the tissue. A non-empty value must occur in ``hpa`` or
        ``transcr``.
    no_gene_score : float
        Keyword-only; default -2.0. Score of a reaction with no genes or no scored genes.
    isozyme_scoring : str
        Keyword-only; default ``"max"``. Combination of genes joined by OR; see
        :func:`~raven_toolbox.init.score.score_reactions_from_genes`.
    complex_scoring : str
        Keyword-only; default ``"min"``. Combination of genes joined by AND; accepts the
        same values as ``isozyme_scoring``.
    multiple_cell_scoring : str
        Keyword-only; default ``"max"``. Combination of the values of several cell types
        (or samples) of the tissue for one gene: ``"max"`` takes the maximum and
        ``"average"`` the mean. Any other value raises ``ValueError``.
    hpa_level_scores : Mapping[str, float] or None
        Keyword-only. Score of each HPA level name, compared case-insensitively. ``None`` and
        an empty mapping use :data:`RAVEN_HPA_LEVEL_SCORES`.
    data_precedence : str
        Keyword-only; default ``"gene"``. ``"gene"`` lets an HPA gene score replace an
        expression score of the same gene. ``"reaction"`` scores a reaction from HPA alone if
        any of its genes has an HPA score and from expression otherwise. Any other value
        raises ``ValueError``.
    """
    if hpa is None and transcr is None:
        raise ValueError("Must supply hpa, transcr or both")
    if multiple_cell_scoring not in ("max", "average"):
        raise ValueError('multiple_cell_scoring must be "max" or "average"')
    if data_precedence not in ("gene", "reaction"):
        raise ValueError('data_precedence must be "gene" or "reaction"')
    if isinstance(transcr, HPARnaData):
        transcr = TranscrData.from_rna(transcr)
    if transcr is not None and len(set(transcr.tissues)) < 2 and transcr.threshold is None:
        raise ValueError("transcr must contain measurements for at least two celltypes/tissues "
                         "since the score is calculated based on the expression level compared "
                         "to the overall average")
    tissues = {t.upper() for t in (hpa.df["tissue"] if hpa is not None else [])}
    tissues |= {t.upper() for t in (transcr.tissues if transcr is not None else [])}
    if tissue.upper() not in tissues:
        raise ValueError("The tissue name does not match")
    if celltype:
        types = {c.upper() for c in (hpa.df["celltype"] if hpa is not None else [])}
        types |= {c.upper() for c in (transcr.celltypes if transcr is not None else [])}
        if celltype.upper() not in types:
            raise ValueError("The cell type name does not match")

    genes = {g.id for g in model.genes}
    array = (_array_gene_scores(transcr, genes, tissue, celltype, multiple_cell_scoring)
             if transcr is not None else {})
    protein = (_hpa_gene_scores(hpa, genes, tissue, celltype, multiple_cell_scoring,
                                hpa_level_scores or RAVEN_HPA_LEVEL_SCORES)
               if hpa is not None else {})
    gene_scores = {**array, **protein}
    kw = {"isozyme_scoring": isozyme_scoring, "complex_scoring": complex_scoring}
    if data_precedence == "gene":
        return score_reactions_from_genes(model, gene_scores, no_gene_score=no_gene_score, **kw), gene_scores

    nan = float("nan")
    from_hpa = score_reactions_from_genes(model, protein, no_gene_score=nan, **kw)
    from_array = score_reactions_from_genes(model, array, no_gene_score=nan, **kw)
    scores: dict[str, float] = {}
    for rxn in model.reactions:
        use_hpa = any(g.id in protein for g in rxn.genes)
        value = from_hpa[rxn.id] if use_hpa else from_array[rxn.id]
        scores[rxn.id] = no_gene_score if math.isnan(value) else value
    return scores, gene_scores
