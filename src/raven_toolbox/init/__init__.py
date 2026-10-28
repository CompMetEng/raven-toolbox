"""Context-specific model extraction (ftINIT).

* :func:`run_ftinit` — the single-step ftINIT MILP (continuous indicators for
  positive-score reactions; binaries only on negatives).
* :func:`ftinit` — the full pipeline (``prep_init_model`` → staged ``run_ftinit`` →
  ``fill_tasks`` → ``remove_low_score_genes``).
* :func:`score_reactions_from_genes` / :func:`gene_scores_from_expression` —
  gene → reaction scoring (RNA-seq is the common upstream).
* :mod:`raven_toolbox.init.hpa` — HPA proteomics/RNA-seq parsing and gene-scoring
  adapters, the typical (f)tINIT entry point for tissue-specific runs.
"""
from raven_toolbox.init.ftinit import FtInitResult, ftinit, run_ftinit
from raven_toolbox.init.genes import remove_low_score_genes
from raven_toolbox.init.hpa import (
    HPA_LEVEL_SCORES,
    HPAData,
    HPARnaData,
    hpa_gene_scores,
    parse_hpa,
    parse_hpa_rna,
    rna_gene_scores,
)
from raven_toolbox.init.merge import group_rxn_scores, merge_linear
from raven_toolbox.init.model_scores import RAVEN_HPA_LEVEL_SCORES, TranscrData, score_for_init
from raven_toolbox.init.prep import PrepData, ReactionMasks, classify_reactions, prep_init_model
from raven_toolbox.init.score import gene_scores_from_expression, score_reactions_from_genes
from raven_toolbox.init.steps import InitStep, get_init_steps
from raven_toolbox.init.taskfill import TaskFillResult, fill_tasks

__all__ = [
    "HPA_LEVEL_SCORES",
    "FtInitResult",
    "HPAData",
    "HPARnaData",
    "InitStep",
    "PrepData",
    "RAVEN_HPA_LEVEL_SCORES",
    "ReactionMasks",
    "TaskFillResult",
    "TranscrData",
    "classify_reactions",
    "fill_tasks",
    "ftinit",
    "gene_scores_from_expression",
    "get_init_steps",
    "group_rxn_scores",
    "hpa_gene_scores",
    "merge_linear",
    "parse_hpa",
    "parse_hpa_rna",
    "prep_init_model",
    "remove_low_score_genes",
    "rna_gene_scores",
    "run_ftinit",
    "score_for_init",
    "score_reactions_from_genes",
]
