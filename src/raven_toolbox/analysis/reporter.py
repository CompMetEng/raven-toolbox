"""Reporter Metabolites: metabolites around which transcriptional change concentrates.

Patil & Nielsen, PNAS 2005. Each gene's differential-expression p-value becomes a
Z-score ``z = -Φ⁻¹(p)``; for every metabolite the Z-scores of the genes on its
neighbouring reactions are aggregated (``Σz / √n``), background-corrected, and turned
back into a p-value.

The background correction has an exact closed form (sampling with replacement from the
scored-gene pool: a random ``Σz/√n`` has mean ``√n·μ`` and standard deviation ``σ``
with μ, σ the mean and standard deviation of the scored Z-scores), so the corrected
score is ``(metZ − √n·μ) / σ`` and needs no Monte-Carlo sampling.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import cobra
import numpy as np
import pandas as pd
from scipy.stats import norm

_CLAMP = 15.0  # |Z| cap for p-values of exactly 0 or 1 (RAVEN's ±15)


@dataclass
class ReporterResult:
    """Reporter-metabolite scores for one gene set.

    ``test`` is ``"all"``, ``"up"`` or ``"down"``; ``table`` is a DataFrame with
    columns ``metabolite, name, z_score, p_value, n_genes, mean_z, std_z`` sorted by
    descending ``z_score``.
    """

    test: str
    table: pd.DataFrame


def _gene_z(pvalues: dict[str, float]) -> dict[str, float]:
    genes = list(pvalues)
    z = -norm.ppf([pvalues[g] for g in genes])
    z = np.where(np.isposinf(z), _CLAMP, z)
    z = np.where(np.isneginf(z), -_CLAMP, z)
    return dict(zip(genes, z, strict=True))


def _reporter_one(model: cobra.Model, gene_z: dict[str, float], test: str) -> ReporterResult:
    z_values = np.fromiter(gene_z.values(), dtype=float)
    # Moments are taken on z-scores centred on c, the first z-score plus the mean of the
    # z-scores relative to it (two passes). A spread of a few ulps between otherwise identical
    # z-scores is then not lost in the rounding of the mean, also when the first z-score is an outlier.
    z_first = float(z_values[0]) if z_values.size else 0.0
    centre = z_first + float((z_values - z_first).mean()) if z_values.size else 0.0
    centred = z_values - centre
    mu_c = float(centred.mean()) if z_values.size else 0.0
    sigma = float(centred.std(ddof=0)) if z_values.size else 0.0
    # The spread is tested for being 0 up to 100 * eps(max|z|), as in RAVEN.
    spread = float(z_values.max() - z_values.min()) if z_values.size else 0.0
    has_spread = spread > 100 * float(np.spacing(np.abs(z_values).max())) if z_values.size else False

    rows = []
    for met in model.metabolites:
        neighbours = {g.id for rxn in met.reactions for g in rxn.genes if g.id in gene_z}
        if not neighbours:
            continue
        zs = np.array([gene_z[g] for g in neighbours])
        n = zs.size
        raw_c = (zs - centre).sum() / math.sqrt(n)
        # Exact background correction for sampling-with-replacement (see module doc).
        corrected = (raw_c - math.sqrt(n) * mu_c) / sigma if has_spread else 0.0
        if not math.isfinite(corrected):  # never on clamped z-scores; guards the p-value
            corrected = 0.0
        rows.append(
            {
                "metabolite": met.id,
                "name": met.name or met.id,
                "z_score": corrected,
                "p_value": float(1.0 - norm.cdf(corrected)),
                "n_genes": n,
                "mean_z": float(zs.mean()),
                "std_z": float(zs.std(ddof=1)) if n > 1 else 0.0,
            }
        )
    table = pd.DataFrame(rows, columns=["metabolite", "name", "z_score", "p_value", "n_genes", "mean_z", "std_z"])
    table = table.sort_values("z_score", ascending=False, ignore_index=True)
    return ReporterResult(test, table)


def reporter_metabolites(
    model: cobra.Model,
    gene_pvalues: Mapping[str, float],
    *,
    gene_fold_changes: Mapping[str, float] | None = None,
) -> list[ReporterResult]:
    """Compute Reporter Metabolites from per-gene differential-expression p-values.

    ``gene_pvalues`` maps gene id to p-value. Genes not in the model and genes with a
    NaN (or ``None``) p-value are dropped. Any other p-value outside ``[0, 1]``,
    including ±inf and for genes that are not in the model, raises ``ValueError``
    (RAVEN raises ``RAVEN:badInput``). If ``gene_fold_changes`` (gene id to log fold change)
    has at least one non-zero value, two extra results are returned for the up- (fc ≥ 0)
    and down- (fc < 0) regulated gene subsets, in addition to ``"all"``; with all fold
    changes zero (or none given) only ``"all"`` is returned, as in RAVEN. A metabolite
    with a single scored neighbouring gene has ``std_z`` 0.

    The background correction uses the population standard deviation
    (``ddof=0``) of the scored Z-scores, which the closed form requires. The
    reported ``p_value`` is the one-sided enrichment ``1 - Φ(z)`` and each
    table is sorted by ``z_score`` in descending order.

    Parity with RAVEN's ``reporterMetabolites``: the same closed-form
    correction, one-sided p-value and sort order, with ``"up"`` and ``"down"``
    matching its ``"only up"`` and ``"only down"`` entries.
    """
    given = {g: float(p) for g, p in gene_pvalues.items() if p is not None and not math.isnan(p)}
    invalid = [(g, p) for g, p in given.items() if not 0.0 <= p <= 1.0]
    if invalid:
        gene, value = invalid[0]
        msg = (
            f"The P-values must lie between 0 and 1, but {len(invalid)} of them do not. "
            f"The first is {value:g} (gene {gene})"
        )
        raise ValueError(msg)
    model_genes = {g.id for g in model.genes}
    scored = {g: p for g, p in given.items() if g in model_genes}
    gene_z = _gene_z(scored)
    results = [_reporter_one(model, gene_z, "all")]

    # RAVEN's any(geneFoldChanges) runs after genes outside the model or without a
    # p-value are dropped, and ignores NaN.
    if gene_fold_changes is not None and any(
        v != 0 and not math.isnan(v) for g, v in gene_fold_changes.items() if g in scored
    ):
        up = {g: z for g, z in gene_z.items() if gene_fold_changes.get(g, 0.0) >= 0}
        down = {g: z for g, z in gene_z.items() if gene_fold_changes.get(g, 0.0) < 0}
        results.append(_reporter_one(model, up, "up"))
        results.append(_reporter_one(model, down, "down"))
    return results
