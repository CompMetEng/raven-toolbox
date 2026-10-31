"""Curation triage for a localisation assignment: the calls that need review by a curator.

This is an optional companion output of the compartment-assignment step. Given the assignment
proposal and the scores it used, it ranks the genes and reactions with the least certain
localisation, each with a reason in text, so that a curator can review those first.

The triage needs no solver and reads only the proposal and the scores. Each candidate gets an
``uncertainty`` in ``[0, 1]`` blended from the signals that apply, and the list is ranked by it. The
signals that fire differ per gene:

* **low confidence**: the raw top probability of the predictor is low (in the yeast-GEM
  benchmark, the share of calls that other evidence corroborates rises with the raw
  probability). This signal carries the largest default weight (0.35) of the signals in the
  blended ``uncertainty``. Needs ``scores.raw_confidence``
  (``load_deeploc(keep_raw_confidence=True)``) or an explicit ``raw_confidence=``, because
  normalisation hides the raw probability.
* **borderline**: the top two compartments are within ``margin_threshold``, so the placement can
  flip.
* **diffuse**: the score mass is spread across many compartments.
* **source conflict**: the supplied per-source tables (DeepLoc, UniProt, COMPARTMENTS) disagree.
* **no evidence**: the gene has no predictor score and is placed by function only.
* **weak compartment**: the assigned compartment is one that the predictor calls with low accuracy
  (see :data:`DEEPLOC_COMPARTMENT_TRUST`).
* **multi-localised**: the gene is placed in several compartments.

The triage has no impact signal: it does not use the essentiality of a gene or the flux of a
reaction.

See the `DeepLoc yeast-GEM benchmark
<https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/localization/deeploc_yeast_benchmark.md>`_
on raven-gecko-parity for the source of the thresholds.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import cobra
import pandas as pd

from raven_toolbox.localization.scores import LocalizationScores

__all__ = ["ReviewReport", "triage_localization", "DEEPLOC_COMPARTMENT_TRUST", "confidence_bin"]

#: Per-compartment reliability of a DeepLoc 2.1 organelle call, finetuned on the slow (ProtT5)
#: yeast-GEM run (organelle-collapsed accuracy; `localisation finetuning
#: <https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/localization/localization_finetuning.md>`_ on raven-gecko-parity,
#: regenerate with ``scripts/finetune_localization_yeast.py``). ``mm`` inherits ``m`` because the
#: mitochondrial split is the one validated routing (AUC ~0.93); the other organelle membranes and
#: ``ce``/``lp`` stay 0 — DeepLoc cannot reach them reliably. Yeast/DeepLoc-specific — override via
#: ``compartment_trust=`` for other models/predictors. Compartments absent here are trusted (1.0).
DEEPLOC_COMPARTMENT_TRUST: dict[str, float] = {
    "er": 0.88, "m": 0.86, "mm": 0.86, "p": 0.83, "c": 0.79, "e": 0.78,
    "v": 0.36, "n": 0.18, "ce": 0.11, "g": 0.01,
    "lp": 0.0, "erm": 0.0, "gm": 0.0, "vm": 0.0,
}

_DEFAULT_WEIGHTS = {"confidence": 0.35, "disagreement": 0.25, "margin": 0.20,
                    "weak_compartment": 0.10, "entropy": 0.10}


def confidence_bin(p: float) -> str:
    """Calibration bin of a raw confidence, with the bins of the benchmark's corroboration rates."""
    for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9)):
        if lo <= p < hi:
            return f"[{lo:.1f},{hi:.1f})"
    return "[0.9,1]"


@dataclass
class ReviewReport:
    """Ranked curation suggestions from :func:`triage_localization` (one row per flagged item)."""

    items: pd.DataFrame                       # columns: level,id,uncertainty,signals,reasons,...
    signals_used: list[str] = field(default_factory=list)

    def top(self, n: int = 15) -> pd.DataFrame:
        """Return the ``n`` items with the highest uncertainty (``items`` is sorted by it)."""
        return self.items.head(n)

    def __str__(self) -> str:
        if self.items.empty:
            return "ReviewReport: nothing flagged."
        n_flag = int((self.items["signals"].str.len() > 0).sum())
        lines = [f"ReviewReport: {len(self.items)} items, {n_flag} with at least one flag. Top:"]
        for _, r in self.top(10).iterrows():
            lines.append(f"  {r['uncertainty']:.2f}  {r['level']:<8} {r['id']:<16} "
                         f"[{', '.join(r['signals'])}]")
        return "\n".join(lines)


def _entropy(row: pd.Series) -> float:
    vals = row[row > 0].to_numpy(dtype=float)
    if len(vals) < 2:
        return 0.0
    p = vals / vals.sum()
    return float(-(p * [math.log(x) for x in p]).sum() / math.log(len(vals)))


def triage_localization(
    proposal: object,
    scores: LocalizationScores,
    *,
    model: cobra.Model | None = None,
    sources: Mapping[str, LocalizationScores] | None = None,
    raw_confidence: pd.Series | None = None,
    compartment_trust: Mapping[str, float] | None = None,
    confidence_threshold: float = 0.7,
    margin_threshold: float = 0.15,
    entropy_threshold: float = 0.7,
    weights: Mapping[str, float] | None = None,
    top_n: int | None = None,
) -> ReviewReport:
    """Rank the localisation calls in ``proposal`` for manual curation, without a solver.

    ``proposal`` is a localisation or assignment proposal with the attributes ``gene_compartments``
    (gene -> [compartments]) and ``unplaced_reactions``; both :class:`LocalizationProposal` and
    :class:`raven_toolbox.localization.assign.AssignmentProposal` provide them. ``scores`` is the
    table that the proposal is based on.

    ``raw_confidence`` (a gene -> top-probability Series) enables the low-confidence signal. If it
    is omitted, it is read from ``scores.raw_confidence`` (set by
    ``load_deeploc(keep_raw_confidence=True)``), and if that is also absent, the confidence signal
    is skipped. ``sources`` (the per-source :class:`LocalizationScores` before fusion, for example
    ``{"DeepLoc": ..., "UniProt": ...}``) enables the source-conflict signal, which is not available
    after :func:`combine_scores`. ``model`` (a cobra model) adds a ``reactions`` column that lists
    the reactions each gene gates. ``compartment_trust`` overrides
    :data:`DEEPLOC_COMPARTMENT_TRUST`.

    ``confidence_threshold``, ``margin_threshold`` and ``entropy_threshold`` set the cutoffs that
    trigger the low-confidence, borderline and diffuse signals respectively (see the module
    docstring for each signal). ``weights`` sets the weight of each signal in the blended
    ``uncertainty`` score. ``top_n`` limits the number of returned rows. Returns a
    :class:`ReviewReport` ranked by ``uncertainty``.
    """
    df = scores.df
    trust = dict(DEEPLOC_COMPARTMENT_TRUST if compartment_trust is None else compartment_trust)
    w = dict(_DEFAULT_WEIGHTS if weights is None else weights)
    if raw_confidence is None:
        raw_confidence = scores.raw_confidence
    gene_comps: dict[str, list[str]] = dict(getattr(proposal, "gene_compartments", {}) or {})
    unplaced = list(getattr(proposal, "unplaced_reactions", []) or [])

    gene_to_rxns: dict[str, list[str]] = {}
    if model is not None:
        for r in model.reactions:
            for g in (gg.id for gg in r.genes):
                gene_to_rxns.setdefault(g, []).append(r.id)

    src_top: dict[str, dict[str, str]] = {}
    if sources:
        for name, s in sources.items():
            src_top[name] = {g: s.df.loc[g].idxmax() for g in s.df.index if s.df.loc[g].max() > 0}

    used: set[str] = set()
    rows: list[dict] = []

    for gene, assigned in gene_comps.items():
        parts: dict[str, float] = {}
        signals: list[str] = []
        reasons: list[str] = []
        has_row = gene in df.index and float(df.loc[gene].max()) > 0
        if not has_row:
            rows.append(_row("gene", gene, 1.0, ["no_evidence"],
                             ["no predictor score for this gene; placed by function only"],
                             assigned, gene_to_rxns.get(gene), None))
            used.add("no_evidence")
            continue
        row = df.loc[gene]

        rc = None if raw_confidence is None or gene not in raw_confidence.index else float(raw_confidence[gene])
        if rc is not None:
            parts["confidence"] = 1.0 - rc
            if rc < confidence_threshold:
                signals.append("low_confidence")
                reasons.append(f"raw confidence {rc:.2f} < {confidence_threshold} "
                               f"(~corroboration of bin {confidence_bin(rc)})")

        ordered = row.sort_values(ascending=False)
        if (ordered > 0).sum() >= 2:
            margin = float(ordered.iloc[0] - ordered.iloc[1])
            parts["margin"] = 1.0 - margin
            if margin < margin_threshold:
                signals.append("borderline")
                reasons.append(f"top two within {margin:.2f}: {ordered.index[0]} vs {ordered.index[1]}")
            ent = _entropy(row)
            parts["entropy"] = ent
            if ent > entropy_threshold:
                signals.append("diffuse")
                reasons.append(f"score spread across compartments (entropy {ent:.2f})")

        if src_top:
            tops = {n: m[gene] for n, m in src_top.items() if gene in m}
            if len(set(tops.values())) > 1:
                parts["disagreement"] = (len(set(tops.values())) - 1) / max(1, len(src_top) - 1)
                signals.append("source_conflict")
                reasons.append("sources disagree: "
                               + ", ".join(f"{n}->{c}" for n, c in tops.items()))

        if assigned:
            min_trust = min(trust.get(c, 1.0) for c in assigned)
            parts["weak_compartment"] = 1.0 - min_trust
            if min_trust < 0.5:
                weak = min(assigned, key=lambda c: trust.get(c, 1.0))
                signals.append("weak_compartment")
                reasons.append(f"assigned to '{weak}' (predictor accuracy {min_trust:.0%} there)")
        if len(assigned) > 1:
            signals.append("multi_localized")
            reasons.append(f"placed in {len(assigned)} compartments: {assigned}")

        unc = _blend(parts, w)
        rows.append(_row("gene", gene, unc, signals, reasons, assigned,
                         gene_to_rxns.get(gene), rc))
        used.update(signals)

    for rid in unplaced:
        rows.append(_row("reaction", rid, 1.0, ["no_evidence"],
                         ["no scored gene; placed by function only"], None, [rid], None))
        used.add("no_evidence")

    items = pd.DataFrame(rows)
    if not items.empty:
        items = items.sort_values("uncertainty", ascending=False, kind="mergesort").reset_index(drop=True)
        if top_n is not None:
            items = items.head(top_n)
    return ReviewReport(items=items, signals_used=sorted(used))


def _blend(parts: dict[str, float], weights: dict[str, float]) -> float:
    if not parts:
        return 0.0
    num = sum(weights.get(k, 0.1) * v for k, v in parts.items())
    den = sum(weights.get(k, 0.1) for k in parts)
    return round(num / den, 4) if den else 0.0


def _row(level, id_, unc, signals, reasons, comps, rxns, rc) -> dict:
    return {"level": level, "id": id_, "uncertainty": round(float(unc), 4),
            "signals": list(signals), "reasons": list(reasons),
            "compartment": (",".join(comps) if comps else None),
            "reactions": (list(rxns) if rxns else []),
            "raw_confidence": rc}
