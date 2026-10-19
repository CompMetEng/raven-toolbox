#!/usr/bin/env python3
"""Benchmark DeepLoc 2.1 predictions against a curated model's compartmentalisation.

Direct predictor accuracy (no MILP): for each curated single-compartment GPR reaction, the predicted
compartment is its highest-confidence gene's top DeepLoc organelle, mapped to the model's compartment
ids by a per-species config. Up to three views:

1. **As-is** — organelle call vs curation, per compartment. DeepLoc predicts 9 organelles and has no
   label for organelle *membranes* (e.g. yeast erm/mm/gm/vm) or the lipid particle (lp).
2. **Membrane-type discrimination** — for organelles split into lumen/membrane in the model, how well
   does DeepLoc's membrane-type signal separate them (ROC AUC), among reactions DeepLoc placed in the
   right organelle? Only emitted for species whose config defines membrane sub-compartments.
3. **Collapsed** — merge each organelle membrane into its lumen and re-score at organelle level — a
   fair target for a predictor with no membrane label. Only emitted when membranes are configured.

Species are configured in ``SPECIES`` (DeepLoc label → (lumen id, membrane id or None)). Add a model
by adding an entry. ASCII-only output (Windows console is cp1252).

Usage:
    python scripts/benchmark_deeploc.py --species yeast --model <path>/yeast-GEM.xml \
        --csv data/deeploc/yeast-GEM_deeploc_*.csv --doc /tmp/deeploc_yeast.md
    python scripts/benchmark_deeploc.py --species aracore --model <path>/AraCore_v2_0.xml \
        --csv data/deeploc/aracore/AraCore_deeploc_*.csv --doc /tmp/deeploc_aracore.md
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cobra
import numpy as np
import pandas as pd

#: Per-species map: DeepLoc organelle label -> (model lumen compartment id, membrane id or None).
#: Labels absent from a species' map are compartments that model does not have, so a gene predicted
#: there counts as a miss. ``comp_order`` is just the report row order.
SPECIES: dict[str, dict] = {
    "yeast": {
        "label": "yeast-GEM",
        "org": {"Cytoplasm": ("c", None), "Nucleus": ("n", None), "Extracellular": ("e", None),
                "Cell membrane": ("ce", None), "Peroxisome": ("p", None),
                "Mitochondrion": ("m", "mm"), "Endoplasmic reticulum": ("er", "erm"),
                "Golgi apparatus": ("g", "gm"), "Lysosome/Vacuole": ("v", "vm")},
        "comp_order": ["c", "m", "mm", "er", "erm", "p", "ce", "lp", "n", "g", "gm", "v", "vm", "e"],
    },
    "aracore": {
        "label": "AraCore",
        # Arabidopsis core-metabolism model. Chloroplast/plastid (h) is the organelle yeast-GEM could
        # not exercise. Thylakoid lumen (l) and mito intermembrane space (i) have no DeepLoc label,
        # but neither hosts a single-compartment GPR reaction in the truth set, so every truth
        # compartment (h/c/m/p) is DeepLoc-addressable -- a cleaner test than yeast.
        "org": {"Cytoplasm": ("c", None), "Mitochondrion": ("m", None),
                "Peroxisome": ("p", None), "Plastid": ("h", None)},
        "comp_order": ["c", "h", "m", "p", "l", "i"],
    },
    "icre1355": {
        "label": "iCre1355",
        # Chlamydomonas (green alga) genome-scale model. Plastid->chloroplast (h), peroxisome->the
        # algal glyoxysome (x). The model has no ER / lysosome-vacuole / plasma-membrane compartment,
        # so those DeepLoc labels map to nothing (count as misses). Thylakoid lumen (u), flagellum
        # (f), eyespot (s) and the inner-mito space (i) are out of DeepLoc's scope. NB the SBML keeps
        # GPRs in legacy notes, so cobra's ``model.genes`` has junk entries -- but ``reaction.genes``
        # still resolves to clean Cre ids (1368/1368 match DeepLoc), which is all the benchmark uses.
        "org": {"Cytoplasm": ("c", None), "Mitochondrion": ("m", None),
                "Plastid": ("h", None), "Peroxisome": ("x", None),
                "Nucleus": ("n", None), "Golgi apparatus": ("g", None),
                "Extracellular": ("e", None)},
        "comp_order": ["c", "h", "m", "x", "n", "g", "e", "u", "f", "s", "i"],
    },
}


def combine_deeploc(csvs: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(c) for c in csvs]).set_index("Protein_ID")


def build_truth(model: cobra.Model) -> tuple[dict[str, str], dict[str, list[str]]]:
    truth, rxn_genes = {}, {}
    for r in model.reactions:
        if r.boundary or not r.genes:
            continue
        comps = {m.compartment for m in r.metabolites if m.compartment}
        if len(comps) == 1:
            truth[r.id] = next(iter(comps))
            rxn_genes[r.id] = [g.id for g in r.genes]
    return truth, rxn_genes


def collapse_membranes(model: cobra.Model, membrane_to_lumen: dict[str, str]) -> cobra.Model:
    """Copy the model with each organelle membrane merged into its lumen (e.g. erm->er, mm->m)."""
    out = model.copy()
    for m in out.metabolites:
        if m.compartment in membrane_to_lumen:
            m.compartment = membrane_to_lumen[m.compartment]
    out.compartments = {c: n for c, n in out.compartments.items() if c not in membrane_to_lumen}
    return out


def auc(sig: list[float], lab: list[int]) -> float:
    s, y = np.asarray(sig), np.asarray(lab)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


class DeepLoc:
    """DeepLoc results: per-gene top organelle (mapped to model ids by ``org``) and membrane signal."""

    def __init__(self, df: pd.DataFrame, org: dict[str, tuple[str, str | None]]):
        self.df = df
        self.org = org
        self.org_cols = [c for c in org if c in df.columns]
        self.top_org = df[self.org_cols].astype(float).idxmax(axis=1)        # DeepLoc label per gene
        self.org_conf = df[self.org_cols].astype(float).max(axis=1)
        self.membrane = (1.0 - df["Soluble"].astype(float)).clip(0, 1)       # P(membrane)

    def covers(self, gid: str) -> bool:
        return gid in self.df.index

    def rxn_organelle(self, genes: list[str]) -> str | None:
        best, bestp = None, -1.0
        for gid in genes:
            if self.covers(gid) and self.org_conf[gid] > bestp:
                bestp, best = self.org_conf[gid], self.org[self.top_org[gid]][0]
        return best

    def rxn_membrane(self, genes: list[str]) -> float:
        return max((self.membrane[g] for g in genes if self.covers(g)), default=0.0)


def organelle_accuracy(truth, rxn_genes, dl: DeepLoc) -> dict[str, tuple[int, int]]:
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for rid, tc in truth.items():
        genes = [g for g in rxn_genes[rid] if dl.covers(g)]
        if not genes:
            continue
        per[tc][0] += 1
        per[tc][1] += (dl.rxn_organelle(genes) == tc)
    return {c: (n, ok) for c, (n, ok) in per.items()}


def fmt_acc(title: str, per: dict[str, tuple[int, int]], comp_order: list[str]) -> list[str]:
    lines = [f"### {title}", "", "| compartment | n | correct | accuracy |", "|---|--:|--:|--:|"]
    tn = tok = 0
    # known order first, then any compartments the config did not list
    order = comp_order + [c for c in per if c not in comp_order]
    for c in order:
        if c in per:
            n, ok = per[c]
            lines.append(f"| {c} | {n} | {ok} | {ok / n:.1%} |")
            tn += n
            tok += ok
    lines.append(f"| **all** | **{tn}** | **{tok}** | **{tok / tn:.1%}** |")
    lines.append("")
    return lines


def membrane_table(truth, rxn_genes, dl: DeepLoc, org: dict) -> list[str]:
    lines = ["### Membrane-type discrimination of lumen vs membrane", "",
             "AUC of DeepLoc's membrane signal (1 - P(Soluble)) separating curated lumen from "
             "membrane, among reactions DeepLoc placed in the correct organelle.", "",
             "| organelle | reactions | organelle recall | lumen / membrane | AUC |",
             "|---|--:|--:|--:|--:|"]
    for label, (lum, mem) in org.items():
        if mem is None:
            continue
        ids = [rid for rid, c in truth.items()
               if c in (lum, mem) and any(dl.covers(g) for g in rxn_genes[rid])]
        org_ok = [rid for rid in ids if dl.rxn_organelle(rxn_genes[rid]) == lum]
        lab = [1 if truth[rid] == mem else 0 for rid in org_ok]
        if ids:
            a = auc([dl.rxn_membrane(rxn_genes[rid]) for rid in org_ok], lab)
            lines.append(f"| {label} ({lum}/{mem}) | {len(ids)} | {len(org_ok) / len(ids):.0%} | "
                         f"{lab.count(0)} / {lab.count(1)} | {a:.3f} |")
    lines.append("")
    return lines


def confidence_calibration(truth, rxn_genes, dl: DeepLoc) -> list[str]:
    """Accuracy stratified by DeepLoc's own top-organelle confidence (calibration check)."""
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    rows: dict[tuple, list[int]] = {b: [0, 0] for b in bins}
    for rid, tc in truth.items():
        genes = [g for g in rxn_genes[rid] if dl.covers(g)]
        if not genes:
            continue
        conf = max(dl.org_conf[g] for g in genes)
        for b in bins:
            if b[0] <= conf < b[1]:
                rows[b][0] += 1
                rows[b][1] += (dl.rxn_organelle(genes) == tc)
                break
    out = ["### Accuracy by DeepLoc confidence", "",
           "Reaction's top-gene organelle confidence vs whether the call is correct.", "",
           "| confidence bin | n | correct | accuracy |", "|---|--:|--:|--:|"]
    for b in bins:
        n, ok = rows[b]
        if n:
            out.append(f"| [{b[0]:.1f}, {min(b[1], 1.0):.1f}] | {n} | {ok} | {ok / n:.1%} |")
    out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", required=True, choices=sorted(SPECIES))
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--csv", required=True, nargs="+", type=Path, help="DeepLoc result CSV(s)")
    ap.add_argument("--doc", type=Path, help="write a markdown study here")
    args = ap.parse_args()

    cfg = SPECIES[args.species]
    org = cfg["org"]
    membrane_to_lumen = {mem: lum for lum, mem in org.values() if mem}

    dl = DeepLoc(combine_deeploc(args.csv), org)
    model = cobra.io.read_sbml_model(str(args.model))
    truth, rxn_genes = build_truth(model)
    covered = {g for genes in rxn_genes.values() for g in genes if dl.covers(g)}
    addressable = {lum for lum, _ in org.values()}
    n_addr = sum(1 for c in truth.values() if c in addressable)

    out = [f"# DeepLoc 2.1 vs {cfg['label']} compartmentalisation", "",
           f"Model `{model.id or args.model.name}`, {len(dl.df)} DeepLoc predictions; "
           f"{len(truth)} single-compartment GPR reactions ({len(covered)} genes covered, "
           f"{n_addr} in DeepLoc-addressable compartments). DeepLoc 2.1 maps to "
           f"{sorted(addressable)} for this model.", ""]
    out += ["## 1. As-is (organelle call vs curation)", ""]
    out += fmt_acc("Per-compartment accuracy", organelle_accuracy(truth, rxn_genes, dl),
                   cfg["comp_order"])
    out += confidence_calibration(truth, rxn_genes, dl)

    if membrane_to_lumen:
        out += ["## 2. Can membrane-type recover the lumen/membrane split?", ""]
        out += membrane_table(truth, rxn_genes, dl, org)
        cmodel = collapse_membranes(model, membrane_to_lumen)
        ctruth, crxn_genes = build_truth(cmodel)
        collapse_desc = ", ".join(f"{mem}->{lum}" for mem, lum in membrane_to_lumen.items())
        out += [f"## 3. Collapsed membranes ({collapse_desc})",
                "", "Collapsing also turns lumen/membrane-spanning reactions (multi-compartment, "
                "so excluded, on the uncollapsed model) into single-compartment ones, so the "
                "truth set grows.", ""]
        out += fmt_acc("Per-compartment accuracy on the organelle-collapsed model",
                       organelle_accuracy(ctruth, crxn_genes, dl), cfg["comp_order"])

    text = "\n".join(out) + "\n"
    print(text)
    if args.doc:
        args.doc.write_text(text, encoding="utf-8", newline="")
        print(f"wrote {args.doc}")


if __name__ == "__main__":
    main()
