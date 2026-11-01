#!/usr/bin/env python3
"""Head-to-head: RAVEN `predictLocalization` (stochastic SA) vs the deterministic MILP, same inputs.

Both methods take the *same* yeast-GEM and the *same* normalised DeepLoc gene-localization scores and
assign genes to compartments. We compare at the **gene level** (the native output of
`predictLocalization`): a gene is correct if its assigned compartment is in the gene's curated
compartment set (the collapsed compartments of its reactions in yeast-GEM).

Three arms:
* **argmax** — each gene to its top DeepLoc compartment (no network awareness). The naive floor.
* **MILP** — `raven_toolbox.localization.predict_localization` on the flattened model; deterministic.
* **predictLocalization** — RAVEN's simulated-annealing assigner, run N times (stochastic, no seed,
  wall-clock budget) to characterise its run-to-run variance. Run separately in MATLAB via
  `scripts/run_predictlocalization.m`; this script writes its inputs (`--prep`) and scores its
  outputs (`--score`).

Why gene-level: `predictLocalization` emits `geneLocalization` (gene->one compartment); scoring genes
(not reactions) is the common denominator across all three arms and avoids re-deriving reaction
placement from each method's idiosyncratic output. ASCII-only output.

Usage:
    python scripts/compare_predictlocalization.py --prep   --model <yeast-GEM.xml> --out .research_tmp/pl
    # ... run scripts/run_predictlocalization.m in MATLAB (writes geneloc_run_*.csv to that dir) ...
    python scripts/compare_predictlocalization.py --score  --model <yeast-GEM.xml> --out .research_tmp/pl \
        --doc ../raven-gecko-parity/docs/localization/predictlocalization_comparison.md
"""
from __future__ import annotations

import argparse
import glob
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cobra  # noqa: E402
import pandas as pd  # noqa: E402

from raven_toolbox.localization import (  # noqa: E402
    DEFAULT_COMPARTMENT_MAP,
    LocalizationScores,
    load_deeploc,
    predict_localization,
)
from raven_toolbox.manipulation.compartments import merge_compartments  # noqa: E402

CSVS = [f"data/deeploc/yeast-GEM_deeploc_{i:03d}.csv" for i in (1, 2, 3)]
COLLAPSE = {"erm": "er", "mm": "m", "gm": "g", "vm": "v"}  # organelle membrane -> lumen
ADDRESSABLE = {"c", "ce", "e", "er", "g", "m", "n", "p", "v"}  # DeepLoc-reachable


def load_scores() -> LocalizationScores:
    parts = [load_deeploc(c, compartment_map=DEFAULT_COMPARTMENT_MAP).df for c in CSVS]
    df = pd.concat(parts).fillna(0.0)
    df.index.name = "gene_id"
    return LocalizationScores(df)


def gene_truth(model: cobra.Model) -> dict[str, set[str]]:
    """Each gene -> the set of (collapsed) compartments of its non-boundary reactions."""
    out: dict[str, set[str]] = {}
    for g in model.genes:
        comps: set[str] = set()
        for r in g.reactions:
            if r.boundary:
                continue
            for m in r.metabolites:
                if m.compartment:
                    comps.add(COLLAPSE.get(m.compartment, m.compartment))
        if comps:
            out[g.id] = comps
    return out


def score_gene_assignment(assign: dict[str, str], truth: dict[str, set[str]]) -> dict:
    """assign: gene -> single compartment. Correct if in the gene's curated set."""
    ids = [g for g in assign if g in truth]
    ok = sum(assign[g] in truth[g] for g in ids)
    addr = [g for g in ids if assign[g] in ADDRESSABLE]
    ok_addr = sum(assign[g] in truth[g] for g in addr)
    return {"n": len(ids), "acc": ok / max(1, len(ids)),
            "n_addr": len(addr), "acc_addr": ok_addr / max(1, len(addr))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("C:/Work/GitHub/yeast-GEM/model/yeast-GEM.xml"))
    ap.add_argument("--out", type=Path, default=Path(".research_tmp/pl"))
    ap.add_argument("--prep", action="store_true", help="write GSS + gene-truth CSVs for MATLAB")
    ap.add_argument("--score", action="store_true", help="score MATLAB outputs + MILP + argmax")
    ap.add_argument("--transport-cost", type=float, default=0.01)
    ap.add_argument("--mip-gap", type=float, default=0.01)
    ap.add_argument("--doc", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scores = load_scores()
    model = cobra.io.read_sbml_model(str(args.model))
    truth = gene_truth(model)

    if args.prep:
        # GSS table for MATLAB: gene_id + one column per DeepLoc-addressable compartment.
        gss = scores.df.reindex(columns=sorted(ADDRESSABLE)).fillna(0.0)
        gss.to_csv(args.out / "deeploc_gss_yeast.csv")
        pd.Series({g: ";".join(sorted(s)) for g, s in truth.items()}).to_csv(
            args.out / "gene_truth_yeast.csv", header=["compartments"])
        print(f"wrote {args.out/'deeploc_gss_yeast.csv'} ({gss.shape[0]} genes x {gss.shape[1]} comps)")
        print(f"wrote {args.out/'gene_truth_yeast.csv'} ({len(truth)} genes)")
        print("Next: run scripts/run_predictlocalization.m in MATLAB, then --score.")
        return

    if not args.score:
        ap.error("pass --prep or --score")

    # ---- argmax arm ----
    df = scores.df
    argmax = {g: df.loc[g].idxmax() for g in df.index if df.loc[g].max() > 0}
    a_arg = score_gene_assignment(argmax, truth)

    # ---- MILP arm (deterministic) ----
    flat, _, _ = merge_compartments(model, merged_id="c", merged_name="c",
                                    drop_single_metabolite_reactions=False,
                                    deduplicate_reactions=False)
    surviving = {r.id for r in flat.reactions}
    rel = [r.id for r in model.reactions
           if not r.boundary and r.genes and r.id in surviving
           and len({m.compartment for m in r.metabolites if m.compartment}) == 1]
    t0 = time.time()
    prop = predict_localization(flat, scores, rel, default_compartment="c",
                                transport_cost=args.transport_cost, multi_compartment_penalty=0.5,
                                apply=False, mip_gap=args.mip_gap, time_limit=600)
    milp_secs = time.time() - t0
    # gene -> its highest-DeepLoc-score assigned compartment (genes may be multi-placed)
    milp = {}
    for g, comps in prop.gene_compartments.items():
        if comps:
            milp[g] = max(comps, key=lambda c: float(df.loc[g, c]) if (g in df.index and c in df.columns) else 0.0)
    a_milp = score_gene_assignment(milp, truth)

    # ---- predictLocalization arm (read MATLAB run outputs) ----
    runs = sorted(glob.glob(str(args.out / "geneloc_run_*.csv")))
    pl_rows = []
    pl_meta = pd.read_csv(args.out / "pl_runs_meta.csv") if (args.out / "pl_runs_meta.csv").exists() else None
    per_gene_runs: dict[str, list[str]] = {}
    for rp in runs:
        gl = pd.read_csv(rp)
        assign = dict(zip(gl["gene"], gl["comp"], strict=True))
        s = score_gene_assignment(assign, truth)
        pl_rows.append(s)
        for g, c in assign.items():
            per_gene_runs.setdefault(g, []).append(c)
    pl = pd.DataFrame(pl_rows)

    # run-to-run instability: fraction of genes assigned >1 distinct compartment across runs
    multi = sum(1 for v in per_gene_runs.values() if len(set(v)) > 1)
    instability = multi / max(1, len(per_gene_runs))

    # strict common gene set: genes scored by every arm (argmax, MILP, every PL run) and in truth
    pl_common = set.intersection(*[{g for g in pd.read_csv(rp)["gene"]} for rp in runs]) if runs else set()
    common = set(truth) & set(argmax) & set(milp) & (pl_common if runs else set(truth))
    def _acc_on(assign, gene_set):
        ids = [g for g in gene_set if g in assign]
        return (sum(assign[g] in truth[g] for g in ids) / len(ids)) if ids else float("nan")
    c_arg = _acc_on(argmax, common)
    c_milp = _acc_on(milp, common)
    c_pl = [_acc_on(dict(zip(pd.read_csv(rp)["gene"], pd.read_csv(rp)["comp"], strict=True)), common)
            for rp in runs]

    # ---- report ----
    L = ["# RAVEN predictLocalization vs the deterministic MILP (yeast-GEM)", "",
         "Same model, same normalised DeepLoc gene scores, same default compartment (`c`); gene-level "
         "agreement with curated yeast-GEM (a gene is correct if its assigned compartment is in the "
         "collapsed compartment set of its reactions). `acc` is over all scored genes; `acc_addr` "
         "restricts to genes assigned to a DeepLoc-addressable compartment.", "",
         f"* Model `{model.id or args.model.name}`; {len(truth)} genes with a curated compartment.",
         f"* MILP: `predict_localization`, transport_cost={args.transport_cost}, mip_gap={args.mip_gap}, "
         f"deterministic, solved in {milp_secs:.0f}s.",
         f"* predictLocalization: RAVEN simulated annealing, {len(runs)} independent runs"
         + (f" (maxTime={pl_meta['maxTime_min'].iloc[0]} min each)" if pl_meta is not None else "")
         + ".", "",
         "## Gene-level agreement", "",
         "| method | n | accuracy | addressable acc |", "|---|--:|--:|--:|",
         f"| argmax (DeepLoc top, no network) | {a_arg['n']} | {a_arg['acc']:.1%} | {a_arg['acc_addr']:.1%} |",
         f"| **MILP (deterministic)** | {a_milp['n']} | **{a_milp['acc']:.1%}** | **{a_milp['acc_addr']:.1%}** |"]
    if not pl.empty:
        L.append(f"| predictLocalization (SA, mean of {len(runs)}) | {int(pl['n'].mean())} | "
                 f"{pl['acc'].mean():.1%} | {pl['acc_addr'].mean():.1%} |")
        L += ["", "## predictLocalization run-to-run variability (the determinism gap)", "",
              f"* accuracy across {len(runs)} runs: min {pl['acc'].min():.1%}, max {pl['acc'].max():.1%}, "
              f"mean {pl['acc'].mean():.1%}, spread {(pl['acc'].max()-pl['acc'].min())*100:.1f} pp.",
              f"* **{instability:.1%}** of genes were assigned to *different* compartments across runs "
              f"({multi}/{len(per_gene_runs)}) -- the MILP returns one reproducible answer.",
              "", "| run | accuracy | addressable acc |", "|---|--:|--:|"]
        for i, r in pl.iterrows():
            L.append(f"| {i+1} | {r['acc']:.1%} | {r['acc_addr']:.1%} |")
    else:
        L += ["", "_No predictLocalization runs found (run scripts/run_predictlocalization.m first)._"]
    if runs and common:
        import statistics
        L += ["", "## Strict comparison on the common gene set", "",
              f"All arms restricted to the {len(common)} genes scored by *every* method and present "
              "in the curated truth (removes the differing-coverage confound).", "",
              "| method | accuracy on common set |", "|---|--:|",
              f"| argmax | {c_arg:.1%} |",
              f"| **MILP (deterministic)** | **{c_milp:.1%}** |",
              f"| predictLocalization (mean of {len(c_pl)}) | {statistics.fmean(c_pl):.1%} "
              f"(min {min(c_pl):.1%}, max {max(c_pl):.1%}) |"]
    L.append("")

    text = "\n".join(L) + "\n"
    print(text)
    if args.doc:
        args.doc.write_text(text, encoding="utf-8", newline="")
        print(f"wrote {args.doc}")


if __name__ == "__main__":
    main()
