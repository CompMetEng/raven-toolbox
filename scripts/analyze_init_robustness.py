#!/usr/bin/env python3
"""Robustness of ftINIT to degraded transcriptomics input.

The metabolic-task layer is *always part of the pipeline* — it is what makes the output a
functional model. The experimental variable here is therefore the **transcriptomics
input**, not whether tasks are used. This script holds the task + gap-fill layer fixed and
asks: as the expression data gets noisier or sparser, (a) does the model stay functional,
and (b) how much does the *reaction content* drift from what clean data would give — and
which parameters keep it stable?

Metrics, per run (tasks always on):

* ``frac``    — fraction of essential metabolic tasks the model performs (``check_tasks``).
                The task+gap-fill layer should hold this at 1.0; a drop is a real failure.
* ``Jaccard`` — reaction-set overlap with the **clean-data** model. This is the real cost
                of bad input: even when all tasks still pass, degraded data changes *which*
                reactions are kept. The primary robustness signal.
* ``n_rxns``  — model size (does degraded data bloat or shrink it).

Three independent degradations of the gene-expression vector (severity = higher is worse):

* ``dropout``    — set a random fraction of genes to 0 (→ gene score -5, a strong *remove*
                   signal). Simulates shallow sequencing / single-cell dropout.
* ``noise``      — multiply each level by ``exp(N(0, sigma))`` (sigma = severity).
* ``downsample`` — drop a random fraction of genes entirely (→ ``no_gene_score``).

Two phases:

* **gradient** — task pipeline across degradation levels; shows functional integrity and
  reaction-set drift vs the clean-data model.
* **levers**   — at a fixed severe degradation, vary the robustness parameters
  (``no_gene_score``, ``force_on``) to see which keeps the model closest to the
  clean-data result / most functional.

Resumable; reuses the cached Human-GEM task prep (``rg_prep_tasks.pkl``). Loose MIP
gap for speed (functionality + set overlap, not the exact optimum, are the metrics).

Usage
-----
    python scripts/analyze_init_robustness.py --cell HCT116
"""
from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import cobra
import numpy as np

from raven_toolbox.init import ftinit, gene_scores_from_expression, score_reactions_from_genes
from raven_toolbox.tasks import check_tasks, parse_task_list

# Degradation grid (severity per kind). A mild and a severe point per kind.
GRADIENT = {
    "dropout": (0.5, 0.7),    # moderate + severe-but-realistic (single-cell dropout ~50-70%);
    "noise": (1.0, 2.0),      # 90%+ dropout breaks ~all tasks so gap-fill rebuilds the model
    "downsample": (0.5, 0.7),  # (a per-task MILP each) — pathologically slow and unrealistic.
}
LEVER_KIND, LEVER_LEVEL = "dropout", 0.7      # severe-but-tractable point for the levers
NO_GENE_SCORES = (-1.0, -0.5)                 # vs the default -2 (the gradient row)
FORCE_ONS = (0.2,)                            # vs the default 0.1

# Loose solver tolerances (speed; functionality + set overlap, not the exact optimum).
MIP_GAP, TIME_LIMIT = 0.02, 120.0


@dataclass
class Result:
    label: str
    seconds: float
    status: str
    n_rxns: int
    n_pass: int
    n_tasks: int
    frac_pass: float
    reactions: list[str] = field(default_factory=list)
    jaccard_clean: float | None = None


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def degrade(expr: dict[str, float], kind: str, level: float, seed: int) -> dict[str, float]:
    """Return a degraded copy of the expression dict (severity ``level``)."""
    if level <= 0:
        return dict(expr)
    rng = np.random.default_rng(seed)
    genes = list(expr)
    if kind == "dropout":
        out = dict(expr)
        for g in rng.choice(genes, size=int(level * len(genes)), replace=False):
            out[g] = 0.0
        return out
    if kind == "noise":
        return {g: max(v * float(np.exp(rng.normal(0.0, level))), 0.0) for g, v in expr.items()}
    if kind == "downsample":
        keep = set(rng.choice(genes, size=int((1 - level) * len(genes)), replace=False))
        return {g: v for g, v in expr.items() if g in keep}
    raise ValueError(f"unknown degradation kind {kind!r}")


def functionality(model: cobra.Model, tasks) -> tuple[int, int]:
    """(passed, total) essential tasks the extracted model can perform."""
    results = check_tasks(model, tasks)
    return sum(t.passed for t in results), len(results)


def _measure(label, builder, tasks, clean_set=None) -> Result:
    t = time.time()
    try:
        model = builder()
        n_pass, n_tasks = functionality(model, tasks)
        rset = sorted(x.id for x in model.reactions)
        r = Result(label, time.time() - t, "ok", len(rset), n_pass, n_tasks,
                   n_pass / n_tasks if n_tasks else 0.0, rset)
        if clean_set is not None:
            r.jaccard_clean = _jaccard(set(rset), clean_set)
    except Exception as ex:  # noqa: BLE001  (infeasible/failed build is itself a finding)
        msg = str(ex)[:80].replace("\n", " ") or type(ex).__name__
        print(f"  FAIL {label}: {type(ex).__name__}: {ex}", flush=True)
        r = Result(label, time.time() - t, f"FAIL:{msg}", 0, 0, len(tasks), 0.0)
    return r


def _table(title, results, note="") -> list[str]:
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines.append("| config | time (s) | status | n_rxns | tasks passed | frac | Jaccard vs clean |")
    lines.append("|---|--:|---|--:|--:|--:|--:|")
    for r in results:
        jac = f"{r.jaccard_clean:.3f}" if r.jaccard_clean is not None else "ref"
        lines.append(f"| {r.label} | {r.seconds:.0f} | {r.status} | {r.n_rxns} | "
                     f"{r.n_pass}/{r.n_tasks} | {r.frac_pass:.3f} | {jac} |")
    lines.append("")
    for ln in lines:
        print(ln)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=Path.home() / "hgem_compare")
    ap.add_argument("--human-gem", type=Path, default=Path.home() / "github" / "Human-GEM")
    ap.add_argument("--cell", default="HCT116")
    ap.add_argument("--phase", default="gradient,levers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--doc", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or args.work / f"init_robustness_{args.cell}.pkl"
    store: dict = pickle.load(open(out, "rb")) if out.exists() else {}

    def save():
        tmp = Path(f"{out}.part")
        pickle.dump(store, open(tmp, "wb"))
        tmp.replace(out)

    def cached(key, fn):
        if key not in store:
            print(f"[ftinit] {key[1]} ...", flush=True)
            store[key] = fn()
            save()
        return store[key]

    t0 = time.time()
    ref = cobra.io.read_sbml_model(str(args.work / "raven_refModel.xml"))
    ref.solver = cobra.Configuration().solver
    expr: dict[str, float] = {}
    with open(args.human_gem / "data" / "datasets" / "Hart2015_RNAseq.txt") as f:
        h = f.readline().rstrip("\n").split("\t")
        c = h.index(args.cell)
        for line in f:
            p = line.rstrip("\n").split("\t")
            expr[p[0]] = float(p[c])
    tasks = parse_task_list(str(args.human_gem / "data" / "metabolicTasks" /
                                "metabolicTasks_Essential.txt"))
    prep = pickle.load(open(args.work / "rg_prep_tasks.pkl", "rb"))  # ftINIT uses task layer
    print(f"[{time.time()-t0:.0f}s] ref {len(ref.reactions)} rxns, {len(tasks)} tasks, "
          f"cell={args.cell} (task layer always on)", flush=True)

    def model_for(e, **kw):
        g = gene_scores_from_expression(e, 1.0)
        r = score_reactions_from_genes(ref, g, no_gene_score=kw.get("no_gene_score", -2.0))
        return ftinit(prep, rxn_scores=r, gene_scores=g, series="1+1",
                      force_on=kw.get("force_on", 0.1), mip_gap=MIP_GAP, time_limit=TIME_LIMIT)

    phases = set(args.phase.split(","))
    doc = [f"# ftINIT robustness to degraded transcriptomics — Human-GEM / {args.cell}",
           "", "Task + gap-fill layer is always on (it is part of the pipeline); the variable is the "
           "expression input. Functional = fraction of essential tasks performed (check_tasks); "
           "Jaccard is reaction-set overlap with the clean-data model. Generated by "
           "`scripts/analyze_init_robustness.py`.", ""]

    clean = cached(("clean", "clean"), lambda: _measure("clean", lambda: model_for(expr), tasks))
    clean_set = set(clean.reactions)
    clean.jaccard_clean = None  # it is the reference
    doc += _table("Clean-data baseline", [clean])

    if "gradient" in phases:
        for kind, levels in GRADIENT.items():
            rows = [clean]
            for lvl in levels:
                e = degrade(expr, kind, lvl, args.seed)
                rows.append(cached((f"grad_{kind}", f"{kind}={lvl}"), lambda e=e, lvl=lvl, kind=kind:
                            _measure(f"{kind}={lvl}", lambda: model_for(e), tasks, clean_set)))
            doc += _table(f"Gradient: {kind} (task pipeline always on)", rows,
                          "Higher severity = noisier/sparser input. frac should stay ~1.0 (the task "
                          "layer's job); the Jaccard drop is how much degraded data changes the model.")

    if "levers" in phases:
        e = degrade(expr, LEVER_KIND, LEVER_LEVEL, args.seed)
        tag = f"{LEVER_KIND}={LEVER_LEVEL}"
        rows = []
        for ngs in NO_GENE_SCORES:
            rows.append(cached(("lever", f"no_gene_score={ngs}"), lambda ngs=ngs:
                        _measure(f"no_gene_score={ngs}", lambda: model_for(e, no_gene_score=ngs),
                                 tasks, clean_set)))
        for fo in FORCE_ONS:
            rows.append(cached(("lever", f"force_on={fo}"), lambda fo=fo:
                        _measure(f"force_on={fo}", lambda: model_for(e, force_on=fo),
                                 tasks, clean_set)))
        doc += _table(f"Levers at {tag}: which parameter keeps the model closest to clean?", rows,
                      "Compare against the default-parameter row for this severity in the gradient "
                      "table above (no_gene_score=-2, force_on=0.1).")

    if args.doc:
        args.doc.write_text("\n".join(doc) + "\n")
        print(f"\nwrote {args.doc}", flush=True)


if __name__ == "__main__":
    main()
