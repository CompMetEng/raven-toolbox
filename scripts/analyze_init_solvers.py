#!/usr/bin/env python3
"""Cross-solver benchmark for ftINIT on a genome-scale model.

The clean-data calibration and robustness studies tuned (and ran) on Gurobi. The CI
``tests/test_init_solvers.py`` checks correctness on toy models for every installed MILP
solver; this script measures **genome-scale tractability and reaction-set agreement** —
does the same ftINIT pipeline that works in seconds on Gurobi also complete on HiGHS or
GLPK, in what time, and producing the same model?

For each installed MILP-capable optlang interface (Gurobi, ``hybrid`` for HiGHS, GLPK) it
runs the *same* ftINIT call (cached Human-GEM no-task prep + HCT116 scores) with the same
``mip_gap``/``time_limit``, records (status, wall time, reaction set), and computes the
pairwise Jaccard of the resulting reaction sets. Solvers that fail (the optlang
``hybrid_interface`` ``clone`` bug, or GLPK timing out at genome scale) are recorded as
such — that *is* the cross-solver picture.

Usage
-----
    python scripts/analyze_init_solvers.py --cell HCT116 --time-limit 900 \
        --doc docs/init_solver_benchmark.md
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import cobra

from raven_toolbox.init import ftinit, gene_scores_from_expression, score_reactions_from_genes

_INTERFACES = {"gurobi": "gurobi_interface", "hybrid": "hybrid_interface", "glpk": "glpk_interface"}


def _available_solvers() -> list[str]:
    return [name for name, mod in _INTERFACES.items()
            if importlib.util.find_spec(f"optlang.{mod}") is not None]


@dataclass
class Result:
    solver: str
    seconds: float
    status: str
    n_rxns: int
    reactions: list[str] = field(default_factory=list)


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=Path.home() / "hgem_compare")
    ap.add_argument("--human-gem", type=Path, default=Path.home() / "github" / "Human-GEM")
    ap.add_argument("--cell", default="HCT116")
    ap.add_argument("--mip-gap", type=float, default=0.001)
    ap.add_argument("--time-limit", type=float, default=900.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--doc", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or args.work / f"init_solver_bench_{args.cell}.pkl"
    store: dict = pickle.load(open(out, "rb")) if out.exists() else {}

    def save():
        tmp = Path(f"{out}.part")
        pickle.dump(store, open(tmp, "wb"))
        tmp.replace(out)

    expr: dict[str, float] = {}
    with open(args.human_gem / "data" / "datasets" / "Hart2015_RNAseq.txt") as f:
        h = f.readline().rstrip("\n").split("\t")
        c = h.index(args.cell)
        for line in f:
            p = line.rstrip("\n").split("\t")
            expr[p[0]] = float(p[c])
    if not (args.work / "rg_prep.pkl").exists():
        raise SystemExit(f"missing prep at {args.work / 'rg_prep.pkl'} — run the validation first")

    solvers = _available_solvers()
    print(f"available MILP solvers: {solvers}", flush=True)

    def run(solver: str) -> Result:
        if solver in store:
            print(f"[{solver}] cached, skip", flush=True)
            return store[solver]
        print(f"[{solver}] running ...", flush=True)
        t = time.time()
        try:
            # Fresh ref + prep load per solver so a broken interface (e.g. the optlang
            # hybrid_interface clone bug at .solver=) doesn't pollute the next solver's state.
            ref = cobra.io.read_sbml_model(str(args.work / "raven_refModel.xml"))
            ref.solver = solver
            local_prep = pickle.load(open(args.work / "rg_prep.pkl", "rb"))
            local_prep.min_model.solver = solver
            g = gene_scores_from_expression(expr, 1.0)
            r = score_reactions_from_genes(ref, g)
            model = ftinit(local_prep, rxn_scores=r, gene_scores=g, series="1+1",
                           mip_gap=args.mip_gap, time_limit=args.time_limit)
            rset = sorted(x.id for x in model.reactions)
            res = Result(solver, time.time() - t, "ok", len(rset), rset)
        except Exception as ex:  # noqa: BLE001 - failure mode is the finding
            res = Result(solver, time.time() - t,
                         f"FAIL:{type(ex).__name__}: {str(ex)[:80]}", 0, [])
        store[solver] = res
        save()
        return res

    results: dict[str, Result] = {s: run(s) for s in solvers}

    # Reporting.
    lines = [f"# Cross-solver ftINIT benchmark — Human-GEM / {args.cell}", "",
             f"Same `ftinit()` call (no-task scaled prep; `mip_gap={args.mip_gap}`, "
             f"`time_limit={args.time_limit}s`) run with each installed MILP-capable "
             f"optlang interface. Generated by `scripts/analyze_init_solvers.py`.", "",
             "## Per-solver result", "",
             "| solver | time (s) | status | n_rxns |",
             "|--------|---------:|--------|-------:|"]
    for s, r in results.items():
        lines.append(f"| {s} | {r.seconds:.0f} | {r.status} | {r.n_rxns} |")
    lines.append("")

    ok = {s: r for s, r in results.items() if r.status == "ok" and r.reactions}
    if len(ok) >= 2:
        lines += ["## Reaction-set agreement (Jaccard)", "",
                  "| solvers | shared | only A | only B | Jaccard |",
                  "|---------|-------:|-------:|-------:|--------:|"]
        names = sorted(ok)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sa, sb = set(ok[a].reactions), set(ok[b].reactions)
                lines.append(f"| {a} vs {b} | {len(sa & sb)} | {len(sa - sb)} | "
                             f"{len(sb - sa)} | {_jaccard(sa, sb):.3f} |")
        lines.append("")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.doc:
        args.doc.write_text(text)
        print(f"wrote {args.doc}", flush=True)


if __name__ == "__main__":
    main()
