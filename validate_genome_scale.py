"""Real ftINIT extraction against genome-scale Human-GEM, for cross-language parity
with the same pipeline (prepINITModel, then ftINIT) run in RAVEN.

Not a committed test -- a script to run once per side of a comparison, using the full
pipeline (prep_init_model with a real task list, then ftinit() with real RNA-seq
expression), not synthetic scores.

Two independent knobs control parallelism, and only one of them is what you think:

* --processes parallelises prep_init_model's task-essential-discovery loop, across
  *tasks* (each task's own LP solve stays single-threaded, for reproducibility --
  see raven_toolbox.tasks.check.find_task_essential_reactions). This is the phase
  that took ~99 minutes on 1 core for 57 Human-GEM tasks; it should scale close to
  linearly with core count since the tasks are fully independent of each other.
* --cobra-processes sets cobra.Configuration().processes, which controls the two
  simplify_model FVA passes inside prep_init_model (unrelated code path, no
  connection to --processes). Both default to the same value for convenience; pass
  them separately if you want to bound one but not the other (e.g. to leave memory
  headroom -- prep_init_model's FVA pass has been observed to spawn one worker per
  CPU with no memory awareness and OOM on a loaded machine).

Run as a file (not `python -c`): both parallel paths use multiprocessing, which
re-executes top-level module code in each worker on Windows ('spawn') without a
__main__ guard.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cobra

from raven_toolbox.init import (
    ftinit,
    gene_scores_from_expression,
    prep_init_model,
    score_reactions_from_genes,
)
from raven_toolbox.io import read_yaml_model
from raven_toolbox.tasks import parse_task_list


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human-gem", type=Path, default=Path(r"C:\Work\GitHub\Human-GEM"),
                    help="a Human-GEM checkout (contains model/, data/metabolicTasks/, "
                         "data/datasets/)")
    ap.add_argument("--cell", default="HCT116",
                    help="column in Hart2015_RNAseq.txt to use as expression (default HCT116)")
    ap.add_argument("--series", default="1+0", help="ftINIT staging series (default 1+0)")
    ap.add_argument("--processes", type=int, default=1,
                    help="workers for prep_init_model's task-essential-discovery loop "
                         "(default 1 = sequential)")
    ap.add_argument("--cobra-processes", type=int, default=None,
                    help="cobra.Configuration().processes for the FVA-based simplify "
                         "passes (default: same as --processes)")
    ap.add_argument("--time-limit", type=float, default=900.0,
                    help="per-step MILP time limit in seconds (default 900)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "validate_genome_scale_result.json")
    ap.add_argument("--essential-cache", type=Path,
                    default=Path(__file__).parent / "validate_genome_scale_essential_cache.pkl",
                    help="resumable cache for task-essential discovery; delete to force a "
                         "full recompute")
    args = ap.parse_args()

    cobra.Configuration().processes = args.cobra_processes or args.processes
    log(f"processes={args.processes} (task-essential loop), "
        f"cobra.Configuration().processes={cobra.Configuration().processes} (FVA passes)")

    log(f"loading {args.human_gem / 'model' / 'Human-GEM.yml'}")
    ref = read_yaml_model(str(args.human_gem / "model" / "Human-GEM.yml"))
    log(f"reference: {len(ref.reactions)} rxns, {len(ref.genes)} genes")

    tasks = parse_task_list(
        str(args.human_gem / "data" / "metabolicTasks" / "metabolicTasks_Essential.txt")
    )
    log(f"tasks: {len(tasks)}")

    log("prep_init_model starting")
    t0 = time.perf_counter()
    prep = prep_init_model(ref, tasks, essential_cache_path=args.essential_cache,
                           processes=args.processes, verbose=True)
    prep_seconds = time.perf_counter() - t0
    log(f"prep built in {prep_seconds:.0f}s: "
        f"{len(prep.essential_rxns)} essential rxns, min_model {len(prep.min_model.reactions)} rxns")

    expr: dict[str, float] = {}
    with open(args.human_gem / "data" / "datasets" / "Hart2015_RNAseq.txt") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = header.index(args.cell)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            expr[parts[0]] = float(parts[col])
    log(f"expression: {len(expr)} genes, cell={args.cell}")

    gene_scores = gene_scores_from_expression(expr, 1.0)
    rxn_scores = score_reactions_from_genes(ref, gene_scores)

    log(f"ftinit() starting (series={args.series})")
    t0 = time.perf_counter()
    model = ftinit(prep, rxn_scores, gene_scores=gene_scores, series=args.series,
                   seed=1234, time_limit=args.time_limit, verbose=True)
    extract_seconds = time.perf_counter() - t0
    log(f"ftinit built in {extract_seconds:.0f}s: "
        f"{len(model.reactions)} rxns, {len(model.genes)} genes")

    args.out.write_text(json.dumps({
        "cell": args.cell,
        "series": args.series,
        "reference_reactions": len(ref.reactions),
        "reference_genes": len(ref.genes),
        "tasks": len(tasks),
        "essential_rxns": len(prep.essential_rxns),
        "prep_seconds": round(prep_seconds, 1),
        "extract_seconds": round(extract_seconds, 1),
        "kept_reactions": sorted(r.id for r in model.reactions),
        "kept": len(model.reactions),
        "kept_genes": len(model.genes),
    }, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
