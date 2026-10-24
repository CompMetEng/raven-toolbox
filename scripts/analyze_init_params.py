#!/usr/bin/env python3
"""Parameter calibration for ftINIT — intrinsic speed/quality sweeps.

Genome-scale benchmark that sweeps the MILP/conditioning parameters of raven_toolbox's
:func:`raven_toolbox.init.run_ftinit`, :func:`raven_toolbox.init.ftinit`, and
:func:`prep_init_model` and records, for each value, the *intrinsic* trade-off: wall-clock
solve time, the MILP objective, and how far the result drifts from the tightest-setting
("reference") run — both in objective (relative gap) and in the **kept-reaction set**
(Jaccard). No external (RAVEN) reference is used: the question answered here is "what is
the loosest / cheapest setting that still reproduces the tight-setting solution?".

Why reaction-set drift matters: a MIP gap g only guarantees the *objective* is within g of
optimal; the *which-reactions* answer can jump between alternate optima well before the
objective moves. For a model-extraction tool the reaction set is the product, so we track
its stability explicitly.

Sweeps (select with ``--sweeps``; each is resumable — results are pickled per config and a
re-run skips finished ones):

* ``ftinit_milp``  — single staged-MILP step (step 0 of series '1+1') on the merged model.
                     Cheap (~30-200 s each); the core sweep for ``mip_gap``/``big_m``/``force_on``.
* ``prep_scale``   — rescaleModelForINIT on/off and its ``max_stoich_diff``, fed into the
                     same step-0 MILP. Shows why scaling is needed for a fixed big-M.
* ``ftinit_full``  — the whole ``ftinit`` pipeline (both steps + gap-fill). Sweeps
                     ``mip_gap``/``big_m``. Expensive (~200 s+/config).

Usage
-----
    python scripts/analyze_init_params.py \
        --work ~/hgem_compare --cell HCT116 --sweeps ftinit_milp,prep_scale

``--work`` holds ``raven_refModel.xml`` and the Human-GEM-derived spont/custom inputs
(see the Human-GEM validation run). Requires a MILP solver (Gurobi/HiGHS) on the cobra
config. Produces a results pickle and prints a table per sweep; feed the tables into
docs/init_param_calibration.md.
"""
from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import cobra

from raven_toolbox.init import (
    ftinit,
    gene_scores_from_expression,
    prep_init_model,
    score_reactions_from_genes,
)
from raven_toolbox.init.ftinit import run_ftinit
from raven_toolbox.init.merge import group_rxn_scores
from raven_toolbox.init.prep import rescale_for_init
from raven_toolbox.init.steps import get_init_steps

# Sweep grids (first value of each tolerance sweep is the tight "reference").
MIP_GAPS = (0.0002, 0.001, 0.003, 0.01, 0.03, 0.1)
BIG_MS = (100.0, 50.0, 25.0, 250.0, 1000.0)
FORCE_ONS = (0.1, 0.02, 0.05, 0.2, 0.5)
MAX_STOICH = (25.0, 10.0, 50.0, 100.0)

# "Recommended = cheapest config within these of the reference" thresholds.
TOL_OBJ = 0.005   # relative objective gap
TOL_JAC = 0.99    # kept-reaction-set Jaccard


@dataclass
class Result:
    """One config's outcome (reaction set stored sorted for pickling/Jaccard)."""

    label: str
    seconds: float
    status: str
    objective: float
    n_kept: int
    reactions: list[str] = field(default_factory=list)
    rel_obj_gap: float | None = None  # vs the sweep reference
    jaccard: float | None = None      # vs the sweep reference


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def _load_inputs(work: Path, human_gem: Path, cell: str):
    ref = cobra.io.read_sbml_model(str(work / "raven_refModel.xml"))
    ref.solver = cobra.Configuration().solver
    spont = []
    with open(human_gem / "model" / "reactions.tsv") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ci = hdr.index("spontaneous")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[ci] == "1":
                spont.append(p[0])
    protein = [f"MAR0{n}" for n in (5155, 5156, 5161, 5167, 5168, 5169, 5170, 5171, 5172,
               5174, 5260, 5262, 5264, 5266, 5267, 5268, 5269, 5270, 5271, 5273, 5275, 5277,
               5279, 5281, 5283, 5291)] + ["MAR09817", "MAR09818"]
    pool = ["MAR00011", "MAR00012", "MAR00477", "MAR05233", "MAR05234", "MAR05238",
            "MAR05239", "MAR05243", "MAR05244", "MAR05247", "MAR09022", "MAR00015",
            "MAR00016", "MAR00017", "MAR10033", "MAR10035", "MAR10036", "MAR10037",
            "MAR10038", "MAR10062", "MAR10063", "MAR10064", "MAR10065", "MAR13082"]
    custom = sorted(set(protein) | set(pool))
    expr: dict[str, float] = {}
    with open(human_gem / "data" / "datasets" / "Hart2015_RNAseq.txt") as f:
        h = f.readline().rstrip("\n").split("\t")
        c = h.index(cell)
        for line in f:
            p = line.rstrip("\n").split("\t")
            expr[p[0]] = float(p[c])
    gene_scores = gene_scores_from_expression(expr, 1.0)
    rxn_scores = score_reactions_from_genes(ref, gene_scores)
    return ref, spont, custom, gene_scores, rxn_scores


def _step0(prep, rxn_scores):
    """The scores/flags for step 0 of series '1+1' (the cheap single-MILP probe)."""
    step = get_init_steps("1+1")[0]
    to_zero = prep.masks.ignored(step.ignore_mask)
    scores = group_rxn_scores(prep.min_model, rxn_scores, prep.orig_rxn_ids,
                              prep.group_ids, to_zero)
    return step, scores


def _run_step0(min_model, scores, prep, step, **kw) -> Result:
    t = time.time()
    res = run_ftinit(min_model, scores, essential_rxns=set(prep.essential_rxns),
                     allow_excretion=step.allow_met_secr, rem_pos_rev=step.pos_rev_off,
                     ignore_mets=step.mets_to_ignore, **kw)
    return Result(label="", seconds=time.time() - t, status="ok",
                  objective=res.objective, n_kept=len(res.on_reactions),
                  reactions=sorted(res.on_reactions))


def _finalize(results: list[Result]) -> None:
    """Fill rel_obj_gap / jaccard against the first result (the reference)."""
    ref = results[0]
    ref_set = set(ref.reactions)
    for r in results:
        r.rel_obj_gap = (ref.objective - r.objective) / abs(ref.objective) if ref.objective else 0.0
        r.jaccard = _jaccard(set(r.reactions), ref_set)


def _recommend(results: list[Result]) -> str:
    """Cheapest config (after the reference) within both tolerances; '-' if none."""
    ok = [r for r in results[1:]
          if r.status == "ok" and abs(r.rel_obj_gap or 1) <= TOL_OBJ and (r.jaccard or 0) >= TOL_JAC]
    return min(ok, key=lambda r: r.seconds).label if ok else "-"


def _print_table(title: str, results: list[Result], note: str = "") -> list[str]:
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines.append("| config | time (s) | status | objective | n_kept | rel.obj.gap | Jaccard vs ref |")
    lines.append("|---|--:|---|--:|--:|--:|--:|")
    for r in results:
        gap = "ref" if r is results[0] else (f"{r.rel_obj_gap:+.4f}" if r.rel_obj_gap is not None else "")
        jac = "ref" if r is results[0] else (f"{r.jaccard:.4f}" if r.jaccard is not None else "")
        lines.append(f"| {r.label} | {r.seconds:.0f} | {r.status} | {r.objective:.1f} | "
                     f"{r.n_kept} | {gap} | {jac} |")
    rec = _recommend(results)
    lines += ["", f"Cheapest config within obj≤{TOL_OBJ:.1%} & Jaccard≥{TOL_JAC} of ref: **{rec}**", ""]
    for ln in lines:
        print(ln)
    return lines


# --------------------------------------------------------------------------- sweeps

def sweep_ftinit_milp(prep, rxn_scores, store, save) -> list:
    step, scores = _step0(prep, rxn_scores)
    mm = prep.min_model
    doc: list[str] = []

    def cfg(label, **kw):
        key = ("ftinit_milp", label)
        if key not in store:
            print(f"[ftinit_milp] {label} ...", flush=True)
            r = _run_step0(mm, scores, prep, step, **kw)
            r.label = label
            store[key] = r
            save()
        return store[key]

    # mip_gap sweep (big_m=100, force_on=0.1)
    res = [cfg(f"gap={g}", mip_gap=g, big_m=100.0, force_on=0.1, time_limit=900) for g in MIP_GAPS]
    _finalize(res)
    doc += _print_table("ftINIT step-0: mip_gap (big_m=100, force_on=0.1)", res)

    # big_m sweep (gap=0.001, force_on=0.1)
    res = [cfg(f"big_m={int(b)}", mip_gap=0.001, big_m=b, force_on=0.1, time_limit=900) for b in BIG_MS]
    _finalize(res)
    doc += _print_table("ftINIT step-0: big_m (gap=0.001, force_on=0.1)", res,
                        "big_m caps a scored reaction's flux; large values weaken the LP relaxation.")

    # force_on sweep (gap=0.001, big_m=100) — changes the model (connectivity threshold)
    res = [cfg(f"force_on={fo}", mip_gap=0.001, big_m=100.0, force_on=fo, time_limit=900) for fo in FORCE_ONS]
    _finalize(res)
    doc += _print_table("ftINIT step-0: force_on (gap=0.001, big_m=100)", res,
                        "force_on changes the *model* (min flux to count as 'on'), not just tolerance — "
                        "Jaccard here measures sensitivity, not error.")
    return doc


def sweep_prep_scale(ref, spont, custom, rxn_scores, store, save) -> list:
    doc: list[str] = []
    # One unscaled prep; rescale copies of its min_model for each setting.
    base = prep_init_model(ref, ext_comp="e", spontaneous=spont, custom=custom, scale=False)
    step, scores = _step0(base, rxn_scores)

    def cfg(label, msd):
        key = ("prep_scale", label)
        if key not in store:
            print(f"[prep_scale] {label} ...", flush=True)
            mm = base.min_model.copy()
            if msd is not None:
                rescale_for_init(mm, msd)
            # group_rxn_scores keys are merged ids — identical across copies, so reuse `scores`.
            t = time.time()
            try:
                r = _run_step0(mm, scores, base, step, mip_gap=0.001, big_m=100.0,
                               force_on=0.1, time_limit=600)
            except Exception as ex:  # noqa: BLE001  (infeasible/intractable is a finding)
                r = Result(label=label, seconds=time.time() - t, status=f"FAIL:{type(ex).__name__}",
                           objective=0.0, n_kept=0)
            r.label = label
            store[key] = r
            save()
        return store[key]

    res = [cfg("scale=on,msd=25", 25.0)]  # reference = production default
    res += [cfg(f"msd={int(m)}", m) for m in MAX_STOICH if m != 25.0]
    res.append(cfg("scale=off", None))
    _finalize(res)
    doc += _print_table("prep scaling: rescaleModelForINIT max_stoich_diff (+scale off), big_m=100", res,
                        "With big_m=100 fixed, scale=off / poor conditioning is expected to be "
                        "infeasible or far slower — that is the reason scaling is on by default.")
    return doc


def sweep_ftinit_full(prep, rxn_scores, gene_scores, store, save) -> list:
    doc: list[str] = []

    def cfg(label, **kw):
        key = ("ftinit_full", label)
        if key not in store:
            print(f"[ftinit_full] {label} ...", flush=True)
            t = time.time()
            try:
                out = ftinit(prep, rxn_scores=rxn_scores, gene_scores=gene_scores, series="1+1", **kw)
                r = Result(label=label, seconds=time.time() - t, status="ok",
                           objective=0.0, n_kept=len(out.reactions),
                           reactions=sorted(x.id for x in out.reactions))
            except Exception as ex:  # noqa: BLE001
                r = Result(label=label, seconds=time.time() - t, status=f"FAIL:{type(ex).__name__}",
                           objective=0.0, n_kept=0)
            store[key] = r
            save()
        return store[key]

    res = [cfg(f"gap={g}", mip_gap=g, time_limit=600) for g in (0.001, 0.003, 0.01)]
    res += [cfg(f"big_m={int(b)}", mip_gap=0.003, big_m=b, time_limit=600) for b in (50.0, 250.0)]
    _finalize(res)
    doc += _print_table("ftINIT full pipeline ('1+1'): mip_gap & big_m — final model size/stability", res)
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=Path.home() / "hgem_compare")
    ap.add_argument("--human-gem", type=Path, default=Path.home() / "github" / "Human-GEM")
    ap.add_argument("--cell", default="HCT116")
    ap.add_argument("--sweeps", default="ftinit_milp,prep_scale,ftinit_full",
                    help="comma-separated subset of: ftinit_milp,prep_scale,ftinit_full")
    ap.add_argument("--out", type=Path, default=None, help="results pickle (resumable)")
    ap.add_argument("--doc", type=Path, default=None, help="write the markdown tables here")
    args = ap.parse_args()

    out = args.out or args.work / f"init_param_sweep_{args.cell}.pkl"
    store: dict = pickle.load(open(out, "rb")) if out.exists() else {}

    def save():
        tmp = Path(f"{out}.part")
        pickle.dump(store, open(tmp, "wb"))
        tmp.replace(out)

    sweeps = set(args.sweeps.split(","))
    t0 = time.time()
    ref, spont, custom, gene_scores, rxn_scores = _load_inputs(args.work, args.human_gem, args.cell)
    print(f"[{time.time()-t0:.0f}s] loaded {len(ref.reactions)} rxns, cell={args.cell}", flush=True)

    prep = None
    if sweeps & {"ftinit_milp", "ftinit_full"}:
        prep = prep_init_model(ref, ext_comp="e", spontaneous=spont, custom=custom, scale=True)
        print(f"[{time.time()-t0:.0f}s] scaled prep: min_model {len(prep.min_model.reactions)} rxns",
              flush=True)

    doc: list[str] = [f"# ftINIT parameter calibration — Human-GEM / {args.cell}", "",
                      "Generated by `scripts/analyze_init_params.py`. Reference (first) row of each "
                      "tolerance sweep is the tightest setting; gaps/Jaccard are measured against it.", ""]
    if "ftinit_milp" in sweeps:
        doc += sweep_ftinit_milp(prep, rxn_scores, store, save)
    if "prep_scale" in sweeps:
        doc += sweep_prep_scale(ref, spont, custom, rxn_scores, store, save)
    if "ftinit_full" in sweeps:
        doc += sweep_ftinit_full(prep, rxn_scores, gene_scores, store, save)

    if args.doc:
        args.doc.write_text("\n".join(doc) + "\n")
        print(f"\nwrote {args.doc}", flush=True)


if __name__ == "__main__":
    main()
