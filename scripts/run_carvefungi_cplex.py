#!/usr/bin/env python3
"""CarveFungi vs ours assignment head-to-head, using CarveFungi's ACTUAL MILP (CPLEX).

Runs CarveFungi's own ``minmax_reduction`` (imported, unmodified) so the comparison is maximally
faithful. The carve is a hard big-M MILP: even full CPLEX leaves the bound loose (~18-27% gap here),
because CarveFungi's big-M reversibility coupling gives a weak LP relaxation. CPLEX nonetheless finds
a single incumbent within seconds that never changes as the bound slowly descends, so each arm's kept
set is a **deterministic, time-budget-stable incumbent -- reproducible, but NOT a proven optimum.**
We report the achieved gap with every result and frame conclusions accordingly: the transport-
parsimony effect is large and direction-robust; the assignment-accuracy difference is within noise.

RUN THIS IN A CPLEX-ENABLED PYTHON (3.10-3.12 where ``import cplex`` works). The bundled PyPI ``cplex``
is Community-capped at 1000 constraints; for the full model, point its ``_internal/cplexXXXX.dll`` at a
licensed CPLEX Studio runtime of the same version (see raven-gecko-parity/docs/localization/carvefungi_milp_benchmark.md). No
Gurobi needed.

Arms (only the objective's parsimony differs; same candidate set, same scores, same connectivity):
* Arm A (CarveFungi): ``minmax_reduction(model, scores)`` unmodified.
* Arm B (ours): same, with each inter-compartment transport reaction's score reduced by
  ``--transport-cost`` (our transport-minimisation term, fed through their objective).

Each kept reaction copy's compartment is read from its METABOLITES (ground truth), not its id suffix
(the universal model's suffixes are mixed-case and overloaded with transport codes).

Inputs:
* ``--carvefungi-dir`` : a clone of github.com/SandraCastilloPriego/CarveFungi (for bin/ + the
  universal model data/reactionDatabase/bigModelv2.21b.sbml + universal_v2.21.csv).
* ``--scores`` : the DeepLoc-injected CarveFungi yeast score dict (reaction_id,score) -- produced by
  running CarveFungi's unmodified scoring with a DeepLoc-derived loc file (see
  raven-gecko-parity/docs/localization/carvefungi_milp_benchmark.md). Provided alongside this script.
* ``--yeast-gem`` : yeast-GEM.xml, for the EC-mapped curated gold reference.

ASCII-only output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import cobra  # noqa: E402
import pandas as pd  # noqa: E402

UNIV2YEAST = {"c": "c", "m": "m", "x": "p", "r": "er", "n": "n", "g": "g", "e": "e", "l": "lp"}
COLLAPSE = {"erm": "er", "mm": "m", "gm": "g", "vm": "v"}


def base_id(rid: str) -> str:
    # Universal-model compartment suffixes are mixed-case (_C/_m/_x/_M ...) and include transport
    # codes (_TCE/_TCM ...). Strip the trailing _suffix case-insensitively so a reaction's copies in
    # different compartments collapse to one base id -- an uppercase-only strip leaves the lowercase
    # copies (_m/_x/_r/_c) unstripped and double-counts them as separate reactions.
    return re.sub(r"_[A-Za-z]+$", "", rid)


def active_reactions(sol: dict[str, float]) -> set[str]:
    """Reaction ids whose forward/reverse keep-binary is on (CarveFungi's own >0.5 threshold)."""
    return {name[3:] for name, val in sol.items()
            if (name.startswith("yf_") or name.startswith("yr_")) and val > 0.5}


def analyse(active, model):
    """Kept reaction copies -> (base reaction ids; base -> set of model compartments from its
    single-compartment copies; number of inter-compartment transport copies).

    Each copy's compartment is read from its METABOLITES, not the id suffix: the suffix is mixed-case
    and overloaded with transport codes, so suffix parsing both mislabels compartments (assigning the
    empty string) and double-counts bases. Transport copies (>1 compartment) are counted, not given a
    single compartment -- the curated gold reference likewise keeps only single-compartment reactions."""
    comp_of, is_transport = {}, {}
    for r in model.reactions:
        cs = {m.compartment for m in r.metabolites}
        is_transport[r.id] = len(cs) > 1
        if len(cs) == 1:
            comp_of[r.id] = next(iter(cs))
    bases: set[str] = set()
    comps: dict[str, set[str]] = defaultdict(set)
    n_transport = 0
    for rid in active:
        bases.add(base_id(rid))
        if is_transport.get(rid):
            n_transport += 1
        elif rid in comp_of:
            comps[base_id(rid)].add(comp_of[rid])
    return bases, dict(comps), n_transport


def gold_reference(yeast_gem: Path, universal_csv: Path):
    y = cobra.io.read_sbml_model(str(yeast_gem))
    ec2comp: dict[str, set[str]] = defaultdict(set)
    for r in y.reactions:
        if r.boundary or not r.genes:
            continue
        comps = {m.compartment for m in r.metabolites if m.compartment}
        if len(comps) != 1:
            continue
        c = COLLAPSE.get(next(iter(comps)), next(iter(comps)))
        for key in ("ec-code", "eccodes"):
            v = r.annotation.get(key, [])
            for e in (v if isinstance(v, list) else [v]):
                ec2comp[e].add(c)
    u = pd.read_csv(universal_csv)
    base2ec: dict[str, list[str]] = {}
    for rid, ecv in zip(u["IDs"], u["ECs"].fillna(""), strict=True):
        ecs = [e.strip() for e in re.split(r"[;, ]+", str(ecv)) if e.strip() and e[0].isdigit()]
        if ecs:
            base2ec[base_id(str(rid))] = ecs
    return dict(ec2comp), base2ec


def ec_eval(kept, ec2comp, base2ec, restrict=None):
    n = recall = exact = 0
    for base, comps in kept.items():
        if restrict is not None and base not in restrict:
            continue
        ecs = base2ec.get(base, [])
        gold = set().union(*[ec2comp[e] for e in ecs if e in ec2comp]) if ecs else set()
        if not gold:
            continue
        assigned = {UNIV2YEAST.get(c, c) for c in comps}
        n += 1
        recall += bool(assigned & gold)
        exact += (assigned == gold)
    return {"n": n, "recall": recall / max(1, n), "exact": exact / max(1, n)}


def best_active_and_obj(solutions):
    """Best (max-objective) pool solution -> (active reaction ids, objective value).

    Tracking the objective lets us report how stable the incumbent is across time budgets: CPLEX
    finds it early and it does not move as the (weak big-M) bound slowly descends, so the kept set is
    deterministic and reproducible -- though, with the gap left at 18-27%, not proven optimal."""
    if not solutions:
        raise RuntimeError("minmax_reduction returned no solutions (CPLEX populate failed?)")
    best = max(solutions, key=lambda s: s.get_obj())
    return active_reactions(best.get_solution()), best.get_obj()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carvefungi-dir", required=True, type=Path)
    ap.add_argument("--scores", required=True, type=Path, help="reaction_id,score CSV")
    ap.add_argument("--yeast-gem", required=True, type=Path)
    ap.add_argument("--universal", type=Path, help="bigModelv2.21b.sbml (default: under carvefungi-dir)")
    ap.add_argument("--universal-csv", type=Path, help="universal_v2.21.csv (default: under carvefungi-dir)")
    ap.add_argument("--transport-cost", type=float, default=0.3)
    ap.add_argument("--time-limit", type=float, default=1200.0,
                    help="CPLEX seconds per arm; the carve is a hard big-M MILP that will not "
                         "reach CarveFungi's 0.1%% pool gap unbounded. We report the achieved gap.")
    ap.add_argument("--arm", choices=["A", "B", "both"], default="both",
                    help="which arm(s) to (re)solve this run; the other loads from --cache-dir so a "
                         "long solve can be split across separate invocations.")
    ap.add_argument("--cache-dir", type=Path,
                    help="where per-arm solutions are persisted (default: alongside --scores).")
    args = ap.parse_args()

    sys.path.insert(0, str(args.carvefungi_dir / "bin"))
    import CarveMeFuncPool as cf  # noqa: E402  (needs cplex)
    from CarveMeFuncPool import minmax_reduction  # noqa: E402  (needs cplex)

    # CarveFungi's generate_soln_pool sets no time limit and prints every pool solution's full
    # ~17k-variable vector (floods stdout). Replace it with the same logic -- same pool.relgap
    # (0.001), same extraction -- quiet, and with the one necessary change: a CPLEX time limit so
    # this hard big-M carve terminates. We surface the achieved MIP gap so the comparison's
    # trustworthiness is explicit rather than assumed.
    gap_holder: dict[str, float] = {}

    def _quiet_time_limited_pool(solver):
        cpx = solver
        cpx.parameters.mip.pool.relgap.set(0.001)  # CarveFungi's setting, unchanged
        cpx.parameters.timelimit.set(float(args.time_limit))
        try:
            cpx.populate_solution_pool()
        except Exception as exc:  # noqa: BLE001
            print("  populate raised:", exc, flush=True)
            return []
        try:
            gap_holder["gap"] = cpx.solution.MIP.get_mip_relative_gap()
            print(f"  CPLEX stopped at {gap_holder['gap']:.3%} gap", flush=True)
        except Exception:  # noqa: BLE001
            pass
        names = cpx.variables.get_names()
        numsol = cpx.solution.pool.get_num()
        print(f"  pool: {numsol} solutions", flush=True)
        return [cf.solution(cpx.solution.pool.get_objective_value(i),
                            dict(zip(names, cpx.solution.pool.get_values(i), strict=True)))
                for i in range(numsol)]

    cf.generate_soln_pool = _quiet_time_limited_pool

    rdb = args.carvefungi_dir / "data" / "reactionDatabase"
    universal = args.universal or rdb / "bigModelv2.21b.sbml"
    universal_csv = args.universal_csv or rdb / "universal_v2.21.csv"

    model = cobra.io.read_sbml_model(str(universal))
    scores = dict(zip(*[pd.read_csv(args.scores)[c] for c in ("reaction_id", "score")], strict=True))
    is_transport = {r.id: len({m.compartment for m in r.metabolites}) > 1 for r in model.reactions}
    scores_by_arm = {
        "A": scores,
        "B": {rid: (s - args.transport_cost if is_transport.get(rid) else s)
              for rid, s in scores.items()},
    }
    labels = {"A": "CarveFungi minmax_reduction (unmodified)",
              "B": f"ours (transports penalised by {args.transport_cost})"}

    cache_dir = args.cache_dir or args.scores.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    to_solve = {"A", "B"} if args.arm == "both" else {args.arm}

    def get_arm(name):
        """Solve arm `name` (if selected) and cache its active set, else load from cache. Returns
        (active set, record) or (None, None) if neither solved nor cached yet. Only the raw active
        set is cached; base/compartment/transport stats are derived from it + the model so a parsing
        fix never needs a re-solve."""
        path = cache_dir / f"arm_{name}.json"
        if name in to_solve:
            print(f"Arm {name}: {labels[name]} (CPLEX, time-limit {args.time_limit:.0f}s) ...",
                  flush=True)
            gap_holder.clear()
            active, obj = best_active_and_obj(minmax_reduction(model, scores_by_arm[name]))
            rec = {"active": sorted(active), "obj": obj, "gap": gap_holder.get("gap"),
                   "time_limit": args.time_limit}
            path.write_text(json.dumps(rec))
            print(f"Arm {name}: solved -> {len(active)} active reactions, obj {obj:.3f}", flush=True)
            return set(active), rec
        if path.exists():
            rec = json.loads(path.read_text())
            print(f"Arm {name}: cache -> {len(rec['active'])} active reactions, obj {rec['obj']:.3f}, "
                  f"gap {rec['gap']:.1%} ({rec['time_limit']:.0f}s)", flush=True)
            return set(rec["active"]), rec
        return None, None

    act_a, rec_a = get_arm("A")
    act_b, rec_b = get_arm("B")
    if act_a is None or act_b is None:
        missing = "A" if act_a is None else "B"
        print(f"\nArm {missing} not available yet; run with --arm {missing} (cache dir {cache_dir}), "
              "then re-run to combine.")
        return

    bases_a, comps_a, tr_a = analyse(act_a, model)
    bases_b, comps_b, tr_b = analyse(act_b, model)
    ec2comp, base2ec = gold_reference(args.yeast_gem, universal_csv)
    common = bases_a & bases_b
    both_assigned = [x for x in common if x in comps_a and x in comps_b]
    same = sum(1 for x in both_assigned if comps_a[x] == comps_b[x])
    ea, eb = ec_eval(comps_a, ec2comp, base2ec), ec_eval(comps_b, ec2comp, base2ec)
    eac = ec_eval(comps_a, ec2comp, base2ec, common)
    ebc = ec_eval(comps_b, ec2comp, base2ec, common)
    rate_a, rate_b = tr_a / max(1, len(bases_a)), tr_b / max(1, len(bases_b))

    def gap_str(rec):
        return f"{rec['gap']:.1%} gap @ {rec['time_limit']:.0f}s" if rec.get("gap") is not None \
            else "gap n/a"

    print("\n=== CPLEX head-to-head (CarveFungi's own MILP; compartments read from metabolites) ===")
    print("note: each arm is a stable, deterministic incumbent at the gap shown, NOT a proven "
          "optimum -- CarveFungi's big-M relaxation leaves the bound loose (see study doc).")
    print(f"Arm A (CarveFungi): {len(bases_a)} base reactions, {tr_a} transports "
          f"({rate_a:.3f}/base); obj {rec_a['obj']:.3f} [{gap_str(rec_a)}]")
    print(f"Arm B (ours):       {len(bases_b)} base reactions, {tr_b} transports "
          f"({rate_b:.3f}/base); obj {rec_b['obj']:.3f} [{gap_str(rec_b)}]")
    print(f"transport reduction: {tr_a}->{tr_b} ({(tr_a-tr_b)/max(1,tr_a):.0%} fewer; "
          f"per-base {rate_a:.3f}->{rate_b:.3f} = {rate_a/max(1e-9,rate_b):.2f}x)")
    print(f"common base reactions: {len(common)}; with a compartment in both: {len(both_assigned)}; "
          f"same compartment set: {same}/{len(both_assigned)} = {same/max(1,len(both_assigned)):.1%}")
    print("gold reference (EC-mapped to curated yeast-GEM), each arm's own kept set:")
    print(f"  Arm A: n={ea['n']} recall={ea['recall']:.1%} exact={ea['exact']:.1%}")
    print(f"  Arm B: n={eb['n']} recall={eb['recall']:.1%} exact={eb['exact']:.1%}")
    print("gold reference on the COMMON kept set (fair, same denominator):")
    print(f"  Arm A: n={eac['n']} recall={eac['recall']:.1%} exact={eac['exact']:.1%}")
    print(f"  Arm B: n={ebc['n']} recall={ebc['recall']:.1%} exact={ebc['exact']:.1%}")


if __name__ == "__main__":
    main()
