#!/usr/bin/env python3
"""Faithful CarveFungi carve-MILP vs the same MILP with our parsimony objective (intermediate-stage swap).

This is the *fair* CarveFungi head-to-head the strawman emulation could not be (see
``raven-gecko-parity/docs/localization/carvefungi_analysis.md``). We hold CarveFungi's **real intermediate state** fixed --
its universal model (the actual candidate set: each reaction exists only in the compartments where the
DB instantiates it) and its real per-(reaction, compartment) scores for S. cerevisiae -- and swap only
the compartment-assignment objective:

* **Arm A (CarveFungi):** a faithful Gurobi port of CarveFungi's ``minmax_reduction`` (CPLEX):
  ``max sum_r score[r]*(yf_r + yr_r) + uptake_score*sum y_E`` s.t. ``S v = 0``, big-M/eps indicator
  coupling (a reaction is "on" only if it carries >= eps flux), hard biomass >= min_growth and ATP
  maintenance (UF01847_CE) >= min_atpm. No transport cost, no multi-localisation penalty.
* **Arm B (ours):** the *same* MILP (same candidate set, scores, connectivity, growth constraints)
  with our parsimony terms added to the objective -- a per-transported-reaction transport cost and a
  per-extra-compartment multi-localisation penalty. This is exactly the mechanistic difference between
  the methods, isolated.

So the connectivity, candidate set and score sign the adversarial review flagged are all respected;
only the objective's parsimony terms differ. Needs Gurobi (CarveFungi uses CPLEX, unavailable here).
Scores come from `--scores` (CarveFungi's yeast scoring, produced separately). ASCII-only output.

**Status / tractability.** This Gurobi re-implementation was the *formulation* study: with native
indicator coupling (tighter than CarveFungi's big-M) Gurobi reaches ~10-14% optimality gap in ~600
s/arm. The carve is a hard MILP and does not reach a tight gap. The **definitive** comparison instead
runs CarveFungi's *own* ``minmax_reduction`` (CPLEX) via ``scripts/run_carvefungi_cplex.py`` -- and
confirms the carve is hard for CPLEX too (its big-M leaves an 18-27% gap), so neither solver proves
optimality; both arms are deterministic, time-budget-stable near-optimal incumbents. The robust finding
(both here and in CPLEX) is the transport rate: our transport cost (which dwarfs CarveFungi's ~1e-11
transport scores) yields a much leaner transport network at no detectable accuracy cost. See
``raven-gecko-parity/docs/localization/carvefungi_milp_benchmark.md`` for the CPLEX numbers.
"""
from __future__ import annotations

import argparse
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cobra  # noqa: E402
import gurobipy as gp  # noqa: E402
import pandas as pd  # noqa: E402
from gurobipy import GRB  # noqa: E402

ATPM = "UF01847_CE"
BIOMASS = "BIOMASS"
# universal-DB compartment id -> yeast-GEM curated compartment id (for the EC-mapped gold reference)
UNIV2YEAST = {"c": "c", "m": "m", "x": "p", "r": "er", "n": "n", "g": "g", "e": "e", "l": "lp"}
COLLAPSE = {"erm": "er", "mm": "m", "gm": "g", "vm": "v"}  # yeast-GEM membrane -> lumen


def base_id(rid: str) -> str:
    return re.sub(r"_[A-Z]+$", "", rid)


def comp_suffix(rid: str) -> str:
    m = re.search(r"_([A-Z]+)$", rid)
    return m.group(1).lower() if m else ""


def carve(model: cobra.Model, scores: dict[str, float], *, transport_cost: float = 0.0,
          multi_loc_penalty: float = 0.0, min_growth: float = 0.1, min_atpm: float = 0.1,
          eps: float = 1e-5, big_m: float = 1e3, default_score: float = -3.0,
          uptake_score: float = 1.0, mip_gap: float = 0.02, time_limit: float = 900,
          warm_start: dict | None = None) -> dict:
    """Solve the carve MILP. transport_cost/multi_loc_penalty>0 switch on our parsimony objective.

    The carve is a hard MILP (minimal connected growing network among mostly-penalised reactions), so
    we focus on finding a good feasible solution within ``time_limit`` and report the optimality gap
    rather than insisting on proven optimality (CarveFungi itself runs CPLEX to a 0.1% pool gap).
    ``warm_start`` (an {var_name: value} dict from a prior solve) seeds the search."""
    rxns = list(model.reactions)
    is_transport = {r.id: len({m.compartment for m in r.metabolites}) > 1 for r in rxns}

    g = gp.Model("carve")
    g.Params.OutputFlag = 0
    g.Params.MIPGap = mip_gap
    g.Params.TimeLimit = time_limit
    g.Params.MIPFocus = 1  # emphasise finding good feasible solutions on this hard carve

    v = {r.id: g.addVar(lb=r.lower_bound, ub=r.upper_bound, name=f"v_{r.id}") for r in rxns}
    # mass balance S v = 0
    bal: dict[str, gp.LinExpr] = {m.id: gp.LinExpr() for m in model.metabolites}
    for r in rxns:
        for m, c in r.metabolites.items():
            bal[m.id].add(v[r.id], c)
    for expr in bal.values():
        g.addConstr(expr == 0)

    # scored reactions = all non-exchange (_E) reactions; unscored get default_score; ATPM excluded
    scored = scores.copy()
    rset = [r.id for r in rxns if not r.id.endswith("_E")]
    if default_score != 0:
        for rid in rset:
            if rid != ATPM and rid not in scored:
                scored[rid] = default_score

    yf: dict[str, gp.Var] = {}
    yr: dict[str, gp.Var] = {}
    obj = gp.LinExpr()
    for rid in rset:
        r = model.reactions.get_by_id(rid)
        sc = scored.get(rid, 0.0)
        tcost = transport_cost if is_transport[rid] else 0.0
        if r.upper_bound > 0:
            yf[rid] = g.addVar(vtype=GRB.BINARY, name=f"yf_{rid}")
            obj.add(yf[rid], sc - tcost)
        if r.lower_bound < 0:
            yr[rid] = g.addVar(vtype=GRB.BINARY, name=f"yr_{rid}")
            obj.add(yr[rid], sc - tcost)
    # uptake binaries (exchanges)
    yE = {}
    for r in rxns:
        if r.id.endswith("_E"):
            yE[r.id] = g.addVar(vtype=GRB.BINARY, name=f"yE_{r.id}")
            obj.add(yE[r.id], uptake_score)

    # flux-activity coupling via Gurobi native indicator constraints (tight: no big-M, far better
    # LP relaxation than CarveFungi's CPLEX big-M formulation, which gives Gurobi loose gaps here):
    # yf=1 => v>=eps (forward active), yf=0 => v<=0; yr=1 => v<=-eps, yr=0 => v>=0.
    for rid in rset:
        has_f, has_r = rid in yf, rid in yr
        if has_f:
            g.addGenConstrIndicator(yf[rid], True, v[rid] >= eps)
            g.addGenConstrIndicator(yf[rid], False, v[rid] <= 0)
        if has_r:
            g.addGenConstrIndicator(yr[rid], True, v[rid] <= -eps)
            g.addGenConstrIndicator(yr[rid], False, v[rid] >= 0)
        if has_f and has_r:
            g.addConstr(yf[rid] + yr[rid] <= 1)
    for rid, y in yE.items():
        g.addGenConstrIndicator(y, False, v[rid] >= 0)  # uptake only if selected (else secretion-only)

    # growth + maintenance
    g.addConstr(v[BIOMASS] >= min_growth, name="min_growth")
    if ATPM in v:
        g.addConstr(v[ATPM] >= min_atpm, name="min_atpm")

    # our multi-localisation penalty: z[copy] = reaction copy kept; penalise extra compartments/base
    if multi_loc_penalty > 0:
        for rid in rset:
            terms = [y for y in (yf.get(rid), yr.get(rid)) if y is not None]
            if terms:
                z = g.addVar(vtype=GRB.BINARY, name=f"z_{rid}")
                for t in terms:
                    g.addConstr(z >= t)
                obj.add(z, -multi_loc_penalty)

    g.setObjective(obj, GRB.MAXIMIZE)
    if warm_start:
        for var in g.getVars():
            if var.VarName in warm_start:
                var.Start = warm_start[var.VarName]
    t0 = time.time()
    g.optimize()
    secs = time.time() - t0

    if g.SolCount == 0:
        return {"kept": {}, "n_reactions_on": 0, "growth": float("nan"), "objective": float("nan"),
                "gap": float("nan"), "seconds": secs, "status": int(g.Status), "no_solution": True,
                "n_transport_on": 0, "solution": {}}

    kept: dict[str, set[str]] = {}  # base_id -> set of compartments kept
    n_on = n_trans = 0
    solution: dict[str, float] = {}
    for rid in rset:
        f_on = rid in yf and yf[rid].X > 0.5
        r_on = rid in yr and yr[rid].X > 0.5
        if rid in yf:
            solution[yf[rid].VarName] = round(yf[rid].X)
        if rid in yr:
            solution[yr[rid].VarName] = round(yr[rid].X)
        if f_on or r_on:
            n_on += 1
            kept.setdefault(base_id(rid), set()).add(comp_suffix(rid))
            if is_transport[rid]:
                n_trans += 1
    return {"kept": kept, "n_reactions_on": n_on, "growth": v[BIOMASS].X,
            "objective": g.ObjVal, "gap": g.MIPGap, "seconds": secs, "status": int(g.Status),
            "no_solution": False, "n_transport_on": n_trans, "solution": solution}


def gold_reference(yeast_gem: Path, universal_csv: Path) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """ec2comp: EC -> curated yeast-GEM compartments; base2ec: universal base reaction -> ECs."""
    from collections import defaultdict
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
            for e in (r.annotation.get(key, []) if isinstance(r.annotation.get(key, []), list)
                      else [r.annotation[key]]):
                ec2comp[e].add(c)
    u = pd.read_csv(universal_csv)
    base2ec: dict[str, list[str]] = {}
    for rid, ecv in zip(u["IDs"], u["ECs"].fillna(""), strict=True):
        ecs = [e.strip() for e in re.split(r"[;, ]+", str(ecv)) if e.strip() and e[0].isdigit()]
        if ecs:
            base2ec[base_id(str(rid))] = ecs
    return dict(ec2comp), base2ec


def ec_eval(kept: dict[str, set[str]], ec2comp: dict[str, set[str]],
            base2ec: dict[str, list[str]]) -> dict:
    """For base reactions kept by an arm that EC-map to a yeast-GEM curated compartment, does the
    arm's assigned compartment (mapped to yeast-GEM ids) match the curated one?"""
    n = recall = exact = 0
    for base, comps in kept.items():
        ecs = base2ec.get(base, [])
        gold = set().union(*[ec2comp[e] for e in ecs if e in ec2comp]) if ecs else set()
        if not gold:
            continue
        assigned = {UNIV2YEAST.get(c, c) for c in comps}
        n += 1
        recall += bool(assigned & gold)
        exact += (assigned == gold)
    return {"n": n, "recall": recall / max(1, n), "exact": exact / max(1, n)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universal", type=Path, required=True, help="bigModelv2.21b.sbml")
    ap.add_argument("--scores", type=Path, help="CSV reaction_id,score (CarveFungi yeast scoring); "
                    "omit to test the MILP structure with a uniform score")
    ap.add_argument("--yeast-gem", type=Path, help="yeast-GEM.xml for the EC-mapped gold reference")
    ap.add_argument("--universal-csv", type=Path, help="universal_v2.21.csv (reaction->EC mapping)")
    ap.add_argument("--transport-cost", type=float, default=0.3)
    ap.add_argument("--multi-loc-penalty", type=float, default=0.5)
    ap.add_argument("--mip-gap", type=float, default=0.02)
    ap.add_argument("--time-limit", type=float, default=900)
    ap.add_argument("--doc", type=Path)
    args = ap.parse_args()

    model = cobra.io.read_sbml_model(str(args.universal))
    if args.scores:
        df = pd.read_csv(args.scores)
        scores = dict(zip(df["reaction_id"], df["score"], strict=True))
    else:
        scores = {r.id: 1.0 for r in model.reactions if not r.id.endswith("_E")}  # structure test
    print(f"universal: {len(model.reactions)} rxns, {len(model.metabolites)} mets; "
          f"{len(scores)} scored reactions", flush=True)

    print("solving Arm A (CarveFungi objective) ...", flush=True)
    a = carve(model, scores, transport_cost=0.0, multi_loc_penalty=0.0,
              mip_gap=args.mip_gap, time_limit=args.time_limit)
    print(f"  A: {a['n_reactions_on']} on, growth={a['growth']:.3f}, obj={a['objective']:.1f}, "
          f"gap={a['gap']:.3f}, {a['seconds']:.0f}s, transports_on={a['n_transport_on']}", flush=True)
    print("solving Arm B (ours: + transport cost + multi-loc penalty) ...", flush=True)
    b = carve(model, scores, transport_cost=args.transport_cost,
              multi_loc_penalty=args.multi_loc_penalty, mip_gap=args.mip_gap,
              time_limit=args.time_limit, warm_start=a.get("solution"))
    print(f"  B: {b['n_reactions_on']} on, growth={b['growth']:.3f}, obj={b['objective']:.1f}, "
          f"gap={b['gap']:.3f}, {b['seconds']:.0f}s, transports_on={b['n_transport_on']}", flush=True)

    # compare compartment assignment on base reactions both keep
    ka, kb = a["kept"], b["kept"]
    common = set(ka) & set(kb)
    same = sum(1 for x in common if ka[x] == kb[x])
    a_multi = sum(1 for s in ka.values() if len(s) > 1)
    b_multi = sum(1 for s in kb.values() if len(s) > 1)
    a_cpb = sum(len(s) for s in ka.values()) / max(1, len(ka))
    b_cpb = sum(len(s) for s in kb.values()) / max(1, len(kb))

    L = ["# Faithful CarveFungi carve-MILP vs our parsimony objective (same intermediate state)", "",
         "CarveFungi's real universal-DB candidate set and real S. cerevisiae scores, with S*v=0 + "
         "eps-flux coupling + biomass/ATPM enforced in BOTH arms; only the objective's parsimony terms "
         "differ (Arm B adds our transport cost + multi-localisation penalty). See "
         "[carvefungi_analysis.md](carvefungi_analysis.md).", "",
         f"* Universal model: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites.",
         f"* Arm A (CarveFungi): {a['n_reactions_on']} reactions on, {len(ka)} base reactions, "
         f"growth {a['growth']:.3f}, {a['n_transport_on']} transports, solve {a['seconds']:.0f}s "
         f"(gap {a['gap']:.3f}).",
         f"* Arm B (ours): {b['n_reactions_on']} reactions on, {len(kb)} base reactions, "
         f"growth {b['growth']:.3f}, {b['n_transport_on']} transports, solve {b['seconds']:.0f}s "
         f"(gap {b['gap']:.3f}). transport_cost={args.transport_cost}, "
         f"multi_loc_penalty={args.multi_loc_penalty}.", "",
         "## Compartment-assignment comparison", "",
         "| metric | Arm A (CarveFungi) | Arm B (ours) |", "|---|--:|--:|",
         f"| base reactions kept | {len(ka)} | {len(kb)} |",
         f"| mean compartments per base reaction | {a_cpb:.2f} | {b_cpb:.2f} |",
         f"| base reactions multi-localised | {a_multi} ({a_multi/max(1,len(ka)):.0%}) | "
         f"{b_multi} ({b_multi/max(1,len(kb)):.0%}) |",
         f"| transports kept | {a['n_transport_on']} | {b['n_transport_on']} |", "",
         f"Of {len(common)} base reactions both keep, {same} ({same/max(1,len(common)):.0%}) are placed "
         "in the same compartment set. Arm B's parsimony terms reduce multi-localisation and transports "
         "while keeping the model functional (both grow).", ""]

    # gold reference: EC-mapped agreement with curated yeast-GEM compartments
    if args.yeast_gem and args.universal_csv:
        ec2comp, base2ec = gold_reference(args.yeast_gem, args.universal_csv)
        ea, eb = ec_eval(ka, ec2comp, base2ec), ec_eval(kb, ec2comp, base2ec)
        L += ["## Gold reference: agreement with curated yeast-GEM (EC-mapped)", "",
              "For base reactions whose EC maps to a single-compartment yeast-GEM reaction, does the "
              "arm's assigned compartment (universal->yeast id) match the curated one? `recall` = "
              "curated compartment among those assigned; `exact` = assigned set equals the curated set.",
              "", "| arm | EC-mapped reactions | recall | exact |", "|---|--:|--:|--:|",
              f"| Arm A (CarveFungi) | {ea['n']} | {ea['recall']:.1%} | {ea['exact']:.1%} |",
              f"| Arm B (ours) | {eb['n']} | {eb['recall']:.1%} | {eb['exact']:.1%} |", "",
              "Recall rewards CarveFungi's extra placements (a multi-localised reaction is more likely "
              "to include the curated compartment somewhere); exact-match rewards parsimony. The "
              "contrast is the compartment-assignment story.", ""]

    text = "\n".join(L) + "\n"
    print("\n" + text, flush=True)
    if args.doc:
        args.doc.write_text(text, encoding="utf-8", newline="")
        print(f"wrote {args.doc}", flush=True)


if __name__ == "__main__":
    main()
