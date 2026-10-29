#!/usr/bin/env python3
"""P1 multi-organism benchmark: certified compartment assignment across four kingdoms.

Runs `assign_compartments` on Human-GEM (mammalian, 9 compartments), AraCore (plant,
chloroplast), and iCre1355 (green alga, 11 compartments incl. chloroplast) — establishing that the
method scales past 4 compartments and handles the **chloroplast**, a compartment CarveFungi's fungal
4-category scheme cannot represent. Yeast is benchmarked separately (with the CarveFungi head-to-head)
by `benchmark_certified_yeast.py`.

Each organism supplies: a model loader, its DeepLoc-label -> compartment-code map, a metabolite base
key (so `merge_compartments` unifies a species across compartments), the biomass reaction, and a
membrane/sub-compartment normalisation. Agreement is scored against the model's OWN compartment
annotation (a circular truth; P2 replaces it with organism-specific experimental ground truth
— HPA, SUBA5, the Chlamydomonas Chloroplast Protein Atlas). Here the goal is scaling + certification
+ chloroplast handling, not the accuracy headline. ASCII-only.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import cobra  # noqa: E402
import pandas as pd  # noqa: E402
from cobra.flux_analysis import find_blocked_reactions  # noqa: E402

from raven_toolbox.localization import (  # noqa: E402
    LocalizationScores,
    apply_assignment,
    assign_compartments,
    load_deeploc,
)
from raven_toolbox.manipulation.compartments import merge_compartments  # noqa: E402
from raven_toolbox.utils.gpr import set_gene_rule  # noqa: E402


def _fix_notes_gpr(model):
    """Re-derive gene_reaction_rules from legacy SBML-L2 ``GENE_ASSOCIATION`` notes (uppercase AND/OR,
    which cobra mis-parses into fragment 'genes'). Rebuilds a clean gene set."""
    for r in model.reactions:
        ga = r.notes.get("GENE_ASSOCIATION")
        if ga:
            rule = re.sub(r"\bOR\b", "or", re.sub(r"\bAND\b", "and", ga)).strip().strip("()").strip()
            set_gene_rule(r, rule)
    cobra.manipulation.remove_genes(model, [g for g in model.genes if not g.reactions], remove_reactions=False)


def _base_name(m):
    return m.name


def _base_bracket(m):
    return m.id.split("[")[0]


def _base_suffix(comps):
    def f(m):
        return m.id[:-2] if len(m.id) > 2 and m.id[-2] == "_" and m.id[-1] in comps else m.id
    return f


ORGANISMS = {
    "humangem": dict(
        path="C:/Work/GitHub/Human-GEM/model/Human-GEM.yml", loader="yaml",
        glob="Human-GEM_deeploc_*.csv", data="data/deeploc/humangem", biomass=None, base="name",
        cmap={"cytoplasm": "c", "cytosol": "c", "nucleus": "n", "mitochondrion": "m", "mitochondria": "m",
              "peroxisome": "x", "endoplasmic reticulum": "r", "golgi apparatus": "g", "golgi": "g",
              "lysosome/vacuole": "l", "lysosome": "l", "extracellular": "e", "cell membrane": "e"},
        parent={"i": "m"}, gpr_fix=False),
    "aracore": dict(
        path=".research_tmp/AraCore_v2_1.xml", loader="sbml",
        glob="AraCore_deeploc_*.csv", data="data/deeploc/aracore", biomass="Bio_opt", base="bracket",
        cmap={"cytoplasm": "c", "cytosol": "c", "plastid": "h", "mitochondrion": "m", "mitochondria": "m",
              "peroxisome": "p"},
        parent={"i": "m", "l": "h"}, gpr_fix=False),
    "icre1355": dict(
        path=".research_tmp/iCre1355_auto.xml", loader="sbml",
        glob="iCre1355_deeploc_*.csv", data="data/deeploc/icre1355", biomass="Biomass_Chlamy_auto",
        base="suffix",
        cmap={"cytoplasm": "c", "cytosol": "c", "plastid": "h", "mitochondrion": "m", "mitochondria": "m",
              "peroxisome": "x", "nucleus": "n", "golgi apparatus": "g", "golgi": "g",
              "extracellular": "e", "cell membrane": "e"},
        parent={"i": "m", "u": "h"}, gpr_fix=True),
}


def run(key, cfg, time_limit):
    model = (cobra.io.load_yaml_model(cfg["path"]) if cfg["loader"] == "yaml"
             else cobra.io.read_sbml_model(cfg["path"]))
    if cfg["gpr_fix"]:
        _fix_notes_gpr(model)
    comps = set(model.compartments)
    base = (_base_name if cfg["base"] == "name" else _base_bracket if cfg["base"] == "bracket"
            else _base_suffix(comps))
    biomass = cfg["biomass"] or next(r.id for r in model.reactions if r.objective_coefficient != 0)
    model.objective = biomass
    parent = cfg["parent"]
    norm = lambda c: parent.get(c, c)  # noqa: E731

    def sole(r):
        cs = {x.compartment for x in r.metabolites}
        return next(iter(cs)) if len(cs) == 1 else None

    curated_rxn = {r.id: norm(c) for r in model.reactions if not r.boundary and (c := sole(r))}
    curated_gene = {}
    for rid, c in curated_rxn.items():
        for g in model.reactions.get_by_id(rid).genes:
            curated_gene.setdefault(g.id, set()).add(c)
    cg = model.slim_optimize() or 0.0

    draft, _d, _u = merge_compartments(model, merged_id="c", merged_name="cyt", base_metabolite=base,
                                       drop_single_metabolite_reactions=False)
    draft.objective = biomass
    frames = [load_deeploc(c, compartment_map=cfg["cmap"]).df
              for c in sorted(Path(cfg["data"]).glob(cfg["glob"]))]
    scores = LocalizationScores(pd.concat(frames))
    relocate = [r.id for r in draft.reactions if not r.boundary and r.id != biomass]
    mg = 0.5 * cg
    t0 = time.monotonic()
    prop = assign_compartments(draft, scores, relocate, default_compartment="c",
                                         base_metabolite=base, biomass_reaction=biomass,
                                         min_growth=mg, time_limit=time_limit)
    wall = round(time.monotonic() - t0, 1)

    out = {"organism": key, "reactions": len(model.reactions), "compartments": len(comps),
           "relocate": len(relocate), "chloroplast": "h" in comps, "wall_s": wall,
           "status": prop.status, "certified": prop.certified, "curated_growth": round(cg, 4),
           "transports": len(prop.added_transports)}
    if prop.placements:
        res = {rid: norm(cs[0]) for rid, cs in prop.placements.items() if cs}
        common = set(res) & set(curated_rxn)
        out["reaction_agreement"] = round(sum(res[r] == curated_rxn[r] for r in common) / len(common), 4)
        rg = {g: {norm(c) for c in cs} for g, cs in prop.gene_compartments.items() if cs}
        cg2 = set(rg) & set(curated_gene)
        out["gene_agreement"] = round(sum(bool(rg[g] & curated_gene[g]) for g in cg2) / len(cg2), 4)
        applied = apply_assignment(draft, prop, default_compartment="c", base_metabolite=base)
        applied.objective = biomass
        out["growth"] = round(applied.slim_optimize(error_value=0.0) or 0.0, 4)
        out["blocked_fraction"] = round(len(set(find_blocked_reactions(applied)) & set(relocate))
                                        / len(relocate), 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organisms", nargs="+", default=["humangem", "aracore", "icre1355"])
    ap.add_argument("--time-limit", type=float, default=900.0)
    ap.add_argument("--out", type=Path, default=Path(".research_tmp/certified_multiorg.json"))
    args = ap.parse_args(argv)

    results = {}
    for key in args.organisms:
        print(f"\n=== {key} ===", flush=True)
        try:
            results[key] = run(key, ORGANISMS[key], args.time_limit)
            r = results[key]
            print(f"  {r['reactions']} rxns, {r['compartments']} compartments (chloroplast={r['chloroplast']}); "
                  f"{r['wall_s']}s certified={r['certified']}", flush=True)
            print(f"  reaction agr {r.get('reaction_agreement')}; gene agr {r.get('gene_agreement')}; "
                  f"transports {r['transports']}; growth {r.get('growth')}; blocked {r.get('blocked_fraction')}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            results[key] = {"error": f"{type(exc).__name__}: {exc}"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwritten -> {args.out}  |  DONE")


if __name__ == "__main__":
    main()
