#!/usr/bin/env python3
"""Comparison 1 — can our approach replicate curated yeast-GEM's compartmentalisation?

A controlled, same-content test that calls **only the shipped public API** — no benchmark-specific
draft construction, annotation wrappers, or metric reimplementations:

1. `manipulation.merge_compartments` flattens curated yeast-GEM to a single compartment (holding the
   reaction/gene content exactly fixed; `base_metabolite=lambda m: m.name` because yeast-GEM keys the
   same species to a different `s_####` id per compartment).
2. `load_deeploc` (+ `DEFAULT_COMPARTMENT_MAP`) reads the committed yeast DeepLoc predictions;
   `annotate_proteome` annotates the yeast proteome's transporters; `evidence_aware_transport_cost`
   turns both into the per-metabolite `transport_cost` mapping.
3. `assign_compartments` re-places every internal reaction on that flattened draft; `apply_assignment`
   materialises the result.
4. The recovered compartmentalisation is scored against curated yeast-GEM: reaction- and gene-level
   compartment agreement, added-transport count, growth, and — crucially — **functional connectivity**
   via cobra's own `find_blocked_reactions` (what fraction of re-placed reactions can actually carry
   flux, so a placement that looks right but strands the reaction in a disconnected compartment is
   counted honestly, not hidden behind a positive growth number).

Curated yeast-GEM distinguishes membrane sub-compartments (erm/mm/vm/gm) that DeepLoc does not, so for a
fair head-to-head the curated truth is collapsed onto its parent organelle (the granularity the scores
can express) before agreement is measured.

Usage::

    python scripts/benchmark_replicate_yeast_gem.py \\
        --yeast-gem C:/Work/GitHub/yeast-GEM/model/yeast-GEM.xml

ASCII-only output.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import cobra  # noqa: E402
import pandas as pd  # noqa: E402
from cobra.flux_analysis import find_blocked_reactions  # noqa: E402

from raven_toolbox.comparison import compare_models  # noqa: E402
from raven_toolbox.localization import (  # noqa: E402
    DEFAULT_COMPARTMENT_MAP,
    LocalizationScores,
    SubstrateOntology,
    annotate_proteome,
    apply_assignment,
    assign_compartments,
    default_substrate_of,
    evidence_aware_transport_cost,
    load_deeploc,
)
from raven_toolbox.manipulation.compartments import merge_compartments  # noqa: E402


def _name(m: cobra.Metabolite) -> str:
    """Compartment-agnostic key for yeast-GEM (same species => same name, different `s_####` id)."""
    return m.name


# Curated yeast-GEM resolves organelle membranes as their own compartments; DeepLoc does not, so
# collapse each to its parent organelle before measuring agreement (the resolution the scores have).
_PARENT = {"erm": "er", "mm": "m", "vm": "v", "gm": "g"}


def _norm(compartment: str) -> str:
    return _PARENT.get(compartment, compartment)


def _sole_compartment(rxn: cobra.Reaction) -> str | None:
    comps = {m.compartment for m in rxn.metabolites}
    return next(iter(comps)) if len(comps) == 1 else None


def load_yeast_scores(data_dir: Path) -> LocalizationScores:
    """The committed yeast DeepLoc predictions, mapped onto yeast-GEM's own compartment codes."""
    frames = [load_deeploc(c, compartment_map=DEFAULT_COMPARTMENT_MAP).df
              for c in sorted(data_dir.glob("yeast-GEM_deeploc_*.csv"))]
    return LocalizationScores(pd.concat(frames))


def annotate_yeast_proteome(data_dir: Path):
    """`annotate_proteome` over the committed yeast proteome FASTA (the real Pfam/TCDB back-end)."""
    import tempfile

    fastas = sorted(data_dir.glob("yeast-GEM_proteins_*.fasta"))
    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as tf:
        tf.write("".join(p.read_text() for p in fastas))
        proteome = Path(tf.name)
    ann = annotate_proteome(proteome, threads=4)
    proteome.unlink(missing_ok=True)
    return ann


def curated_reaction_compartments(model: cobra.Model) -> dict[str, str]:
    """Curated single-compartment, non-boundary reaction -> its (parent-normalised) compartment."""
    out = {}
    for r in model.reactions:
        if r.boundary:
            continue
        c = _sole_compartment(r)
        if c is not None:
            out[r.id] = _norm(c)
    return out


def curated_gene_compartments(model: cobra.Model) -> dict[str, set[str]]:
    """Curated gene -> the (parent-normalised) compartments of the single-compartment reactions it
    catalyses."""
    out: dict[str, set[str]] = {}
    for rid, comp in curated_reaction_compartments(model).items():
        for g in model.reactions.get_by_id(rid).genes:
            out.setdefault(g.id, set()).add(comp)
    return out


def build_draft(yeast: cobra.Model) -> tuple[cobra.Model, str]:
    """Flatten curated yeast-GEM to one compartment via the real `merge_compartments`, and return the
    reaction to use as the biomass/growth objective on the flattened draft.

    `drop_single_metabolite_reactions=False` (the default, passed explicitly) is essential here:
    exchange reactions are single-metabolite, so `True` would delete every nutrient uptake and the
    flattened draft couldn't grow (it keeps the collapsed one-metabolite transports too, but those
    are negligible; the only material effect is preserving the medium). With them kept, yeast-GEM's objective
    ``r_2111`` "growth" (a ``biomass ->`` drain) survives directly. The `add_boundary` branch is a
    defensive fallback for models whose objective still doesn't survive: re-add an equivalent demand
    sink on the merged biomass metabolite so its producer isn't mass-balance-blocked."""
    obj = next(r for r in yeast.reactions if r.objective_coefficient != 0)
    obj_substrates = [m for m, c in obj.metabolites.items() if c < 0]
    # drop_single_metabolite_reactions=False preserves the exchange reactions (single-metabolite), so
    # the flattened draft can still take up nutrients and grow (see above). deduplicate_reactions is
    # left at its default True: two isozyme copies (one in c, one in m) become identical after
    # flattening and one is dropped, so ~1900 of the 2362 curated single-compartment reactions remain
    # to be placed and compared. This does not bias the agreement *rate* (only the sample size) and
    # keeps the MILP small enough for the hyperparameter sweeps to be practical.
    draft, _deleted, _dupes = merge_compartments(
        yeast, merged_id="c", merged_name="cytoplasm", base_metabolite=_name,
        drop_single_metabolite_reactions=False)
    if obj.id in draft.reactions:
        biomass_id = obj.id
    else:
        if len(obj_substrates) != 1:
            raise SystemExit(f"objective {obj.id} was dropped by the merge and has "
                             f"{len(obj_substrates)} substrates, so its biomass sink is ambiguous")
        bm = next((m for m in draft.metabolites if m.name == obj_substrates[0].name), None)
        if bm is None:
            raise SystemExit(f"biomass metabolite {obj_substrates[0].name!r} not found after merge")
        biomass_id = draft.add_boundary(bm, type="demand").id
    draft.objective = biomass_id
    return draft, biomass_id


def run_once(draft, biomass_id, scores, relocate, cost, *, min_growth, time_limit,
             multi_localization):
    """One `assign_compartments` -> `apply_assignment` pass. Returns (proposal, result, wall)."""
    t0 = time.monotonic()
    proposal = assign_compartments(
        draft, scores, relocate, transport_cost=cost, default_compartment="c",
        base_metabolite=_name, biomass_reaction=biomass_id, min_growth=min_growth,
        time_limit=time_limit, multi_localization=multi_localization)
    wall = time.monotonic() - t0
    result = apply_assignment(draft, proposal, default_compartment="c", base_metabolite=_name)
    result.objective = biomass_id
    return proposal, result, wall


def compute_metrics(result, proposal, relocate, curated_rxn, curated_gene, curated_transports, wall):
    # reaction-level agreement over reactions single-compartment in BOTH result and curated
    res_comp = {rid: _norm(cs[0]) for rid, cs in proposal.placements.items() if cs}
    common = set(res_comp) & set(curated_rxn)
    rxn_ok = sum(1 for rid in common if res_comp[rid] == curated_rxn[rid])
    rxn_rate = rxn_ok / len(common) if common else float("nan")

    # gene-level agreement
    res_gene = {g: {_norm(c) for c in cs} for g, cs in proposal.gene_compartments.items() if cs}
    common_g = set(res_gene) & set(curated_gene)
    gene_ok = sum(1 for g in common_g if res_gene[g] & curated_gene[g])
    gene_rate = gene_ok / len(common_g) if common_g else float("nan")

    growth = result.slim_optimize()
    # functional connectivity: of the re-placed reactions, how many can carry flux at all?
    blocked = set(find_blocked_reactions(result))
    relocated_blocked = sorted(set(relocate) & blocked)
    blocked_rate = len(relocated_blocked) / len(relocate) if relocate else float("nan")

    print(f"  status={proposal.status}  objective={proposal.objective:.3f}  solve={wall:.1f}s")
    print(f"  growth (materialised): {growth}")
    print(f"  reaction agreement {rxn_rate:.1%} ({rxn_ok}/{len(common)}); "
          f"gene agreement {gene_rate:.1%} ({gene_ok}/{len(common_g)})")
    print(f"  transports added: {len(proposal.added_transports)}  (curated inter-compartment: "
          f"{curated_transports})")
    print(f"  blocked re-placed reactions: {blocked_rate:.1%} ({len(relocated_blocked)}/{len(relocate)})")
    print(f"  unplaced (no scored gene): {len(proposal.unplaced_reactions)}")

    return {
        "status": proposal.status, "objective": proposal.objective, "solve_seconds": wall,
        "growth": growth, "curated_growth_note": "curated yeast-GEM grows 0.0809/h",
        "reaction_agreement_rate": rxn_rate, "reaction_agreement_n": len(common),
        "gene_agreement_rate": gene_rate, "gene_agreement_n": len(common_g),
        "transports_added": len(proposal.added_transports), "curated_transports": curated_transports,
        "blocked_replaced_rate": blocked_rate, "blocked_replaced_n": len(relocated_blocked),
        "relocate_n": len(relocate), "unplaced_reactions": len(proposal.unplaced_reactions),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yeast-gem", type=Path,
                    default=Path("C:/Work/GitHub/yeast-GEM/model/yeast-GEM.xml"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/deeploc"))
    ap.add_argument("--min-growth-fraction", type=float, default=0.5,
                    help="growth floor as a fraction of curated yeast-GEM's own growth (default 0.5)")
    ap.add_argument("--sibling-weight", type=float, default=0.0)
    ap.add_argument("--base-cost", type=float, default=0.5)
    ap.add_argument("--multi-localization", action="store_true",
                    help="allow flux-active multi-compartment placement (else mono-localisation)")
    ap.add_argument("--time-limit", type=float, default=900.0, help="per-solve seconds")
    ap.add_argument("--out", type=Path,
                    default=Path(".research_tmp/replicate_yeast_gem_results.json"))
    args = ap.parse_args(argv)

    yeast = cobra.io.read_sbml_model(str(args.yeast_gem))
    curated_growth = yeast.slim_optimize()
    print(f"curated yeast-GEM: {len(yeast.reactions)} reactions, {len(yeast.genes)} genes, "
          f"{len(yeast.compartments)} compartments, growth={curated_growth:.4f}")
    curated_rxn = curated_reaction_compartments(yeast)
    curated_gene = curated_gene_compartments(yeast)
    curated_transports = sum(1 for r in yeast.reactions
                             if not r.boundary and _sole_compartment(r) is None)
    print(f"curated single-compartment reactions: {len(curated_rxn)}; "
          f"inter-compartment (transport) reactions: {curated_transports}")

    draft, biomass_id = build_draft(yeast)
    print(f"flattened draft (merge_compartments): {len(draft.reactions)} reactions, "
          f"{len(draft.metabolites)} metabolites; draft growth={draft.slim_optimize():.4f} "
          f"(objective {biomass_id})")

    scores = load_yeast_scores(args.data_dir)
    print(f"DeepLoc scores: {len(scores.df)} genes x {list(scores.df.columns)} compartments")
    ann = annotate_yeast_proteome(args.data_dir)
    print(f"annotate_proteome: {len(ann)} transporter genes")
    gene_comps = {g: {scores.df.loc[g].astype(float).idxmax()} for g in scores.df.index if g in ann}
    cost = evidence_aware_transport_cost(
        draft, ann, gene_comps, substrate_of=default_substrate_of, ontology=SubstrateOntology.load(),
        sibling_weight=args.sibling_weight, base_cost=args.base_cost, base_metabolite=_name)

    relocate = [r.id for r in draft.reactions if not r.boundary and r.id != biomass_id]
    print(f"relocate set (internal, non-biomass): {len(relocate)} reactions")

    min_growth = args.min_growth_fraction * curated_growth
    print(f"\n=== assign_compartments (min_growth={min_growth:.4f} = "
          f"{args.min_growth_fraction:g} x curated; multi_localization={args.multi_localization}) ===")
    proposal, result, wall = run_once(
        draft, biomass_id, scores, relocate, cost, min_growth=min_growth,
        time_limit=args.time_limit, multi_localization=args.multi_localization)

    if not proposal.placements:
        print(f"  status={proposal.status}: NO incumbent found in {wall:.0f}s "
              "(this MILP was too hard for the budget -- e.g. multi_localization at genome scale)")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"status": proposal.status, "no_incumbent": True, "solve_seconds": wall,
             "multi_localization": args.multi_localization,
             "min_growth_fraction": args.min_growth_fraction}, indent=2, default=str))
        print(f"\nwritten -> {args.out}")
        return

    metrics = compute_metrics(result, proposal, relocate, curated_rxn, curated_gene,
                              curated_transports, wall)

    # structural sanity: the re-placed model must still carry the same reaction set as curated
    # (content is fixed by construction, so this should be ~1.0 — a guard against merge/apply loss).
    cmp = compare_models([yeast, result])
    metrics["structural_similarity"] = float(cmp.similarity.iloc[0, 1])
    print(f"  structural similarity vs curated: {metrics['structural_similarity']:.3f}")

    metrics.update({"min_growth": min_growth, "min_growth_fraction": args.min_growth_fraction,
                    "sibling_weight": args.sibling_weight, "base_cost": args.base_cost,
                    "multi_localization": args.multi_localization})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nwritten -> {args.out}")

    # Persist the raw placements (yeast-GEM compartment codes) so the CarveFungi comparison can reuse
    # our per-gene decision without re-running this genome-scale MILP. Derived from --out so a sweep
    # run doesn't clobber the main run's placements.
    placements_out = args.out.with_name(args.out.stem + "_placements.json")
    placements_out.write_text(json.dumps({
        "gene_compartments": {g: cs for g, cs in proposal.gene_compartments.items() if cs},
        "reaction_placements": {rid: cs for rid, cs in proposal.placements.items() if cs},
    }, indent=2, default=str))
    print(f"written -> {placements_out}")


if __name__ == "__main__":
    main()
