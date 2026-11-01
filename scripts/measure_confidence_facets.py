#!/usr/bin/env python3
"""Regenerate the confidence-facet tables in ``raven-gecko-parity/docs/localization/confidence_tracking.md``.

Runs the shipped ``equation`` and ``gene_association`` scorers over a model and prints:

1. the exemption counts (which facets even apply, and why),
2. the ``facet x basis x score`` distribution -- the table quoted in the study doc,
3. the review queue's head (``confidence_report``, lowest confidence first),
4. an **independent check** of the gene rubric: the rubric reads only the GPR and a ``pubmed``
   annotation, never the model's own Thiele-Palsson ``Confidence Level`` note, which therefore serves as
   held-out ground truth rather than as an input.

The doc's numbers are this script's output, not prose arithmetic. ASCII-only output.
"""
from __future__ import annotations

import argparse
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
import cobra  # noqa: E402

from raven_toolbox.confidence import (  # noqa: E402
    confidence_report,
    equation_exempt,
    facet_summary,
    gene_association_exempt,
    get_confidence,
    score_equation_confidence,
    score_gene_association_confidence,
)


def _recorded(rxn) -> int | None:
    try:
        return int(float(str((rxn.notes or {}).get("Confidence Level"))))
    except (TypeError, ValueError):
        return None


def _load_model(path: Path):
    """Load a model by extension: SBML (.xml/.sbml), cobra YAML (.yml/.yaml) or cobra JSON (.json)."""
    suffix = path.suffix.lower()
    if suffix in (".yml", ".yaml"):
        return cobra.io.load_yaml_model(str(path))
    if suffix == ".json":
        return cobra.io.load_json_model(str(path))
    return cobra.io.read_sbml_model(str(path))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, nargs="?",
                    default=Path("C:/Work/GitHub/yeast-GEM/model/yeast-GEM.xml"))
    ap.add_argument("--head", type=int, default=12, help="rows of the review queue to print")
    args = ap.parse_args(argv)

    model = _load_model(args.model)
    total = len(model.reactions)
    print(f"=== {model.id or args.model.name}: {total} reactions, {len(model.metabolites)} metabolites ===")

    eq_ex = Counter(r for r in map(equation_exempt, model.reactions) if r)
    ga_ex = Counter(r for r in map(gene_association_exempt, model.reactions) if r)
    print(f"\nequation exempt        {sum(eq_ex.values()):5d}  {dict(eq_ex)}")
    print(f"gene_association exempt {sum(ga_ex.values()):4d}  {dict(ga_ex)}")

    n_eq = score_equation_confidence(model)
    n_ga = score_gene_association_confidence(model)
    print(f"\nscored: equation {n_eq}, gene_association {n_ga}")

    print("\n=== facet x basis distribution ===")
    print(facet_summary(model).to_string(index=False))

    rep = confidence_report(model)
    print(f"\n=== review queue ({len(rep)} rows), lowest confidence first ===")
    print(rep.head(args.head).to_string(index=False))
    zeros = list(rep[rep["overall"] == 0.0]["reaction"])
    print(f"\noverall == 0.0 (evidence contradicts the model): {len(zeros)} -> {zeros}")

    print("\n=== independent check: our gene rubric vs the model's own 'Confidence Level' note ===")
    cross: Counter = Counter()
    for rxn in model.reactions:
        entry = get_confidence(rxn).facets.get("gene_association")
        if entry is not None:
            cross[(entry.basis, _recorded(rxn))] += 1
    if not any(rec is not None for _, rec in cross):
        print("  (this model records no Confidence Level notes)")
        return
    print(f"  {'our basis':18s} {'recorded TP':>11s}      n")
    for (basis, rec), n in sorted(cross.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        print(f"  {basis:18s} {str(rec):>11s}  {n:5d}")


if __name__ == "__main__":
    main()
