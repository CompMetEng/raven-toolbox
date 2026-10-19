"""Tier 2 at genome scale: the extraction the studies report, re-run nightly.

The Human-GEM, yeast and multi-organism validations in raven-gecko-parity and raven-docs are the
strongest evidence raven-toolbox agrees with MATLAB RAVEN, and they are also the
least protected: they were measured once, by hand, on a model too large for a
free runner's solver licence. Nothing re-checks them.

These tests close that gap on a licensed runner (see
``.github/workflows/parity-nightly.yml``). They need a genome-scale model, given
by ``$HUMAN_GEM_YML``, and a Gurobi licence that will accept it -- the
size-limited licence bundled with pip-installed gurobipy will not.

Unlike the small fixture, this is where an overlap *band* is the honest
assertion rather than exact equality: at this size the MILP has many optima of
equal value, and a bounded solve (``time_limit``, ``mip_gap``) may legitimately
stop at a different one. The band floor is recorded in the baseline alongside
the run that produced it, so it can be tightened as evidence accumulates rather
than guessed now.

This exercises ``run_ftinit`` directly (the single-step MILP, not the staged
``ftinit()`` pipeline) on the raw model with synthetic scores.
``time_limit``/``mip_gap`` are the genome-scale-staged values ``run_ftinit``'s
own docstring recommends, since nothing here goes through ``prep_init_model``'s
rescaling.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from raven_toolbox.init import run_ftinit
from raven_toolbox.io import read_yaml_model

pytestmark = [pytest.mark.parity, pytest.mark.slow]

BASELINE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "parity"
    / "baselines"
    / "init_genome_scale.json"
)

# Bounded so a nightly job cannot run for hours. A solve that hits the limit is
# still useful -- it is compared as a band, not as an exact answer.
TIME_LIMIT_SECONDS = 600.0
MIP_GAP = 0.005


@pytest.fixture(scope="module")
def genome_scale_model():
    path = os.environ.get("HUMAN_GEM_YML")
    if not path:
        pytest.skip(
            "HUMAN_GEM_YML is not set. The nightly workflow downloads "
            "Human-GEM.yml and points at it; to run locally, download "
            "model/Human-GEM.yml from SysBioChalmers/Human-GEM and set the "
            "variable."
        )
    model_path = Path(path)
    if not model_path.is_file():
        pytest.skip(f"HUMAN_GEM_YML={model_path} does not exist")
    return read_yaml_model(model_path)


@pytest.fixture(scope="module")
def extraction(genome_scale_model):
    """Run the extraction once and share it across the assertions below.

    The result is also written to disk. pytest captures stdout for a *passing*
    test, so a print here is invisible in exactly the case that matters most:
    the successful nightly run whose numbers are supposed to become the next
    baseline. The report file survives capture and is uploaded as an artefact.
    """
    model = genome_scale_model
    scores = {
        rxn.id: (10.0 if i % 3 == 0 else (-5.0 if i % 3 == 1 else 1.0))
        for i, rxn in enumerate(model.reactions)
    }
    started = time.perf_counter()
    # allow_excretion=True: unlike run_init, run_ftinit rewards no connectivity by
    # default (prod_weight only applies to metabolomics-detected mets, not used here),
    # so with strict balance and no essential_rxns/tasks a legitimate optimum can be
    # "everything off" regardless of scores. Net production allowed gives positively
    # scored reactions a way to actually carry flux.
    result = run_ftinit(
        model.copy(), scores, allow_excretion=True,
        mip_gap=MIP_GAP, time_limit=TIME_LIMIT_SECONDS,
    )
    elapsed = time.perf_counter() - started
    kept = sorted(r.id for r in result.model.reactions)

    summary = (
        f"kept {len(kept)} of {len(model.reactions)} reactions in "
        f"{elapsed:.0f}s (mip_gap={MIP_GAP}, "
        f"time_limit={TIME_LIMIT_SECONDS:.0f}s)"
    )
    print(f"\ngenome-scale extraction: {summary}")

    out_dir = Path(os.environ.get("RAVEN_PARITY_REPORT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "genome_scale_run.json").write_text(
        json.dumps(
            {
                "kept_reactions": kept,
                "kept": len(kept),
                "total": len(model.reactions),
                "seconds": round(elapsed, 1),
                "mip_gap": MIP_GAP,
                "time_limit": TIME_LIMIT_SECONDS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Genome-scale extraction\n\n{summary}\n")

    return {"kept": kept, "total": len(model.reactions), "seconds": elapsed}


def test_extraction_produces_a_plausible_model(extraction):
    """A sanity floor that holds regardless of which optimum was found."""
    kept, total = extraction["kept"], extraction["total"]

    assert kept, "extraction returned an empty model"
    assert len(kept) < total, (
        "extraction kept every reaction, so it did not discriminate at all"
    )
    fraction = len(kept) / total
    assert 0.02 < fraction < 0.98, (
        f"extraction kept {fraction:.1%} of the model, which is outside any "
        f"plausible range for a context-specific extraction"
    )
    assert len(set(kept)) == len(kept), "extraction returned duplicate reaction ids"


def test_extraction_overlaps_the_recorded_baseline(extraction):
    """Compare as a band: alternate optima are legitimate at this size."""
    if not BASELINE.is_file():
        pytest.skip(
            f"no genome-scale baseline at {BASELINE}. The first successful "
            f"nightly run reports the numbers; record them (with the band floor "
            f"the evidence supports) and commit the file."
        )

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    floor = baseline["jaccard_floor"]
    expected = set(baseline["kept_reactions"])
    got = set(extraction["kept"])

    overlap = len(got & expected) / len(got | expected)
    print(f"genome-scale Jaccard vs baseline: {overlap:.4f} (floor {floor})")

    assert overlap >= floor, (
        f"extraction overlap with the baseline fell to {overlap:.4f}, below the "
        f"recorded floor of {floor} "
        f"(baseline recorded {baseline['recorded']['date']} with "
        f"{baseline['recorded']['solver']}).\n"
        f"  kept now: {len(got)}, baseline: {len(expected)}, "
        f"shared: {len(got & expected)}\n"
        f"A drop this size is not solver noise -- something in the extraction "
        f"changed. If it is intended, re-record the baseline and say why."
    )
