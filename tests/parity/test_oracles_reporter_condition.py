"""Tier 1: reporter-metabolite subsets and empty condition files against recorded MATLAB answers.

The oracles ``reporter_fold_changes`` and ``empty_condition`` are written by
``scripts/parity/generate_oracles.m``; each test skips while its oracle is absent.
"""
from __future__ import annotations

import pytest

from raven_toolbox.analysis import reporter_metabolites
from raven_toolbox.conditions import apply_condition, load_condition
from raven_toolbox.io import read_yaml_model

pytestmark = pytest.mark.parity

_TEST_NAMES = {"all": "all", "only up": "up", "only down": "down"}


@pytest.fixture
def tiny(parity_fixture_dir):
    return read_yaml_model(parity_fixture_dir / "tiny.yml")


def test_reporter_up_and_down_subsets_match_matlab(tiny, oracle):
    """One fold change per gene: the up (>= 0) and down (< 0) subsets score like RAVEN."""
    expected = oracle("reporter_fold_changes")
    pvalues = dict(zip(expected["genes"], expected["pvalues"], strict=True))
    fold_changes = dict(zip(expected["genes"], expected["foldChanges"], strict=True))

    ours = {r.test: r.table for r in reporter_metabolites(
        tiny, pvalues, gene_fold_changes=fold_changes)}

    assert {_TEST_NAMES[t["test"]] for t in expected["tests"]} == set(ours)
    for entry in expected["tests"]:
        table = ours[_TEST_NAMES[entry["test"]]].set_index("metabolite")
        mets = entry["mets"] if isinstance(entry["mets"], list) else [entry["mets"]]
        z_scores = entry["metZScores"]
        z_scores = z_scores if isinstance(z_scores, list) else [z_scores]
        assert set(table.index) == set(mets), entry["test"]
        for met, z in zip(mets, z_scores, strict=True):
            assert table.loc[met, "z_score"] == pytest.approx(z, abs=1e-9), (entry["test"], met)


def test_empty_condition_file_matches_matlab(tiny, oracle, tmp_path):
    """RAVEN reads an empty condition file as an empty struct and changes nothing."""
    expected = oracle("empty_condition")
    path = tmp_path / "empty.yml"
    path.write_text("")

    cond = load_condition(path)
    assert len(cond) == expected["nFields"]

    before = {r.id: (r.lower_bound, r.upper_bound) for r in tiny.reactions}
    apply_condition(tiny, path)
    changed = {r.id: (r.lower_bound, r.upper_bound) for r in tiny.reactions} != before
    assert changed == expected["modelChanged"]
