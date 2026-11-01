"""Tests for run_blast / run_diamond / blast_from_table + the tabular parser."""
import shutil

import pandas as pd
import pytest

from raven_toolbox.reconstruction.homology import (
    HIT_COLUMNS,
    blast_from_table,
    run_blast,
    run_diamond,
)
from raven_toolbox.reconstruction.homology.blast import _parse_tabular

_SEQ = (
    "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQPRGRRQPIPKARRPEGRTWAQPGYPWPLYGNEGCGWAGWLLSPRG"
)


def test_parse_tabular_csv():
    text = "tg1,ng1,1e-50,99.0,120,250.0,99.5\ntg2,ng2,0.0,100.0,200,400.0,100.0\n"
    df = _parse_tabular(text, "templ", "org", sep=",")
    assert list(df.columns) == HIT_COLUMNS
    assert df.iloc[0].from_gene == "tg1" and df.iloc[0].to_gene == "ng1"
    assert df.iloc[0].from_id == "templ" and df.iloc[0].to_id == "org"
    assert df.iloc[1].identity == 100.0 and df.iloc[1].align_len == 200


def test_parse_tabular_empty():
    assert _parse_tabular("", "a", "b", sep=",").empty


def test_blast_from_table_dataframe_roundtrip():
    df = pd.DataFrame(
        [["templ", "org", "tg1", "ng1", 0.0, 100.0, 100, 200.0, 100.0]],
        columns=HIT_COLUMNS + ["extra"][:0],  # exactly HIT_COLUMNS
    )
    out = blast_from_table(df)
    assert list(out.columns) == HIT_COLUMNS
    assert len(out) == 1


def test_blast_from_table_csv(tmp_path):
    p = tmp_path / "hits.csv"
    pd.DataFrame(
        [["templ", "org", "tg1", "ng1", 0.0, 100.0, 100, 200.0, 100.0]], columns=HIT_COLUMNS
    ).to_csv(p, index=False)
    out = blast_from_table(p)
    assert out.iloc[0].from_gene == "tg1"


def test_blast_from_table_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        blast_from_table(pd.DataFrame({"from_id": ["x"]}))


def test_blast_from_table_csv_numeric_gene_ids_stay_str(tmp_path):
    """All-numeric gene ids (e.g. Entrez) read as str, so they match model gene ids."""
    p = tmp_path / "hits.csv"
    pd.DataFrame(
        [["templ", "org", 125, 4790, 0.0, 100.0, 100, 200.0, 100.0]], columns=HIT_COLUMNS
    ).to_csv(p, index=False)
    out = blast_from_table(p)
    assert out.iloc[0].from_gene == "125" and out.iloc[0].to_gene == "4790"


@pytest.mark.skipif(
    not (shutil.which("blastp") and shutil.which("makeblastdb")), reason="BLAST+ not installed"
)
def test_run_blast_integration(tmp_path):
    org = tmp_path / "org.faa"
    ref = tmp_path / "templ.faa"
    org.write_text(f">ngene\n{_SEQ}\n")
    ref.write_text(f">tgene\n{_SEQ}\n")  # identical sequence -> strong reciprocal hit

    hits = run_blast("org", org, ["templ"], [ref])
    assert list(hits.columns) == HIT_COLUMNS
    assert not hits.empty
    # both directions present
    assert {("templ", "org"), ("org", "templ")} <= set(zip(hits.from_id, hits.to_id, strict=False))
    # the reciprocal pair tgene<->ngene is found
    fwd = hits[(hits.from_gene == "tgene") & (hits.to_gene == "ngene")]
    assert not fwd.empty


@pytest.mark.parametrize("runner", [run_blast, run_diamond])
def test_threads_default_to_every_available_cpu(runner, monkeypatch):
    """As RAVEN's getBlast/getDiamond, which pass feature('numcores')'s assigned count."""
    import raven_toolbox.reconstruction.homology.blast as blast

    calls = []
    monkeypatch.setattr(blast, "_available_cpus", lambda: 7)
    monkeypatch.setattr(blast, "resolve_binary", lambda name, binary=None: name)
    monkeypatch.setattr(blast, "_run", lambda cmd: calls.append(cmd) or "")
    runner("org", "org.faa", ["m"], ["m.faa"])
    searches = [c for c in calls if c[0] in ("blastp", "diamond") and "makedb" not in c]
    assert searches
    for cmd in searches:
        flag = "-num_threads" if cmd[0] == "blastp" else "--threads"
        assert cmd[cmd.index(flag) + 1] == "7"
