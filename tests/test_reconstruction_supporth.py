"""Tests for reading SuppOrth output (hits_from_supporth) and running it (run_supporth)."""
import json
import subprocess
import sys
from pathlib import Path

import cobra
import pandas as pd
import pytest

from raven_toolbox.manipulation import add_reactions_from_equations
from raven_toolbox.reconstruction.homology import (
    HIT_COLUMNS,
    get_model_from_homology,
    hits_from_supporth,
    run_supporth,
)
from raven_toolbox.reconstruction.homology import supporth as supporth_mod

HEADER = ("Species1\tSpecies2\tProtein1\tProtein2\tLv_support\tSupportedBy\t"
          "orthofinder\tsonicparanoid\tbroccoli\tIdentity\te_val\tBitscore")

# Input order templateGEM, bug, other: bug is Species2 against templateGEM and
# Species1 against other, as SuppOrth orients pairs by input order.
ROWS = [
    "templateGEM\tbug\ttg1\tng1\t3\tF,S,B\t1\t1\t1\t85.2\t1e-150\t512.0",
    "templateGEM\tbug\ttg2\tng2\t2\tF,S\t1\t1\t0\t55.0\t1e-80\t300.0",
    "templateGEM\tbug\ttg3\tng2\t1\tB\t0\t0\t1\t\t\t",
    "templateGEM\tbug\ttg4\tsp|P1|NG4_BUG\t1\tF\t1\t0\t0\t30.0\t1e-20\t120.0",
    "templateGEM\tbug\ttg5\tNA\t1\tS\t0\t1\t0\t82.0\t1e-90\t280.0",
    "templateGEM\tother\ttg1\to1\t3\tF,S,B\t1\t1\t1\t90.0\t0.0\t600.0",
    "bug\tother\tng1\to1\t2\tF,B\t1\t0\t1\t70.0\t1e-100\t400.0",
    "bug\tother\t0123\to2\t1\tF\t1\t0\t0\t81.0\t1e-50\t200.0",
]


def _write(tmp_path, rows=ROWS, header=HEADER, name="suppOrth.tsv"):
    path = tmp_path / name
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def _pairs(hits, from_id, to_id):
    sel = hits[(hits.from_id == from_id) & (hits.to_id == to_id)]
    return set(zip(sel.from_gene, sel.to_gene, strict=True))


def _template():
    m = cobra.Model("templateGEM")
    m.compartments = {"c": "cytoplasm"}
    m.add_metabolites([cobra.Metabolite(x, name=x.upper(), compartment="c") for x in ("a", "b", "d")])
    add_reactions_from_equations(
        m,
        [
            {"id": "R_single", "equation": "a --> b", "gene_reaction_rule": "tg1"},
            {"id": "R_iso", "equation": "b --> d", "gene_reaction_rule": "tg2 or tg3"},
            {"id": "R_cplx", "equation": "a --> d", "gene_reaction_rule": "tg4 and tg5"},
        ],
    )
    return m


# --- reading -----------------------------------------------------------------

def test_columns_directions_and_source(tmp_path):
    hits = hits_from_supporth(_write(tmp_path), "bug")
    assert list(hits.columns[: len(HIT_COLUMNS)]) == HIT_COLUMNS
    assert list(hits.columns[len(HIT_COLUMNS):]) == [
        "support", "supported_by", "identity_raw", "evalue_raw", "bitscore_raw"]
    assert hits.attrs["source"] == "SuppOrth"
    fwd = _pairs(hits, "bug", "templateGEM")
    rev = _pairs(hits, "templateGEM", "bug")
    assert {(b, a) for a, b in fwd} == rev
    assert ("ng1", "tg1") in fwd


def test_both_orientations_and_template_pairs_ignored(tmp_path):
    hits = hits_from_supporth(_write(tmp_path), "bug")
    assert _pairs(hits, "bug", "other") == {("ng1", "o1"), ("0123", "o2")}
    # templateGEM-other pairs do not involve the organism.
    assert _pairs(hits, "templateGEM", "other") == set()
    assert set(hits.from_id) == {"bug", "templateGEM", "other"}


def test_placeholder_metrics_and_raw_values(tmp_path):
    hits = hits_from_supporth(_write(tmp_path), "bug")
    assert (hits.evalue == 0).all() and (hits.identity == 100).all()
    assert (hits.align_len == 1000).all() and (hits.ppos == 100).all()
    row = hits[(hits.from_gene == "ng1") & (hits.to_gene == "tg1")].iloc[0]
    assert row.bitscore == 512.0 and row.identity_raw == 85.2 and row.support == 3
    assert row.supported_by == "F,S,B"
    no_hit = hits[(hits.from_gene == "ng2") & (hits.to_gene == "tg3")].iloc[0]
    assert pd.isna(no_hit.identity_raw) and pd.isna(no_hit.evalue_raw)
    assert no_hit.bitscore == 0.0


def test_ids_kept_as_strings(tmp_path):
    hits = hits_from_supporth(_write(tmp_path), "bug")
    genes = set(hits.from_gene)
    assert {"NA", "0123", "sp|P1|NG4_BUG"} <= genes


def test_dataframe_input(tmp_path):
    df = pd.read_csv(_write(tmp_path), sep="\t", keep_default_na=False,
                     na_values={"Identity": [""], "e_val": [""], "Bitscore": [""]})
    hits = hits_from_supporth(df, "bug")
    assert ("ng1", "tg1") in _pairs(hits, "bug", "templateGEM")


def test_without_alignment_columns(tmp_path):
    header = "\t".join(HEADER.split("\t")[:9])
    rows = ["\t".join(r.split("\t")[:9]) for r in ROWS]
    hits = hits_from_supporth(_write(tmp_path, rows, header), "bug")
    assert hits.identity_raw.isna().all()
    assert (hits.bitscore == 0).all()


# --- filters -----------------------------------------------------------------

def test_min_support(tmp_path):
    hits = hits_from_supporth(_write(tmp_path), "bug", min_support=2)
    assert _pairs(hits, "bug", "templateGEM") == {("ng1", "tg1"), ("ng2", "tg2")}


def test_require_tools(tmp_path):
    hits = hits_from_supporth(_write(tmp_path), "bug", require_tools=["broccoli"])
    assert _pairs(hits, "bug", "templateGEM") == {("ng1", "tg1"), ("ng2", "tg3")}
    with pytest.raises(ValueError, match="require_tools"):
        hits_from_supporth(_write(tmp_path), "bug", require_tools=["oma"])


def test_identity_and_evalue_filters_drop_missing(tmp_path):
    hits = hits_from_supporth(_write(tmp_path), "bug", min_identity=50)
    pairs = _pairs(hits, "bug", "templateGEM")
    assert ("ng2", "tg3") not in pairs  # no alignment hit
    assert ("sp|P1|NG4_BUG", "tg4") not in pairs  # 30 %
    assert ("NA", "tg5") in pairs
    hits = hits_from_supporth(_write(tmp_path), "bug", max_evalue=1e-85)
    assert _pairs(hits, "bug", "templateGEM") == {("ng1", "tg1"), ("NA", "tg5")}


# --- species labels ------------------------------------------------------------

def test_species_to_model(tmp_path):
    mapping = {"templateGEM": "iTemplate", "bug": "bugGEM", "other": None}
    hits = hits_from_supporth(_write(tmp_path), "bug", mapping)
    assert set(hits.from_id) == {"bugGEM", "iTemplate"}


def test_species_to_model_missing_label(tmp_path):
    with pytest.raises(ValueError, match="no entry for \\['other'\\]"):
        hits_from_supporth(_write(tmp_path), "bug", {"templateGEM": "t", "bug": "b"})


def test_unknown_organism(tmp_path):
    with pytest.raises(ValueError, match="not a species in the table"):
        hits_from_supporth(_write(tmp_path), "yeast")


def test_missing_columns(tmp_path):
    path = _write(tmp_path, ["g1\tg2\t1"], "GeneID1\tGeneID2\tLv_support")
    with pytest.raises(ValueError, match="missing columns"):
        hits_from_supporth(path, "bug")


def test_empty_after_filtering(tmp_path):
    with pytest.raises(ValueError, match="No SuppOrth pairs"):
        hits_from_supporth(_write(tmp_path), "bug", min_support=4)


# --- protein to gene -----------------------------------------------------------

ISOFORM_ROWS = [
    "templateGEM\tbug\ttg1\tng1.1\t1\tF\t1\t0\t0\t60.0\t1e-60\t250.0",
    "templateGEM\tbug\ttg1\tng1.2\t3\tF,S,B\t1\t1\t1\t75.0\t1e-90\t350.0",
    "templateGEM\tbug\ttg2\tng2.1\t2\tS,B\t0\t1\t1\t50.0\t1e-40\t150.0",
]
GENE_MAP = {"ng1.1": "ng1", "ng1.2": "ng1", "ng2.1": "ng2"}


def _check_isoforms(hits):
    assert _pairs(hits, "bug", "templateGEM") == {("ng1", "tg1"), ("ng2", "tg2")}
    row = hits[(hits.from_gene == "ng1") & (hits.from_id == "bug")].iloc[0]
    assert row.support == 3
    assert row.supported_by == "F,S,B"
    assert row.identity_raw == 75.0 and row.evalue_raw == 1e-90 and row.bitscore == 350.0


def test_protein_to_gene_dict(tmp_path):
    with pytest.warns(UserWarning, match="2 of 5 proteins"):  # tg1, tg2 are unmapped
        hits = hits_from_supporth(_write(tmp_path, ISOFORM_ROWS), "bug", protein_to_gene=GENE_MAP)
    _check_isoforms(hits)


def test_protein_to_gene_files(tmp_path):
    full = {**GENE_MAP, "tg1": "tg1", "tg2": "tg2"}
    tsv = tmp_path / "map.tsv"
    tsv.write_text("# protein\tgene\n" + "".join(f"{k}\t{v}\n" for k, v in full.items()),
                   encoding="utf-8")
    js = tmp_path / "map.json"
    js.write_text(json.dumps(full), encoding="utf-8")
    table = _write(tmp_path, ISOFORM_ROWS)
    for gene_map in (tsv, js):
        _check_isoforms(hits_from_supporth(table, "bug", protein_to_gene=gene_map))


# --- through get_model_from_homology -------------------------------------------------

def test_model_from_supporth_hits(tmp_path):
    hits = hits_from_supporth(_write(tmp_path), "bug", min_support=2)
    res = get_model_from_homology([_template()], hits, "bug")
    rxns = {r.id: r.gene_reaction_rule for r in res.model.reactions}
    assert rxns == {"R_single": "ng1", "R_iso": "ng2"}


def test_support_threshold_changes_model(tmp_path):
    hits = hits_from_supporth(_write(tmp_path), "bug", min_support=3)
    res = get_model_from_homology([_template()], hits, "bug")
    assert {r.id for r in res.model.reactions} == {"R_single"}


def test_one_to_many_and_strictness(tmp_path):
    hits = hits_from_supporth(_write(tmp_path), "bug", protein_to_gene={"sp|P1|NG4_BUG": "ng4"})
    for strictness in (1, 2, 3):
        res = get_model_from_homology([_template()], hits, "bug", strictness=strictness)
        assert res.model.reactions.get_by_id("R_iso").gene_reaction_rule == "ng2"


def test_gene_ids_cobra_splits_rejected(tmp_path):
    hits = hits_from_supporth(_write(tmp_path), "bug")
    with pytest.raises(ValueError, match=r"cannot be written into a GPR.*sp\|P1\|NG4_BUG"):
        get_model_from_homology([_template()], hits, "bug")


def test_review_identity_warns(tmp_path):
    hits = hits_from_supporth(_write(tmp_path), "bug", min_support=2)
    with pytest.warns(UserWarning, match="review_identity has no effect"):
        res = get_model_from_homology([_template()], hits, "bug", review_identity=25)
    assert res.candidates.empty


def test_pickle_map_rejected(tmp_path):
    with pytest.raises(ValueError, match="pickle"):
        hits_from_supporth(_write(tmp_path), "bug", protein_to_gene=tmp_path / "map.pkl")


def test_supported_by_order(tmp_path):
    rows = [
        "templateGEM\tbug\ttg1\tng1.1\t1\tB\t0\t0\t1\t60.0\t1e-60\t250.0",
        "templateGEM\tbug\ttg1\tng1.2\t1\tF\t1\t0\t0\t75.0\t1e-90\t350.0",
    ]
    hits = hits_from_supporth(_write(tmp_path, rows), "bug",
                              protein_to_gene={"ng1.1": "ng1", "ng1.2": "ng1", "tg1": "tg1"})
    assert set(hits.supported_by) == {"F,B"}


# --- run_supporth ----------------------------------------------------------------

def _fasta(path, ids):
    path.write_text("".join(f">{i} some description\nMKV\n" for i in ids), encoding="utf-8")
    return path


@pytest.fixture
def fastas(tmp_path):
    return {
        "bug": _fasta(tmp_path / "bug.faa", ["ng1", "ng2"]),
        "templateGEM": _fasta(tmp_path / "t.faa", ["tg1", "tg2", "tg3"]),
        "other": _fasta(tmp_path / "o.faa", ["o1", "o2"]),
    }


@pytest.fixture
def fake_run(monkeypatch):
    """Replace the subprocess call: record the command, write a table, return a code."""
    monkeypatch.setattr(sys, "platform", "linux")
    calls = {"returncode": 0, "rows": ROWS}

    def run(cmd, capture_output, text):
        calls["cmd"] = cmd
        outdir = Path(cmd[cmd.index("-o") + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        _write(outdir, calls["rows"])
        return subprocess.CompletedProcess(cmd, calls["returncode"], "log line",
                                           "stage broccoli failed")

    monkeypatch.setattr(supporth_mod.subprocess, "run", run)
    return calls


def test_run_supporth_command(fake_run, fastas, tmp_path):
    hits = run_supporth("bug", fastas["bug"], ["templateGEM", "other"],
                        [fastas["templateGEM"], fastas["other"]], tmp_path / "out",
                        tools=["orthofinder"], threads=2, extra_args=["--resume"],
                        supporth="/opt/suppOrth", min_support=2)
    cmd = fake_run["cmd"]
    assert cmd[:5] == ["/opt/suppOrth", "run", str(fastas["bug"]), str(fastas["templateGEM"]),
                       str(fastas["other"])]
    assert cmd[cmd.index("--labels") + 1] == "bug,templateGEM,other"
    assert cmd[cmd.index("--tools") + 1] == "orthofinder"
    assert cmd[cmd.index("--supportLv") + 1] == "1"  # one tool: support cannot exceed 1
    assert cmd[cmd.index("-t") + 1] == "2"
    assert cmd[-1] == "--resume"
    assert _pairs(hits, "bug", "templateGEM") == {("ng1", "tg1"), ("ng2", "tg2")}
    assert not (tmp_path / "out" / "prefixed_inputs").exists()


def test_run_supporth_threads_default_to_every_available_cpu(fake_run, fastas, tmp_path, monkeypatch):
    """As RAVEN's runSuppOrth, which passes feature('numcores')'s assigned count."""
    import os

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(7)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    run_supporth("bug", fastas["bug"], "templateGEM", fastas["templateGEM"], tmp_path / "out",
                 supporth="suppOrth")
    assert fake_run["cmd"][fake_run["cmd"].index("-t") + 1] == "7"


def test_run_supporth_default_support(fake_run, fastas, tmp_path):
    run_supporth("bug", fastas["bug"], "templateGEM", fastas["templateGEM"], tmp_path / "out",
                 supporth="suppOrth")
    cmd = fake_run["cmd"]
    assert cmd[cmd.index("--labels") + 1] == "bug,templateGEM"
    assert cmd[cmd.index("--supportLv") + 1] == "3"
    assert cmd[cmd.index("--tools") + 1] == "orthofinder,sonicparanoid,broccoli"


def test_run_supporth_tools_string(fake_run, fastas, tmp_path):
    run_supporth("bug", fastas["bug"], ["templateGEM"], [fastas["templateGEM"]],
                 tmp_path / "out", tools="orthofinder, broccoli", supporth="suppOrth")
    cmd = fake_run["cmd"]
    assert cmd[cmd.index("--tools") + 1] == "orthofinder,broccoli"
    assert cmd[cmd.index("--supportLv") + 1] == "2"


def test_run_supporth_support_above_tools_warns(fake_run, fastas, tmp_path):
    with pytest.warns(UserWarning, match="more than the 1 predictor"):
        run_supporth("bug", fastas["bug"], ["templateGEM"], [fastas["templateGEM"]],
                     tmp_path / "out", tools="orthofinder", support_lv=3, supporth="s")


def test_run_supporth_shared_ids_prefixed(fake_run, tmp_path):
    bug = _fasta(tmp_path / "bug.faa", ["ng1", "ng2"])
    t1 = _fasta(tmp_path / "t1.faa", ["tg1", "tg2"])
    t2 = _fasta(tmp_path / "t2.faa", ["tg1", "tg2"])  # same species, same gene ids
    fake_run["rows"] = [
        "t1\tbug\tt1__tg1\tbug__ng1\t3\tF,S,B\t1\t1\t1\t85.0\t1e-150\t500.0",
        "t2\tbug\tt2__tg2\tbug__ng2\t2\tF,S\t1\t1\t0\t60.0\t1e-80\t300.0",
    ]
    hits = run_supporth("bug", bug, ["t1", "t2"], [t1, t2], tmp_path / "out", supporth="s")
    inputs = tmp_path / "out" / "prefixed_inputs"
    assert fake_run["cmd"][2:5] == [str(inputs / f"{x}.faa") for x in ("bug", "t1", "t2")]
    assert (inputs / "t2.faa").read_text(encoding="utf-8").startswith(">t2__tg1 some description\n")
    assert _pairs(hits, "bug", "t1") == {("ng1", "tg1")}
    assert _pairs(hits, "bug", "t2") == {("ng2", "tg2")}


def test_run_supporth_failure_not_read(fake_run, fastas, tmp_path):
    fake_run["returncode"] = 1
    with pytest.raises(RuntimeError, match="exited with code 1.*\nlog line\nstage broccoli failed"):
        run_supporth("bug", fastas["bug"], ["templateGEM"], [fastas["templateGEM"]],
                     tmp_path / "out", supporth="suppOrth")
    assert (tmp_path / "out" / "run_supporth.log").read_text(encoding="utf-8").endswith(
        "\n\nlog line\nstage broccoli failed\n")


def test_run_supporth_failed_stage_kept_in_error(fake_run, fastas, monkeypatch, tmp_path):
    def run(cmd, capture_output, text):
        summary = "\n".join(f"summary {i}" for i in range(30))
        return subprocess.CompletedProcess(
            cmd, 1, "[2/2] broccoli\n  FAILED after 0.0 min: exit code 2\n" + summary, "")

    monkeypatch.setattr(supporth_mod.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="FAILED after 0.0 min: exit code 2\nsummary 10\n"):
        run_supporth("bug", fastas["bug"], ["templateGEM"], [fastas["templateGEM"]],
                     tmp_path / "out", supporth="suppOrth")


@pytest.mark.parametrize("bad", ["a,b", ".", "x/y", "", " lead"])
def test_run_supporth_bad_labels(fake_run, fastas, tmp_path, bad):
    with pytest.raises(ValueError, match="cannot be a SuppOrth species label"):
        run_supporth("bug", fastas["bug"], [bad], [fastas["templateGEM"]], tmp_path / "out",
                     supporth="s")


def test_run_supporth_duplicate_labels(fake_run, fastas, tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        run_supporth("bug", fastas["bug"], ["bug"], [fastas["templateGEM"]], tmp_path / "out",
                     supporth="s")


def test_run_supporth_not_found(fake_run, fastas, monkeypatch, tmp_path):
    monkeypatch.delenv("RAVEN_SUPPORTH", raising=False)
    monkeypatch.setattr(supporth_mod.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="suppOrth install"):
        run_supporth("bug", fastas["bug"], ["templateGEM"], [fastas["templateGEM"]],
                     tmp_path / "out")


def test_run_supporth_env_var(fake_run, fastas, monkeypatch, tmp_path):
    monkeypatch.setenv("RAVEN_SUPPORTH", "/env/suppOrth")
    run_supporth("bug", fastas["bug"], ["templateGEM"], [fastas["templateGEM"]], tmp_path / "out")
    assert fake_run["cmd"][0] == "/env/suppOrth"


def test_run_supporth_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(NotImplementedError, match="WSL"):
        run_supporth("bug", "bug.faa", ["templateGEM"], ["t.faa"], tmp_path / "out")


