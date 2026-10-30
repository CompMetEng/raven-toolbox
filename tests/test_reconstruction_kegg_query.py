"""Tests for the KEGG HMM-query path (reconstruction/kegg/query.py, step 3b.5).

The ``kegg_dump`` fixture (tests/conftest.py) is a small, fully fictional dump —
no real KEGG content is committed.
"""
import pandas as pd
import pytest

from raven_toolbox.reconstruction.kegg import (
    assign_kos,
    build_kegg_tables,
    build_reference_model,
    get_kegg_model_from_sequences,
    parse_hmmsearch_tblout,
    parse_kegg_compounds,
    parse_kegg_kos,
    parse_kegg_reactions,
)

# A minimal hmmsearch --tblout excerpt: target(gene) accession query(KO) ... evalue ...
TBLOUT = """\
#                                                               --- full sequence ----
# target name        accession  query name  accession   E-value  score  bias
#------------------- ---------- ----------- ---------- --------- ------ -----
gene1                -          K90001      -          1e-120     400.0   0.0
gene2                -          K90001      -          1e-100     350.0   0.0
gene1                -          K90002      -          1e-10      40.0    0.0
"""


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_tblout_skips_comments():
    hits = parse_hmmsearch_tblout(TBLOUT)
    assert list(hits.columns) == ["ko", "gene", "evalue"]
    assert len(hits) == 3
    assert set(hits["ko"]) == {"K90001", "K90002"}
    assert hits.iloc[0]["evalue"] == 1e-120


def test_parse_tblout_empty():
    assert parse_hmmsearch_tblout("# only a header\n").empty


# --------------------------------------------------------------------------- #
# assign_kos scoring/filters
# --------------------------------------------------------------------------- #
def test_cutoff_excludes_weak_hits():
    hits = parse_hmmsearch_tblout(TBLOUT)
    # gene1->K90002 has evalue 1e-10, above the default cutoff 1e-30: dropped.
    assigned = assign_kos(hits)
    assert "K90002" not in assigned
    assert set(assigned["K90001"]) == {"gene1", "gene2"}


def test_loose_cutoff_keeps_hit():
    hits = parse_hmmsearch_tblout(TBLOUT)
    assigned = assign_kos(hits, cutoff=1e-5, min_score_ratio_g=0.0, min_score_ratio_ko=0.0)
    assert assigned.get("K90002") == ["gene1"]


def test_min_score_ratio_ko_prunes_weak_member():
    # In one KO: best 1e-200, weak 1e-20. log(1e-20)/log(1e-200)=0.1 < 0.3 -> pruned.
    hits = pd.DataFrame(
        [("K1", "strong", 1e-200), ("K1", "weak", 1e-20)],
        columns=["ko", "gene", "evalue"],
    )
    assigned = assign_kos(hits, cutoff=1e-5, min_score_ratio_ko=0.3, min_score_ratio_g=0.0)
    assert assigned["K1"] == ["strong"]


def test_min_score_ratio_g_keeps_gene_in_best_ko_only():
    # gene g hits K1 strongly (1e-200) and K2 weakly (1e-20).
    # For the gene: log(1e-20)/log(1e-200)=0.1 < 0.8 -> K2 assignment dropped.
    hits = pd.DataFrame(
        [("K1", "g", 1e-200), ("K2", "g", 1e-20)],
        columns=["ko", "gene", "evalue"],
    )
    assigned = assign_kos(hits, cutoff=1e-5, min_score_ratio_ko=0.0, min_score_ratio_g=0.8)
    assert assigned == {"K1": ["g"]}


def test_zero_evalue_does_not_crash():
    hits = pd.DataFrame([("K1", "g", 0.0)], columns=["ko", "gene", "evalue"])
    assert assign_kos(hits) == {"K1": ["g"]}


def test_cutoff_ge_one_rejected():
    """cutoff >= 1 would let log(best_evalue)=0 through and cause a ZeroDivisionError
    later. Reject up front with a clear message."""
    hits = pd.DataFrame([("K1", "g", 0.5)], columns=["ko", "gene", "evalue"])
    with pytest.raises(ValueError, match="cutoff must be < 1"):
        assign_kos(hits, cutoff=1.0)


# --------------------------------------------------------------------------- #
# Model assembly via the HMM path (hmmsearch mocked)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def reference_and_tables(kegg_dump):
    reactions = parse_kegg_reactions(kegg_dump)
    compounds = parse_kegg_compounds(kegg_dump)
    linked = {ko for r in reactions for ko in r.kos}
    kos = parse_kegg_kos(kegg_dump, keep=linked)
    return build_reference_model(reactions, compounds), build_kegg_tables(reactions, kos)


def test_get_model_from_sequences(reference_and_tables, monkeypatch):
    model_ref, tables = reference_and_tables
    # Mock the HMM search: K90001 -> myGeneA/myGeneB (-> R90010).
    monkeypatch.setattr(
        "raven_toolbox.reconstruction.kegg.query.run_hmmsearch",
        lambda *a, **k: (
            "myGeneA - K90001 - 1e-120 400 0\n"
            "myGeneB - K90001 - 1e-110 380 0\n"
        ),
    )
    model = get_kegg_model_from_sequences(
        "ignored.fasta",
        model_ref,
        tables["ko_reaction"],
        "ignored.hmm",
        rxn_flags=tables["rxn_flags"],
        model_id="myorg",
    )
    assert model.id == "myorg"
    r = model.reactions.get_by_id("R90010")
    assert set(r.gene_reaction_rule.split(" or ")) == {"myGeneA", "myGeneB"}
    assert r.notes["note"] == "Included by KEGG HMM reconstruction"
    # Its single KO matched, so the kegg.orthology annotation is unchanged.
    assert r.annotation["kegg.orthology"] == ["K90001"]
    # R90200/R90300 had no matched KOs and are not spontaneous -> absent.
    assert "R90200" not in model.reactions


def test_model_id_defaults_to_fasta_stem(reference_and_tables, monkeypatch):
    """RAVEN always sets model.id; with no model_id we default it to the FASTA stem
    rather than inheriting the reference model's id."""
    model_ref, tables = reference_and_tables
    monkeypatch.setattr(
        "raven_toolbox.reconstruction.kegg.query.run_hmmsearch",
        lambda *a, **k: "myGeneA - K90001 - 1e-120 400 0\n",
    )
    model = get_kegg_model_from_sequences(
        "/some/path/eco.faa", model_ref, tables["ko_reaction"], "ignored.hmm",
        rxn_flags=tables["rxn_flags"],
    )
    assert model.id == "eco"


def test_prune_orthology_keeps_only_matched_kos():
    """The FASTA path prunes a kept reaction's kegg.orthology to the KOs that
    matched a gene (RAVEN getKEGGModelForOrganism HMM branch), preserving order."""
    import cobra

    from raven_toolbox.reconstruction.kegg.assemble import assemble_model_from_ko_genes

    ref = cobra.Model("ref")
    met = cobra.Metabolite("C1", compartment="s")
    rxn = cobra.Reaction("R1")
    rxn.add_metabolites({met: -1.0})
    rxn.annotation["kegg.orthology"] = ["K1", "K2", "K3"]
    ref.add_reactions([rxn])
    ko_reaction = pd.DataFrame(
        [("K1", "R1"), ("K2", "R1"), ("K3", "R1")], columns=["ko", "reaction"]
    )
    ko_to_genes = {"K2": ["g"]}  # only K2 matched a gene

    pruned, _ = assemble_model_from_ko_genes(
        ref, ko_reaction, ko_to_genes, prune_orthology=True
    )
    assert pruned.reactions.get_by_id("R1").annotation["kegg.orthology"] == ["K2"]
    # Default (organism path) keeps the full reference KO list.
    full, _ = assemble_model_from_ko_genes(ref, ko_reaction, ko_to_genes)
    assert full.reactions.get_by_id("R1").annotation["kegg.orthology"] == ["K1", "K2", "K3"]


def _fake_hmmsearch(monkeypatch, calls):
    """Make hmmsearch a stub that records its command; the process may run on 7 of 16 CPUs."""
    import os
    import subprocess

    import raven_toolbox.reconstruction.kegg.query as query

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(7)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.setattr(query, "resolve_binary", lambda name, binary=None: name)

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        open(cmd[cmd.index("--tblout") + 1], "w").close()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(query.subprocess, "run", fake_run)
    return query


def test_hmmsearch_threads_default_to_every_available_cpu(tmp_path, monkeypatch):
    """As RAVEN's getKEGGModelForOrganism, which passes feature('numcores')'s assigned count."""
    calls: list = []
    query = _fake_hmmsearch(monkeypatch, calls)
    library = tmp_path / "lib.hmm"
    library.write_text("HMMER3/f\n")
    query.run_hmmsearch("org.faa", library)
    assert calls[0][calls[0].index("--cpu") + 1] == "7"
    query.run_hmmsearch("org.faa", library, threads=3)
    assert calls[1][calls[1].index("--cpu") + 1] == "3"


def test_kegg_model_from_sequences_passes_the_default_threads_on(tmp_path, monkeypatch, reference_and_tables):
    calls: list = []
    query = _fake_hmmsearch(monkeypatch, calls)
    model_ref, tables = reference_and_tables
    library = tmp_path / "lib.hmm"
    library.write_text("HMMER3/f\n")
    query.get_kegg_model_from_sequences("org.faa", model_ref, tables["ko_reaction"], library,
                                        rxn_flags=tables["rxn_flags"])
    assert calls[0][calls[0].index("--cpu") + 1] == "7"
