"""Tests for init/hpa.py — HPA parsing + score adapters (Phase 5)."""
from __future__ import annotations

from textwrap import dedent

import pytest

from raven_toolbox.init import (
    HPA_LEVEL_SCORES,
    HPAData,
    HPARnaData,
    hpa_gene_scores,
    parse_hpa,
    parse_hpa_rna,
    rna_gene_scores,
)


@pytest.fixture
def hpa_tsv(tmp_path):
    """Minimal HPA proteomics TSV with two genes × two tissues × two cell types."""
    p = tmp_path / "hpa.tsv"
    p.write_text(dedent("""\
        Gene\tGene name\tTissue\tCell type\tLevel\tReliability
        ENSG1\tGeneA\tliver\thepatocytes\tHigh\tEnhanced
        ENSG1\tGeneA\tliver\tbile duct cells\tLow\tApproved
        ENSG1\tGeneA\tkidney\ttubular cells\tNot detected\tApproved
        ENSG2\tGeneB\tliver\thepatocytes\tMedium\tSupported
        ENSG2\tGeneB\tkidney\ttubular cells\tHigh\tEnhanced
        ENSG3\tGeneC\tliver\thepatocytes\tMixed\tUncertain
    """))
    return p


@pytest.fixture
def rna_tsv(tmp_path):
    """Tidy HPA-style RNA-seq TSV (Gene/Gene name/Tissue/TPM)."""
    p = tmp_path / "rna.tsv"
    p.write_text(dedent("""\
        Gene\tGene name\tTissue\tTPM
        ENSG1\tGeneA\tliver\t100.0
        ENSG1\tGeneA\tkidney\t10.0
        ENSG2\tGeneB\tliver\t5.0
        ENSG2\tGeneB\tkidney\t50.0
    """))
    return p


# ---------------------------------------------------------------------- parsers

def test_parse_hpa_basic(hpa_tsv):
    hpa = parse_hpa(hpa_tsv)
    assert isinstance(hpa, HPAData)
    assert hpa.tissues() == ["kidney", "liver"]
    assert hpa.celltypes("liver") == ["bile duct cells", "hepatocytes"]
    # one row per (gene, tissue, celltype):
    assert len(hpa.df) == 6
    assert set(hpa.df.columns) == {"gene_id", "gene_name", "tissue", "celltype",
                                    "level", "reliability"}


def test_parse_hpa_missing_columns(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("Gene\tTissue\nx\ty\n")
    with pytest.raises(ValueError, match="missing HPA columns"):
        parse_hpa(p)


def test_parse_hpa_rna_tidy(rna_tsv):
    rna = parse_hpa_rna(rna_tsv)
    assert isinstance(rna, HPARnaData)
    assert rna.tissues() == ["kidney", "liver"]
    assert rna.expression("liver") == {"ENSG1": 100.0, "ENSG2": 5.0}


def test_parse_hpa_rna_v18_layout(tmp_path):
    p = tmp_path / "rna_v18.tsv"
    p.write_text("Gene\tGene name\tSample\tValue\nENSG1\tGeneA\tliver\t100\nENSG1\tGeneA\tkidney\t10\n")
    rna = parse_hpa_rna(p)
    assert rna.tissues() == ["kidney", "liver"]
    assert rna.expression("liver") == {"ENSG1": 100.0}


def test_parse_hpa_rna_wide_layout_raises(tmp_path):
    p = tmp_path / "rna_wide.tsv"
    p.write_text("Gene\tGene name\tliver\tkidney\nENSG1\tGeneA\t100\t10\n")
    with pytest.raises(ValueError, match="tissue column"):
        parse_hpa_rna(p)


def test_parse_hpa_rna_without_value_column_raises(tmp_path):
    p = tmp_path / "rna_novalue.tsv"
    p.write_text("Gene\tGene name\tTissue\nENSG1\tGeneA\tliver\n")
    with pytest.raises(ValueError, match="value column"):
        parse_hpa_rna(p)


def test_parse_hpa_rna_repeated_gene_tissue_keeps_last(tmp_path):
    p = tmp_path / "rna_dup.tsv"
    p.write_text("Gene\tGene name\tTissue\tTPM\nENSG1\tGeneA\tliver\t1\nENSG1\tGeneA\tliver\t7\n")
    with pytest.warns(UserWarning, match="repeat a gene and tissue"):
        rna = parse_hpa_rna(p)
    assert rna.expression("liver") == {"ENSG1": 7.0}


# ---------------------------------------------------------------------- scoring

def test_hpa_gene_scores_best_picks_max(hpa_tsv):
    """In liver, ENSG1 is High (hepatocytes) + Low (bile duct) → best = 20."""
    g = hpa_gene_scores(parse_hpa(hpa_tsv), "liver", multiple_celltype="best")
    assert g["ENSG1"] == HPA_LEVEL_SCORES["High"]      # 20
    assert g["ENSG2"] == HPA_LEVEL_SCORES["Medium"]    # 15


def test_hpa_gene_scores_average(hpa_tsv):
    """Average across cell types: ENSG1 in liver = mean(20, 10) = 15."""
    g = hpa_gene_scores(parse_hpa(hpa_tsv), "liver", multiple_celltype="average")
    assert g["ENSG1"] == pytest.approx(15.0)


def test_hpa_gene_scores_celltype_filter(hpa_tsv):
    """Restricting to a celltype gives only that celltype's score."""
    g = hpa_gene_scores(parse_hpa(hpa_tsv), "liver", celltype="bile duct cells")
    assert g == {"ENSG1": HPA_LEVEL_SCORES["Low"]}     # 10; GeneB has no bile-duct row


def test_hpa_level_none_scores_minus_eight(tmp_path):
    p = tmp_path / "hpa_none.tsv"
    p.write_text("Gene\tGene name\tTissue\tCell type\tLevel\tReliability\n"
                 "ENSG9\tGeneZ\tliver\thepatocytes\tNone\tApproved\n")
    assert hpa_gene_scores(parse_hpa(p), "liver") == {"ENSG9": -8.0}


def test_hpa_gene_scores_unknown_level_omitted(hpa_tsv):
    """A 'Mixed' / 'N/A' level is not in HPA_LEVEL_SCORES and is dropped (not -inf)."""
    g = hpa_gene_scores(parse_hpa(hpa_tsv), "liver")
    assert "ENSG3" not in g    # the only ENSG3 row in liver has level='Mixed'


def test_hpa_gene_scores_unknown_celltype_returns_empty(hpa_tsv):
    g = hpa_gene_scores(parse_hpa(hpa_tsv), "liver", celltype="cardiomyocytes")
    assert g == {}


def test_hpa_gene_scores_custom_level_table(hpa_tsv):
    """``level_scores`` overrides the default mapping."""
    g = hpa_gene_scores(parse_hpa(hpa_tsv), "liver",
                        level_scores={"High": 1.0, "Medium": 0.5, "Low": 0.1, "Not detected": -1.0})
    assert g == {"ENSG1": 1.0, "ENSG2": 0.5}


def test_hpa_gene_scores_compare_level_names_case_insensitively(tmp_path):
    """RAVEN's scoreModel compares upper-cased level names, for the default and a custom table."""
    p = tmp_path / "hpa_case.tsv"
    p.write_text("Gene\tGene name\tTissue\tCell type\tLevel\tReliability\n"
                 "G1\tA\tliver\thepatocytes\tstrong\tApproved\n"
                 "G2\tB\tliver\thepatocytes\tNOT DETECTED\tApproved\n"
                 "G3\tC\tliver\thepatocytes\tMixed\tApproved\n")
    hpa = parse_hpa(p)
    assert hpa_gene_scores(hpa, "liver") == {"G1": 20.0, "G2": -8.0}
    assert hpa_gene_scores(hpa, "liver", level_scores={"STRONG": 1.0}) == {"G1": 1.0}


def test_rna_gene_scores_against_per_gene_mean(rna_tsv):
    """Default reference is per-gene cross-tissue mean (RAVEN arrayData.threshold default).

    ENSG1 liver TPM=100, mean across tissues=55 → log(100/55) > 0 → positive score.
    ENSG2 liver TPM=5,   mean=27.5            → log(5/27.5) < 0 → negative score.
    """
    g = rna_gene_scores(parse_hpa_rna(rna_tsv), "liver")
    assert g["ENSG1"] > 0
    assert g["ENSG2"] < 0


def test_rna_gene_scores_scalar_reference(rna_tsv):
    """A scalar reference applies to all genes (and reuses gene_scores_from_expression)."""
    g = rna_gene_scores(parse_hpa_rna(rna_tsv), "liver", reference=10.0)
    # ENSG1 TPM=100, ref=10 → ln(10)*5 ≈ 11.5 → clamped to max_score=10.
    assert g["ENSG1"] == 10.0
    assert g["ENSG2"] < 0  # TPM=5 < ref=10


def test_rna_gene_scores_unknown_tissue_raises(rna_tsv):
    with pytest.raises(ValueError, match="not in dataset"):
        rna_gene_scores(parse_hpa_rna(rna_tsv), "spleen")


def test_hpa_gene_scores_invalid_multiple_celltype(hpa_tsv):
    with pytest.raises(ValueError, match="multiple_celltype"):
        hpa_gene_scores(parse_hpa(hpa_tsv), "liver", multiple_celltype="weighted")
