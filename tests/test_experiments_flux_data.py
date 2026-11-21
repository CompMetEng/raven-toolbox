"""Tests for load_flux_data."""
from pathlib import Path

import numpy as np
import pytest

from raven_toolbox.experiments import FluxData, load_flux_data


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(
        "\n".join("\t".join(r) for r in rows) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Pre-flight
# --------------------------------------------------------------------------- #

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="fluxData file"):
        load_flux_data(tmp_path / "missing.tsv")


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.tsv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_flux_data(p)


def test_short_header_raises(tmp_path):
    """Header with fewer than 3 columns -> error."""
    p = tmp_path / "short.tsv"
    _write_tsv(p, [["only", "two"]])
    with pytest.raises(ValueError, match="header has"):
        load_flux_data(p)


def test_no_exchange_columns_raises(tmp_path):
    """Header with exactly 3 columns (no exchanges) -> error."""
    p = tmp_path / "no_exch.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate"],
        ["c1", "0.5", "0.4"],
    ])
    with pytest.raises(ValueError, match="no exchange-flux columns"):
        load_flux_data(p)


# --------------------------------------------------------------------------- #
# Basic happy path
# --------------------------------------------------------------------------- #

def test_an_unknown_column_is_an_exchange_column_with_a_warning(tmp_path, caplog):
    f = tmp_path / "fluxData.tsv"
    _write_tsv(f, [
        ["Condition", "Ptot", "grRate", "glucose (r_1714)", "other"],
        ["c1", "0.5", "0.4", "-10", "1"],
    ])
    with caplog.at_level("WARNING"):
        fd = load_flux_data(f)
    assert fd.weight is None
    assert fd.exch_rxn_ids == ["r_1714", "other"]
    assert "lack the" in caplog.text


def test_weight_column_is_read_and_removed_from_the_fluxes(tmp_path):
    f = tmp_path / "fluxData.tsv"
    _write_tsv(f, [
        ["Condition", "Ptot", "grRate", "glucose (r_1714)", "weight", "source"],
        ["c1", "0.5", "0.4", "-10", "0", "x"],
        ["c2", "0.5", "0.3", "-8", "2", "y"],
    ])
    data = load_flux_data(f)
    assert data.weight.tolist() == [0.0, 2.0]
    assert data.exch_rxn_ids == ["r_1714"]


def test_basic_single_condition(tmp_path):
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate", "glucose (EX_glc)"],
        ["c1", "0.5", "0.4", "-10.0"],
    ])
    fd = load_flux_data(p)
    assert isinstance(fd, FluxData)
    assert fd.conds == ["c1"]
    np.testing.assert_array_equal(fd.p_tot, [0.5])
    np.testing.assert_array_equal(fd.gr_rate, [0.4])
    np.testing.assert_array_equal(fd.exch_fluxes, [[-10.0]])
    assert fd.exch_mets == ["glucose"]
    assert fd.exch_rxn_ids == ["EX_glc"]


def test_multi_condition_multi_exchange(tmp_path):
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate",
         "glucose (EX_glc)", "oxygen (EX_O2)", "ethanol (EX_eth)"],
        ["c1", "0.5", "0.4", "-10.0", "-2.0", "5.0"],
        ["c2", "0.6", "0.5", "-8.5", "-3.0", "7.0"],
    ])
    fd = load_flux_data(p)
    assert fd.conds == ["c1", "c2"]
    np.testing.assert_array_equal(fd.p_tot, [0.5, 0.6])
    np.testing.assert_array_equal(fd.gr_rate, [0.4, 0.5])
    np.testing.assert_array_equal(
        fd.exch_fluxes,
        [[-10.0, -2.0, 5.0], [-8.5, -3.0, 7.0]],
    )
    assert fd.exch_mets == ["glucose", "oxygen", "ethanol"]
    assert fd.exch_rxn_ids == ["EX_glc", "EX_O2", "EX_eth"]


# --------------------------------------------------------------------------- #
# Exchange header parsing
# --------------------------------------------------------------------------- #

def test_exchange_header_with_spaces_in_met_name(tmp_path):
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate", "carbon dioxide (EX_co2)"],
        ["c1", "0.5", "0.4", "5.0"],
    ])
    fd = load_flux_data(p)
    assert fd.exch_mets == ["carbon dioxide"]
    assert fd.exch_rxn_ids == ["EX_co2"]


def test_exchange_header_without_parens_passes_through(tmp_path):
    """Header without `(...)` falls back to met=rxn=full header."""
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate", "EX_glc"],
        ["c1", "0.5", "0.4", "-10.0"],
    ])
    fd = load_flux_data(p)
    assert fd.exch_mets == ["EX_glc"]
    assert fd.exch_rxn_ids == ["EX_glc"]


# --------------------------------------------------------------------------- #
# NaN handling
# --------------------------------------------------------------------------- #

def test_empty_flux_cell_yields_nan(tmp_path):
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate", "glc (EX_glc)", "o2 (EX_O2)"],
        ["c1", "0.5", "0.4", "-10.0", ""],
        ["c2", "0.6", "0.5", "", "-3.0"],
    ])
    fd = load_flux_data(p)
    assert np.isnan(fd.exch_fluxes[0, 1])
    assert np.isnan(fd.exch_fluxes[1, 0])
    assert fd.exch_fluxes[0, 0] == -10.0
    assert fd.exch_fluxes[1, 1] == -3.0


def test_non_numeric_flux_yields_nan(tmp_path):
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate", "glc (EX_glc)"],
        ["c1", "0.5", "0.4", "NA"],
    ])
    fd = load_flux_data(p)
    assert np.isnan(fd.exch_fluxes[0, 0])


# --------------------------------------------------------------------------- #
# Optional columns
# --------------------------------------------------------------------------- #

def test_no_optional_columns_yields_none(tmp_path):
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate", "glc (EX_glc)"],
        ["c1", "0.5", "0.4", "-10.0"],
    ])
    fd = load_flux_data(p)
    assert fd.weight is None
    assert fd.source is None


def test_weight_column_loaded(tmp_path):
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate",
         "glc (EX_glc)", "weight"],
        ["c1", "0.5", "0.4", "-10.0", "1.5"],
        ["c2", "0.6", "0.5", "-8.5", "0.7"],
    ])
    fd = load_flux_data(p)
    assert fd.weight is not None
    np.testing.assert_array_equal(fd.weight, [1.5, 0.7])
    # Tuning-weight column dropped from exch_fluxes.
    assert fd.exch_fluxes.shape == (2, 1)
    assert fd.exch_rxn_ids == ["EX_glc"]


def test_source_column_loaded(tmp_path):
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate", "glc (EX_glc)", "source"],
        ["c1", "0.5", "0.4", "-10.0", "PMID:12345"],
        ["c2", "0.6", "0.5", "-8.5", "in-house"],
    ])
    fd = load_flux_data(p)
    assert fd.source == ["PMID:12345", "in-house"]
    assert fd.exch_fluxes.shape == (2, 1)


def test_both_optional_columns(tmp_path):
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate",
         "glc (EX_glc)", "weight", "source"],
        ["c1", "0.5", "0.4", "-10.0", "1.5", "PMID:12345"],
    ])
    fd = load_flux_data(p)
    assert fd.weight is not None
    np.testing.assert_array_equal(fd.weight, [1.5])
    assert fd.source == ["PMID:12345"]
    assert fd.exch_fluxes.shape == (1, 1)


def test_optional_columns_in_arbitrary_position(tmp_path):
    """Optional columns can appear before, between, or after the
    exchange columns; they are pulled out independently of position."""
    p = tmp_path / "fd.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate",
         "source", "glc (EX_glc)", "weight"],
        ["c1", "0.5", "0.4", "PMID:1", "-10.0", "1.5"],
    ])
    fd = load_flux_data(p)
    assert fd.source == ["PMID:1"]
    assert fd.weight is not None
    np.testing.assert_array_equal(fd.weight, [1.5])
    np.testing.assert_array_equal(fd.exch_fluxes, [[-10.0]])
    assert fd.exch_rxn_ids == ["EX_glc"]


# --------------------------------------------------------------------------- #
# Blank lines
# --------------------------------------------------------------------------- #

def test_blank_lines_skipped(tmp_path):
    p = tmp_path / "fd.tsv"
    p.write_text(
        "condition\tPtot\tgrRate\tglc (EX_glc)\n"
        "\n"
        "c1\t0.5\t0.4\t-10.0\n"
        "\n",
        encoding="utf-8",
    )
    fd = load_flux_data(p)
    assert fd.conds == ["c1"]


def test_ragged_row_raises(tmp_path):
    p = tmp_path / "ragged.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate", "Glc (EX_a)", "O2 (EX_b)"],
        ["c1", "0.5", "0.4", "-1", "2"],
        ["c2", "0.5", "0.4", "-1"],
    ])
    with pytest.raises(ValueError, match="row 3 has 4 cell"):
        load_flux_data(p)


def test_nested_parentheses_in_reaction_id(tmp_path):
    p = tmp_path / "nested.tsv"
    _write_tsv(p, [
        ["condition", "Ptot", "grRate", "X (EX_a(e))", "O2 (EX_b)"],
        ["c1", "0.5", "0.4", "-1", "2"],
    ])
    fd = load_flux_data(p)
    assert fd.exch_mets == ["X", "O2"]
    assert fd.exch_rxn_ids == ["EX_a(e)", "EX_b"]
