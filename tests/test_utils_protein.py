"""Tests for calculate_mw."""
import logging
import math

import pytest

from raven_toolbox.utils.protein import (
    _RESIDUE_MASSES,
    _STANDARD_RESIDUE_MASSES,
    _WATER_MASS,
    _X_MASS,
    calculate_mw,
)

# --------------------------------------------------------------------------- #
# Trivial cases
# --------------------------------------------------------------------------- #

def test_empty_sequence_returns_nan():
    # An empty / all-skipped sequence has no residues; NaN avoids being
    # mistaken for an 18 Da "protein".
    assert math.isnan(calculate_mw(""))
    assert math.isnan(calculate_mw("   "))
    assert math.isnan(calculate_mw("123"))


def test_single_residue():
    assert calculate_mw("A") == pytest.approx(_WATER_MASS + 71.08)


def test_dipeptide():
    expected = _WATER_MASS + 71.08 + 137.14
    assert calculate_mw("AH") == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Known reference cases
# --------------------------------------------------------------------------- #

def test_methionine_only():
    """Met has mass 131.20."""
    assert calculate_mw("M") == pytest.approx(_WATER_MASS + 131.20)


def test_repeated_residues_count_correctly():
    expected = _WATER_MASS + 5 * 71.08
    assert calculate_mw("AAAAA") == pytest.approx(expected)


def test_all_20_standard_residues():
    """A sequence of one of each of the 20 standard residues should
    give water + sum of all 20 masses."""
    seq = "".join(sorted(_STANDARD_RESIDUE_MASSES.keys()))
    expected = _WATER_MASS + sum(_STANDARD_RESIDUE_MASSES.values())
    assert calculate_mw(seq) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Non-standard codes
# --------------------------------------------------------------------------- #

def test_b_is_average_of_d_and_n():
    expected = (_RESIDUE_MASSES["D"] + _RESIDUE_MASSES["N"]) / 2
    assert _RESIDUE_MASSES["B"] == pytest.approx(expected)


def test_z_is_average_of_e_and_q():
    expected = (_RESIDUE_MASSES["E"] + _RESIDUE_MASSES["Q"]) / 2
    assert _RESIDUE_MASSES["Z"] == pytest.approx(expected)


def test_j_equals_l_and_i():
    assert _RESIDUE_MASSES["J"] == _RESIDUE_MASSES["L"] == _RESIDUE_MASSES["I"]


def test_o_pyrrolysine():
    assert _RESIDUE_MASSES["O"] == pytest.approx(255.31)


def test_u_selenocysteine():
    assert _RESIDUE_MASSES["U"] == pytest.approx(150.04)


def test_x_is_mean_of_20_standard_residues():
    expected = sum(_STANDARD_RESIDUE_MASSES.values()) / 20
    assert _X_MASS == pytest.approx(expected)
    assert _X_MASS == pytest.approx(118.885, abs=1e-3)


# --------------------------------------------------------------------------- #
# Case insensitivity
# --------------------------------------------------------------------------- #

def test_lowercase_is_treated_same_as_uppercase():
    assert calculate_mw("ahkc") == calculate_mw("AHKC")


def test_mixed_case_works():
    assert calculate_mw("AhKc") == calculate_mw("AHKC")


# --------------------------------------------------------------------------- #
# Whitespace and digits ignored silently
# --------------------------------------------------------------------------- #

def test_whitespace_ignored():
    assert calculate_mw("A H") == calculate_mw("AH")
    assert calculate_mw("\tA\nH\n") == calculate_mw("AH")


def test_digits_ignored():
    assert calculate_mw("A1H2") == calculate_mw("AH")


def test_whitespace_and_digits_do_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        calculate_mw("A H 1 2 K")
    assert "skipped" not in caplog.text


# --------------------------------------------------------------------------- #
# Unknown characters: warn but do not raise
# --------------------------------------------------------------------------- #

def test_unknown_character_warns(caplog):
    with caplog.at_level(logging.WARNING):
        calculate_mw("A@H")
    assert "skipped" in caplog.text
    assert "@" in caplog.text


def test_unknown_character_does_not_contribute_to_mass(caplog):
    with caplog.at_level(logging.WARNING):
        result = calculate_mw("A@H")
    assert result == calculate_mw("AH")


def test_multiple_unknown_chars_warned_together(caplog):
    with caplog.at_level(logging.WARNING):
        calculate_mw("A@H#K?L")
    assert "skipped" in caplog.text
    assert "@" in caplog.text
    assert "#" in caplog.text
    assert "?" in caplog.text


# --------------------------------------------------------------------------- #
# Realistic sequence
# --------------------------------------------------------------------------- #

def test_short_real_protein_mass():
    """A snippet of a real protein sequence should yield a plausible
    mass in the low-kDa range, not just any positive number."""
    seq = "MENTGTLAGCDVRFGRSGGSGSGSGSAATAACK"
    mw = calculate_mw(seq)
    assert mw > 0
    # Approximately 33 residues at average ~110 g/mol = 3630 + water.
    assert 3000 < mw < 4500


# --------------------------------------------------------------------------- #
# Water mass precision
# --------------------------------------------------------------------------- #

def test_water_mass_is_precise():
    """The water mass used is the precise average, 18.01528 Da."""
    assert _WATER_MASS == pytest.approx(18.01528)
