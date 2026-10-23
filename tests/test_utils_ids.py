"""Tests for generate_new_ids."""
import cobra
import pytest

from raven_toolbox.utils import generate_new_ids


def _model_with_reactions(ids):
    m = cobra.Model("t")
    for rid in ids:
        m.add_reactions([cobra.Reaction(rid)])
    return m


def test_generate_new_ids_starts_at_one_when_none_exist():
    m = _model_with_reactions([])
    assert generate_new_ids(m, "reactions", "r_") == ["r_0001"]


def test_generate_new_ids_continues_after_existing_max():
    m = _model_with_reactions(["r_0001", "r_0003", "unrelated"])
    assert generate_new_ids(m, "reactions", "r_", quantity=2) == ["r_0004", "r_0005"]


def test_generate_new_ids_adopts_existing_width_over_requested_num_length():
    # Existing ids are zero-padded to 6 digits; the default/requested
    # num_length=4 is overridden by that, matching generateNewIds.m.
    m = _model_with_reactions(["r_000042"])
    assert generate_new_ids(m, "reactions", "r_", num_length=4) == ["r_000043"]


def test_generate_new_ids_metabolites():
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite("s_0001")])
    assert generate_new_ids(m, "metabolites", "s_") == ["s_0002"]


def test_generate_new_ids_rejects_unknown_entity_type():
    m = _model_with_reactions([])
    with pytest.raises(ValueError, match="entity_type"):
        generate_new_ids(m, "genes", "g_")


def test_generate_new_ids_non_numeric_suffix_falls_back_to_zero():
    # A stripped id that isn't a plain integer (e.g. hand-edited) can't be
    # parsed as a starting point -- matching str2double producing NaN,
    # coerced to 0 in generateNewIds.m -- so numbering restarts at 1.
    m = _model_with_reactions(["r_abcd"])
    assert generate_new_ids(m, "reactions", "r_") == ["r_0001"]


def test_generate_new_ids_takes_the_numeric_maximum():
    m = _model_with_reactions(["r_9", "r_10", "r_2"])
    assert generate_new_ids(m, "reactions", "r_") == ["r_11"]


def test_generate_new_ids_numeric_beside_non_numeric():
    m = _model_with_reactions(["r_0003", "r_abc"])
    assert generate_new_ids(m, "reactions", "r_") == ["r_0004"]


def test_generate_new_ids_prefix_without_separator():
    m = _model_with_reactions(["r0001", "r_0007", "rx12"])
    # "r" strips to 0001, _0007 and x12; only 0001 reads as a number.
    assert generate_new_ids(m, "reactions", "r") == ["r0002"]


def test_generate_new_ids_all_non_numeric_uses_last_in_string_order_for_width():
    m = _model_with_reactions(["r_ab", "r_xyz"])
    assert generate_new_ids(m, "reactions", "r_") == ["r_001"]
