"""Tests for raven_toolbox.conditions.apply (generic condition mechanism)."""
from __future__ import annotations

import io

import cobra
import pytest
from ruamel.yaml import YAML

from raven_toolbox.conditions import (
    apply_condition,
    load_condition,
    set_reaction_bounds,
)

_SAFE_YAML = YAML(typ="safe")


def _yaml_dump(cfg: dict) -> str:
    """Helper: dump a dict to YAML via ruamel.yaml (PyYAML.safe_dump
    equivalent). Used here because PyYAML isn't a project dependency."""
    buf = io.StringIO()
    _SAFE_YAML.dump(cfg, buf)
    return buf.getvalue()


def _toy_model() -> cobra.Model:
    m = cobra.Model("toy")
    m.compartments = {"c": "cytoplasm", "e": "extracellular"}

    mets = {
        "atp_c": cobra.Metabolite("atp_c", name="ATP", compartment="c", charge=-4),
        "glc_e": cobra.Metabolite("glc_e", name="glucose", compartment="e", charge=0),
        "h_c":   cobra.Metabolite("h_c",   name="H+",      compartment="c", charge=1),
        "fad":   cobra.Metabolite("fad",   name="FAD",     compartment="c", charge=-2),
        "fadh2": cobra.Metabolite("fadh2", name="FADH2",   compartment="c", charge=0),
        "heme":  cobra.Metabolite("heme",  name="heme a",  compartment="c", charge=-2),
    }
    m.add_metabolites(list(mets.values()))

    ex = cobra.Reaction("EX_glc", lower_bound=-10, upper_bound=1000)
    ex.add_metabolites({mets["glc_e"]: -1})
    cofactor = cobra.Reaction("cofac", lower_bound=0, upper_bound=1000)
    cofactor.add_metabolites({mets["heme"]: -1, mets["fad"]: -1, mets["h_c"]: 3})
    biomass = cobra.Reaction("bio", lower_bound=0, upper_bound=1000)
    biomass.add_metabolites({mets["atp_c"]: -1})
    blocked = cobra.Reaction("blocked", lower_bound=-1000, upper_bound=1000)
    blocked.add_metabolites({mets["atp_c"]: -1, mets["glc_e"]: 1})
    m.add_reactions([ex, cofactor, biomass, blocked])
    return m


# --- prelude ----------------------------------------------------------

def test_prelude_reset_exchanges_zeroes_lb_and_caps_ub():
    m = _toy_model()
    apply_condition(m, {"prelude": {"reset_exchanges": "out"}})
    ex = m.reactions.get_by_id("EX_glc")
    assert ex.lower_bound == 0
    assert ex.upper_bound == 1000


# --- bounds -----------------------------------------------------------

def test_bounds_apply_lb_only():
    m = _toy_model()
    apply_condition(m, {"bounds": [{"rxn": "EX_glc", "lb": -1000}]})
    assert m.reactions.get_by_id("EX_glc").lower_bound == -1000
    # ub preserved
    assert m.reactions.get_by_id("EX_glc").upper_bound == 1000


def test_bounds_apply_both():
    m = _toy_model()
    apply_condition(m, {"bounds": [{"rxn": "blocked", "lb": 0, "ub": 0}]})
    assert m.reactions.get_by_id("blocked").bounds == (0, 0)


def test_bounds_lb_gt_ub_bypasses_cobra_validator():
    m = _toy_model()
    apply_condition(m, {"bounds": [{"rxn": "blocked", "lb": 1000, "ub": 0}]})
    rxn = m.reactions.get_by_id("blocked")
    assert rxn.lower_bound == 1000
    assert rxn.upper_bound == 0


def test_bounds_missing_rxn_warns_and_continues():
    m = _toy_model()
    with pytest.warns(UserWarning, match="not_a_real_rxn"):
        apply_condition(m, {"bounds": [{"rxn": "not_a_real_rxn", "lb": 0}]})


# --- cofactor pseudoreaction -----------------------------------------

def test_remove_met_zeroes_coefficient():
    m = _toy_model()
    apply_condition(
        m,
        {
            "cofactor_pseudoreaction": {
                "rxn_id": "cofac",
                "remove_mets": [{"met": "heme"}],
            }
        },
    )
    rxn = m.reactions.get_by_id("cofac")
    heme = m.metabolites.get_by_id("heme")
    assert rxn.metabolites.get(heme, 0) == 0


def test_charge_balance_recomputed_after_removal():
    m = _toy_model()
    apply_condition(
        m,
        {
            "cofactor_pseudoreaction": {
                "rxn_id": "cofac",
                "remove_mets": [{"met": "heme"}],
                "charge_balance_met": "h_c",
            }
        },
    )
    rxn = m.reactions.get_by_id("cofac")
    # heme (-1·-2) and fad (-1·-2) contributed +4 charge.
    # After heme removal: only fad (-1·-2) contributes +2.
    # H+ (charge +1) coef should be -2 to zero the net.
    h_c = m.metabolites.get_by_id("h_c")
    assert rxn.metabolites[h_c] == -2


def test_charge_balance_raises_on_unknown_charge():
    m = _toy_model()
    cofac = m.reactions.get_by_id("cofac")
    mystery = cobra.Metabolite("mystery", name="uncharacterized cofactor", compartment="c")
    m.add_metabolites([mystery])
    cofac.add_metabolites({mystery: -1})
    with pytest.raises(ValueError, match="mystery"):
        apply_condition(
            m,
            {
                "cofactor_pseudoreaction": {
                    "rxn_id": "cofac",
                    "charge_balance_met": "h_c",
                }
            },
        )


# --- biomass stoichiometry delta -------------------------------------

def test_biomass_stoichiometry_delta_combines_with_existing():
    m = _toy_model()
    bio = m.reactions.get_by_id("bio")
    atp = m.metabolites.get_by_id("atp_c")
    before = bio.metabolites[atp]
    apply_condition(
        m,
        {
            "biomass_stoichiometry_delta": {
                "rxn_id": "bio",
                "add": [{"met": "atp_c", "coef": 0.5}],
            }
        },
    )
    assert bio.metabolites[atp] == pytest.approx(before + 0.5)


# --- expected_uptake_count -------------------------------------------

def test_expected_uptake_count_mismatch_warns():
    m = _toy_model()
    cfg = {
        "bounds": [{"rxn": "EX_glc", "lb": -1000}],
        "expected_uptake_count": 5,
    }
    with pytest.warns(UserWarning, match="Expected 5 uptake reactions, applied 1"):
        apply_condition(m, cfg)


def test_expected_uptake_count_match_silent(recwarn):
    m = _toy_model()
    apply_condition(
        m,
        {
            "bounds": [{"rxn": "EX_glc", "lb": -1000}],
            "expected_uptake_count": 1,
        },
    )
    assert not any("Expected" in str(w.message) for w in recwarn)


# --- yaml path entrypoint --------------------------------------------

def test_apply_condition_accepts_yaml_path(tmp_path):
    cfg = {"bounds": [{"rxn": "EX_glc", "lb": -42}]}
    path = tmp_path / "cond.yml"
    path.write_text(_yaml_dump(cfg))
    m = _toy_model()
    apply_condition(m, path)
    assert m.reactions.get_by_id("EX_glc").lower_bound == -42


def test_load_condition_round_trip(tmp_path):
    cfg = {"name": "x", "bounds": [{"rxn": "r1", "lb": 0}]}
    path = tmp_path / "cond.yml"
    path.write_text(_yaml_dump(cfg))
    assert load_condition(path) == cfg


def test_load_condition_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_condition(tmp_path / "nope.yml")


# --- set_reaction_bounds public helper -------------------------------

def test_set_reaction_bounds_normal_case():
    m = _toy_model()
    rxn = m.reactions.get_by_id("EX_glc")
    set_reaction_bounds(rxn, -5, 5)
    assert rxn.bounds == (-5, 5)


def test_set_reaction_bounds_infeasible_case():
    m = _toy_model()
    rxn = m.reactions.get_by_id("EX_glc")
    set_reaction_bounds(rxn, 100, 0)
    assert rxn.lower_bound == 100
    assert rxn.upper_bound == 0


def test_load_condition_of_an_empty_file_is_an_empty_condition(tmp_path):
    path = tmp_path / "empty.yml"
    path.write_text("")
    assert load_condition(path) == {}
    path.write_text("# only a comment\n")
    assert load_condition(path) == {}


def test_apply_condition_with_an_empty_file_leaves_the_model_unchanged(tmp_path):
    path = tmp_path / "empty.yml"
    path.write_text("")
    m = _toy_model()
    before = {r.id: (r.lower_bound, r.upper_bound) for r in m.reactions}
    apply_condition(m, path)
    assert {r.id: (r.lower_bound, r.upper_bound) for r in m.reactions} == before
