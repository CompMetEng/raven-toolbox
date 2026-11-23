"""Tests for the GAM and NGAM fit (``raven_toolbox.biomass.maintenance``).

The toy model makes the uptake analytic. Glucose gives 2 ATP; the biomass reaction takes
5 glucose and ``GAM`` ATP per unit of biomass; the maintenance reaction takes ``NGAM`` ATP.
The glucose uptake at growth ``mu`` is therefore ``5 * mu + (GAM * mu + NGAM) / 2``.
"""
import cobra
import numpy as np
import pandas as pd
import pytest

from raven_toolbox.biomass import (
    MaintenanceConfig,
    bootstrap_gam,
    fit_gam,
    fit_maintenance,
    get_gam,
    huber_loss,
    log_ratio,
    min_uptake,
    ngam_from_maintenance_coefficient,
    set_gam,
    study_weights,
    uptake_grid,
)

NAMES = ("ATP", "ADP", "H2O", "H+", "phosphate")
CONFIG = MaintenanceConfig(
    biomass_rxn="BIO", ngam_rxn="NGAM", growth_rxn="GROWTH",
    substrate_exchange="EX_glc", cofactor_met_names=NAMES,
)
PROTON_OFFSET = 0.3  # balances the charge of the other biomass terms


def _toy(gam=30.0, ngam=1.0):
    model = cobra.Model("toy")
    mets = {
        i: cobra.Metabolite(i, name=n, compartment="c")
        for i, n in [("glc", "glucose"), ("atp", "ATP"), ("adp", "ADP"), ("h2o", "H2O"),
                     ("h", "H+"), ("pi", "phosphate"), ("bio", "biomass")]
    }
    m = mets

    def rxn(rid, stoich, lb=0.0, ub=1000.0):
        r = cobra.Reaction(rid)
        r.add_metabolites({m[k]: v for k, v in stoich.items()})
        r.bounds = (lb, ub)
        return r

    model.add_reactions([
        rxn("EX_glc", {"glc": -1}, -1000, 1000),
        rxn("ATPPROD", {"glc": -1, "adp": -2, "pi": -2, "h": -2, "atp": 2, "h2o": 2}),
        rxn("BIO", {"glc": -5, "atp": -gam, "h2o": -gam, "adp": gam, "pi": gam,
                    "h": gam + PROTON_OFFSET, "bio": 1}),
        rxn("GROWTH", {"bio": -1}),
        rxn("NGAM", {"atp": -1, "h2o": -1, "adp": 1, "pi": 1, "h": 1}, ngam, ngam),
        rxn("EX_h2o", {"h2o": -1}, -1000, 1000),
        rxn("EX_h", {"h": -1}, -1000, 1000),
        rxn("EX_pi", {"pi": -1}, -1000, 1000),
    ])
    model.objective = "GROWTH"
    return model


def _uptake(mu, gam, ngam):
    return 5 * mu + (gam * mu + ngam) / 2


# --- set_gam and get_gam ---------------------------------------------------------------


def test_get_gam_is_the_atp_coefficient():
    assert get_gam(_toy(gam=30.0), biomass_rxn="BIO") == 30.0


def test_get_gam_needs_one_atp_in_the_reaction():
    with pytest.raises(ValueError, match="needs exactly one"):
        get_gam(_toy(), biomass_rxn="BIO", atp_met_name="nothing")


def test_keep_offsets_shifts_every_term_and_keeps_the_proton_offset():
    model = _toy(gam=30.0)
    set_gam(model, 45.0, biomass_rxn="BIO", cofactor_met_names=NAMES, keep_offsets_of="ATP")
    coef = {met.name: c for met, c in model.reactions.BIO.metabolites.items()}
    assert coef["ATP"] == -45.0 and coef["H2O"] == -45.0
    assert coef["ADP"] == 45.0 and coef["phosphate"] == 45.0
    assert coef["H+"] == pytest.approx(45.0 + PROTON_OFFSET)


def test_without_keep_offsets_every_term_is_set_to_the_value():
    model = _toy(gam=30.0)
    set_gam(model, 45.0, biomass_rxn="BIO", cofactor_met_names=NAMES)
    coef = {met.name: c for met, c in model.reactions.BIO.metabolites.items()}
    assert coef["H+"] == 45.0


def test_keep_offsets_reverts_in_a_context():
    model = _toy(gam=30.0)
    with model:
        set_gam(model, 45.0, biomass_rxn="BIO", cofactor_met_names=NAMES,
                ngam_rxn="NGAM", ngam_value=2.0, keep_offsets_of="ATP")
        assert get_gam(model, biomass_rxn="BIO") == 45.0
    assert get_gam(model, biomass_rxn="BIO") == 30.0
    assert model.reactions.NGAM.bounds == (1.0, 1.0)


def test_keep_offsets_of_needs_one_metabolite_of_that_name_in_the_reaction():
    with pytest.raises(ValueError, match="needs exactly one"):
        set_gam(_toy(), 45.0, biomass_rxn="BIO", cofactor_met_names=("nothing", "ATP"),
                keep_offsets_of="nothing")


def test_get_gam_needs_the_biomass_reaction():
    with pytest.raises(KeyError):
        get_gam(_toy(), biomass_rxn="NOPE")


def test_keep_offsets_sets_ngam_too():
    model = _toy(gam=30.0)
    set_gam(model, 45.0, biomass_rxn="BIO", cofactor_met_names=NAMES,
            ngam_rxn="NGAM", ngam_value=2.0, keep_offsets_of="ATP")
    assert model.reactions.NGAM.bounds == (2.0, 2.0)


def test_keep_offsets_of_must_be_a_cofactor():
    with pytest.raises(ValueError, match="not one of the cofactor names"):
        set_gam(_toy(), 45.0, biomass_rxn="BIO", cofactor_met_names=("ADP",), keep_offsets_of="ATP")


# --- loss, ratios, weights ----------------------------------------------------------------


def test_huber_loss_is_quadratic_then_linear():
    r = np.array([0.0, 0.1, -0.1, 0.45])
    assert huber_loss(r, delta=0.15) == pytest.approx(
        [0.0, 0.005, 0.005, 0.15 * (0.45 - 0.075)])


def test_log_ratio_broadcasts_and_penalises_infeasible_rows():
    grid = np.array([[2.0, np.nan], [1.0, 4.0]])
    out = log_ratio(grid, np.array([1.0, 2.0]), infeasible=0.7)
    assert out == pytest.approx(np.array([[np.log(2.0), 0.7], [0.0, np.log(2.0)]]))


def test_study_weights_count_studies_equally_and_discount():
    w = study_weights(["a"] * 4 + ["b"] * 4 + ["c"],
                      discounted=[False] * 4 + [True] * 4 + [False])
    assert w[:4] == pytest.approx(0.5)
    assert w[4:8] == pytest.approx(0.25)
    assert w[8] == pytest.approx(1.0)


def test_study_weights_without_discount():
    assert study_weights(["a", "a", "b"]) == pytest.approx([2 ** -0.5, 2 ** -0.5, 1.0])


# --- simulation ------------------------------------------------------------------------------


def test_min_uptake_matches_the_analytic_value_and_leaves_the_model_as_it_was():
    model = _toy(gam=30.0, ngam=1.0)
    assert min_uptake(model, 0.1, CONFIG) == pytest.approx(_uptake(0.1, 30.0, 1.0))
    assert model.reactions.GROWTH.bounds == (0.0, 1000.0)
    assert model.reactions.EX_glc.lower_bound == -1000.0
    assert model.objective.expression.has(model.reactions.GROWTH.forward_variable)


def test_min_uptake_is_nan_when_the_growth_rate_cannot_be_reached():
    model = _toy()
    model.reactions.ATPPROD.upper_bound = 0.1  # too little ATP for the growth rate
    assert np.isnan(min_uptake(model, 0.8, CONFIG))


def test_min_uptake_runs_the_setup_inside_the_context():
    model = _toy(gam=30.0)

    def setup(m):
        m.reactions.EX_glc.upper_bound = 0.0
        m.reactions.NGAM.bounds = (3.0, 3.0)

    assert min_uptake(model, 0.0, CONFIG, setup=setup) == pytest.approx(1.5)
    assert model.reactions.NGAM.bounds == (1.0, 1.0)


def test_uptake_grid_covers_every_gam_and_row():
    model = _toy()
    data = pd.DataFrame({"growth_rate": [0.05, 0.1, 0.2]})
    grid = uptake_grid(model, data, [20.0, 40.0], 0.8, CONFIG)
    expected = [[_uptake(mu, g, 0.8) for mu in data["growth_rate"]] for g in (20.0, 40.0)]
    assert grid == pytest.approx(np.array(expected))
    assert get_gam(model, biomass_rxn="BIO") == 30.0


def test_uptake_grid_calls_the_condition_setup_for_each_row():
    model = _toy()
    data = pd.DataFrame({"growth_rate": [0.1, 0.1], "extra": [0.0, 10.0]})

    def setup(m, row):
        m.reactions.EX_glc.upper_bound = row["extra"]  # not binding: only checks the call

    seen = []

    def spy(m, row):
        seen.append(row["extra"])
        setup(m, row)

    uptake_grid(model, data, [30.0], 1.0, CONFIG, condition_setup=spy)
    assert seen == [0.0, 10.0]


def test_ngam_from_maintenance_coefficient_inverts_the_uptake_at_zero_growth():
    model = _toy(gam=30.0, ngam=1.0)
    ngam = ngam_from_maintenance_coefficient(model, 0.4, CONFIG)
    assert ngam == pytest.approx(0.8)  # uptake at zero growth is NGAM / 2
    assert model.reactions.NGAM.bounds == (1.0, 1.0)


def test_ngam_from_maintenance_coefficient_with_a_given_gam_keeps_offsets():
    model = _toy(gam=30.0)
    assert ngam_from_maintenance_coefficient(model, 0.4, CONFIG, gam=50.0) == pytest.approx(0.8)
    assert get_gam(model, biomass_rxn="BIO") == 30.0


# --- fit -----------------------------------------------------------------------------------------

TRUE_GAM, TRUE_NGAM = 40.0, 0.8


def _chemostats(noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for study in "abcdef":
        for mu in (0.05, 0.1, 0.2, 0.3):
            up = _uptake(mu, TRUE_GAM, TRUE_NGAM) * (1 + noise * rng.standard_normal())
            rows.append({"study": study, "growth_rate": mu, "uptake": up})
    return pd.DataFrame(rows)


def test_fit_gam_recovers_the_gam_the_data_were_made_with():
    model = _toy(gam=30.0)
    data = _chemostats()
    gam, grid, ratios = fit_gam(model, data, TRUE_NGAM, CONFIG, gams=np.arange(30.0, 50.1, 2.5))
    assert gam == 40.0
    assert ratios.shape == (len(grid), len(data))
    assert ratios[list(grid).index(40.0)] == pytest.approx(0.0, abs=1e-9)


def test_fit_gam_one_outlier_study_moves_the_result_by_less_than_a_grid_step():
    """A study that reads three times the uptake of the others, with a quadratic loss,
    would pull the best GAM far above 40; the Huber loss keeps it within one step."""
    model = _toy(gam=30.0)
    data = _chemostats()
    data.loc[data["study"] == "a", "uptake"] *= 3.0
    weights = study_weights(data["study"])
    gam, _, _ = fit_gam(model, data, TRUE_NGAM, CONFIG, gams=np.arange(30.0, 50.1, 2.5),
                        weights=weights)
    assert abs(gam - 40.0) <= 2.5


def test_bootstrap_gam_is_exact_without_noise_and_wider_with_a_dissenting_study():
    gams = np.arange(30.0, 50.1, 2.5)
    model = _toy(gam=30.0)
    data = _chemostats()
    weights = study_weights(data["study"])
    _, grid, ratios = fit_gam(model, data, TRUE_NGAM, CONFIG, gams=gams, weights=weights)
    assert bootstrap_gam(grid, ratios, data["study"], weights, n=50) == (40.0, 40.0)

    data.loc[data["study"] == "a", "uptake"] *= 1.5
    _, grid, ratios = fit_gam(model, data, TRUE_NGAM, CONFIG, gams=gams, weights=weights)
    lo, hi = bootstrap_gam(grid, ratios, data["study"], weights, n=200)
    assert lo <= 40.0 <= hi and hi > lo


def test_bootstrap_gam_is_reproducible_by_seed():
    gams = np.array([30.0, 40.0, 50.0])
    rng = np.random.default_rng(0)
    ratios = rng.normal(size=(3, 12))
    study = list("aabbccddeeff")
    w = np.ones(12)
    assert bootstrap_gam(gams, ratios, study, w, seed=3) == bootstrap_gam(gams, ratios, study, w, seed=3)


def test_fit_maintenance_returns_the_fit_and_leaves_the_model_alone():
    model = _toy(gam=30.0)
    data = _chemostats()
    fit = fit_maintenance(model, data, TRUE_NGAM / 2, CONFIG,
                          gams=np.arange(30.0, 50.1, 2.5), bootstrap=30)
    assert fit.ngam == pytest.approx(TRUE_NGAM)
    assert fit.gam == 40.0 and fit.gam_p16 == 40.0 and fit.gam_p84 == 40.0
    assert fit.units == 24 and fit.studies == 6
    assert fit.median_error == pytest.approx(0.0, abs=1e-9)
    assert get_gam(model, biomass_rxn="BIO") == 30.0
    assert model.reactions.NGAM.bounds == (1.0, 1.0)


def test_fit_maintenance_discounts_rows_and_sets_up_conditions():
    model = _toy(gam=30.0)
    data = _chemostats()
    data["note"] = np.where(data["study"] == "a", "figure", "table")
    calls = []

    def setup(m, row):
        calls.append(row["study"])

    fit = fit_maintenance(
        model, data, TRUE_NGAM / 2, CONFIG, gams=np.arange(30.0, 50.1, 2.5), bootstrap=5,
        discounted=data["note"] == "figure", condition_setup=setup,
    )
    assert fit.gam == 40.0
    assert len(calls) == 24 * 9  # nine GAM values
