"""Tests for experiments.conditions and experiments.distance.

A tiny two-carbon-source toy model with a known enzyme cap, so every
simulated growth/exchange-flux value used below is hand-predictable
from the model's stoichiometry rather than treated as a black box.
"""
import warnings

import cobra
import numpy as np
import pytest

from raven_toolbox.experiments.data import ExperimentData
from raven_toolbox.experiments.flux_data import FluxData
from raven_toolbox.experiments.distance import (
    BIOMASS_CARBON_EQUIV,
    experiment_distance,
    compute_carbon_weights,
    dataset_rmse,
)
from raven_toolbox.experiments.conditions import (
    ConditionSimResult,
    simulate_conditions,
)


# --------------------------------------------------------------------------- #
# Toy two-carbon-source model fixture
# --------------------------------------------------------------------------- #
#
# glucose and ethanol each feed a shared pool metabolite A_c, which the
# single enzyme-gated reaction R turns into biomass at an enzyme-capped
# rate of kcat*3600/mw = 1*3600/100 = 36 (pool size 1 mg/gDW). R also
# co-produces a small fixed amount of a "byproduct" the experimental
# data assumes is never made (byp_c, 0.1 mol per mol of R flux) -- this
# exercises `zero_flux_rxns`' nonzero-simulated-flux case.
#
# - "glucose" condition: uptake unconstrained relative to the enzyme
#   cap (measured flux -36 exactly matches TR's capacity), so growth
#   is enzyme-limited: 36.
# - "ethanol" condition: uptake capped at -20 (below the enzyme cap),
#   so growth is carbon-limited instead: 20.

_KCAT = 1.0
_MW = 100.0
_ENZYME_CAP = _KCAT * 3600.0 / _MW  # 36.0
_BYPRODUCT_YIELD = 0.1


def _build_toy() -> cobra.Model:
    model = cobra.Model("toy")

    glc_e = cobra.Metabolite("glc_e", compartment="e", formula="C6H12O6")
    eth_e = cobra.Metabolite("eth_e", compartment="e", formula="C2H6O")
    byp_c = cobra.Metabolite("byp_c", compartment="c", formula="C1")
    A_c = cobra.Metabolite("A_c", compartment="c")
    pool = cobra.Metabolite("prot_pool", compartment="c")
    prot_E = cobra.Metabolite("prot_E", compartment="c")
    bio_met = cobra.Metabolite("bio_met", compartment="c")
    model.add_metabolites([glc_e, eth_e, byp_c, A_c, pool, prot_E, bio_met])

    EX_glc = cobra.Reaction("EX_glc")
    EX_glc.add_metabolites({glc_e: -1.0})
    EX_glc.bounds = (-1000.0, 0.0)

    EX_eth = cobra.Reaction("EX_eth")
    EX_eth.add_metabolites({eth_e: -1.0})
    EX_eth.bounds = (-1000.0, 0.0)

    EX_byp = cobra.Reaction("EX_byp")
    EX_byp.add_metabolites({byp_c: -1.0})
    EX_byp.bounds = (0.0, 1000.0)

    TR_glc = cobra.Reaction("TR_glc")
    TR_glc.add_metabolites({glc_e: -1.0, A_c: 1.0})
    TR_glc.bounds = (0.0, 1000.0)

    TR_eth = cobra.Reaction("TR_eth")
    TR_eth.add_metabolites({eth_e: -1.0, A_c: 1.0})
    TR_eth.bounds = (0.0, 1000.0)

    coeff = _MW / (_KCAT * 3600.0)  # mg/mmol per unit flux
    R = cobra.Reaction("R")
    R.add_metabolites(
        {A_c: -1.0, prot_E: -coeff, bio_met: 1.0, byp_c: _BYPRODUCT_YIELD}
    )
    R.bounds = (0.0, 1000.0)

    BIO = cobra.Reaction("biomass")
    BIO.add_metabolites({bio_met: -1.0})
    BIO.bounds = (0.0, 1000.0)

    pool_ex = cobra.Reaction("prot_pool_exchange")
    pool_ex.add_metabolites({pool: 1.0})
    pool_ex.bounds = (0.0, 1.0)  # 1 mg/gDW pool

    usage_E = cobra.Reaction("usage_prot_E")
    usage_E.add_metabolites({pool: -1.0, prot_E: 1.0})
    usage_E.bounds = (0.0, 1000.0)

    model.add_reactions(
        [EX_glc, EX_eth, EX_byp, TR_glc, TR_eth, R, BIO, pool_ex, usage_E]
    )
    model.objective = "biomass"
    return model


def _flux_data(*, ethanol_grrate: float) -> FluxData:
    """Two conditions: glucose (enzyme-limited, everything matches
    exactly) and ethanol (carbon-limited; ``ethanol_grrate`` controls
    whether the measured growth matches the simulated 20.0 exactly or
    deliberately deviates from it)."""
    return FluxData(
        conds=["glucose", "ethanol"],
        p_tot=np.array([np.nan, np.nan]),
        gr_rate=np.array([_ENZYME_CAP, ethanol_grrate]),
        exch_fluxes=np.array(
            [
                [-_ENZYME_CAP, np.nan],
                [np.nan, -20.0],
            ]
        ),
        exch_mets=["glucose", "ethanol"],
        exch_rxn_ids=["EX_glc", "EX_eth"],
        weight=np.array([1.0, 1.0]),
        source=["test", "test"],
    )


def _max_growth_data() -> FluxData:
    """Max-growth dataset: uptake fully opened, so both conditions hit
    the same enzyme-capped growth of 36."""
    return FluxData(
        conds=["glucose", "ethanol"],
        p_tot=np.array([np.nan, np.nan]),
        gr_rate=np.array([_ENZYME_CAP, _ENZYME_CAP]),
        exch_fluxes=np.array(
            [
                [-1000.0, np.nan],
                [np.nan, -1000.0],
            ]
        ),
        exch_mets=["glucose", "ethanol"],
        exch_rxn_ids=["EX_glc", "EX_eth"],
    )


# --------------------------------------------------------------------------- #
# simulate_conditions
# --------------------------------------------------------------------------- #

def test_simulate_flux_data_enzyme_vs_carbon_limited():
    model = _build_toy()
    flux_data = _flux_data(ethanol_grrate=20.0)

    results = simulate_conditions(
        model, flux_data,
        constrain=True, zero_flux_rxns=["EX_byp"], bio_rxn_id="biomass",
    )

    assert len(results) == 2
    glucose, ethanol = results

    # Glucose: enzyme-limited growth of 36, full carbon uptake used.
    assert glucose.feasible
    assert glucose.growth == pytest.approx(_ENZYME_CAP)
    assert glucose.exch_fluxes["EX_glc"] == pytest.approx(-_ENZYME_CAP)
    assert glucose.block_fluxes["EX_byp"] == pytest.approx(
        _ENZYME_CAP * _BYPRODUCT_YIELD
    )

    # Ethanol: carbon-limited to the -20 uptake cap, growth = 20.
    assert ethanol.feasible
    assert ethanol.growth == pytest.approx(20.0)
    assert ethanol.exch_fluxes["EX_eth"] == pytest.approx(-20.0)
    assert ethanol.block_fluxes["EX_byp"] == pytest.approx(20.0 * _BYPRODUCT_YIELD)

    # Bounds fully revert after the call (both `with model:` blocks exited).
    assert model.reactions.get_by_id("EX_glc").bounds == (-1000.0, 0.0)
    assert model.reactions.get_by_id("EX_eth").bounds == (-1000.0, 0.0)


def test_simulate_max_growth_opens_uptake_fully():
    model = _build_toy()
    max_growth = _max_growth_data()

    results = simulate_conditions(
        model, max_growth,
        constrain=False, zero_flux_rxns=[], bio_rxn_id="biomass",
    )

    assert [r.growth for r in results] == pytest.approx([_ENZYME_CAP, _ENZYME_CAP])
    # max_growth rows carry no exchange-flux measurements to report back.
    assert results[0].exch_fluxes == {}
    assert results[0].block_fluxes == {}


def test_simulate_unmatched_condition_name_raises():
    model = _build_toy()
    flux_data = _flux_data(ethanol_grrate=20.0)
    flux_data.conds = ["glucose", "propionate"]

    with pytest.raises(ValueError, match="propionate"):
        simulate_conditions(
            model, flux_data,
            constrain=True, zero_flux_rxns=[], bio_rxn_id="biomass",
        )


def _objective_warnings(model, **kwargs):
    """The warnings about the objective that one call of simulate_conditions issues."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulate_conditions(
            model, _flux_data(ethanol_grrate=20.0),
            constrain=True, zero_flux_rxns=[], bio_rxn_id="biomass", **kwargs,
        )
    return [w for w in caught if "objective of the model" in str(w.message)]


def _clear_objective(model):
    for rxn in model.reactions:
        rxn.objective_coefficient = 0.0


def test_no_warning_when_growth_is_the_objective():
    assert _objective_warnings(_build_toy()) == []


def _another_reaction(model):
    model.objective = "EX_byp"


def _minimised_growth(model):
    model.objective_direction = "min"


def _two_reactions(model):
    model.objective = {model.reactions.get_by_id("biomass"): 1.0,
                       model.reactions.get_by_id("R"): 1.0}


def _no_objective(model):
    _clear_objective(model)


def _negative_growth(model):
    _clear_objective(model)
    model.objective = {model.reactions.get_by_id("biomass"): -1.0}


@pytest.mark.parametrize(
    "spoil",
    [_another_reaction, _minimised_growth, _two_reactions, _no_objective, _negative_growth],
)
def test_warning_once_when_growth_is_not_the_maximised_objective(spoil):
    model = _build_toy()
    spoil(model)

    caught = _objective_warnings(model)

    # One warning for the call, although two conditions are simulated.
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert 'model.objective = "biomass"' in str(caught[0].message)
    assert caught[0].filename == __file__


def test_warning_is_issued_with_an_optimize_function_too():
    model = _build_toy()
    _another_reaction(model)

    caught = _objective_warnings(model, optimize=lambda m: m.optimize())

    assert len(caught) == 1


def test_the_simulation_goes_on_after_the_warning():
    model = _build_toy()
    _another_reaction(model)

    with pytest.warns(UserWarning, match="objective of the model"):
        results = simulate_conditions(
            model, _flux_data(ethanol_grrate=20.0),
            constrain=True, zero_flux_rxns=[], bio_rxn_id="biomass",
        )

    assert len(results) == 2
    assert all(r.feasible for r in results)


# --------------------------------------------------------------------------- #
# compute_carbon_weights
# --------------------------------------------------------------------------- #

def test_compute_carbon_weights():
    model = _build_toy()

    carbon_weights = compute_carbon_weights(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    assert carbon_weights["EX_glc"] == pytest.approx(6.0)   # C6H12O6
    assert carbon_weights["EX_eth"] == pytest.approx(2.0)   # C2H6O
    assert carbon_weights["EX_byp"] == pytest.approx(1.0)   # C1
    assert carbon_weights["biomass"] == pytest.approx(BIOMASS_CARBON_EQUIV)


def test_compute_carbon_weights_zero_carbon_clamps_to_one():
    """A genuinely zero-carbon exchanged metabolite (e.g. O2) still
    gets weight 1, not 0, so those reactions' mismatches are not silently
    excluded from the RMSE."""
    model = _build_toy()
    o2_e = cobra.Metabolite("o2_e", compartment="e", formula="O2")
    model.add_metabolites([o2_e])
    EX_o2 = cobra.Reaction("EX_o2")
    EX_o2.add_metabolites({o2_e: -1.0})
    EX_o2.bounds = (-1000.0, 0.0)
    model.add_reactions([EX_o2])

    carbon_weights = compute_carbon_weights(model, ["EX_o2"], bio_rxn_id="biomass")

    assert carbon_weights["EX_o2"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# dataset_rmse / experiment_distance
# --------------------------------------------------------------------------- #

def test_dataset_rmse_measured_terms_match_only_byproduct_deviates():
    model = _build_toy()
    # ethanol_grrate=20.0 matches the ethanol condition's actual
    # carbon-limited growth exactly, and both conditions' own measured
    # exchange fluxes match the simulation exactly too -- the only
    # nonzero term left is the unmeasured byproduct (assumed
    # zero-flux, but nonzero in the simulation).
    flux_data = _flux_data(ethanol_grrate=20.0)
    sims = simulate_conditions(
        model, flux_data,
        constrain=True, zero_flux_rxns=["EX_byp"], bio_rxn_id="biomass",
    )
    carbon_weights = compute_carbon_weights(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    rmse, per_cond = dataset_rmse(
        flux_data, sims,
        constrain=True, carbon_weights=carbon_weights,
    )

    # The byproduct term is nonzero-simulated-vs-zero-measured, so RMSE
    # is not zero overall -- but every *measured* exchange/growth term
    # matches exactly, and the byproduct term is hand-computable.
    glucose_byp = _ENZYME_CAP * _BYPRODUCT_YIELD * 1.0  # carbon weight 1
    ethanol_byp = 20.0 * _BYPRODUCT_YIELD * 1.0
    # 3 terms per condition: growth (matches exactly -> 0), the
    # condition's own exchange flux (matches exactly -> 0), and the
    # byproduct block term (measured 0 vs the nonzero simulated value).
    expected_glucose_rmse = np.sqrt(np.mean([0.0, 0.0, glucose_byp ** 2]))
    expected_ethanol_rmse = np.sqrt(np.mean([0.0, 0.0, ethanol_byp ** 2]))

    assert per_cond[0] == pytest.approx(expected_glucose_rmse)
    assert per_cond[1] == pytest.approx(expected_ethanol_rmse)
    assert rmse == pytest.approx(np.mean([expected_glucose_rmse, expected_ethanol_rmse]))


def test_dataset_rmse_growth_mismatch_is_hand_computable():
    model = _build_toy()
    # Deliberately wrong measured growth for the ethanol condition
    # (18.0 vs the actual carbon-limited 20.0) -- a known, hand-provable
    # deviation on top of the byproduct term.
    flux_data = _flux_data(ethanol_grrate=18.0)
    sims = simulate_conditions(
        model, flux_data,
        constrain=True, zero_flux_rxns=["EX_byp"], bio_rxn_id="biomass",
    )
    carbon_weights = compute_carbon_weights(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    _, per_cond = dataset_rmse(
        flux_data, sims,
        constrain=True, carbon_weights=carbon_weights,
    )

    growth_diff = (18.0 - 20.0) * BIOMASS_CARBON_EQUIV  # bio_meas - bio_sim
    byp_diff = 0.0 - (20.0 * _BYPRODUCT_YIELD)  # measured 0 vs simulated
    expected = np.sqrt(np.mean([growth_diff ** 2, 0.0, byp_diff ** 2]))
    assert per_cond[1] == pytest.approx(expected)


def test_dataset_rmse_infeasible_condition_gets_penalty():
    flux_data = _flux_data(ethanol_grrate=20.0)
    sims = [ConditionSimResult(feasible=False), ConditionSimResult(feasible=False)]
    carbon_weights = {}

    rmse, per_cond = dataset_rmse(
        flux_data, sims,
        constrain=True, carbon_weights=carbon_weights, penalty=99.0,
    )

    assert list(per_cond) == [99.0, 99.0]
    assert rmse == pytest.approx(99.0)


def test_dataset_rmse_max_growth_scores_growth_only():
    max_growth = _max_growth_data()
    sims = [
        ConditionSimResult(feasible=True, growth=_ENZYME_CAP),
        ConditionSimResult(feasible=True, growth=30.0),  # deliberately off
    ]

    rmse, per_cond = dataset_rmse(
        max_growth, sims,
        constrain=False, carbon_weights={},
    )

    assert per_cond[0] == pytest.approx(0.0)
    expected_second = abs((_ENZYME_CAP - 30.0) * BIOMASS_CARBON_EQUIV)
    assert per_cond[1] == pytest.approx(expected_second)
    assert rmse == pytest.approx(np.mean([0.0, expected_second]))


def test_experiment_distance_combines_both_datasets():
    model = _build_toy()
    flux_data = _flux_data(ethanol_grrate=20.0)
    max_growth = _max_growth_data()
    experiment_data = ExperimentData(flux_data=flux_data, max_growth=max_growth, zero_flux=["EX_byp"])

    flux_sims = simulate_conditions(
        model, flux_data,
        constrain=True, zero_flux_rxns=["EX_byp"], bio_rxn_id="biomass",
    )
    max_growth_sims = simulate_conditions(
        model, max_growth,
        constrain=False, zero_flux_rxns=[], bio_rxn_id="biomass",
    )
    carbon_weights = compute_carbon_weights(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    rmse, detail = experiment_distance(
        experiment_data,
        flux_sims=flux_sims, max_growth_sims=max_growth_sims,
        carbon_weights=carbon_weights,
    )

    expected_flux_rmse, _ = dataset_rmse(
        flux_data, flux_sims, constrain=True, carbon_weights=carbon_weights,
    )
    expected_max_growth_rmse, _ = dataset_rmse(
        max_growth, max_growth_sims, constrain=False, carbon_weights=carbon_weights,
    )
    assert rmse == pytest.approx(
        np.mean([expected_flux_rmse, expected_max_growth_rmse])
    )
    assert "flux_data" in detail and "max_growth" in detail


def test_experiment_distance_max_growth_weight_scales_the_max_growth_term():
    """``max_growth_weight`` scales the max-growth term, so 2 makes the
    eight max-growth conditions count double against the 33 flux ones."""
    model = _build_toy()
    flux_data = _flux_data(ethanol_grrate=20.0)
    max_growth = _max_growth_data()
    experiment_data = ExperimentData(flux_data=flux_data, max_growth=max_growth, zero_flux=["EX_byp"])

    flux_sims = simulate_conditions(
        model, flux_data,
        constrain=True, zero_flux_rxns=["EX_byp"], bio_rxn_id="biomass",
    )
    max_growth_sims = simulate_conditions(
        model, max_growth,
        constrain=False, zero_flux_rxns=[], bio_rxn_id="biomass",
    )
    carbon_weights = compute_carbon_weights(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    expected_flux_rmse, _ = dataset_rmse(
        flux_data, flux_sims, constrain=True, carbon_weights=carbon_weights,
    )
    expected_max_growth_rmse, _ = dataset_rmse(
        max_growth, max_growth_sims, constrain=False, carbon_weights=carbon_weights,
    )

    rmse, _ = experiment_distance(
        experiment_data,
        flux_sims=flux_sims, max_growth_sims=max_growth_sims,
        carbon_weights=carbon_weights, max_growth_weight=2.0,
    )
    assert rmse == pytest.approx(
        (expected_flux_rmse + 2.0 * expected_max_growth_rmse) / 3.0
    )

    # 0.5 weights the flux term twice as much as the max-growth term.
    rmse_half, _ = experiment_distance(
        experiment_data,
        flux_sims=flux_sims, max_growth_sims=max_growth_sims,
        carbon_weights=carbon_weights, max_growth_weight=0.5,
    )
    assert rmse_half == pytest.approx(
        (2.0 * expected_flux_rmse + expected_max_growth_rmse) / 3.0
    )

    # A lone dataset carries the whole score whatever the weight.
    solo = ExperimentData(flux_data=flux_data, max_growth=None, zero_flux=["EX_byp"])
    rmse_solo, _ = experiment_distance(
        solo,
        flux_sims=flux_sims, max_growth_sims=None,
        carbon_weights=carbon_weights, max_growth_weight=2.0,
    )
    assert rmse_solo == pytest.approx(expected_flux_rmse)


def test_experiment_distance_missing_dataset_contributes_nothing():
    model = _build_toy()
    flux_data = _flux_data(ethanol_grrate=20.0)
    experiment_data = ExperimentData(flux_data=flux_data, max_growth=None, zero_flux=["EX_byp"])

    flux_sims = simulate_conditions(
        model, flux_data,
        constrain=True, zero_flux_rxns=["EX_byp"], bio_rxn_id="biomass",
    )
    carbon_weights = compute_carbon_weights(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    rmse, detail = experiment_distance(
        experiment_data,
        flux_sims=flux_sims, max_growth_sims=None,
        carbon_weights=carbon_weights,
    )

    expected_flux_rmse, _ = dataset_rmse(
        flux_data, flux_sims, constrain=True, carbon_weights=carbon_weights,
    )
    assert rmse == pytest.approx(expected_flux_rmse)
    assert set(detail) == {"flux_data"}


def test_simulate_uses_the_given_optimize_function():
    """``optimize`` replaces ``model.optimize()`` for every condition."""
    model = _build_toy()
    calls = []

    def optimize(m):
        calls.append(m)
        return m.optimize()

    sims = simulate_conditions(
        model, _flux_data(ethanol_grrate=20.0),
        constrain=True, zero_flux_rxns=[], bio_rxn_id="biomass",
        optimize=optimize,
    )

    assert len(calls) == 2
    assert all(s.feasible for s in sims)


# --------------------------------------------------------------------------- #
# Conditions with weight 0 are left out
# --------------------------------------------------------------------------- #

def _with_std_row(flux_data: FluxData) -> FluxData:
    """The data with a first condition named ``Std`` that matches no exchange and has
    weight 0, as the condition of a protein measurement in a tutorial file."""
    n_rxns = len(flux_data.exch_rxn_ids)
    weight = np.ones(len(flux_data.conds)) if flux_data.weight is None else flux_data.weight
    return FluxData(
        conds=["Std"] + list(flux_data.conds),
        p_tot=np.concatenate([[0.5], flux_data.p_tot]),
        gr_rate=np.concatenate([[0.1], flux_data.gr_rate]),
        exch_fluxes=np.vstack([np.full((1, n_rxns), np.nan), flux_data.exch_fluxes]),
        exch_mets=list(flux_data.exch_mets),
        exch_rxn_ids=list(flux_data.exch_rxn_ids),
        weight=np.concatenate([[0.0], weight]),
    )


def _simulate(model, flux_data, *, constrain):
    return simulate_conditions(
        model, flux_data,
        constrain=constrain, zero_flux_rxns=["EX_byp"] if constrain else [],
        bio_rxn_id="biomass",
    )


def test_a_condition_with_weight_zero_is_not_simulated_or_matched():
    model = _build_toy()
    flux_data = _flux_data(ethanol_grrate=20.0)
    with_std = _with_std_row(flux_data)

    sims = _simulate(model, with_std, constrain=True)
    reference = _simulate(_build_toy(), flux_data, constrain=True)

    assert len(sims) == 3
    assert sims[0].skipped and not sims[0].feasible
    assert [s.skipped for s in sims[1:]] == [False, False]
    assert [s.growth for s in sims[1:]] == [s.growth for s in reference]


def test_a_condition_with_weight_zero_does_not_block_its_exchange():
    model = _build_toy()
    flux_data = _flux_data(ethanol_grrate=20.0)
    seen = []

    def optimize(m):
        seen.append(m.reactions.get_by_id("EX_eth").lower_bound)
        return m.optimize()

    flux_data.weight = np.array([1.0, 0.0])
    _simulate_with(model, flux_data, optimize)
    assert seen == [-1000.0]

    seen.clear()
    flux_data.weight = np.array([1.0, 1.0])
    _simulate_with(model, flux_data, optimize)
    assert seen == [0.0, -20.0]


def _simulate_with(model, flux_data, optimize):
    return simulate_conditions(
        model, flux_data, constrain=True, zero_flux_rxns=[], bio_rxn_id="biomass",
        optimize=optimize,
    )


@pytest.mark.parametrize("constrain", [True, False])
def test_a_condition_with_weight_zero_does_not_change_the_distance(constrain):
    model = _build_toy()
    flux_data = _flux_data(ethanol_grrate=20.0) if constrain else _max_growth_data()
    with_std = _with_std_row(flux_data)
    carbon_weights = compute_carbon_weights(
        model, ["EX_glc", "EX_eth", "EX_byp", "biomass"], bio_rxn_id="biomass",
    )

    rmse, per_cond = dataset_rmse(
        with_std, _simulate(model, with_std, constrain=constrain),
        constrain=constrain, carbon_weights=carbon_weights,
    )
    expected, expected_per_cond = dataset_rmse(
        flux_data, _simulate(_build_toy(), flux_data, constrain=constrain),
        constrain=constrain, carbon_weights=carbon_weights,
    )

    assert rmse == expected
    assert np.isnan(per_cond[0])
    assert list(per_cond[1:]) == list(expected_per_cond)


def test_experiment_distance_leaves_out_a_dataset_without_active_conditions():
    model = _build_toy()
    flux_data = _flux_data(ethanol_grrate=20.0)
    flux_data.weight = np.array([0.0, 0.0])
    max_growth = _max_growth_data()
    experiment_data = ExperimentData(flux_data=flux_data, max_growth=max_growth)
    carbon_weights = compute_carbon_weights(
        model, ["EX_glc", "EX_eth", "biomass"], bio_rxn_id="biomass",
    )
    flux_sims = _simulate(model, flux_data, constrain=True)
    max_growth_sims = _simulate(model, max_growth, constrain=False)
    assert all(s.skipped for s in flux_sims)

    rmse, detail = experiment_distance(
        experiment_data, flux_sims=flux_sims, max_growth_sims=max_growth_sims,
        carbon_weights=carbon_weights,
    )
    expected, _ = dataset_rmse(
        max_growth, max_growth_sims, constrain=False, carbon_weights=carbon_weights,
    )

    assert rmse == expected
    assert np.isnan(detail["flux_data"]).all()


def test_the_distance_is_nan_when_every_condition_has_weight_zero():
    model = _build_toy()
    flux_data = _flux_data(ethanol_grrate=20.0)
    flux_data.weight = np.array([0.0, 0.0])
    carbon_weights = compute_carbon_weights(
        model, ["EX_glc", "EX_eth", "biomass"], bio_rxn_id="biomass",
    )
    sims = _simulate(model, flux_data, constrain=True)

    rmse, per_cond = dataset_rmse(
        flux_data, sims, constrain=True, carbon_weights=carbon_weights,
    )
    distance, _ = experiment_distance(
        ExperimentData(flux_data=flux_data), flux_sims=sims, max_growth_sims=None,
        carbon_weights=carbon_weights,
    )

    assert np.isnan(rmse) and np.isnan(per_cond).all()
    assert np.isnan(distance)


# --------------------------------------------------------------------------- #
# Missing reactions
# --------------------------------------------------------------------------- #

def test_simulate_missing_reactions_raise_before_anything_is_solved():
    model = _build_toy()
    solved = []

    def optimize(m):
        solved.append(1)
        return m.optimize()

    with pytest.raises(ValueError, match="The reactions EX_nope, nobio are not in the model."):
        simulate_conditions(
            model, _flux_data(ethanol_grrate=20.0),
            constrain=True, zero_flux_rxns=["EX_nope"], bio_rxn_id="nobio",
            optimize=optimize,
        )
    assert solved == []


def test_simulate_missing_exchange_reaction_raises():
    model = _build_toy()
    model.remove_reactions(["EX_eth"])
    with pytest.raises(ValueError, match="The reactions EX_eth are not in the model."):
        simulate_conditions(
            model, _flux_data(ethanol_grrate=20.0),
            constrain=True, zero_flux_rxns=[], bio_rxn_id="biomass",
        )


# --------------------------------------------------------------------------- #
# Carbon atoms of a formula (the same list is tested for computeCarbonWeights in RAVEN)
# --------------------------------------------------------------------------- #

CARBON_FORMULAS = [
    ("C6H12O6", 6.0),
    ("CO2", 1.0),
    ("C", 1.0),
    ("C2.5H5", 2.5),
    ("C6H12O6(H2O)", 1.0),
    ("C6H12O6(H2O)2", 1.0),
    ("CuSO4.5H2O", 1.0),
    ("C6H12O6*H2O", 6.0),
    ("(C5H8)n", 1.0),
    ("C3H5O3-", 3.0),
    ("C6H12O6R", 6.0),
    ("H2O", 1.0),
    ("O2", 1.0),
    ("Co", 1.0),
    ("Xx", 1.0),
    ("c6h12o6", 1.0),
    ("C0H2", 1.0),
    ("C6H12O6 ", 6.0),
    ("", 1.0),
    (None, 1.0),
]


@pytest.mark.parametrize("formula, expected", CARBON_FORMULAS)
def test_compute_carbon_weights_of_a_formula(formula, expected):
    """A formula that is read gives its number of carbon atoms; one that is not read (empty,
    with parentheses, unknown), and one without carbon, gives 1."""
    model = cobra.Model("formulas")
    glc = cobra.Metabolite("m_e", formula="C6H12O6")  # keeps the model from having no formula
    met = cobra.Metabolite("x_e", formula=formula)
    model.add_metabolites([glc, met])
    ex_glc, ex_x = cobra.Reaction("EX_glc"), cobra.Reaction("EX_x")
    model.add_reactions([ex_glc, ex_x])
    ex_glc.add_metabolites({glc: -1.0})
    ex_x.add_metabolites({met: -1.0})

    weights = compute_carbon_weights(model, ["EX_x"], bio_rxn_id="biomass")

    assert weights["EX_x"] == pytest.approx(expected)
