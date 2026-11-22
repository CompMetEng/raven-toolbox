"""Tests for experiments.data."""
from raven_toolbox.experiments import (
    ExperimentData,
    load_experiment_data,
)

_FLUX_DATA_TSV = (
    "Condition\tPtot\tgrRate\tglucose (r_1714)\tethanol (r_1761)\t"
    "weight\tsource\n"
    "glucose\tNaN\t0.36\t-13.33\t19.83\t1\tDLKcat\n"
    "ethanol\tNaN\t0.12\tNaN\t-8.5\t1\tbrenda\n"
)

_MAX_GROWTH_TSV = (
    "Condition\tPtot\tgrRate\tglucose (r_1714)\tfructose (r_1709)\t"
    "weight\tsource\n"
    "glucose\tNaN\t0.41\t-1000\tNaN\t1\tDLKcat\n"
    "fructose\tNaN\t0.338\tNaN\t-1000\t1\tDLKcat\n"
)

_ZERO_EXCH_TSV = "Rxns\nr_1542\nr_1545\nr_1546\n"


def test_load_experiment_data_all_files_present(tmp_path):
    data_dir = tmp_path
    (data_dir / "fluxData.tsv").write_text(_FLUX_DATA_TSV)
    (data_dir / "maxGrowth.tsv").write_text(_MAX_GROWTH_TSV)
    (data_dir / "zeroFlux.tsv").write_text(_ZERO_EXCH_TSV)

    experiment_data = load_experiment_data(tmp_path)

    assert isinstance(experiment_data, ExperimentData)

    assert experiment_data.flux_data is not None
    assert experiment_data.flux_data.conds == ["glucose", "ethanol"]
    assert experiment_data.flux_data.exch_rxn_ids == ["r_1714", "r_1761"]
    assert experiment_data.flux_data.source == ["DLKcat", "brenda"]
    assert list(experiment_data.flux_data.weight) == [1.0, 1.0]

    assert experiment_data.max_growth is not None
    assert experiment_data.max_growth.conds == ["glucose", "fructose"]
    assert experiment_data.max_growth.exch_rxn_ids == ["r_1714", "r_1709"]

    assert experiment_data.zero_flux == ["r_1542", "r_1545", "r_1546"]


def test_load_experiment_data_all_files_missing(tmp_path):

    experiment_data = load_experiment_data(tmp_path)

    assert experiment_data.flux_data is None
    assert experiment_data.max_growth is None
    assert experiment_data.zero_flux == []


def test_load_experiment_data_partial(tmp_path):
    """Only flux_data present: max_growth stays None."""
    data_dir = tmp_path
    (data_dir / "fluxData.tsv").write_text(_FLUX_DATA_TSV)

    experiment_data = load_experiment_data(tmp_path)

    assert experiment_data.flux_data is not None
    assert experiment_data.max_growth is None
    assert experiment_data.zero_flux == []
