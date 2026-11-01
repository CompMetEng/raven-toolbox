"""Tests for raven_toolbox.io.metadata and its use by write_yaml_model / export_for_git."""
import datetime

import cobra
import pytest

from raven_toolbox.io import (
    export_for_git,
    get_model_metadata,
    read_yaml_model,
    set_model_metadata,
    write_yaml_model,
)


@pytest.fixture
def model():
    m = cobra.Model("m1")
    m.add_metabolites([cobra.Metabolite("a", compartment="c"), cobra.Metabolite("b", compartment="c")])
    r = cobra.Reaction("R1")
    m.add_reactions([r])
    r.add_metabolites({m.metabolites.a: -1, m.metabolites.b: 1})
    return m


def test_set_then_get(model):
    set_model_metadata(model, version="1.2.0", date="2026-10-05", taxonomy="10090")
    assert get_model_metadata(model) == {"version": "1.2.0", "date": "2026-10-05", "taxonomy": "10090"}


def test_date_object_is_stored_as_iso(model):
    set_model_metadata(model, date=datetime.date(2026, 10, 5))
    assert get_model_metadata(model)["date"] == "2026-10-05"


def test_unset_arguments_are_left_alone(model):
    set_model_metadata(model, version="1.0.0", date="2026-01-01")
    set_model_metadata(model, version="1.1.0")
    assert get_model_metadata(model) == {"version": "1.1.0", "date": "2026-01-01"}


def test_unknown_field_is_an_error(model):
    with pytest.raises(ValueError, match="unknown metadata"):
        set_model_metadata(model, colour="red")


def test_get_falls_back_to_metadata_version(model):
    model.notes = {"metaData": {"version": "3.0.0"}}
    assert get_model_metadata(model)["version"] == "3.0.0"


def test_yaml_round_trip_carries_version_and_date(model, tmp_path):
    set_model_metadata(model, version="1.2.0", date="2026-10-05")
    write_yaml_model(model, tmp_path / "m.yml")
    meta = get_model_metadata(read_yaml_model(tmp_path / "m.yml"))
    assert (meta["version"], meta["date"]) == ("1.2.0", "2026-10-05")


def test_export_for_git_stamps_a_copy(model, tmp_path):
    root = export_for_git(model, tmp_path, prefix="m", formats=("yml",), version="2.0.0", date="2026-10-06")
    meta = get_model_metadata(read_yaml_model(root / "yml" / "m.yml"))
    assert (meta["version"], meta["date"]) == ("2.0.0", "2026-10-06")
    assert get_model_metadata(model) == {}


def test_datetime_is_reduced_to_its_date(model):
    set_model_metadata(model, date=datetime.datetime(2026, 10, 5, 13, 30))
    assert get_model_metadata(model)["date"] == "2026-10-05"


@pytest.mark.parametrize("bad", ["yesterday", "2026-13-01", "05/10/2026", 20261005])
def test_date_that_is_not_iso_is_an_error(model, bad):
    with pytest.raises(ValueError, match="date must be"):
        set_model_metadata(model, date=bad)
    assert get_model_metadata(model) == {}


def test_version_kept_only_in_metadata_is_written(model, tmp_path):
    model.notes = {"metaData": {"version": "3.0.0"}}
    write_yaml_model(model, tmp_path / "m.yml")
    assert get_model_metadata(read_yaml_model(tmp_path / "m.yml"))["version"] == "3.0.0"


def test_notes_version_wins_over_metadata_version(model, tmp_path):
    model.notes = {"version": "2.0.0", "metaData": {"version": "1.0.0"}}
    write_yaml_model(model, tmp_path / "m.yml")
    assert get_model_metadata(read_yaml_model(tmp_path / "m.yml"))["version"] == "2.0.0"


def test_export_for_git_stamps_sbml_too(model, tmp_path):
    root = export_for_git(model, tmp_path, prefix="m", formats=("xml",), version="2.0.0", date="2026-10-06")
    xml = (root / "xml" / "m.xml").read_text()
    assert "2.0.0" in xml and "2026-10-06" in xml
