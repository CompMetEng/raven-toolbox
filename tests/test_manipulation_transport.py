"""Tests for add_transport_reactions (addTransport port)."""
import cobra
import pytest

from raven_toolbox.manipulation import add_transport_reactions


@pytest.fixture
def model():
    m = cobra.Model("t")
    m.compartments = {"c": "cytoplasm", "m": "mitochondrion", "e": "extracellular"}
    m.add_metabolites(
        [
            cobra.Metabolite("atp_c", name="ATP", formula="C10H16N5O13P3", charge=-4, compartment="c"),
            cobra.Metabolite("h2o_c", name="H2O", formula="H2O", compartment="c"),
            cobra.Metabolite("atp_m", name="ATP", compartment="m"),  # exists in m
        ]
    )
    return m


def test_basic_transport_to_existing(model):
    added = add_transport_reactions(model, "c", "m", ["ATP"])
    assert len(added) == 1
    rxn = added[0]
    assert rxn.id == "tr_0001"
    assert rxn.name == "ATP transport, cytoplasm-mitochondrion"
    assert {m.id: rxn.get_coefficient(m.id) for m in rxn.metabolites} == {
        "atp_c": -1.0,
        "atp_m": 1.0,
    }
    assert rxn.reversibility is True


def test_only_to_existing_skips_missing(model):
    # H2O is not in m; with only_to_existing (default) it's skipped
    added = add_transport_reactions(model, "c", "m", ["ATP", "H2O"])
    assert [r.id for r in added] == ["tr_0001"]  # only ATP


def test_creates_missing_target_metabolite(model):
    added = add_transport_reactions(
        model, "c", "m", ["H2O"], only_to_existing=False
    )
    assert len(added) == 1
    new = [mt for mt in model.metabolites if mt.name == "H2O" and mt.compartment == "m"]
    assert len(new) == 1
    assert new[0].formula == "H2O"  # copied from source


def test_copies_formula_and_charge(model):
    add_transport_reactions(model, "c", "e", ["ATP"], only_to_existing=False)
    new = [mt for mt in model.metabolites if mt.name == "ATP" and mt.compartment == "e"][0]
    assert new.formula == "C10H16N5O13P3"
    assert new.charge == -4


def test_irreversible(model):
    (rxn,) = add_transport_reactions(model, "c", "m", ["ATP"], reversible=False)
    assert rxn.lower_bound == 0
    assert rxn.reversibility is False


def test_default_all_metabolites_in_from(model):
    # default metabolite_names = all in c (ATP, H2O); to m, only_to_existing -> only ATP
    added = add_transport_reactions(model, "c", "m")
    assert [r.id for r in added] == ["tr_0001"]


def test_multiple_target_compartments_and_sequential_ids(model):
    added = add_transport_reactions(
        model, "c", ["m", "e"], ["ATP"], only_to_existing=False
    )
    assert [r.id for r in added] == ["tr_0001", "tr_0002"]


def test_unknown_compartment_raises(model):
    with pytest.raises(ValueError, match="not in the model"):
        add_transport_reactions(model, "x", "m", ["ATP"])


def test_unknown_metabolite_raises(model):
    with pytest.raises(ValueError, match="not found in compartment"):
        add_transport_reactions(model, "c", "m", ["NOPE"])


# --- regression: duplicate name in compartment ------------------------------

def test_duplicate_name_in_source_compartment_warns(model):
    """Two source mets sharing a name in the same compartment warn instead of
    silently collapsing to one."""
    model.add_metabolites([
        cobra.Metabolite("h2o2_c", name="H2O", compartment="c"),  # duplicate name
    ])
    with pytest.warns(UserWarning, match="Multiple metabolites named 'H2O'"):
        added = add_transport_reactions(model, "c", "m", ["H2O"], only_to_existing=False)
    # Transport still works (uses the first match) — the warning is the signal.
    assert len(added) == 1


# --- the subsystem names the two compartments, as in RAVEN's addTransport -----------------

def test_reversible_transport_subsystem(model):
    added = add_transport_reactions(model, "c", "m", ["ATP"])
    assert added[0].subsystem == "Transport between c and m"


def test_irreversible_transport_subsystem(model):
    added = add_transport_reactions(model, "c", "m", ["ATP"], reversible=False)
    assert added[0].subsystem == "Transport from c to m"


def test_subsystem_names_each_target_compartment(model):
    added = add_transport_reactions(model, "c", ["m", "e"], ["ATP"], only_to_existing=False)
    assert [r.subsystem for r in added] == ["Transport between c and m", "Transport between c and e"]
