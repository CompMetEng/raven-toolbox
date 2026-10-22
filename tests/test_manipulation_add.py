"""Tests for raven_toolbox.manipulation.add (addRxns port)."""
import cobra
import pytest

from raven_toolbox.manipulation import add_reactions_from_equations
from raven_toolbox.utils.parse import parse_name_comp


@pytest.fixture
def model():
    m = cobra.Model("t")
    m.add_metabolites(
        [
            cobra.Metabolite("atp_c", name="ATP", compartment="c"),
            cobra.Metabolite("h2o_c", name="H2O", compartment="c"),
            cobra.Metabolite("adp_c", name="ADP", compartment="c"),
            cobra.Metabolite("pi_c", name="phosphate", compartment="c"),
        ]
    )
    return m


# --- parse_name_comp -------------------------------------------------------

@pytest.mark.parametrize(
    "token,expected",
    [
        ("ATP[c]", ("ATP", "c")),
        ("ATP", ("ATP", None)),
        ("  ATP[c] ", ("ATP", "c")),
        ("weird[name][m]", ("weird[name]", "m")),
    ],
)
def test_parse_name_comp(token, expected):
    assert parse_name_comp(token) == expected


# --- id mode (eqnType 1) ---------------------------------------------------

def test_add_by_id_basic_and_reversibility(model):
    (rxn,) = add_reactions_from_equations(
        model, [{"id": "R1", "equation": "atp_c + h2o_c <=> adp_c + pi_c"}]
    )
    assert rxn.id == "R1"
    assert rxn.reversibility is True
    assert {m.id: rxn.get_coefficient(m.id) for m in rxn.metabolites} == {
        "atp_c": -1.0,
        "h2o_c": -1.0,
        "adp_c": 1.0,
        "pi_c": 1.0,
    }


def test_irreversible_arrows(model):
    rxns = add_reactions_from_equations(
        model,
        [
            {"id": "R1", "equation": "atp_c --> adp_c"},
            {"id": "R2", "equation": "atp_c => adp_c"},
        ],
    )
    for r in rxns:
        assert r.lower_bound == 0.0
        assert r.reversibility is False


def test_coefficients(model):
    (rxn,) = add_reactions_from_equations(
        model, [{"id": "R1", "equation": "2 atp_c + 1.5 h2o_c --> adp_c"}]
    )
    assert rxn.get_coefficient("atp_c") == -2.0
    assert rxn.get_coefficient("h2o_c") == -1.5


def test_id_mode_creates_new_met_in_compartment(model):
    add_reactions_from_equations(
        model,
        [{"id": "R1", "equation": "atp_c --> amp_c"}],
        compartment="c",
    )
    assert "amp_c" in model.metabolites
    assert model.metabolites.get_by_id("amp_c").compartment == "c"


def test_id_mode_new_met_without_compartment_errors(model):
    with pytest.raises(ValueError, match="no compartment"):
        add_reactions_from_equations(model, [{"id": "R1", "equation": "atp_c --> amp_c"}])


# --- name mode (eqnType 2) -------------------------------------------------

def test_name_mode_matches_existing_by_name(model):
    (rxn,) = add_reactions_from_equations(
        model,
        [{"id": "R1", "equation": "ATP + H2O <=> ADP + phosphate"}],
        mets_by="name",
        compartment="c",
    )
    # resolved to the existing _c metabolites, not new ones
    assert {m.id for m in rxn.metabolites} == {"atp_c", "h2o_c", "adp_c", "pi_c"}
    assert len(model.metabolites) == 4


def test_name_mode_creates_new_met_with_auto_id(model):
    add_reactions_from_equations(
        model,
        [{"id": "R1", "equation": "ATP --> AMP"}],
        mets_by="name",
        compartment="c",
    )
    new = [m for m in model.metabolites if m.name == "AMP"]
    assert len(new) == 1
    assert new[0].id == "m1"
    assert new[0].compartment == "c"


def test_name_mode_dedups_new_met_across_reactions(model):
    # A new metabolite named on more than one reaction in the same call must be
    # created once and shared — later tokens have to see mets created earlier in
    # the call (the (name, comp) index is seeded once and updated on creation).
    r1, r2 = add_reactions_from_equations(
        model,
        [
            {"id": "R1", "equation": "ATP --> AMP"},
            {"id": "R2", "equation": "AMP --> ADP"},
        ],
        mets_by="name",
        compartment="c",
    )
    amp = [m for m in model.metabolites if m.name == "AMP"]
    assert len(amp) == 1                 # created once, not duplicated
    assert amp[0] in r1.metabolites
    assert amp[0] in r2.metabolites


def test_name_mode_requires_compartment(model):
    with pytest.raises(ValueError, match="needs a compartment"):
        add_reactions_from_equations(
            model, [{"id": "R1", "equation": "ATP --> ADP"}], mets_by="name"
        )


# --- name[comp] mode (eqnType 3) -------------------------------------------

def test_name_comp_syntax(model):
    model.add_metabolites([cobra.Metabolite("atp_m", name="ATP", compartment="m")])
    (rxn,) = add_reactions_from_equations(
        model,
        [{"id": "R1", "equation": "ATP[c] --> ATP[m]"}],
        mets_by="name",
        compartment="c",
    )
    # matched ATP in two different compartments by name[comp]
    assert {m.id for m in rxn.metabolites} == {"atp_c", "atp_m"}


# --- genes -----------------------------------------------------------------

def test_gene_rule_auto_creates_genes(model):
    (rxn,) = add_reactions_from_equations(
        model,
        [{"id": "R1", "equation": "atp_c --> adp_c", "gene_reaction_rule": "G1 and G2"}],
    )
    assert {g.id for g in rxn.genes} == {"G1", "G2"}
    assert {g.id for g in model.genes} == {"G1", "G2"}


def test_strict_genes_errors_on_unknown(model):
    with pytest.raises(ValueError, match="genes not in the model"):
        add_reactions_from_equations(
            model,
            [{"id": "R1", "equation": "atp_c --> adp_c", "gene_reaction_rule": "G1"}],
            allow_new_genes=False,
        )


def test_strict_genes_ok_when_present(model):
    model.genes.append(cobra.core.gene.Gene("G1"))
    (rxn,) = add_reactions_from_equations(
        model,
        [{"id": "R1", "equation": "atp_c --> adp_c", "gene_reaction_rule": "G1"}],
        allow_new_genes=False,
    )
    assert rxn.gene_reaction_rule == "G1"


# --- guards & extras -------------------------------------------------------

def test_duplicate_reaction_id_errors(model):
    model.add_reactions([cobra.Reaction("R1")])
    with pytest.raises(ValueError, match="already exists"):
        add_reactions_from_equations(model, [{"id": "R1", "equation": "atp_c --> adp_c"}])


def test_strict_mets_errors(model):
    with pytest.raises(ValueError, match="allow_new_mets"):
        add_reactions_from_equations(
            model,
            [{"id": "R1", "equation": "atp_c --> amp_c"}],
            compartment="c",
            allow_new_mets=False,
        )


def test_explicit_bounds_override_arrow(model):
    (rxn,) = add_reactions_from_equations(
        model,
        [{"id": "R1", "equation": "atp_c <=> adp_c", "bounds": (0, 50), "name": "myrxn"}],
    )
    assert rxn.bounds == (0, 50)
    assert rxn.name == "myrxn"


def test_net_zero_metabolite_dropped(model):
    # atp_c on both sides nets to zero and is removed.
    (rxn,) = add_reactions_from_equations(
        model, [{"id": "R1", "equation": "atp_c + h2o_c --> atp_c + adp_c"}]
    )
    assert "atp_c" not in {m.id for m in rxn.metabolites}
    assert {m.id for m in rxn.metabolites} == {"h2o_c", "adp_c"}


def test_missing_equation_errors(model):
    with pytest.raises(ValueError, match="missing required 'equation'"):
        add_reactions_from_equations(model, [{"id": "R1"}])


def test_no_arrow_errors(model):
    with pytest.raises(ValueError, match="No reaction arrow"):
        add_reactions_from_equations(model, [{"id": "R1", "equation": "atp_c + h2o_c"}])


# --- regression: leading-number metabolite name -------

def test_name_mode_preserves_leading_number_name(model):
    """A metabolite name that begins with a number isn't misparsed as a coefficient.

    The resolver prefers a full-token match against an existing metabolite name
    before falling back to splitting off a leading number as a coefficient --
    otherwise ``"2 oxoglutarate"`` would parse as ``(coeff=2, name="oxoglutarate")``,
    silently corrupting the stoichiometry.
    """
    model.add_metabolites([
        cobra.Metabolite("akg_c", name="2 oxoglutarate", compartment="c"),
    ])
    (rxn,) = add_reactions_from_equations(
        model,
        [{"id": "R1", "equation": "ATP + 2 oxoglutarate --> ADP"}],
        mets_by="name",
        compartment="c",
    )
    assert rxn.get_coefficient("akg_c") == -1.0  # not -2.0
    assert rxn.get_coefficient("atp_c") == -1.0


def test_name_mode_coefficient_still_works_without_collision(model):
    """If the full token doesn't match anything, fall back to coefficient split."""
    (rxn,) = add_reactions_from_equations(
        model,
        [{"id": "R1", "equation": "2 ATP + H2O --> ADP + phosphate"}],
        mets_by="name",
        compartment="c",
    )
    assert rxn.get_coefficient("atp_c") == -2.0


# --- regression: empty-stoichiometry warning ----------

def test_empty_stoichiometry_warns(model):
    """All-terms-cancel reaction warns instead of silently shipping an empty rxn."""
    with pytest.warns(UserWarning, match="no net metabolites"):
        (rxn,) = add_reactions_from_equations(
            model, [{"id": "R1", "equation": "atp_c --> atp_c"}]
        )
    assert len(rxn.metabolites) == 0


# --- regression: unknown-compartment warning ----------

def test_id_mode_unknown_compartment_warns(model):
    """Id mode validates unregistered compartments, matching the check the
    name/[comp] path already applies."""
    with pytest.warns(UserWarning, match="unregistered compartment 'cyto'"):
        add_reactions_from_equations(
            model,
            [{"id": "R1", "equation": "atp_c --> amp_c"}],
            compartment="cyto",  # typo for 'c'
        )


def test_name_comp_unknown_compartment_warns(model):
    """Same defensive check in the name[comp] path when allow_new_mets=True."""
    with pytest.warns(UserWarning, match="unregistered compartment 'mito'"):
        add_reactions_from_equations(
            model,
            [{"id": "R1", "equation": "ATP[c] --> AMP[mito]"}],
            mets_by="name",
        )


def test_new_genes_are_added_in_sorted_order_whatever_the_hash_seed():
    import os
    import subprocess
    import sys

    code = (
        "import cobra\n"
        "from raven_toolbox.manipulation import add_reactions_from_equations\n"
        "m = cobra.Model('t')\n"
        "add_reactions_from_equations(m, [{'id': 'PX', 'equation': 'a[c] --> b[c]', "
        "'gene_reaction_rule': 'YPROBE2 and YPROBE1 and YPROBE10'}], mets_by='name', compartment='c')\n"
        "print('GENES:' + ','.join(g.id for g in m.genes))\n"
    )
    def genes_line(seed):
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        (line,) = [x for x in out.splitlines() if x.startswith("GENES:")]
        return line.removeprefix("GENES:")

    orders = {genes_line(seed) for seed in ("0", "3")}
    assert orders == {"YPROBE1,YPROBE10,YPROBE2"}
