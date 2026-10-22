"""Tests for raven_toolbox.biomass."""
from __future__ import annotations

import cobra
import pytest

from raven_toolbox.biomass import (
    BiomassComponent,
    BiomassConfig,
    rescale_pseudoreaction,
    scale_biomass,
    set_gam,
    sum_biomass,
)
from raven_toolbox.biomass.scale import _formula_mw

# --- formula MW helper ----------------------------------------------

@pytest.mark.parametrize(
    "formula, expected",
    [
        ("H2O", 18.015),       # 2*1.008 + 15.999
        ("ATP", 0.0),          # invalid: no element 'ATP'... we raise
        ("CO2", 44.008),       # 12.01 + 2*15.999
        ("C6H12O6", 180.156),  # 6*12.01 + 12*1.008 + 6*15.999
    ],
)
def test_formula_mw(formula, expected):
    if formula == "ATP":
        with pytest.raises(ValueError, match="Unknown element"):
            _formula_mw(formula)
    else:
        assert _formula_mw(formula) == pytest.approx(expected, rel=1e-4)


# --- toy biomass model ----------------------------------------------

def _toy_biomass_model() -> tuple[cobra.Model, BiomassConfig]:
    """Tiny but realistic biomass layout:

    - protein pseudoreaction: 2 amino-acid tRNAs → protein (mass_strategy mw_minus_2h)
    - carbohydrate pseudoreaction: glucose → carbohydrate (mass_strategy mw)
    - biomass pseudoreaction: protein + carbohydrate + GAM cofactors → biomass
    """
    m = cobra.Model("toy")
    m.compartments = {"c": "cytoplasm"}

    aa1 = cobra.Metabolite("aa1_c", name="alanine-tRNA", compartment="c",
                            charge=0, formula="C3H6N1O2")  # MW ≈ 88
    aa2 = cobra.Metabolite("aa2_c", name="glycine-tRNA", compartment="c",
                            charge=0, formula="C2H4N1O2")  # MW ≈ 74
    glc = cobra.Metabolite("glc_c", name="glucose", compartment="c",
                            charge=0, formula="C6H12O6")  # MW = 180.156
    protein = cobra.Metabolite("protein_c", name="protein", compartment="c", charge=0)
    carb = cobra.Metabolite("carb_c", name="carbohydrate", compartment="c", charge=0)
    biomass = cobra.Metabolite("biomass_c", name="biomass", compartment="c", charge=0)
    h = cobra.Metabolite("h_c", name="H+", compartment="c", charge=1)
    atp = cobra.Metabolite("atp_c", name="ATP", compartment="c", charge=-4)
    adp = cobra.Metabolite("adp_c", name="ADP", compartment="c", charge=-3)
    h2o = cobra.Metabolite("h2o_c", name="H2O", compartment="c", charge=0)
    pi = cobra.Metabolite("pi_c", name="phosphate", compartment="c", charge=-2)
    m.add_metabolites([aa1, aa2, glc, protein, carb, biomass, h, atp, adp, h2o, pi])

    prot_rxn = cobra.Reaction("PROT_pseudo")
    prot_rxn.name = "protein pseudoreaction"
    prot_rxn.add_metabolites({aa1: -0.5, aa2: -0.5, protein: 1, h: 1})
    carb_rxn = cobra.Reaction("CARB_pseudo")
    carb_rxn.name = "carbohydrate pseudoreaction"
    carb_rxn.add_metabolites({glc: -0.001, carb: 1})  # 1 mmol → 180 mg
    bio_rxn = cobra.Reaction("BIO_pseudo")
    bio_rxn.name = "biomass pseudoreaction"
    bio_rxn.add_metabolites({
        protein: -1, carb: -1, biomass: 1,
        atp: -50, h2o: -50, adp: 50, h: 50, pi: 50,   # GAM = 50
    })
    m.add_reactions([prot_rxn, carb_rxn, bio_rxn])

    cfg = BiomassConfig(
        biomass_rxn="BIO_pseudo",
        proton_met="h_c",
        components=(
            BiomassComponent("protein", "protein pseudoreaction",
                             mass_strategy="mw_minus_2h"),
            BiomassComponent("carbohydrate", "carbohydrate pseudoreaction",
                             mass_strategy="mw"),
        ),
    )
    return m, cfg


# --- sum_biomass -----------------------------------------------------

def test_sum_biomass_keys_match_components_plus_total():
    m, cfg = _toy_biomass_model()
    out = sum_biomass(m, cfg)
    assert set(out) == {"protein", "carbohydrate", "total"}


def test_sum_biomass_protein_uses_minus_2h_offset():
    m, cfg = _toy_biomass_model()
    out = sum_biomass(m, cfg)
    # aa1 (alanine-tRNA, C3H6N1O2, MW ≈ 88.09): MW - 2.016
    # aa2 (glycine-tRNA, C2H4N1O2, MW ≈ 74.07): MW - 2.016
    # contribs: 0.5*(88.09 - 2.016) + 0.5*(74.07 - 2.016) ≈ 79.066, /1000
    assert out["protein"] == pytest.approx(0.079066, rel=1e-3)


def test_sum_biomass_carbohydrate_uses_plain_mw():
    m, cfg = _toy_biomass_model()
    out = sum_biomass(m, cfg)
    # 0.001 mmol glucose * 180.156 g/mol / 1000 = 0.000180156
    assert out["carbohydrate"] == pytest.approx(0.000180156, rel=1e-3)


def test_sum_biomass_total_is_components_sum():
    m, cfg = _toy_biomass_model()
    out = sum_biomass(m, cfg)
    assert out["total"] == pytest.approx(out["protein"] + out["carbohydrate"])


def test_sum_biomass_grams_strategy():
    m, cfg = _toy_biomass_model()
    # Add a lipid component whose stoichiometry is already in grams.
    lipid = cobra.Metabolite("lipid_c", name="lipid backbone", compartment="c")
    m.add_metabolites([lipid])
    rxn = cobra.Reaction("LIPID_pseudo")
    rxn.name = "lipid backbone pseudoreaction"
    rxn.add_metabolites({lipid: 1, m.metabolites.get_by_id("glc_c"): -0.05})
    m.add_reactions([rxn])
    cfg2 = BiomassConfig.from_components(
        cfg.biomass_rxn, cfg.proton_met,
        [*cfg.components,
         BiomassComponent("lipid_backbone", "lipid backbone pseudoreaction",
                          mass_strategy="grams")],
    )
    out = sum_biomass(m, cfg2)
    assert out["lipid_backbone"] == pytest.approx(0.05)


def test_sum_biomass_missing_pseudoreaction_contributes_zero():
    m, cfg = _toy_biomass_model()
    cfg2 = BiomassConfig.from_components(
        cfg.biomass_rxn, cfg.proton_met,
        [*cfg.components, BiomassComponent("DNA", "DNA pseudoreaction")],
    )
    out = sum_biomass(m, cfg2)
    assert out["DNA"] == 0


# --- rescale_pseudoreaction -----------------------------------------

def test_rescale_pseudoreaction_doubles_substrate_coefs():
    m, cfg = _toy_biomass_model()
    rxn = m.reactions.get_by_id("PROT_pseudo")
    aa1 = m.metabolites.get_by_id("aa1_c")
    before = rxn.metabolites[aa1]
    rescale_pseudoreaction(m, cfg, "protein", 2.0)
    assert rxn.metabolites[aa1] == pytest.approx(2 * before)


def test_rescale_pseudoreaction_leaves_product_alone():
    m, cfg = _toy_biomass_model()
    rxn = m.reactions.get_by_id("PROT_pseudo")
    protein = m.metabolites.get_by_id("protein_c")
    before = rxn.metabolites[protein]
    rescale_pseudoreaction(m, cfg, "protein", 3.0)
    assert rxn.metabolites[protein] == before


def test_rescale_pseudoreaction_rebalances_h_for_charge():
    m, cfg = _toy_biomass_model()
    # Add a non-zero-charge substrate so H+ balance is meaningful.
    rxn = m.reactions.get_by_id("PROT_pseudo")
    aa1 = m.metabolites.get_by_id("aa1_c")
    aa1.charge = -1  # negatively charged
    rescale_pseudoreaction(m, cfg, "protein", 2.0)
    total = sum((m_.charge or 0) * c for m_, c in rxn.metabolites.items())
    assert total == pytest.approx(0)


# --- scale_biomass --------------------------------------------------

def test_scale_biomass_lands_on_new_value():
    m, cfg = _toy_biomass_model()
    target = 0.40
    scale_biomass(m, cfg, "protein", target)
    out = sum_biomass(m, cfg)
    assert out["protein"] == pytest.approx(target, rel=1e-6)


def test_scale_biomass_with_balance_keeps_total_at_one():
    m, cfg = _toy_biomass_model()
    # Force a known starting total ≠ 1 by tweaking the toy model.
    scale_biomass(m, cfg, "protein", 0.6, balance_out="carbohydrate")
    out = sum_biomass(m, cfg)
    assert out["total"] == pytest.approx(1.0, rel=1e-6)


def test_scale_biomass_balance_to_target_total():
    m, cfg = _toy_biomass_model()
    scale_biomass(m, cfg, "protein", 0.4, balance_out="carbohydrate", target_total=0.9)
    out = sum_biomass(m, cfg)
    assert out["protein"] == pytest.approx(0.4, rel=1e-6)
    assert out["total"] == pytest.approx(0.9, rel=1e-6)


def test_scale_biomass_negative_balance_uses_target_total():
    m, cfg = _toy_biomass_model()
    # Protein 0.8 fits a total of 1 but not of 0.5.
    with pytest.raises(ValueError, match="bring the total to 0.5"):
        scale_biomass(m, cfg, "protein", 0.8, balance_out="carbohydrate", target_total=0.5)


def test_scale_biomass_zero_current_raises():
    m, cfg = _toy_biomass_model()
    # Build a config that references a missing component → 0 current
    cfg2 = BiomassConfig.from_components(
        cfg.biomass_rxn, cfg.proton_met,
        [*cfg.components, BiomassComponent("DNA", "DNA pseudoreaction")],
    )
    with pytest.raises(ValueError, match="current"):
        scale_biomass(m, cfg2, "DNA", 0.1)


def test_scale_biomass_balance_that_would_go_negative_raises():
    m, cfg = _toy_biomass_model()
    before = {r.id: dict(r.metabolites) for r in m.reactions}
    # Protein above 1 g/gDW leaves carbohydrate a negative share of the total,
    # which would turn its substrates into products.
    with pytest.raises(ValueError, match="carbohydrate"):
        scale_biomass(m, cfg, "protein", 1.5, balance_out="carbohydrate")
    assert {r.id: dict(r.metabolites) for r in m.reactions} == before


def test_scale_biomass_balance_on_empty_component_raises():
    m, cfg = _toy_biomass_model()
    cfg2 = BiomassConfig.from_components(
        cfg.biomass_rxn, cfg.proton_met,
        [*cfg.components, BiomassComponent("DNA", "DNA pseudoreaction")],
    )
    before = {r.id: dict(r.metabolites) for r in m.reactions}
    with pytest.raises(ValueError, match="DNA"):
        scale_biomass(m, cfg2, "protein", 0.5, balance_out="DNA")
    assert {r.id: dict(r.metabolites) for r in m.reactions} == before


# --- set_gam --------------------------------------------------------

def test_set_gam_scales_cofactor_coefficients():
    m, cfg = _toy_biomass_model()
    set_gam(m, 80, biomass_rxn=cfg.biomass_rxn,
            cofactor_met_names=("ATP", "ADP", "H2O", "H+", "phosphate"))
    bio = m.reactions.get_by_id(cfg.biomass_rxn)
    atp = m.metabolites.get_by_id("atp_c")
    adp = m.metabolites.get_by_id("adp_c")
    h2o = m.metabolites.get_by_id("h2o_c")
    pi = m.metabolites.get_by_id("pi_c")
    assert bio.metabolites[atp] == -80
    assert bio.metabolites[adp] == 80
    assert bio.metabolites[h2o] == -80
    assert bio.metabolites[pi] == 80


def test_set_gam_with_ngam_sets_equality_bounds():
    m, _ = _toy_biomass_model()
    ngam = cobra.Reaction("NGAM", lower_bound=0, upper_bound=1000)
    atp = m.metabolites.get_by_id("atp_c")
    ngam.add_metabolites({atp: -1})
    m.add_reactions([ngam])
    set_gam(m, 80, biomass_rxn="BIO_pseudo",
            cofactor_met_names=("ATP",),
            ngam_rxn="NGAM", ngam_value=1.5)
    assert m.reactions.get_by_id("NGAM").bounds == (1.5, 1.5)


def test_set_gam_ignores_non_cofactor_mets():
    m, cfg = _toy_biomass_model()
    bio = m.reactions.get_by_id(cfg.biomass_rxn)
    protein = m.metabolites.get_by_id("protein_c")
    before = bio.metabolites[protein]
    set_gam(m, 80, biomass_rxn=cfg.biomass_rxn,
            cofactor_met_names=("ATP",))
    assert bio.metabolites[protein] == before


@pytest.mark.parametrize("first,value", [(90.2, 0.1), (46.5, 0.7), (59.276, 0.001)])
def test_set_gam_stores_the_requested_value_exactly(first, value):
    m, cfg = _toy_biomass_model()
    names = ("ATP", "ADP", "H2O", "H+", "phosphate")
    set_gam(m, first, biomass_rxn=cfg.biomass_rxn, cofactor_met_names=names)
    set_gam(m, value, biomass_rxn=cfg.biomass_rxn, cofactor_met_names=names)
    bio = m.reactions.get_by_id(cfg.biomass_rxn)
    assert bio.metabolites[m.metabolites.get_by_id("atp_c")] == -value
    assert bio.metabolites[m.metabolites.get_by_id("adp_c")] == value


def test_set_gam_missing_ngam_reaction_raises_before_changing_anything():
    m, cfg = _toy_biomass_model()
    bio = m.reactions.get_by_id(cfg.biomass_rxn)
    before = dict(bio.metabolites)
    with pytest.raises(KeyError):
        set_gam(m, 80, biomass_rxn=cfg.biomass_rxn, cofactor_met_names=("ATP",),
                ngam_rxn="NOPE", ngam_value=1.0)
    assert dict(bio.metabolites) == before


# --- charge handling and the post-rescale balance check ---------------------

def test_rescale_pseudoreaction_unset_charge_raises_and_changes_nothing():
    m, cfg = _toy_biomass_model()
    m.metabolites.get_by_id("aa1_c").charge = None
    rxn = m.reactions.get_by_id("PROT_pseudo")
    before = dict(rxn.metabolites)
    with pytest.raises(ValueError, match="aa1_c"):
        rescale_pseudoreaction(m, cfg, "protein", 2.0)
    assert dict(rxn.metabolites) == before


def _proton_mass_model():
    m = cobra.Model("grams")
    mets = {
        "a_c": cobra.Metabolite("a_c", name="ala", charge=-1, compartment="c"),
        "h_c": cobra.Metabolite("h_c", name="H+", charge=1, compartment="c"),
        "prot_c": cobra.Metabolite("prot_c", name="protein", charge=0, compartment="c"),
        "d_c": cobra.Metabolite("d_c", name="dNMP", charge=0, compartment="c"),
        "dna_c": cobra.Metabolite("dna_c", name="DNA", charge=0, compartment="c"),
        "bio_c": cobra.Metabolite("bio_c", name="biomass", charge=0, compartment="c"),
    }
    m.add_metabolites(list(mets.values()))
    for rid, name, stoich in [
        ("P1", "protein pseudoreaction", {"a_c": -0.4, "prot_c": 1}),
        ("D1", "DNA pseudoreaction", {"d_c": -0.5, "dna_c": 1}),
    ]:
        rxn = cobra.Reaction(rid, name=name)
        m.add_reactions([rxn])
        rxn.add_metabolites({mets[k]: v for k, v in stoich.items()})
    cfg = BiomassConfig.from_components(
        "P1", "h_c",
        [
            BiomassComponent("protein", "protein pseudoreaction", mass_strategy="grams"),
            BiomassComponent("DNA", "DNA pseudoreaction", mass_strategy="grams"),
        ],
    )
    return m, cfg


def test_scale_biomass_balance_check_includes_the_proton_rebalance():
    """The H+ added by the first rescale counts as mass, so protein = 0.8 ends at 1.6 g/gDW
    and DNA cannot absorb it, although a linear estimate would let it through."""
    m, cfg = _proton_mass_model()
    before = {r.id: dict(r.metabolites) for r in m.reactions}
    with pytest.raises(ValueError, match="DNA"):
        scale_biomass(m, cfg, "protein", 0.8, balance_out="DNA")
    assert {r.id: dict(r.metabolites) for r in m.reactions} == before
