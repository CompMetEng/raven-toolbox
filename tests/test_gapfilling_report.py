"""Tests for the gap analysis summary (gapfilling/report.py, gapReport port)."""
import cobra
import pytest

from raven_toolbox.gapfilling import GapReportResult, MinToConnect, gap_report


@pytest.fixture
def gap_report_model():
    """Four independent pieces, each exercising a different section of the report.

    - glc/pyr/co2: a normal, fully-working chain (EX_glc open uptake, R1, R2,
      DM_co2 open demand). Nothing here is blocked or a leak.
    - x/y: R3 (x -> y) alone, with no other reaction touching x or y at all.
      Blocked under both normal and relaxed analysis (nothing produces x
      even once net production elsewhere is allowed for free) -- but giving
      x a free source connects both x and y at once.
    - z: a metabolite with no reactions at all -- an isolated orphan,
      unproducible on its own.
    - P/Q: R4 (P -> Q) and R5 (Q -> 2P), the same unbalanced-cycle leak used
      in test_gapfilling_leak.py. Blocked under normal analysis (only
      v4=v5=0 satisfies exact steady state) but not under relaxed analysis
      (v4=v5=1 satisfies Sv >= 0 for both P and Q) -- and, separately, both
      P and Q can be produced with no external input at all once probed
      directly (checkProduction's own excretion probes are enough on their
      own; no relaxation needed for that part).
    """
    m = cobra.Model("gaps", name="Gaps")
    glc = cobra.Metabolite("glc", compartment="c")
    pyr = cobra.Metabolite("pyr", compartment="c")
    co2 = cobra.Metabolite("co2", compartment="c")
    x = cobra.Metabolite("x", compartment="c")
    y = cobra.Metabolite("y", compartment="c")
    z = cobra.Metabolite("z", compartment="c")
    p = cobra.Metabolite("P", compartment="c")
    q = cobra.Metabolite("Q", compartment="c")
    m.add_metabolites([glc, pyr, co2, x, y, z, p, q])

    ex_glc = cobra.Reaction("EX_glc", lower_bound=-10, upper_bound=1000)
    ex_glc.add_metabolites({glc: -1})
    r1 = cobra.Reaction("R1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({glc: -1, pyr: 1})
    r2 = cobra.Reaction("R2", lower_bound=0, upper_bound=1000)
    r2.add_metabolites({pyr: -1, co2: 1})
    dm_co2 = cobra.Reaction("DM_co2", lower_bound=0, upper_bound=1000)
    dm_co2.add_metabolites({co2: -1})
    r3 = cobra.Reaction("R3", lower_bound=0, upper_bound=1000)
    r3.add_metabolites({x: -1, y: 1})
    r4 = cobra.Reaction("R4", lower_bound=0, upper_bound=1000)
    r4.add_metabolites({p: -1, q: 1})
    r5 = cobra.Reaction("R5", lower_bound=0, upper_bound=1000)
    r5.add_metabolites({q: -1, p: 2})
    m.add_reactions([ex_glc, r1, r2, dm_co2, r3, r4, r5])
    return m


def test_no_flux_reactions(gap_report_model):
    result = gap_report(gap_report_model)
    assert isinstance(result, GapReportResult)
    assert set(result.no_flux_reactions) == {"R3", "R4", "R5"}


def test_relaxation_only_fixes_the_downstream_leak(gap_report_model):
    # R4/R5 net-produce P (an excess with nowhere required to go) and are
    # rescued by relaxation; R3 needs an upstream source relaxation can't
    # invent, so it stays blocked either way.
    result = gap_report(gap_report_model)
    assert set(result.no_flux_reactions_relaxed) == {"R3"}


def test_subgraphs_largest_first_with_encounter_order_ties(gap_report_model):
    result = gap_report(gap_report_model)
    sizes = [len(g) for g in result.subgraphs]
    assert sizes == [3, 2, 2, 1]
    assert set(result.subgraphs[0]) == {"glc", "pyr", "co2"}
    # x/y and P/Q tie at size 2; x/y is encountered first in model order.
    assert set(result.subgraphs[1]) == {"x", "y"}
    assert set(result.subgraphs[2]) == {"P", "Q"}
    assert result.subgraphs[3] == ["z"]


def test_not_produced_metabolites(gap_report_model):
    # P and Q are both producible from nothing via their own unbalanced
    # cycle (no relaxation needed for checkProduction's own probes); glc/
    # pyr/co2 are producible via the open EX_glc import.
    result = gap_report(gap_report_model)
    assert result.not_produced_metabolites == ["x", "y", "z"]


def test_needed_for_production_and_min_to_connect(gap_report_model):
    # x connects itself: flipped to a free source, R3 can run and drain it
    # (y's own excretion reaction absorbs the product), so x's own probe
    # carries flux too. y and z have no such route -- flipped to a source,
    # nothing in the network (or, for z, at all) drains them, so neither
    # connects even itself. z is therefore never reachable through this
    # analysis at all and gets no min_to_connect entry of its own.
    result = gap_report(gap_report_model)
    assert result.needed_for_production["x"] == ["x", "y"]
    assert result.needed_for_production["y"] == []
    assert result.needed_for_production["z"] == []
    assert result.min_to_connect == [MinToConnect(metabolite="x", connects=2)]


def test_mass_balancing_produce_and_consume(gap_report_model):
    result = gap_report(gap_report_model)
    # The P/Q cycle nets +P per cycle, so both can be excreted for free once
    # every boundary reaction (EX_glc, DM_co2) is closed; nothing here can
    # be freely *consumed* -- the cycle only ever produces a net surplus.
    assert set(result.can_produce_without_input) == {"P", "Q"}
    assert result.can_consume_without_output == []


def test_original_model_is_not_modified(gap_report_model):
    n_rxns, n_mets = len(gap_report_model.reactions), len(gap_report_model.metabolites)
    gap_report(gap_report_model)
    assert len(gap_report_model.reactions) == n_rxns
    assert len(gap_report_model.metabolites) == n_mets


def test_no_template_models_skips_gap_fill(gap_report_model):
    result = gap_report(gap_report_model)
    assert result.gap_fill is None


def test_template_models_connects_a_blocked_reaction():
    # add_reactions_from_model (which connect_blocked_reactions uses to merge
    # template reactions) matches metabolites by name[compartment], so x and
    # y need distinct names for the template's exchange to land on the right
    # one rather than an arbitrary same-name match.
    m = cobra.Model("small")
    x = cobra.Metabolite("x", name="ex", compartment="c")
    y = cobra.Metabolite("y", name="why", compartment="c")
    m.add_metabolites([x, y])
    r3 = cobra.Reaction("R3", lower_bound=0, upper_bound=1000)
    r3.add_metabolites({x: -1, y: 1})
    dm_y = cobra.Reaction("DM_y", lower_bound=0, upper_bound=1000)
    dm_y.add_metabolites({y: -1})
    m.add_reactions([r3, dm_y])

    template = cobra.Model("template")
    tx = cobra.Metabolite("x", name="ex", compartment="c")
    template.add_metabolites([tx])
    ex_x = cobra.Reaction("EX_x", lower_bound=0, upper_bound=1000)
    ex_x.add_metabolites({tx: 1})
    template.add_reactions([ex_x])

    result = gap_report(m, template_models=template)
    assert result.gap_fill is not None
    assert "R3" in result.gap_fill.newly_connected
    assert "EX_x" in result.gap_fill.added_reactions


def test_infeasible_model_raises_with_maintenance_hint(gap_report_model):
    atpm = cobra.Reaction("ATPM", lower_bound=5.0, upper_bound=1000.0)
    atpm.add_metabolites({gap_report_model.metabolites.get_by_id("z"): -1})
    gap_report_model.add_reactions([atpm])
    with pytest.raises(ValueError, match="infeasible under its own bounds.*ATPM.*ATP maintenance"):
        gap_report(gap_report_model)


def test_forced_flux_infeasible_once_closed_raises(gap_report_model):
    # Feasible with the uptake open, infeasible once every boundary is closed.
    gap_report_model.reactions.get_by_id("R1").lower_bound = 1.0
    with pytest.raises(ValueError, match="once its boundary reactions are closed.*R1"):
        gap_report(gap_report_model)
