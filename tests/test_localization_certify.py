"""Tests for materialised-FBA-certified compartment assignment.

These use single-compartment toys to exercise the *certified* contract: the returned proposal is
``certified=True`` **iff** the model ``apply_assignment`` builds actually reaches the growth floor (the
certificate is a real FBA on the materialised model, not a growth row inside the placement solve), and
functionality still drives placement against a reaction's top localisation score when a confined
metabolite requires it.
"""
import cobra
import pandas as pd
import pytest

from raven_toolbox.localization import (
    AssignmentProposal,
    GrowthCondition,
    apply_assignment,
    assign_compartments,
)
from raven_toolbox.localization.scores import LocalizationScores


def _met(mid):
    return cobra.Metabolite(mid, name=mid, compartment="c")


def _grows(model, floor=1e-6):
    v = model.slim_optimize(error_value=0.0)
    return v is not None and v > floor


def _linear():
    """EX_A -> r1: A->B -> bio: B->. B produced only by r1."""
    m = cobra.Model("linear")
    A, B = _met("A_c"), _met("B_c")
    m.add_metabolites([A, B])
    ex = cobra.Reaction("EX_A", lower_bound=-10, upper_bound=1000)
    ex.add_metabolites({A: -1})
    r1 = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({A: -1, B: 1})
    r1.gene_reaction_rule = "g1"
    bio = cobra.Reaction("bio", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({B: -1})
    m.add_reactions([ex, r1, bio])
    m.objective = "bio"
    return m


def _chain():
    """EX_A -> r1: A->X -> r2: X->B -> bio: B->. X is the pathway intermediate."""
    m = cobra.Model("chain")
    A, X, B = _met("A_c"), _met("X_c"), _met("B_c")
    m.add_metabolites([A, X, B])
    ex = cobra.Reaction("EX_A", lower_bound=-10, upper_bound=1000)
    ex.add_metabolites({A: -1})
    r1 = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({A: -1, X: 1})
    r1.gene_reaction_rule = "g1"
    r2 = cobra.Reaction("r2", lower_bound=0, upper_bound=1000)
    r2.add_metabolites({X: -1, B: 1})
    r2.gene_reaction_rule = "g2"
    bio = cobra.Reaction("bio", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({B: -1})
    m.add_reactions([ex, r1, r2, bio])
    m.objective = "bio"
    return m


def _gap_draft():
    """EX_A -> r1: A->B; bio: B + C -> biomass. Nothing makes C, so the draft cannot grow."""
    m = cobra.Model("gap")
    A, B, C = _met("A_c"), _met("B_c"), _met("C_c")
    m.add_metabolites([A, B, C])
    ex = cobra.Reaction("EX_A", lower_bound=-10, upper_bound=1000)
    ex.add_metabolites({A: -1})
    r1 = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({A: -1, B: 1})
    r1.gene_reaction_rule = "g1"
    bio = cobra.Reaction("bio", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({B: -1, C: -1})
    m.add_reactions([ex, r1, bio])
    m.objective = "bio"
    return m


def _universal():
    """Candidate database with rC: A -> C (the missing producer of C)."""
    u = cobra.Model("universal")
    A, C = _met("A_c"), _met("C_c")
    u.add_metabolites([A, C])
    rc = cobra.Reaction("rC", lower_bound=0, upper_bound=1000)
    rc.add_metabolites({A: -1, C: 1})
    u.add_reactions([rc])
    return u


def _relay():
    """EX_A -> r1: A->I -> r2: I->B -> bio: B->. I only ever lives where r1/r2 are placed."""
    m = cobra.Model("relay")
    A, mid, B = _met("A_c"), _met("Int_c"), _met("B_c")
    m.add_metabolites([A, mid, B])
    ex = cobra.Reaction("EX_A", lower_bound=-10, upper_bound=1000)
    ex.add_metabolites({A: -1})
    r1 = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({A: -1, mid: 1})
    r1.gene_reaction_rule = "g1"
    r2 = cobra.Reaction("r2", lower_bound=0, upper_bound=1000)
    r2.add_metabolites({mid: -1, B: 1})
    r2.gene_reaction_rule = "g2"
    bio = cobra.Reaction("bio", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({B: -1})
    m.add_reactions([ex, r1, r2, bio])
    m.objective = "bio"
    return m


def test_pool_split_across_two_nondefault_compartments_is_bridged():
    # Regression: the intermediate I is produced in m (r1) and consumed in p (r2) and never exists in
    # the default compartment c. The star-topology transport must still create the c hub so I is
    # reconnected (m<->c<->p); otherwise both transports are silently dropped and the model is dead.
    m = _relay()
    scores = LocalizationScores(pd.DataFrame(
        {"c": [0.1, 0.1], "m": [0.9, 0.0], "p": [0.0, 0.9]}, index=["g1", "g2"]))
    res = assign_compartments(m, scores, ["r1", "r2"])
    assert res.certified
    assert res.placements["r1"] == ["m"]
    assert res.placements["r2"] == ["p"]
    assert _grows(apply_assignment(m, res))


def _multiloc():
    """EX_A -> A_c; tA: A_c<->A_m; r1: A->B (g1 scores c and m equally); bio: B_c->; sink: B_m->.

    Both compartments are connected (transport of A, a demand for B_m), so a duplicate of r1 in the
    non-primary compartment can carry flux — a genuine, evidence-backed dual placement.
    """
    m = cobra.Model("multiloc")
    Ac = cobra.Metabolite("A_c", compartment="c")
    Bc = cobra.Metabolite("B_c", compartment="c")
    Am = cobra.Metabolite("A_m", compartment="m")
    Bm = cobra.Metabolite("B_m", compartment="m")
    m.add_metabolites([Ac, Bc, Am, Bm])
    ex = cobra.Reaction("EX_A", lower_bound=-10, upper_bound=1000)
    ex.add_metabolites({Ac: -1})
    tA = cobra.Reaction("tA", lower_bound=-1000, upper_bound=1000)
    tA.add_metabolites({Ac: -1, Am: 1})
    r1 = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({Ac: -1, Bc: 1})
    r1.gene_reaction_rule = "g1"
    bio = cobra.Reaction("bio", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({Bc: -1})
    sink = cobra.Reaction("sinkB_m", lower_bound=0, upper_bound=1000)
    sink.add_metabolites({Bm: -1})
    m.add_reactions([ex, tA, r1, bio, sink])
    m.objective = "bio"
    return m


def test_multilocalize_recovers_evidence_backed_dual():
    # g1 scores c and m equally and BOTH compartments are functionally connected, so the FVA-validated
    # enrichment places r1 in both.
    m = _multiloc()
    scores = LocalizationScores(pd.DataFrame({"c": [0.9], "m": [0.9]}, index=["g1"]))
    res = assign_compartments(m, scores, ["r1"], multi_localize=True)
    assert res.certified
    assert set(res.placements["r1"]) == {"c", "m"}
    assert _grows(apply_assignment(m, res))


def test_multilocalize_rejects_dead_duplicate():
    # Soundness: g1 scores m strongly (0.9 > threshold) so an m-duplicate of r1 IS proposed, but with
    # nothing transportable the m copy is disconnected (no A in m, no B sink) and carries no flux -> the
    # FVA gate drops it, so r1 stays mono. No dead duplicate can harvest the m-score.
    m = _linear()
    scores = LocalizationScores(pd.DataFrame({"c": [0.9], "m": [0.9]}, index=["g1"]))
    res = assign_compartments(m, scores, ["r1"], transportable=[], multi_localize=True)
    assert res.certified
    assert res.placements["r1"] == ["c"]
    assert _grows(apply_assignment(m, res))


def _two_carbon():
    """Grows on A (rA) or B (rB), both feeding the shared core M -> bio."""
    m = cobra.Model("two_carbon")
    A, B, M = _met("A_c"), _met("B_c"), _met("M_c")
    m.add_metabolites([A, B, M])
    exA = cobra.Reaction("EX_A", lower_bound=-10, upper_bound=1000)
    exA.add_metabolites({A: -1})
    exB = cobra.Reaction("EX_B", lower_bound=0, upper_bound=1000)  # closed by default (B medium opens)
    exB.add_metabolites({B: -1})
    rA = cobra.Reaction("rA", lower_bound=0, upper_bound=1000)
    rA.add_metabolites({A: -1, M: 1})
    rA.gene_reaction_rule = "gA"
    rB = cobra.Reaction("rB", lower_bound=0, upper_bound=1000)
    rB.add_metabolites({B: -1, M: 1})
    rB.gene_reaction_rule = "gB"
    bio = cobra.Reaction("bio", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({M: -1})
    m.add_reactions([exA, exB, rA, rB, bio])
    m.objective = "bio"
    return m


# --------------------------------------------------------------------------- basics
def test_result_type_and_certified():
    m = _linear()
    scores = LocalizationScores(pd.DataFrame({"c": [0.4], "m": [0.9]}, index=["g1"]))
    res = assign_compartments(m, scores, ["r1"])
    assert isinstance(res, AssignmentProposal)
    assert res.certified is True
    assert res.status == "certified"


def test_score_respected_when_functional():
    # r1 prefers m (0.9 > 0.4); B is transportable (default), so r1 can follow its score to m and a
    # transport keeps the model functional.
    m = _linear()
    scores = LocalizationScores(pd.DataFrame({"c": [0.4], "m": [0.9]}, index=["g1"]))
    res = assign_compartments(m, scores, ["r1"])
    assert res.placements["r1"] == ["m"]
    assert res.certified


def test_minimize_transports_stays_certified_and_functional():
    # The transport-subset minimisation must keep the model certified and functional, and never
    # increase the transport count. (On this toy both transports are essential, so the minimum equals
    # the certified set; the point is that minimisation is sound and does not break certification.)
    m = _linear()
    scores = LocalizationScores(pd.DataFrame({"c": [0.4], "m": [0.9]}, index=["g1"]))
    base = assign_compartments(m, scores, ["r1"])
    mini = assign_compartments(m, scores, ["r1"], minimize_transports=True)
    assert mini.certified
    assert mini.placements["r1"] == ["m"]
    assert len(mini.added_transports) <= len(base.added_transports)
    assert _grows(apply_assignment(m, mini))


def _isozyme():
    """EX_A -> {r1: A->B (g1), r2: A->B (g2)} -> bio: B_c->. Two isozymes make B from A."""
    m = cobra.Model("isozyme")
    A, B = _met("A_c"), _met("B_c")
    m.add_metabolites([A, B])
    ex = cobra.Reaction("EX_A", lower_bound=-10, upper_bound=1000)
    ex.add_metabolites({A: -1})
    r1 = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({A: -1, B: 1})
    r1.gene_reaction_rule = "g1"
    r2 = cobra.Reaction("r2", lower_bound=0, upper_bound=1000)
    r2.add_metabolites({A: -1, B: 1})
    r2.gene_reaction_rule = "g2"
    bio = cobra.Reaction("bio", lower_bound=0, upper_bound=1000)
    bio.add_metabolites({B: -1})
    m.add_reactions([ex, r1, r2, bio])
    m.objective = "bio"
    return m


def test_minimize_transports_drops_a_redundant_transport():
    # g1 sends its isozyme r1 to m (needing transports for A and B); g2 keeps r2 in c (no transport).
    # Both make B, so the cytosolic r2 alone feeds biomass and r1's route is redundant -> pFBA carries
    # no flux through r1's transports, and minimisation drops them. A real reduction, still functional.
    scores = LocalizationScores(pd.DataFrame(
        {"c": [0.1, 0.9], "m": [0.9, 0.1]}, index=["g1", "g2"]))
    base = assign_compartments(_isozyme(), scores, ["r1", "r2"])
    mini = assign_compartments(_isozyme(), scores, ["r1", "r2"], minimize_transports=True)
    assert base.certified and mini.certified
    assert base.added_transports  # the score placement did add redundant transports
    assert len(mini.added_transports) < len(base.added_transports)  # and minimisation dropped them
    assert _grows(apply_assignment(_isozyme(), mini))


def test_certified_implies_materialised_growth():
    # The soundness invariant: a certified proposal, when materialised, actually grows. (This is exactly
    # what a placement certified only by a flux model inside the placement solve could get wrong.)
    m = _linear()
    scores = LocalizationScores(pd.DataFrame({"c": [0.4], "m": [0.9]}, index=["g1"]))
    res = assign_compartments(m, scores, ["r1"])
    assert res.certified
    assert _grows(apply_assignment(m, res))


# --------------------------------------------------- functionality overrides the score
def test_functionality_overrides_score():
    # B is NOT transportable, so placing r1 in m would strand B in m where biomass (in c) cannot reach
    # it. The confinement repair pins r1 to c, against its higher m-score. Materialised model grows.
    m = _linear()
    scores = LocalizationScores(pd.DataFrame({"c": [0.4], "m": [0.9]}, index=["g1"]))
    res = assign_compartments(m, scores, ["r1"], transportable=["A"])
    assert res.certified
    assert res.placements["r1"] == ["c"]
    assert _grows(apply_assignment(m, res))


def test_pathway_coherence_overrides_individual_score():
    # g1 prefers c (0.9), g2 prefers m (0.9). The intermediate X is non-transportable, so r1 and r2
    # must co-locate; the score objective then anchors the whole pathway in c (0.9+0.5 > 0.1+0.9),
    # so r2 lands in c against its own m-preference.
    m = _chain()
    scores = LocalizationScores(pd.DataFrame(
        {"c": [0.9, 0.5], "m": [0.1, 0.9]}, index=["g1", "g2"]))
    res = assign_compartments(m, scores, ["r1", "r2"], transportable=["A", "B"])
    assert res.certified
    assert res.placements["r1"] == ["c"]
    assert res.placements["r2"] == ["c"]
    assert _grows(apply_assignment(m, res))


# --------------------------------------------------------------------- gap-fill coupling
def test_gapfill_restores_function():
    # The draft cannot grow (C has no producer). Certification fails, so the feedback loop pulls rC
    # from the universal model and re-certifies.
    m = _gap_draft()
    scores = LocalizationScores(pd.DataFrame({"c": [1.0], "m": [0.0]}, index=["g1"]))
    res = assign_compartments(m, scores, ["r1"], universal=_universal(), min_growth=1.0)
    assert res.certified
    assert "rC" in res.added_reactions
    assert _grows(apply_assignment(m, res, universal=_universal()))


def test_gapfill_returns_a_growth_restoring_set():
    # Directly exercise the flux-based _gapfill: on a draft that cannot grow it returns the universal
    # reaction that restores biomass, and nothing spurious. As a plain LP it is reliable where cobra's
    # indicator gap-fill MILP is not (that MILP fails to find such a fill in most genome-scale cases even
    # when it exists) -- and a returned set always actually restores growth.
    from raven_toolbox.localization.certify import _gapfill

    applied = _gap_draft()  # EX_A -> r1: A->B; bio: B + C ->  (C has no producer)
    added = _gapfill(applied, _universal(), "bio", min_growth=1.0)
    assert added == ["rC"]  # the exact missing producer of C, nothing extra

    ur = _universal().reactions.get_by_id("rC")
    nr = cobra.Reaction(ur.id, lower_bound=ur.lower_bound, upper_bound=ur.upper_bound)
    applied.add_reactions([nr])
    nr.add_metabolites({applied.metabolites.get_by_id(m.id): c for m, c in ur.metabolites.items()})
    assert _grows(applied)


def test_gapfill_offers_nothing_when_the_universal_cannot_restore_growth():
    # If even adding every candidate cannot reach the floor, _gapfill returns [] (no false fill).
    from raven_toolbox.localization.certify import _gapfill

    applied = _gap_draft()
    empty = cobra.Model("empty")  # no candidate produces C
    assert _gapfill(applied, empty, "bio", min_growth=1.0) == []


def test_gapfill_warns_on_namespace_mismatch_instead_of_failing_silently():
    # A universal whose metabolite ids don't match the model can't connect; _gapfill returns [] but must
    # WARN, so "found nothing" is distinguishable from "wrong namespace".
    from raven_toolbox.localization.certify import _gapfill

    applied = _gap_draft()
    foreign = cobra.Model("foreign")  # same chemistry (A->C) under foreign ids
    a, c = cobra.Metabolite("A_x", compartment="c"), cobra.Metabolite("C_x", compartment="c")
    foreign.add_metabolites([a, c])
    rc = cobra.Reaction("rC", lower_bound=0, upper_bound=1000)
    rc.add_metabolites({a: -1, c: 1})
    foreign.add_reactions([rc])
    with pytest.warns(UserWarning, match="namespace"):
        assert _gapfill(applied, foreign, "bio", min_growth=1.0) == []


def test_no_gratuitous_gapfill():
    # When the draft already grows, no universal candidate is pulled (it is only reached on a real
    # growth failure).
    m = _linear()
    scores = LocalizationScores(pd.DataFrame({"c": [0.9], "m": [0.1]}, index=["g1"]))
    res = assign_compartments(m, scores, ["r1"], universal=_universal())
    assert res.certified
    assert res.added_reactions == []


def test_gapfill_reuses_relocated_compartment_metabolite():
    # Gap-fill candidates must resolve metabolite ids through the shared base-id/compartment resolver
    # that _move_reaction uses, not match them verbatim. A relocated reaction materialises its
    # non-default-compartment metabolites under a *generated* id ("A_c__m"), never the universal
    # candidate's own "A_m" -- matching verbatim silently creates a second, disconnected "A_m" instead
    # of raising, leaving the gap-fill reaction an island nothing else in the model touches.
    m = _linear()  # EX_A -> r1: A_c->B_c (g1) -> bio: B_c->
    proposal = AssignmentProposal(placements={"r1": ["m"]}, added_reactions=["rD"])

    u = cobra.Model("universal")
    A_m = cobra.Metabolite("A_m", name="A", compartment="m")
    D_m = cobra.Metabolite("D_m", name="D", compartment="m")
    u.add_metabolites([A_m, D_m])
    rD = cobra.Reaction("rD", lower_bound=0, upper_bound=1000)
    rD.add_metabolites({A_m: -1, D_m: 1})
    u.add_reactions([rD])

    out = apply_assignment(m, proposal, universal=u)

    r1_out, rD_out = out.reactions.get_by_id("r1"), out.reactions.get_by_id("rD")
    a_resolved = next(met for met, coeff in r1_out.metabolites.items() if coeff < 0)
    assert a_resolved in rD_out.metabolites  # rD's "A" reuses the same node r1 relocated to

    # Exactly 3 compartment-m metabolites should exist: A and B (from relocating r1) and D (a
    # genuinely new species from the gap-fill reaction) -- not a 4th, disconnected copy of A.
    in_m = [met for met in out.metabolites if met.compartment == "m"]
    assert len(in_m) == 3


# ------------------------------------------------------- honest partial (no false positive)
def test_unreachable_floor_is_uncertified_not_falsely_certified():
    # An unreachable growth floor must yield an *uncertified* proposal (with the shortfall visible),
    # never a false certificate. This is the property the whole redesign exists to guarantee.
    m = _linear()
    scores = LocalizationScores(pd.DataFrame({"c": [0.4], "m": [0.9]}, index=["g1"]))
    res = assign_compartments(m, scores, ["r1"], min_growth=1e6)
    assert res.certified is False
    assert res.status == "uncertified"
    assert res.growths["__primary__"] < 1e6  # the real (small) growth is reported, not hidden


def test_no_growth_draft_raises():
    m = _linear()
    m.reactions.EX_A.lower_bound = 0  # no uptake -> no growth
    scores = LocalizationScores(pd.DataFrame({"c": [0.4], "m": [0.9]}, index=["g1"]))
    with pytest.raises(ValueError, match="does not grow"):
        assign_compartments(m, scores, ["r1"])


# --------------------------------------------------------------- multi-medium certification
def test_growth_conditions_are_certified_per_medium():
    # Certification runs on the primary medium (A) and each GrowthCondition medium (B); both grow, so
    # the proposal is certified and every medium's biomass flux is reported.
    m = _two_carbon()
    scores = LocalizationScores(pd.DataFrame(
        {"c": [0.9, 0.9], "m": [0.1, 0.1]}, index=["gA", "gB"]))
    on_b = GrowthCondition(name="onB", medium={"EX_B": 10.0}, min_growth=1.0)
    res = assign_compartments(m, scores, ["rA", "rB"], min_growth=1.0,
                                        growth_conditions=[on_b])
    assert res.certified
    assert res.growths["__primary__"] >= 1.0
    assert res.growths["onB"] >= 1.0


def _merged_two_compartment_draft():
    """A draft merged by merge_compartments: A has copies A_c and A_m, so the merged A keeps the id
    A_c in compartment s."""
    from raven_toolbox.manipulation import merge_compartments

    m = cobra.Model("draft")
    a_c = cobra.Metabolite("A_c", name="A", compartment="c")
    a_m = cobra.Metabolite("A_m", name="A", compartment="m")
    x_c = cobra.Metabolite("X_c", name="X", compartment="c")
    r1 = cobra.Reaction("R1", lower_bound=0, upper_bound=1000)
    r2 = cobra.Reaction("R2", lower_bound=0, upper_bound=1000)
    ex = cobra.Reaction("EX_X", lower_bound=-10, upper_bound=1000)
    m.add_reactions([r1, r2, ex])
    r1.add_metabolites({x_c: -1, a_c: 1})
    r2.add_metabolites({a_m: -1, x_c: 1})
    ex.add_metabolites({x_c: -1})
    merged, _, _ = merge_compartments(m)
    assert merged.metabolites.A_c.compartment == "s"
    return merged


def test_apply_assignment_matches_a_merged_metabolite_to_the_universal_copy():
    """The merged A_c (in s) and the universal A_m share the key A, so the gap-fill reaction lands
    on the A that R1 makes in m instead of on a second copy."""
    merged = _merged_two_compartment_draft()
    universal = cobra.Model("u")
    u1 = cobra.Reaction("U1", lower_bound=0, upper_bound=1000)
    universal.add_reactions([u1])
    u1.add_metabolites({cobra.Metabolite("A_m", name="A", compartment="m"): -1,
                        cobra.Metabolite("Y_m", name="Y", compartment="m"): 1})
    proposal = AssignmentProposal(placements={"R1": ["m"]}, added_reactions=["U1"])
    out = apply_assignment(merged, proposal, universal=universal)
    a_in_m = [x for x in out.metabolites if x.name == "A" and x.compartment == "m"]
    assert len(a_in_m) == 1
    assert a_in_m[0] in out.reactions.R1.metabolites
    assert a_in_m[0] in out.reactions.U1.metabolites


def _draft_merged_into_c():
    """A draft merged into c by merge_compartments: the intermediate I had copies I_g and I_x only,
    so the merged I keeps the id I_g in c."""
    from raven_toolbox.manipulation import merge_compartments

    o = cobra.Model("orig")
    a = cobra.Metabolite("A_c", name="A", compartment="c")
    b = cobra.Metabolite("B_c", name="B", compartment="c")
    ig = cobra.Metabolite("I_g", name="I", compartment="g")
    ix = cobra.Metabolite("I_x", name="I", compartment="x")
    ex = cobra.Reaction("EX_A", lower_bound=-10, upper_bound=1000)
    r1 = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r2 = cobra.Reaction("r2", lower_bound=0, upper_bound=1000)
    t = cobra.Reaction("tI", lower_bound=-1000, upper_bound=1000)
    bio = cobra.Reaction("bio", lower_bound=0, upper_bound=1000)
    o.add_reactions([ex, r1, r2, t, bio])
    ex.add_metabolites({a: -1})
    r1.add_metabolites({a: -1, ig: 1})
    r2.add_metabolites({ix: -1, b: 1})
    t.add_metabolites({ig: -1, ix: 1})
    bio.add_metabolites({b: -1})
    r1.gene_reaction_rule = "g1"
    r2.gene_reaction_rule = "g2"
    o.objective = "bio"
    draft, _, _ = merge_compartments(o, merged_id="c", merged_name="cytosol")
    assert draft.metabolites.I_g.compartment == "c"
    scores = LocalizationScores(pd.DataFrame(
        {"c": [0.1, 0.1], "m": [0.9, 0.0], "p": [0.0, 0.9], "g": [0.0, 0.0], "x": [0.0, 0.0]},
        index=["g1", "g2"]))
    return draft, scores


def test_default_key_of_a_merged_metabolite_is_the_same_in_every_localization_function():
    """assign_compartments and apply_assignment both key the merged I_g as I, so the model built
    from the proposal has every transport the proposal lists and grows."""
    from raven_toolbox.localization import curation_priority, relocate_reactions

    draft, scores = _draft_merged_into_c()
    res = assign_compartments(draft, scores, ["r1", "r2"])
    assert res.certified
    assert {b for b, _ in res.added_transports} == {"A", "I", "B"}
    out = apply_assignment(draft, res)
    assert len([r for r in out.reactions if r.id.startswith("tr_")]) == len(res.added_transports)
    assert out.slim_optimize(error_value=0.0) > 1e-6
    moved = relocate_reactions(draft, res, {"r2": "m"})
    assert {b for b, _ in moved.transports_added + moved.transports_removed} <= {"A", "I", "B"}
    curation_priority(draft, res, scores, check_essential=False)


def test_name_keyed_merge_keeps_distinct_species_distinct_in_apply_assignment():
    """X_c (name X) and X_m (name X mito) are two species; after a name-keyed merge both are in s
    and apply_assignment makes a copy of each in m."""
    from raven_toolbox.manipulation import merge_compartments

    m = cobra.Model("m")
    xc = cobra.Metabolite("X_c", name="X", compartment="c")
    xm = cobra.Metabolite("X_m", name="X mito", compartment="m")
    yc = cobra.Metabolite("Y_c", name="Y", compartment="c")
    ra = cobra.Reaction("Ra", lower_bound=0, upper_bound=1000)
    rb = cobra.Reaction("Rb", lower_bound=0, upper_bound=1000)
    m.add_reactions([ra, rb])
    ra.add_metabolites({yc: -1, xc: 1})
    rb.add_metabolites({xm: -1, yc: 1})
    merged, _, _ = merge_compartments(m, base_metabolite=lambda x: x.name)
    out = apply_assignment(merged, AssignmentProposal(placements={"Ra": ["m"], "Rb": ["m"]}))
    in_m = sorted(x.name for x in out.metabolites if x.compartment == "m")
    assert in_m == ["X", "X mito", "Y"]


def test_id_that_ends_in_another_compartment_id_keeps_its_own_key():
    """vitamin_a in c and vitamin_c in e stay two species when the universal model has a
    compartment a: only the suffix of a metabolite's own compartment is stripped."""
    m = cobra.Model("m")
    va = cobra.Metabolite("vitamin_a", name="retinol", compartment="c")
    vc = cobra.Metabolite("vitamin_c", name="ascorbate", compartment="e")
    z = cobra.Metabolite("Z_c", name="Z", compartment="c")
    rv = cobra.Reaction("Rv", lower_bound=0, upper_bound=1000)
    rw = cobra.Reaction("Rw", lower_bound=0, upper_bound=1000)
    m.add_reactions([rv, rw])
    rv.add_metabolites({va: -1, z: 1})
    rw.add_metabolites({vc: -1, z: 1})
    u = cobra.Model("u")
    u.add_metabolites([cobra.Metabolite("Q_a", name="Q", compartment="a")])
    out = apply_assignment(m, AssignmentProposal(placements={"Rv": ["m"], "Rw": ["m"]}), universal=u)
    assert sorted(x.name for x in out.metabolites if x.compartment == "m") == ["Z", "ascorbate", "retinol"]


def test_two_metabolites_with_one_key_in_a_compartment_raise():
    m = cobra.Model("m")
    h1 = cobra.Metabolite("h_c", name="H+", compartment="c")
    h2 = cobra.Metabolite("h", name="proton", compartment="c")
    r = cobra.Reaction("R", lower_bound=0, upper_bound=1000)
    m.add_reactions([r])
    r.add_metabolites({h1: -1, h2: 1})
    with pytest.raises(ValueError, match="same key"):
        apply_assignment(m, AssignmentProposal(placements={"R": ["m"]}))
    scores = LocalizationScores(pd.DataFrame({"c": [0.1], "m": [0.9]}, index=["g1"]))
    with pytest.raises(ValueError, match="same key"):
        assign_compartments(m, scores, ["R"], biomass_reaction="R")
