"""Tests for manipulation/compartments.py — merge_compartments + copy_to_compartment."""
from __future__ import annotations

import cobra
import pytest

from raven_toolbox import manipulation
from raven_toolbox.manipulation import copy_to_compartment, merge_compartments


def test_exported_from_the_package():
    """Both are public API, not just importable from the submodule.

    Importing directly from ``manipulation.compartments`` would pass even if the functions
    were missing from ``manipulation.__all__``, since that import path doesn't exercise the
    one users actually take.
    """
    assert "merge_compartments" in manipulation.__all__
    assert "copy_to_compartment" in manipulation.__all__


def _two_compartment_model() -> cobra.Model:
    """A_c → B_c, A_m → B_m, and a transport A_c ↔ A_m. Multi-compartment toy."""
    m = cobra.Model("toy")
    A_c = cobra.Metabolite("A_c", name="A", compartment="c")
    A_m = cobra.Metabolite("A_m", name="A", compartment="m")
    B_c = cobra.Metabolite("B_c", name="B", compartment="c")
    B_m = cobra.Metabolite("B_m", name="B", compartment="m")
    m.add_metabolites([A_c, A_m, B_c, B_m])

    def rxn(rid, lb, ub, mets, gpr=None):
        r = cobra.Reaction(rid, lower_bound=lb, upper_bound=ub)
        r.add_metabolites(mets)
        if gpr:
            r.gene_reaction_rule = gpr
        return r
    m.add_reactions([rxn("r_c", 0, 1000, {A_c: -1, B_c: 1}, "g1"),
                     rxn("r_m", 0, 1000, {A_m: -1, B_m: 1}, "g2"),
                     rxn("tr_A", -1000, 1000, {A_c: -1, A_m: 1})])
    return m


def _name_keyed_model() -> cobra.Model:
    """Same topology as ``_two_compartment_model`` but keyed like yeast-GEM: the same species
    gets a *different opaque id per compartment* (``s_1``..``s_4``), unified only by ``name``.
    The default ``_base_id`` (suffix strip) can't collapse these — only a name key can."""
    m = cobra.Model("named")
    s1 = cobra.Metabolite("s_1", name="A", compartment="c")
    s2 = cobra.Metabolite("s_2", name="A", compartment="m")
    s3 = cobra.Metabolite("s_3", name="B", compartment="c")
    s4 = cobra.Metabolite("s_4", name="B", compartment="m")
    m.add_metabolites([s1, s2, s3, s4])

    def rxn(rid, lb, ub, mets, gpr=None):
        r = cobra.Reaction(rid, lower_bound=lb, upper_bound=ub)
        r.add_metabolites(mets)
        if gpr:
            r.gene_reaction_rule = gpr
        return r
    m.add_reactions([rxn("r_c", 0, 1000, {s1: -1, s3: 1}, "g1"),
                     rxn("r_m", 0, 1000, {s2: -1, s4: 1}, "g2"),
                     rxn("tr_A", -1000, 1000, {s1: -1, s2: 1})])
    return m


# ----------------------------------------------------------------- merge_compartments

def test_merge_compartments_collapses_to_one():
    """A_c + A_m collapse to one species; B_c + B_m to another; transport A_c↔A_m self-cancels and is
    dropped. No copy lived in the merged compartment 's', so each merged metabolite keeps the id that
    sorts first: A_c, B_c."""
    m = _two_compartment_model()
    merged, deleted, dupes = merge_compartments(m)
    assert {x.id for x in merged.metabolites} == {"A_c", "B_c"}
    # The transport reaction collapsed (A → A) and was deleted.
    assert "tr_A" in deleted
    # r_c and r_m are now both A → B; one of them gets deduplicated.
    surviving = {r.id for r in merged.reactions}
    assert len(surviving & {"r_c", "r_m"}) == 1
    assert (set(dupes) | (surviving & {"r_c", "r_m"})) == {"r_c", "r_m"}


def test_merge_compartments_preserves_gpr_and_subsystem():
    m = _two_compartment_model()
    m.reactions.r_c.subsystem = "carbo"
    merged, _, _ = merge_compartments(m)
    survivor = next(r for r in merged.reactions if r.id in {"r_c", "r_m"})
    # The survivor keeps its gene rule + subsystem (cobra may sometimes lose them
    # through copy; we set them explicitly).
    assert survivor.gene_reaction_rule in {"g1", "g2"}
    if survivor.id == "r_c":
        assert survivor.subsystem == "carbo"


def test_merge_compartments_keeps_single_met_reactions_when_asked():
    """drop_single_metabolite_reactions=False keeps the collapsed transport (now A → A,
    which is empty stoichiometry after net-cancellation — still dropped, but the *one-met*
    case is the more interesting one). Use a uniport pattern to exercise it."""
    m = cobra.Model("uniport")
    A_c = cobra.Metabolite("A_c", name="A", compartment="c")
    A_m = cobra.Metabolite("A_m", name="A", compartment="m")
    H_c = cobra.Metabolite("H_c", name="H", compartment="c")
    m.add_metabolites([A_c, A_m, H_c])
    # H+ symport: A_c + H_c → A_m. After merge: A + H → A → leaves H.
    sym = cobra.Reaction("sym", lower_bound=0, upper_bound=1000)
    sym.add_metabolites({A_c: -1, H_c: -1, A_m: 1})
    m.add_reactions([sym])
    merged_drop, deleted_drop, _ = merge_compartments(m, drop_single_metabolite_reactions=True)
    assert "sym" in deleted_drop
    merged_keep, deleted_keep, _ = merge_compartments(m, drop_single_metabolite_reactions=False)
    # With keep, sym survives as a one-met reaction (consumes H).
    assert "sym" not in deleted_keep
    assert "sym" in {r.id for r in merged_keep.reactions}


def test_merge_compartments_keeps_pre_existing_exchange_reactions():
    """An exchange had one metabolite *before* merging, so it never became trivial and must
    survive. Dropping these silently strips a model of its medium and its biomass reaction:
    smallYeast grows at 0.1222 before flattening and at 0.0000 after, with no warning."""
    m = _two_compartment_model()
    uptake = cobra.Reaction("EX_A", lower_bound=-1000, upper_bound=1000)
    uptake.add_metabolites({m.metabolites.A_c: 1})
    m.add_reactions([uptake])

    merged, deleted, _ = merge_compartments(m)

    assert "EX_A" not in deleted
    assert "EX_A" in {r.id for r in merged.reactions}
    assert len(merged.boundary) == 1
    # the genuinely-collapsed transport is still dropped
    assert "tr_A" in deleted


def test_merge_compartments_carries_the_objective_over():
    """The merged model is rebuilt from scratch, so the objective must be copied across.
    Without this the caller gets a model that silently optimises to 0.0."""
    m = _two_compartment_model()
    m.objective = "r_c"

    merged, _, dupes = merge_compartments(m)

    # r_c and r_m merge to the same reaction; whichever survives carries the objective.
    survivor = next(r for r in merged.reactions if r.id in {"r_c", "r_m"})
    assert str(merged.objective.expression) != "0"
    if survivor.id == "r_c":
        assert survivor.objective_coefficient == 1.0
    assert merged.objective_direction == m.objective_direction


def test_merge_compartments_deduplicate_off_keeps_both():
    m = _two_compartment_model()
    merged, _, dupes = merge_compartments(m, deduplicate_reactions=False)
    assert dupes == []
    assert {"r_c", "r_m"} <= {r.id for r in merged.reactions}


def test_merge_compartments_default_key_cannot_unify_name_keyed_model():
    """On yeast-GEM-style ids (different id per compartment, shared only by name), the default
    suffix-stripping key leaves every metabolite distinct — nothing collapses. This is the failure
    the base_metabolite override exists to fix."""
    merged, deleted, _ = merge_compartments(_name_keyed_model())
    # nothing unifies (4 distinct species); each metabolite keeps its own id.
    assert {x.id for x in merged.metabolites} == {"s_1", "s_2", "s_3", "s_4"}
    assert deleted == []  # tr_A (s_1 -> s_2) still spans two distinct mets, not collapsed


def test_merge_compartments_base_metabolite_unifies_by_name():
    """With base_metabolite=lambda m: m.name the compartment copies unify by species: A_c/A_m
    collapse to one, the A-transport self-cancels and is dropped, and the two now-identical
    A->B reactions deduplicate — the same outcome as suffix-keyed ids get by default."""
    merged, deleted, dupes = merge_compartments(
        _name_keyed_model(), base_metabolite=lambda m: m.name)
    # merged id is the id of a member, NOT synthesised from the name key.
    assert {x.id for x in merged.metabolites} == {"s_1", "s_3"}
    assert "tr_A" in deleted
    surviving = {r.id for r in merged.reactions}
    assert len(surviving & {"r_c", "r_m"}) == 1
    assert (set(dupes) | (surviving & {"r_c", "r_m"})) == {"r_c", "r_m"}
    # The survivor keeps a real gene rule.
    survivor = next(r for r in merged.reactions if r.id in {"r_c", "r_m"})
    assert survivor.gene_reaction_rule in {"g1", "g2"}


def test_merge_by_name_inherits_legal_ids_despite_punctuated_names():
    """Real metabolite names carry whitespace AND punctuation (e.g.
    '1-acyl-sn-glycerol 3-phosphate (16:0)'), which would crash the solver if used as a metabolite id
    (cobra names each constraint after the id, and optlang forbids whitespace). Because the merged id
    is *inherited* from an existing (legal) member id and never minted from the name key, merging by
    name yields solver-safe ids by construction — the punctuated names only drive grouping."""
    m = cobra.Model("named_punct")
    a1 = cobra.Metabolite("x1", name="1-acyl-sn-glycerol 3-phosphate (16:0)", compartment="c")
    a2 = cobra.Metabolite("x2", name="1-acyl-sn-glycerol 3-phosphate (16:0)", compartment="m")
    b1 = cobra.Metabolite("x3", name="gamma", compartment="c")
    m.add_metabolites([a1, a2, b1])
    r = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r.add_metabolites({a1: -1, b1: 1})
    tr = cobra.Reaction("tr", lower_bound=-1000, upper_bound=1000)
    tr.add_metabolites({a1: -1, a2: 1})
    m.add_reactions([r, tr])
    merged, deleted, _ = merge_compartments(m, base_metabolite=lambda mm: mm.name)
    ids = {x.id for x in merged.metabolites}
    assert ids == {"x1", "x3"}           # inherited from x1/x3, never the raw name
    assert "tr" in deleted               # the two punctuated copies unified, so the transport cancels
    merged.slim_optimize()               # must not raise while building the solver problem


def test_merge_into_existing_compartment_keeps_that_compartments_ids():
    """When a copy already lives in the merged-into compartment, its id is kept verbatim (it already
    reflects the compartment); other copies collapse into it."""
    m = _two_compartment_model()  # A_c/A_m/B_c/B_m
    merged, _, _ = merge_compartments(m, merged_id="c", merged_name="cytosol")
    assert {x.id for x in merged.metabolites} == {"A_c", "B_c"}  # the 'c' copies' ids survive


def _acetyl_coa_model(first: tuple[str, str], second: tuple[str, str]) -> cobra.Model:
    """Two copies of acetyl-CoA, given as (id, compartment) in model order, and X_c:
    R1 X_c => first copy, R2 second copy => X_c."""
    m = cobra.Model("accoa")
    a = cobra.Metabolite(first[0], name="acetyl-CoA", compartment=first[1])
    b = cobra.Metabolite(second[0], name="acetyl-CoA", compartment=second[1])
    x = cobra.Metabolite("X_c", name="X", compartment="c")
    m.add_metabolites([a, b, x])
    r1 = cobra.Reaction("R1", lower_bound=0, upper_bound=1000)
    r1.add_metabolites({x: -1, a: 1})
    r2 = cobra.Reaction("R2", lower_bound=0, upper_bound=1000)
    r2.add_metabolites({b: -1, x: 1})
    m.add_reactions([r1, r2])
    return m


def test_merge_compartments_keeps_the_id_that_sorts_first():
    """The merged metabolite keeps the id that sorts first, whatever the order of the copies in
    the model (RAVEN's mergeCompartments applies the same rule)."""
    for order in ((("ACCOA_m", "m"), ("ACCOA_c", "c")), (("ACCOA_c", "c"), ("ACCOA_m", "m"))):
        merged, _, _ = merge_compartments(_acetyl_coa_model(*order),
                                          base_metabolite=lambda mm: mm.name)
        assert {x.id for x in merged.metabolites} == {"ACCOA_c", "X_c"}
        assert merged.metabolites.ACCOA_c.compartment == "s"


def test_merge_compartments_prefers_the_copy_in_the_merged_compartment():
    """A copy already in the merged compartment keeps its id, even when another copy's id sorts
    before it."""
    merged, _, _ = merge_compartments(_acetyl_coa_model(("ACCOA_c", "c"), ("ACCOA_z", "s")),
                                      base_metabolite=lambda mm: mm.name)
    assert {x.id for x in merged.metabolites} == {"ACCOA_z", "X_c"}


# ----------------------------------------------------------------- copy_to_compartment

def test_copy_to_compartment_basic():
    """Copy r_c into 'p' (peroxisome): a new reaction r_c_p with metabolites in p."""
    m = _two_compartment_model()
    out, new_rxns, new_mets = copy_to_compartment(m, ["r_c"], "p",
                                                    target_compartment_name="peroxisome")
    assert "r_c_p" in [r.id for r in out.reactions]
    new_r = out.reactions.r_c_p
    assert {x.compartment for x in new_r.metabolites} == {"p"}
    assert "A_p" in [x.id for x in out.metabolites]
    assert "B_p" in [x.id for x in out.metabolites]
    assert new_rxns == ["r_c_p"]
    assert set(new_mets) == {"A_p", "B_p"}
    # Original still there.
    assert "r_c" in [r.id for r in out.reactions]


def test_copy_to_compartment_preserves_gpr_and_bounds():
    m = _two_compartment_model()
    out, _, _ = copy_to_compartment(m, ["r_c"], "p")
    new_r = out.reactions.r_c_p
    assert new_r.gene_reaction_rule == "g1"
    assert new_r.lower_bound == 0 and new_r.upper_bound == 1000


def test_copy_to_compartment_delete_original_is_a_move():
    m = _two_compartment_model()
    out, _, _ = copy_to_compartment(m, ["r_c"], "p", delete_original=True)
    assert "r_c" not in [r.id for r in out.reactions]
    assert "r_c_p" in [r.id for r in out.reactions]


def test_copy_to_compartment_idempotent():
    """Calling twice doesn't add the reaction twice."""
    m = _two_compartment_model()
    out, _, _ = copy_to_compartment(m, ["r_c"], "p")
    out2, new_rxns, _ = copy_to_compartment(out, ["r_c"], "p")
    assert new_rxns == []  # nothing added on second call
    assert len([r for r in out2.reactions if r.id == "r_c_p"]) == 1


def test_copy_to_compartment_unknown_reaction_raises():
    m = _two_compartment_model()
    with pytest.raises(ValueError, match="not in model"):
        copy_to_compartment(m, ["does_not_exist"], "p")


def test_copy_to_compartment_custom_suffix():
    m = _two_compartment_model()
    out, new_rxns, _ = copy_to_compartment(m, ["r_c"], "p", id_suffix="copy1")
    assert new_rxns == ["r_c_copy1"]
    assert "A_copy1" in [x.id for x in out.metabolites]
