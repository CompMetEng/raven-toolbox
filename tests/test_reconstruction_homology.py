"""Tests for homology reconstruction core (make_ortholog_hits + get_model_from_homology)."""
import cobra
import pandas as pd
import pytest

from raven_toolbox.manipulation import add_reactions_from_equations
from raven_toolbox.reconstruction.homology import (
    HIT_COLUMNS,
    get_model_from_homology,
    make_ortholog_hits,
)

# --- make_ortholog_hits ----------------------------------------------------

def test_make_ortholog_hits_bidirectional():
    hits = make_ortholog_hits([("tA", "nA"), ("tB", "nB")], "template", "neworg")
    assert list(hits.columns) == HIT_COLUMNS
    assert len(hits) == 4  # 2 pairs x 2 directions
    fwd = hits[(hits.from_id == "template") & (hits.from_gene == "tA")]
    assert fwd.iloc[0].to_gene == "nA"
    rev = hits[(hits.from_id == "neworg") & (hits.from_gene == "nA")]
    assert rev.iloc[0].to_gene == "tA"


def test_make_ortholog_hits_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        make_ortholog_hits([], "t", "n")


# --- template model fixture ------------------------------------------------

def _template():
    m = cobra.Model("templateGEM")
    m.compartments = {"c": "cytoplasm"}
    m.add_metabolites([cobra.Metabolite(x, name=x.upper(), compartment="c") for x in ("a", "b", "d")])
    add_reactions_from_equations(
        m,
        [
            {"id": "R_single", "equation": "a --> b", "gene_reaction_rule": "tg1"},
            {"id": "R_iso", "equation": "b --> d", "gene_reaction_rule": "tg2 or tg3"},
            {"id": "R_cplx", "equation": "a --> d", "gene_reaction_rule": "tg4 and tg5"},
        ],
    )
    return m


# --- one-to-one transfer ---------------------------------------------------

def test_single_gene_reaction_transferred():
    t = _template()
    hits = make_ortholog_hits([("tg1", "ng1")], "templateGEM", "bug")
    res = get_model_from_homology([t], hits, "bug")
    assert res.model.id == "bug"
    assert "R_single" in {r.id for r in res.model.reactions}
    assert res.model.reactions.get_by_id("R_single").gene_reaction_rule == "ng1"


def test_unsupported_reaction_dropped():
    t = _template()
    hits = make_ortholog_hits([("tg1", "ng1")], "templateGEM", "bug")  # only tg1 mapped
    res = get_model_from_homology([t], hits, "bug")
    # R_iso (tg2/tg3) and R_cplx (tg4/tg5) have no ortholog -> dropped
    assert {r.id for r in res.model.reactions} == {"R_single"}


def test_one_to_many_orthologs_become_or():
    t = _template()
    hits = make_ortholog_hits([("tg1", "ngA"), ("tg1", "ngB")], "templateGEM", "bug")
    res = get_model_from_homology([t], hits, "bug")
    assert res.model.reactions.get_by_id("R_single").gene_reaction_rule == "ngA or ngB"


# --- isozyme (OR) handling -------------------------------------------------

def test_isozyme_branch_without_ortholog_dropped():
    t = _template()
    hits = make_ortholog_hits([("tg2", "ng2")], "templateGEM", "bug")  # only one isozyme maps
    res = get_model_from_homology([t], hits, "bug")
    assert res.model.reactions.get_by_id("R_iso").gene_reaction_rule == "ng2"


# --- complex (AND) policies ------------------------------------------------

def _complex_hits():
    # only tg4 of the tg4-and-tg5 complex has an ortholog
    return make_ortholog_hits([("tg4", "ng4")], "templateGEM", "bug")


def test_complex_policy_flag_keeps_old_marker():
    res = get_model_from_homology([_template()], _complex_hits(), "bug", complex_policy="flag")
    gpr = res.model.reactions.get_by_id("R_cplx").gene_reaction_rule
    assert "ng4" in gpr and "OLD_templateGEM_tg5" in gpr and " and " in gpr


def test_complex_policy_keep_drops_unmapped_subunit():
    res = get_model_from_homology([_template()], _complex_hits(), "bug", complex_policy="keep")
    assert res.model.reactions.get_by_id("R_cplx").gene_reaction_rule == "ng4"


def test_complex_policy_drop_removes_reaction():
    res = get_model_from_homology([_template()], _complex_hits(), "bug", complex_policy="drop")
    assert "R_cplx" not in {r.id for r in res.model.reactions}


def _one_rule_template(model_id, rxn_id, rule):
    m = cobra.Model(model_id)
    m.add_metabolites([cobra.Metabolite(x, name=x, compartment="c") for x in ("a_c", "b_c")])
    add_reactions_from_equations(m, [{"id": rxn_id, "equation": "a_c --> b_c", "gene_reaction_rule": rule}])
    return m


def _rules(res):
    return {r.id: r.gene_reaction_rule for r in res.model.reactions}


def _policy_draft(rule, mapped, policy):
    hits = make_ortholog_hits([(g, f"n{g}") for g in mapped], "T", "N")
    return _rules(get_model_from_homology([_one_rule_template("T", "R1", rule)], hits, "N", complex_policy=policy))


def test_isozymes_without_ortholog_leave_one_flagged_subunit():
    assert _policy_draft("(A or B) and C", ["C"], "flag") == {"R1": "OLD_T_B and nC"}
    assert _policy_draft("(A or B) and C", ["C"], "keep") == {"R1": "nC"}
    assert _policy_draft("(A or B) and C", ["C"], "drop") == {}


def test_flagged_subunit_stays_beside_a_complete_isozyme_branch():
    flag = _policy_draft("A or (B and C)", ["A", "C"], "flag")
    assert flag == {"R1": "nA or (OLD_T_B and nC)"}
    assert _policy_draft("A or (B and C)", ["A", "C"], "keep") == {"R1": "nA or nC"}


def test_complex_policy_drop_removes_reaction_with_any_flagged_subunit():
    assert _policy_draft("A or (B and C)", ["A", "C"], "drop") == {}
    assert _policy_draft("(A and B) or C", ["C"], "drop") == {}
    assert _policy_draft("(A and B) or C", ["C"], "flag") == {"R1": "(OLD_T_A and OLD_T_B) or nC"}
    assert _policy_draft("(A and B) or C", ["C"], "keep") == {"R1": "nC"}


# --- strictness ------------------------------------------------------------

def test_strictness_bidirectional_and_best_hits():
    t = _template()
    hits = make_ortholog_hits([("tg1", "ng1")], "templateGEM", "bug")
    res = get_model_from_homology([t], hits, "bug", strictness=3)  # bidir + best-hits
    assert "R_single" in {r.id for r in res.model.reactions}


def test_strictness_3_keeps_tied_best_hits():
    hits = make_ortholog_hits([("tg1", "ng1"), ("tg2", "ng1")], "templateGEM", "bug")
    res = get_model_from_homology([_template()], hits, "bug", strictness=3)
    assert {r.id for r in res.model.reactions} == {"R_single", "R_iso"}

    weaker = (hits.from_gene == "ng1") & (hits.to_gene == "tg2")
    hits.loc[weaker, "bitscore"] = 500.0
    res = get_model_from_homology([_template()], hits, "bug", strictness=3)
    assert {r.id for r in res.model.reactions} == {"R_single"}


def test_max_evalue_is_exclusive():
    hits = make_ortholog_hits([("tg1", "ng1")], "templateGEM", "bug")
    hits["evalue"] = 1e-30
    assert not get_model_from_homology([_template()], hits, "bug").model.reactions
    res = get_model_from_homology([_template()], hits, "bug", max_evalue=1e-29)
    assert "R_single" in {r.id for r in res.model.reactions}


def test_evalue_at_the_cutoff_is_parsed_exactly():
    from raven_toolbox.reconstruction.homology.blast import _parse_tabular

    hits = _parse_tabular("ng1\ttg1\t1e-30\t100\t1000\t300\t100\n", "bug", "templateGEM", "\t")
    assert hits.evalue.iloc[0] == 1e-30


def test_only_genes_in_models_uses_the_template_gene():
    # A new-organism gene that shares its id with a template gene ("tg2") must not
    # keep a hit to a gene the template lacks, which would be its best hit.
    hits = make_ortholog_hits([("tg1", "tg2"), ("tgX", "tg2")], "templateGEM", "bug")
    hits.loc[hits.to_gene == "tgX", "bitscore"] = 2000.0
    res = get_model_from_homology([_template()], hits, "bug", strictness=3,
                                  only_genes_in_models=True)
    assert res.model.reactions.get_by_id("R_single").gene_reaction_rule == "tg2"


def test_only_genes_in_models_rejects_an_emptied_direction():
    hits = make_ortholog_hits([("tgX", "ng1"), ("tg1", "ng2")], "templateGEM", "bug")
    hits.loc[hits.to_gene.isin(["tg1", "ng2"]), "identity"] = 10.0
    assert not get_model_from_homology([_template()], hits, "bug").model.reactions
    with pytest.raises(ValueError, match="No genes in matching from"):
        get_model_from_homology([_template()], hits, "bug", only_genes_in_models=True)


def test_miscased_organism_ids_are_rejected():
    hits = make_ortholog_hits([("tg1", "ng1")], "templateGEM", "bug")
    with pytest.raises(ValueError, match=r"only when case is ignored: \['bug'\]"):
        get_model_from_homology([_template()], hits, "Bug")
    other = make_ortholog_hits([("tg1", "ng1")], "otherGEM", "bug")
    res = get_model_from_homology([_template()], pd.concat([hits, other]), "bug")
    assert "R_single" in {r.id for r in res.model.reactions}


def test_one_directional_non_reciprocal():
    # build hits with only the new->old direction present
    hits = make_ortholog_hits([("tg1", "ng1")], "templateGEM", "bug")
    one_way = hits[hits.from_id == "bug"]  # drop the template->new rows
    t = _template()
    # bidirectional default would find nothing; one-directional should map
    assert "R_single" not in {r.id for r in get_model_from_homology([t], one_way, "bug").model.reactions}
    res = get_model_from_homology([t], one_way, "bug", strictness=2, map_direction="new_to_old")
    assert "R_single" in {r.id for r in res.model.reactions}


# --- preferred order -------------------------------------------------------

def test_preferred_order_routes_gene_to_one_model():
    t1 = _template()
    t1.id = "modelA"
    t2 = _template()
    t2.id = "modelB"
    hits1 = make_ortholog_hits([("tg1", "ng1")], "modelA", "bug")
    hits2 = make_ortholog_hits([("tg1", "ng1")], "modelB", "bug")
    hits = pd.concat([hits1, hits2], ignore_index=True)
    res = get_model_from_homology([t1, t2], hits, "bug", preferred_order=["modelA", "modelB"])
    # ng1's reaction comes only from modelA
    sources = {r.notes.get("homology_source") for r in res.model.reactions if r.id.startswith("R_single")}
    assert sources == {"modelA"}


def _two_templates(pairs1, pairs2, rxn1="R_T1", rxn2="R_T2"):
    t1 = _one_rule_template("T1", rxn1, "Y")
    t2 = _one_rule_template("T2", rxn2, "X")
    hits = pd.concat(
        [make_ortholog_hits(pairs1, "T1", "N"), make_ortholog_hits(pairs2, "T2", "N")], ignore_index=True
    )
    return [t1, t2], hits


def test_preferred_order_removes_a_gene_that_maps_to_a_taken_gene():
    templates, hits = _two_templates([("Y", "N1")], [("X", "N1"), ("X", "N2")])
    first = get_model_from_homology(templates, hits, "N", preferred_order=["T1", "T2"])
    assert _rules(first) == {"R_T1": "N1"}
    assert "X" not in first.gene_map.get("T2", {})
    second = get_model_from_homology(templates, hits, "N", preferred_order=["T2", "T1"])
    assert _rules(second) == {"R_T2": "N1 or N2"}
    assert _rules(get_model_from_homology(templates, hits, "N")) == {"R_T1": "N1", "R_T2": "N1 or N2"}


def test_preferred_order_without_shared_genes_merges_in_given_order():
    templates, hits = _two_templates([("Y", "N1")], [("X", "N2")], rxn1="R1", rxn2="R1")
    plain = _rules(get_model_from_homology(templates, hits, "N"))
    ordered = _rules(get_model_from_homology(templates, hits, "N", preferred_order=["T2", "T1"]))
    assert ordered == plain
    assert plain["R1"] == "N1"


def test_preferred_order_keeps_a_reaction_that_survives_without_the_removed_gene():
    t1 = _one_rule_template("T1", "R_T1", "Y")
    t2 = _one_rule_template("T2", "R_T2", "X or Z")
    add_reactions_from_equations(t2, [{"id": "R_T3", "equation": "b_c --> a_c", "gene_reaction_rule": "X and Z"}])
    hits = pd.concat(
        [make_ortholog_hits([("Y", "N1")], "T1", "N"),
         make_ortholog_hits([("X", "N1"), ("Z", "N3")], "T2", "N")],
        ignore_index=True,
    )
    res = get_model_from_homology([t1, t2], hits, "N", preferred_order=["T1", "T2"])
    assert _rules(res) == {"R_T1": "N1", "R_T2": "N3"}  # R_T3 needs the removed X


# --- gene-free reactions, notes, duplicate genes, metadata ------------------------

def _template_with_extras():
    m = _template()
    m.add_metabolites(cobra.Metabolite("e", name="E", compartment="c"))
    add_reactions_from_equations(
        m, [{"id": "R_free", "equation": "d --> e"}], allow_new_mets=False,
    )
    m.reactions.R_single.notes = {"references": "PMID:1", "confidence_score": 4}
    m.notes = {"metaData": {"id": "templateGEM", "version": "9.9.9"}, "version": "9.9.9", "keep": "me"}
    return m


def test_gene_free_reactions_dropped_by_default():
    hits = make_ortholog_hits([("tg1", "ng1")], "templateGEM", "bug")
    res = get_model_from_homology([_template_with_extras()], hits, "bug")
    assert "R_free" not in {r.id for r in res.model.reactions}


def test_keep_gene_free_transfers_reactions_without_a_gpr():
    hits = make_ortholog_hits([("tg1", "ng1")], "templateGEM", "bug")
    res = get_model_from_homology([_template_with_extras()], hits, "bug", keep_gene_free=True)
    ids = {r.id for r in res.model.reactions}
    assert {"R_single", "R_free"} <= ids
    assert res.model.reactions.get_by_id("R_free").gene_reaction_rule == ""


def test_preserve_notes_keeps_template_notes_and_adds_source():
    hits = make_ortholog_hits([("tg1", "ng1")], "templateGEM", "bug")
    kept = get_model_from_homology([_template_with_extras()], hits, "bug", preserve_notes=True)
    notes = kept.model.reactions.R_single.notes
    assert notes["references"] == "PMID:1" and notes["confidence_score"] == 4
    assert notes["homology_source"] == "templateGEM"
    default = get_model_from_homology([_template_with_extras()], hits, "bug")
    assert default.model.reactions.R_single.notes["note"] == "Included by get_model_from_homology"


def test_duplicate_orthologs_are_not_repeated_in_a_rule():
    hits = make_ortholog_hits([("tg2", "ng"), ("tg3", "ng")], "templateGEM", "bug")
    res = get_model_from_homology([_template()], hits, "bug")
    assert res.model.reactions.R_iso.gene_reaction_rule == "ng"


def test_isozymes_mapping_to_shared_orthologs_are_deduplicated():
    hits = make_ortholog_hits([("tg2", "na"), ("tg2", "nb"), ("tg3", "nb")], "templateGEM", "bug")
    res = get_model_from_homology([_template()], hits, "bug")
    assert res.model.reactions.R_iso.gene_reaction_rule == "na or nb"


def test_draft_does_not_inherit_template_version_and_date():
    hits = make_ortholog_hits([("tg1", "ng1")], "templateGEM", "bug")
    res = get_model_from_homology([_template_with_extras()], hits, "bug")
    assert res.model.notes == {"keep": "me"}
