"""Tests for expand_model (RAVEN expandModel.m) — splitting isozymes into reactions."""
import cobra
import pytest

from raven_toolbox.manipulation import expand_model
from raven_toolbox.manipulation.expand import gpr_to_dnf

# --------------------------------------------------------------------------- #
# DNF conversion (internal helper, worth testing directly)
# --------------------------------------------------------------------------- #

def _dnf_from_gpr_string(gpr_str: str) -> list[list[str]]:
    from cobra.core.gene import GPR

    gpr = GPR.from_string(gpr_str)
    return gpr_to_dnf(gpr)


def test_dnf_empty_gpr():
    assert _dnf_from_gpr_string("") == []


def test_dnf_single_gene():
    assert _dnf_from_gpr_string("g1") == [["g1"]]


def test_dnf_simple_and():
    assert _dnf_from_gpr_string("g1 and g2") == [["g1", "g2"]]


def test_dnf_simple_or():
    assert _dnf_from_gpr_string("g1 or g2") == [["g1"], ["g2"]]


def test_dnf_or_of_ands():
    assert _dnf_from_gpr_string("(g1 and g2) or (g3 and g4)") == [
        ["g1", "g2"],
        ["g3", "g4"],
    ]


def test_dnf_distributes_and_over_or():
    result = _dnf_from_gpr_string("g1 and (g2 or g3)")
    assert result == [["g1", "g2"], ["g1", "g3"]]


def test_dnf_triple_or():
    assert _dnf_from_gpr_string("g1 or g2 or g3") == [
        ["g1"], ["g2"], ["g3"],
    ]


def test_dnf_preserves_gene_order_within_clause():
    result = _dnf_from_gpr_string("g3 and g1 and g2")
    assert result == [["g3", "g1", "g2"]]


# --------------------------------------------------------------------------- #
# expand_model
# --------------------------------------------------------------------------- #

def _build_model(
    reactions: list[tuple[str, dict[str, float], float, float, str]],
) -> cobra.Model:
    """Build from (rxn_id, {met_id: coef}, lb, ub, gpr) tuples."""
    model = cobra.Model("test")
    mets: dict[str, cobra.Metabolite] = {}
    for _, stoich, _, _, _ in reactions:
        for met_id in stoich:
            if met_id not in mets:
                mets[met_id] = cobra.Metabolite(met_id, compartment="c")

    for rxn_id, stoich, lb, ub, gpr in reactions:
        rxn = cobra.Reaction(rxn_id)
        rxn.lower_bound = lb
        rxn.upper_bound = ub
        rxn.add_metabolites({mets[m]: c for m, c in stoich.items()})
        if gpr:
            rxn.gene_reaction_rule = gpr
        model.add_reactions([rxn])
    return model


def test_does_not_expand_reaction_without_gpr():
    model = _build_model([("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0, "")])
    added = expand_model(model)
    assert added == []
    assert "r1" in {r.id for r in model.reactions}


def test_does_not_expand_single_and_clause():
    model = _build_model([
        ("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0, "g1 and g2"),
    ])
    added = expand_model(model)
    assert added == []
    r1 = model.reactions.get_by_id("r1")
    assert r1.gene_reaction_rule == "g1 and g2"


def test_does_not_expand_single_gene():
    model = _build_model([("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0, "g1")])
    added = expand_model(model)
    assert added == []
    assert model.reactions.get_by_id("r1").gene_reaction_rule == "g1"


def test_splits_simple_or_into_two_reactions():
    model = _build_model([
        ("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0, "g1 or g2"),
    ])
    added = expand_model(model)

    assert added == ["r1_EXP_1", "r1_EXP_2"]
    rxn_ids = {r.id for r in model.reactions}
    assert "r1" not in rxn_ids
    assert "r1_EXP_1" in rxn_ids
    assert "r1_EXP_2" in rxn_ids

    assert model.reactions.get_by_id("r1_EXP_1").gene_reaction_rule == "g1"
    assert model.reactions.get_by_id("r1_EXP_2").gene_reaction_rule == "g2"


def test_splits_or_of_ands():
    model = _build_model([
        ("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0,
         "(g1 and g2) or (g3 and g4)"),
    ])
    added = expand_model(model)

    assert added == ["r1_EXP_1", "r1_EXP_2"]
    assert model.reactions.get_by_id("r1_EXP_1").gene_reaction_rule == "g1 and g2"
    assert model.reactions.get_by_id("r1_EXP_2").gene_reaction_rule == "g3 and g4"


def test_distributes_and_over_or():
    model = _build_model([
        ("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0,
         "g1 and (g2 or g3)"),
    ])
    added = expand_model(model)

    assert added == ["r1_EXP_1", "r1_EXP_2"]
    assert model.reactions.get_by_id("r1_EXP_1").gene_reaction_rule == "g1 and g2"
    assert model.reactions.get_by_id("r1_EXP_2").gene_reaction_rule == "g1 and g3"


def test_expanded_reactions_inherit_stoichiometry_and_bounds():
    model = _build_model([
        ("r1", {"A": -1.0, "B": 2.0}, -500.0, 1500.0, "g1 or g2"),
    ])
    expand_model(model)

    for suffix in ("_EXP_1", "_EXP_2"):
        rxn = model.reactions.get_by_id(f"r1{suffix}")
        assert rxn.bounds == (-500.0, 1500.0)
        stoich = {m.id: c for m, c in rxn.metabolites.items()}
        assert stoich == {"A": -1.0, "B": 2.0}


def test_expanded_reactions_inherit_name_and_subsystem():
    model = _build_model([
        ("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0, "g1 or g2"),
    ])
    r1 = model.reactions.get_by_id("r1")
    r1.name = "an isozyme-catalyzed reaction"
    r1.subsystem = "central metabolism"

    expand_model(model)

    for suffix in ("_EXP_1", "_EXP_2"):
        rxn = model.reactions.get_by_id(f"r1{suffix}")
        assert rxn.name == "an isozyme-catalyzed reaction"
        assert rxn.subsystem == "central metabolism"


def test_multiple_reactions_expand_independently():
    model = _build_model([
        ("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0, "g1 or g2"),
        ("r2", {"B": -1.0, "C": 1.0}, 0.0, 1000.0, "g3 and g4"),
        ("r3", {"C": -1.0, "D": 1.0}, 0.0, 1000.0,
         "(g5 and g6) or g7 or (g8 and g9)"),
    ])
    added = expand_model(model)

    assert added == sorted([
        "r1_EXP_1", "r1_EXP_2",
        "r3_EXP_1", "r3_EXP_2", "r3_EXP_3",
    ])

    rxn_ids = {r.id for r in model.reactions}
    assert "r2" in rxn_ids
    assert "r1" not in rxn_ids
    assert "r3" not in rxn_ids

    assert model.reactions.get_by_id("r2").gene_reaction_rule == "g3 and g4"
    assert model.reactions.get_by_id("r3_EXP_1").gene_reaction_rule == "g5 and g6"
    assert model.reactions.get_by_id("r3_EXP_2").gene_reaction_rule == "g7"
    assert model.reactions.get_by_id("r3_EXP_3").gene_reaction_rule == "g8 and g9"


def test_expanded_reaction_has_correct_gene_set():
    model = _build_model([
        ("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0,
         "(g1 and g2) or (g3 and g4)"),
    ])
    expand_model(model)

    r1_1 = model.reactions.get_by_id("r1_EXP_1")
    assert {g.id for g in r1_1.genes} == {"g1", "g2"}

    r1_2 = model.reactions.get_by_id("r1_EXP_2")
    assert {g.id for g in r1_2.genes} == {"g3", "g4"}


def test_expansion_is_idempotent_in_the_no_op_sense():
    model = _build_model([
        ("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0, "g1 or g2"),
        ("r2", {"B": -1.0, "C": 1.0}, 0.0, 1000.0, "g3 and g4"),
    ])
    expand_model(model)
    ids_before = {r.id for r in model.reactions}

    second = expand_model(model)
    assert second == []

    ids_after = {r.id for r in model.reactions}
    assert ids_after == ids_before


def test_empty_model_is_unchanged():
    model = cobra.Model("empty")
    assert expand_model(model) == []


# --------------------------------------------------------------------------- #
# Annotation and notes propagation
# --------------------------------------------------------------------------- #

def test_expanded_reactions_inherit_annotation_and_notes():
    model = _build_model([
        ("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0, "g1 or g2"),
    ])
    r1 = model.reactions.get_by_id("r1")
    r1.annotation["ec-code"] = "1.2.3.4"
    r1.annotation["sbo"] = "SBO:0000176"
    r1.notes["custom"] = "hello"

    expand_model(model)

    for suffix in ("_EXP_1", "_EXP_2"):
        rxn = model.reactions.get_by_id(f"r1{suffix}")
        assert rxn.annotation["ec-code"] == "1.2.3.4"
        assert rxn.annotation["sbo"] == "SBO:0000176"
        assert rxn.notes["custom"] == "hello"


def test_expanded_reaction_annotation_is_independent_of_parent():
    """Mutating one expanded reaction's annotation must not affect siblings."""
    model = _build_model([
        ("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0, "g1 or g2"),
    ])
    model.reactions.get_by_id("r1").annotation["ec-code"] = ["1.2.3.4"]

    expand_model(model)

    r1_1 = model.reactions.get_by_id("r1_EXP_1")
    r1_2 = model.reactions.get_by_id("r1_EXP_2")
    r1_1.annotation["ec-code"].append("9.9.9.9")
    assert r1_2.annotation["ec-code"] == ["1.2.3.4"]


def test_objective_coefficient_preserved_on_expansion():
    """An expanded reaction's isozyme copies retain the original objective coefficient."""
    m = cobra.Model("o")
    a, b = (cobra.Metabolite(x, compartment="c") for x in "ab")
    m.add_metabolites([a, b])
    r = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    r.add_metabolites({a: -1, b: 1})
    r.gene_reaction_rule = "g1 or g2"
    m.add_reactions([r])
    m.objective = "r1"  # objective on the soon-to-be-expanded reaction

    expand_model(m)
    coeffs = {rx.id: rx.objective_coefficient for rx in m.reactions}
    assert coeffs == {"r1_EXP_1": 1.0, "r1_EXP_2": 1.0}  # objective survives on both copies


def test_first_copy_keeps_the_original_position_and_other_copies_are_appended():
    model = _build_model([
        ("r1", {"A": -1.0, "B": 1.0}, 0.0, 1000.0, "g1 or g2"),
        ("r2", {"B": -1.0, "C": 1.0}, 0.0, 1000.0, "g3"),
        ("r3", {"C": -1.0, "D": 1.0}, 0.0, 1000.0, "g4 or g5 or g6"),
        ("r4", {"D": -1.0, "E": 1.0}, 0.0, 1000.0, "g7"),
    ])
    expand_model(model)
    assert [r.id for r in model.reactions] == [
        "r1_EXP_1", "r2", "r3_EXP_1", "r4", "r1_EXP_2", "r3_EXP_2", "r3_EXP_3",
    ]
    assert model.reactions.index("r4") == 3


@pytest.mark.parametrize(
    "rule, expected",
    [
        ("G1 && (G2 || G3)", [["G1", "G2"], ["G1", "G3"]]),
        ("G1 Or G2", [["G1"], ["G2"]]),
        ("G1&G2 or G3", [["G1&G2"], ["G3"]]),
        ("", []),
    ],
)
def test_dnf_from_string_uses_the_parsegrrule_grammar(rule, expected):
    assert gpr_to_dnf(rule) == expected


@pytest.mark.parametrize("rule", ["G1 and", "(G1 or G2", "G1 G2", "()", "G1 or or G2"])
def test_dnf_from_string_raises_on_a_malformed_rule(rule):
    with pytest.raises(ValueError, match="grRule"):
        gpr_to_dnf(rule)
