"""Tests for raven_toolbox.utils.gpr (GPR linting)."""
import cobra
import pytest

from raven_toolbox.utils import GPRIssue, find_non_dnf_grrules, is_dnf


@pytest.mark.parametrize(
    "rule",
    [
        "",
        "G1",
        "G1 and G2",
        "G1 or G2",
        "G1 and G2 and G3",
        "G1 or G2 or G3",
        "(G1 and G2) or G3",
        "(G1 and G2) or (G3 and G4)",
        "G1 or (G2 and G3)",
    ],
)
def test_is_dnf_true(rule):
    assert is_dnf(rule) is True


@pytest.mark.parametrize(
    "rule",
    [
        "(G1 or G2) and G3",
        "G1 and (G2 or G3)",
        "(G1 or G2) and (G3 or G4)",
        "G1 and (G2 or (G3 and G4))",
    ],
)
def test_is_dnf_false(rule):
    assert is_dnf(rule) is False


def test_is_dnf_accepts_gpr_and_none():
    from cobra.core.gene import GPR

    assert is_dnf(GPR.from_string("(G1 or G2) and G3")) is False
    assert is_dnf(GPR.from_string("G1 or G2")) is True
    assert is_dnf(None) is True


def test_is_dnf_independent_of_formatting():
    # cobra normalises on assignment, so casing/whitespace cannot change the verdict.
    assert is_dnf("(G1 OR G2)   AND   G3") is False
    assert is_dnf("( G1 and G2 )  or  G3") is True


def _model_with_rules(rules: dict[str, str]) -> cobra.Model:
    model = cobra.Model("t")
    model.add_reactions([cobra.Reaction(rid) for rid in rules])
    for rid, rule in rules.items():
        model.reactions.get_by_id(rid).gene_reaction_rule = rule
    return model


def test_find_non_dnf_grrules_flags_only_offenders():
    model = _model_with_rules(
        {
            "R_ok_single": "G1",
            "R_ok_complex": "G1 and G2",
            "R_ok_dnf": "(G1 and G2) or G3",
            "R_no_gpr": "",
            "R_bad_1": "(G1 or G2) and G3",
            "R_bad_2": "(G1 or G2) and (G3 or G4)",
        }
    )

    issues = find_non_dnf_grrules(model)

    assert [i.reaction_id for i in issues] == ["R_bad_1", "R_bad_2"]
    assert all(isinstance(i, GPRIssue) for i in issues)
    assert all("disjunctive normal form" in i.reason for i in issues)
    # the reported GPR is the cobra-normalised string
    assert issues[0].gpr == "(G1 or G2) and G3"


def test_find_non_dnf_grrules_empty_when_all_clean():
    model = _model_with_rules({"R1": "G1 or G2", "R2": "(G1 and G2) or G3"})
    assert find_non_dnf_grrules(model) == []


@pytest.mark.parametrize(
    "rule",
    ["G1 and", "(G1 or G2", "G1 or or G2", "G1 G2", "and G1", "(G1 and G2)) or G3", "()"],
)
def test_is_dnf_raises_on_malformed_rule(rule):
    """RAVEN's isDnfGrRule errors on these (RAVEN:badGrRule); cobra alone returns an empty GPR."""
    with pytest.raises(ValueError, match="Malformed grRule"):
        is_dnf(rule)


@pytest.mark.parametrize("rule", ["", "   ", None])
def test_is_dnf_blank_rule_is_still_true(rule):
    assert is_dnf(rule) is True
