"""Tests for raven_toolbox.utils.gpr (GPR linting)."""
import re

import cobra
import pytest

from raven_toolbox.utils import GPRIssue, find_non_dnf_grrules, is_dnf, parse_gr_rule


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
    with pytest.raises(ValueError, match="grRule"):
        is_dnf(rule)


@pytest.mark.parametrize("rule", ["", "   ", None])
def test_is_dnf_blank_rule_is_still_true(rule):
    assert is_dnf(rule) is True


@pytest.mark.parametrize(
    "rule, expected",
    [
        ("G1 && G2", "G1 and G2"),
        ("G1 || G2", "G1 or G2"),
        ("G1 Or G2 aNd G3", "G1 or (G2 and G3)"),
        ("G1 AND (G2 | G3)", "G1 and (G2 or G3)"),
        ("10 and 100", "10 and 100"),
    ],
)
def test_parse_gr_rule_takes_the_parsegrrule_operators(rule, expected):
    assert parse_gr_rule(rule).to_string() == expected


def test_parse_gr_rule_reads_an_operator_only_as_a_whole_word():
    """``G1&G2`` is one gene id in RAVEN's parseGrRule; cobra alone would read it as an AND."""
    assert sorted(parse_gr_rule("G1&G2 or RAND1").genes) == ["G1&G2", "RAND1"]


@pytest.mark.parametrize("rule", ["", "   ", None])
def test_parse_gr_rule_blank_is_none(rule):
    assert parse_gr_rule(rule) is None


@pytest.mark.parametrize(
    "rule, message",
    [
        ("G1 and", "grRule ends after an operator"),
        ("(G1 or G2", "Unbalanced parentheses in grRule"),
        ("G1 G2", 'Unexpected "G2" in grRule'),
        ("()", 'Expected a gene but found ")" in grRule'),
        ("G1 xor G2", 'Unexpected "xor" in grRule'),
    ],
)
def test_parse_gr_rule_raises_ravens_message(rule, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_gr_rule(rule)


# The code points for which MATLAB's isspace is true, enumerated over char(0:65535).
_MATLAB_ISSPACE = [9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 5760, *range(8192, 8199), 8200, 8201,
                   8202, 8232, 8233, 8287, 12288]


def test_parse_gr_rule_whitespace_is_matlab_isspace():
    for code in range(65536):
        rule = f"G1{chr(code)}and G2"
        if code in _MATLAB_ISSPACE:
            assert sorted(parse_gr_rule(rule).genes) == ["G1", "G2"], code
        elif chr(code) not in "()":
            with pytest.raises(ValueError):
                parse_gr_rule(rule)


@pytest.mark.parametrize("char", [" ", "\u0085", " ", " "])
def test_parse_gr_rule_a_space_that_isspace_refuses_is_part_of_a_word(char):
    with pytest.raises(ValueError, match=re.escape(f'Unexpected "G2" in grRule: G1{char}and G2')):
        parse_gr_rule(f"G1{char}and G2")
    assert sorted(parse_gr_rule(f"G1{char}G2").genes) == [f"G1{char}G2"]
    assert parse_gr_rule(char) is not None


@pytest.mark.parametrize("char", ["\x1c", " ", "　"])
def test_parse_gr_rule_isspace_characters_separate_words(char):
    assert parse_gr_rule(f"G1{char}or{char}G2").to_string() == "G1 or G2"
    assert parse_gr_rule(char * 2) is None
