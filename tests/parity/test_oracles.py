"""Tier 1: compare against answers recorded from MATLAB RAVEN.

Some behaviour cannot be inferred from a file -- how RAVEN grades a reaction's
elemental balance, what it normalises a gene rule to. For those, MATLAB is run
once by ``scripts/parity/generate_oracles.m`` and its answers are committed as
JSON; these tests assert raven-toolbox agrees.

Each test skips, with instructions, when its oracle has not been generated, so
a checkout without oracles still has a green suite. The trade-off is deliberate:
a skipped test is visible in the report, a fabricated oracle is not.
"""
from __future__ import annotations

import pytest

from raven_toolbox.io import read_yaml_model
from raven_toolbox.manipulation import gpr_to_dnf
from raven_toolbox.utils import get_elemental_balance

pytestmark = pytest.mark.parity


@pytest.fixture
def tiny(parity_fixture_dir):
    return read_yaml_model(parity_fixture_dir / "tiny.yml")


def test_model_structure_matches_matlab(tiny, oracle):
    """The two readers see the same model in the same file."""
    expected = oracle("model_structure")

    assert sorted(r.id for r in tiny.reactions) == sorted(expected["rxns"])
    assert sorted(m.id for m in tiny.metabolites) == sorted(expected["mets"])
    assert sorted(g.id for g in tiny.genes) == sorted(expected["genes"])

    bounds = {r.id: (r.lower_bound, r.upper_bound) for r in tiny.reactions}
    for rid, lb, ub in zip(
        expected["rxns"], expected["lb"], expected["ub"], strict=True
    ):
        assert bounds[rid] == pytest.approx((lb, ub)), f"{rid}: bounds differ"

    ours = {
        (r.id, m.id): coeff
        for r in tiny.reactions
        for m, coeff in r.metabolites.items()
    }
    theirs = {
        (e["rxn"], e["met"]): e["coefficient"] for e in expected["stoichiometry"]
    }
    assert ours.keys() == theirs.keys(), "different reaction/metabolite pairs"
    for key, coeff in theirs.items():
        assert ours[key] == pytest.approx(coeff), f"{key}: coefficient differs"


def test_elemental_balance_matches_matlab(tiny, oracle):
    """Balanced / unbalanced / undecidable must be graded the same way.

    RAVEN reports 1, 0 and -1; raven-toolbox reports the same three states by
    name. The fixture contains one of each, including a reaction whose
    metabolite has no formula -- the case plain cobrapy silently miscounts.
    """
    expected = oracle("elemental_balance")
    matlab = dict(zip(expected["rxns"], expected["balanceStatus"], strict=True))

    to_status = {1: "balanced", 0: "unbalanced", -1: "unknown"}
    report = get_elemental_balance(tiny)
    ours = {entry.reaction_id: entry.status for entry in report}

    for rid, code in matlab.items():
        assert rid in ours, f"{rid} missing from the Python report"
        assert ours[rid] == to_status[int(code)], (
            f"{rid}: MATLAB says {to_status[int(code)]}, Python says {ours[rid]}"
        )


def test_gpr_normalisation_matches_matlab(tiny, oracle):
    """Gene rules normalise to the same disjunctive normal form."""
    expected = oracle("gpr_dnf")
    matlab = {
        rid: dnf
        for rid, dnf in zip(expected["rxns"], expected["dnf"], strict=True)
        if dnf
    }

    for rid, their_dnf in matlab.items():
        ours = gpr_to_dnf(tiny.reactions.get_by_id(rid).gpr)
        ours_clauses = {frozenset(clause) for clause in ours}
        assert ours_clauses == _clauses(their_dnf), (
            f"{rid}: MATLAB normalised to {their_dnf!r}, Python to {ours!r}"
        )


def _clauses(rule: str | list) -> set[frozenset[str]]:
    """A DNF rule as a set of AND-clauses, so ordering carries no weight.

    RAVEN returns the rule as a list of clauses, each a list of genes (a lone clause or gene
    collapses to a shorter list or a string in JSON); a string rule is parsed.
    """
    if not rule:
        return set()
    if isinstance(rule, list):
        return {
            frozenset([clause] if isinstance(clause, str) else clause)
            for clause in rule
        }
    stripped = rule.replace("(", " ").replace(")", " ")
    return {
        frozenset(part.strip() for part in clause.split(" and ") if part.strip())
        for clause in stripped.split(" or ")
        if clause.strip()
    }
