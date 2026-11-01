"""Tests for guess_composition (guessComposition port)."""
import cobra
import pytest

from raven_toolbox.utils import GuessCompositionResult, guess_composition


def _met(mid, name, formula=None, compartment="c"):
    return cobra.Metabolite(mid, name=name, formula=formula, compartment=compartment)


def test_accounts_for_own_stoichiometric_coefficient():
    """A -> 2 B: B's own coefficient is 2, so its atom counts are half of
    A's, not identical to A's, as in RAVEN's guessComposition."""
    m = cobra.Model("t")
    a = _met("a_c", "A", formula="CH4")
    b = _met("b_c", "B")
    m.add_metabolites([a, b])
    r = cobra.Reaction("R1")
    m.add_reactions([r])
    r.add_metabolites({a: -1, b: 2})

    result = guess_composition(m, print_results=False)

    assert result == GuessCompositionResult(guessed=["B"], could_not_guess=[])
    assert b.formula == "C0.5H2"


def test_resolves_fixed_point_chain():
    """C depends on B depending on A: resolving A unlocks B in a later pass,
    which unlocks C in a later pass still."""
    m = cobra.Model("t")
    a = _met("a_c", "A", formula="CH4")
    b = _met("b_c", "B")
    c = _met("c_c", "C")
    m.add_metabolites([a, b, c])
    r1 = cobra.Reaction("R1")
    r2 = cobra.Reaction("R2")
    m.add_reactions([r1, r2])
    r1.add_metabolites({a: -1, b: 1})
    r2.add_metabolites({b: -1, c: 1})

    result = guess_composition(m, print_results=False)

    assert set(result.guessed) == {"B", "C"}
    assert result.could_not_guess == []
    assert b.formula == "CH4"
    assert c.formula == "CH4"


def test_same_name_different_compartments_share_one_resolution():
    """B's cytosol copy has no reactions of its own; its formula is still
    filled in once the extracellular copy is resolved, since composition is
    tracked per name, not per compartment instance."""
    m = cobra.Model("t")
    a = _met("a_e", "A", formula="CH4", compartment="e")
    b_e = _met("b_e", "B", compartment="e")
    b_c = _met("b_c", "B", compartment="c")
    m.add_metabolites([a, b_e, b_c])
    r = cobra.Reaction("R1")
    m.add_reactions([r])
    r.add_metabolites({a: -1, b_e: 1})

    result = guess_composition(m, print_results=False)

    assert result.guessed == ["B"]
    assert b_e.formula == "CH4"
    assert b_c.formula == "CH4"


def test_metabolite_instance_without_reactions_does_not_block_resolution():
    m = cobra.Model("t")
    a = _met("a_c", "A", formula="CH4")
    b = _met("b_c", "B")
    orphan = _met("orphan_c", "Orphan")
    m.add_metabolites([a, b, orphan])
    r = cobra.Reaction("R1")
    m.add_reactions([r])
    r.add_metabolites({a: -1, b: 1})

    result = guess_composition(m, print_results=False)

    assert "B" in result.guessed
    assert "Orphan" in result.could_not_guess


def test_contradiction_warns_and_leaves_unresolved():
    """Two reactions imply different compositions for B: R1 says B == A
    (CH4), R2 says B == D (C2H6). Neither is trusted."""
    m = cobra.Model("t")
    a = _met("a_c", "A", formula="CH4")
    d = _met("d_c", "D", formula="C2H6")
    b = _met("b_c", "B")
    m.add_metabolites([a, d, b])
    r1 = cobra.Reaction("R1")
    r2 = cobra.Reaction("R2")
    m.add_reactions([r1, r2])
    r1.add_metabolites({a: -1, b: 1})
    r2.add_metabolites({d: -1, b: 1})

    with pytest.warns(UserWarning, match="inconsistencies"):
        result = guess_composition(m, print_results=False)

    assert result.guessed == []
    assert result.could_not_guess == ["B"]
    assert b.formula is None or b.formula == ""


def test_unparseable_formula_treated_as_unknown():
    """A polymer formula makes cobra's Metabolite.elements return None; that
    counts as "not known" rather than crashing the balance computation."""
    m = cobra.Model("t")
    poly = _met("poly_c", "Poly", formula="(C5H8)n")
    b = _met("b_c", "B")
    m.add_metabolites([poly, b])
    r = cobra.Reaction("R1")
    m.add_reactions([r])
    r.add_metabolites({poly: -1, b: 1})

    result = guess_composition(m, print_results=False)

    assert result.guessed == []
    assert result.could_not_guess == ["B"]


def test_print_results(capsys):
    m = cobra.Model("t")
    a = _met("a_c", "A", formula="CH4")
    b = _met("b_c", "B")
    m.add_metabolites([a, b])
    r = cobra.Reaction("R1")
    m.add_reactions([r])
    r.add_metabolites({a: -1, b: 1})

    guess_composition(m)

    assert 'Predicted composition for "B" to be CH4' in capsys.readouterr().out


def test_print_results_false_is_silent(capsys):
    m = cobra.Model("t")
    a = _met("a_c", "A", formula="CH4")
    b = _met("b_c", "B")
    m.add_metabolites([a, b])
    r = cobra.Reaction("R1")
    m.add_reactions([r])
    r.add_metabolites({a: -1, b: 1})

    guess_composition(m, print_results=False)

    assert capsys.readouterr().out == ""


def test_already_complete_model_guesses_nothing():
    m = cobra.Model("t")
    a = _met("a_c", "A", formula="CH4")
    m.add_metabolites([a])

    result = guess_composition(m, print_results=False)

    assert result == GuessCompositionResult(guessed=[], could_not_guess=[])
