"""Tests for comparison/compare.py — N-model comparison (Phase 5)."""
from __future__ import annotations

import cobra
import pytest

from raven_toolbox.comparison import ModelComparison, compare_models
from raven_toolbox.tasks import Task


def _mk(model_id: str, reactions: list[tuple[str, dict[str, int], str | None]],
        genes: list[str] | None = None) -> cobra.Model:
    """Tiny helper: build a model with the given reactions + optional gene rules + subsystems."""
    m = cobra.Model(model_id)
    mets: dict[str, cobra.Metabolite] = {}
    for _rid, stoich, _ in reactions:
        for mid in stoich:
            if mid not in mets:
                mets[mid] = cobra.Metabolite(mid, name=mid.split("_")[0], compartment="c")
                m.add_metabolites([mets[mid]])
    for (rid, stoich, sub), gpr in zip(reactions, genes or [None] * len(reactions), strict=True):
        r = cobra.Reaction(rid, lower_bound=-1000, upper_bound=1000)
        r.add_metabolites({mets[mid]: c for mid, c in stoich.items()})
        if sub is not None:
            r.subsystem = sub
        if gpr is not None:
            r.gene_reaction_rule = gpr
        m.add_reactions([r])
    return m


@pytest.fixture
def two_models():
    """Two models sharing r1/r2 but each with one unique reaction; different subsystems."""
    a = _mk("A", [("r1", {"A_c": -1, "B_c": 1}, "carbo"),
                  ("r2", {"B_c": -1, "C_c": 1}, "amino"),
                  ("r3", {"C_c": -1, "D_c": 1}, "carbo")],
            genes=["g1", "g2", "g3"])
    b = _mk("B", [("r1", {"A_c": -1, "B_c": 1}, "carbo"),
                  ("r2", {"B_c": -1, "C_c": 1}, "amino"),
                  ("r4", {"B_c": -1, "E_c": 1}, "lipid")],
            genes=["g1", "g2", "g4"])
    return [a, b]


def test_returns_dataclass(two_models):
    res = compare_models(two_models)
    assert isinstance(res, ModelComparison)
    assert res.model_ids == ["A", "B"]


def test_reactions_matrix_shape_and_values(two_models):
    res = compare_models(two_models)
    # union = {r1, r2, r3, r4}; both have r1+r2, only A has r3, only B has r4.
    assert set(res.reactions.index) == {"r1", "r2", "r3", "r4"}
    assert res.reactions.loc["r1", "A"] == 1 and res.reactions.loc["r1", "B"] == 1
    assert res.reactions.loc["r3", "A"] == 1 and res.reactions.loc["r3", "B"] == 0
    assert res.reactions.loc["r4", "A"] == 0 and res.reactions.loc["r4", "B"] == 1


def test_metabolites_and_genes_union(two_models):
    res = compare_models(two_models)
    assert set(res.metabolites.index) == {"A_c", "B_c", "C_c", "D_c", "E_c"}
    assert set(res.genes.index) == {"g1", "g2", "g3", "g4"}
    assert res.genes.loc["g3", "A"] == 1 and res.genes.loc["g3", "B"] == 0


def test_subsystems_counts(two_models):
    res = compare_models(two_models)
    # A: carbo=2 (r1+r3), amino=1; B: carbo=1, amino=1, lipid=1.
    assert res.subsystems.loc["carbo", "A"] == 2
    assert res.subsystems.loc["carbo", "B"] == 1
    assert res.subsystems.loc["lipid", "B"] == 1
    assert res.subsystems.loc["lipid", "A"] == 0


def test_subsystems_empty_falls_under_none():
    a = _mk("A", [("r1", {"X_c": -1, "Y_c": 1}, None)])
    b = _mk("B", [("r1", {"X_c": -1, "Y_c": 1}, "")])
    res = compare_models([a, b])
    assert res.subsystems.loc["(none)", "A"] == 1
    assert res.subsystems.loc["(none)", "B"] == 1


def test_jaccard_similarity_diagonal_and_symmetry(two_models):
    res = compare_models(two_models)
    # Diagonal = 1 (self vs self).
    assert res.similarity.loc["A", "A"] == 1.0
    assert res.similarity.loc["B", "B"] == 1.0
    # Symmetric.
    assert res.similarity.loc["A", "B"] == res.similarity.loc["B", "A"]
    # Shared r1+r2; total union 4 → Jaccard 2/4 = 0.5.
    assert res.similarity.loc["A", "B"] == pytest.approx(0.5)


def test_tasks_optional_and_passed_through(two_models):
    """Both models export E → expect both to pass the make-E task."""
    # Add a sink so E can be excreted (otherwise it accumulates → infeasible at steady state).
    for m in two_models:
        if "E_c" in [x.id for x in m.metabolites]:
            m.add_boundary(m.metabolites.get_by_id("E_c"), type="demand")
    res = compare_models(two_models, tasks=[
        Task(id="make_E", inputs=[("A[c]", 0.0, 1000.0)], outputs=[("E[c]", 1.0, 1.0)]),
    ])
    assert res.tasks is not None
    assert list(res.tasks.index) == ["make_E"]
    # Only B has r4 (which makes E), so only B passes.
    assert bool(res.tasks.loc["make_E", "B"]) is True
    assert bool(res.tasks.loc["make_E", "A"]) is False


def test_tasks_with_a_repeated_id_keep_their_own_results(two_models):
    """The task matrix is filled by position: two tasks with one id get their own rows."""
    for m in two_models:
        for met in ("D_c", "E_c"):
            if met in [x.id for x in m.metabolites]:
                m.add_boundary(m.metabolites.get_by_id(met), type="demand")
    makes_e = Task(id="BS", inputs=[("A[c]", 0.0, 1000.0)], outputs=[("E[c]", 1.0, 1.0)])
    makes_d = Task(id="BS", inputs=[("A[c]", 0.0, 1000.0)], outputs=[("D[c]", 1.0, 1.0)])
    res = compare_models(two_models, tasks=[makes_e, makes_d])
    assert res.tasks is not None
    assert list(res.tasks.index) == ["BS", "BS"]
    assert res.tasks.iloc[0].tolist() == [False, True]    # only B makes E
    assert res.tasks.iloc[1].tolist() == [True, False]    # only A makes D


def test_duplicate_or_missing_model_id_disambiguated():
    """Two models with the same id (or empty id) should get distinct labels."""
    a = _mk("", [("r1", {"X_c": -1, "Y_c": 1}, None)])
    b = _mk("", [("r1", {"X_c": -1, "Y_c": 1}, None)])
    res = compare_models([a, b])
    assert res.model_ids[0] == "model_0"
    assert res.model_ids[1] != "model_0"      # disambiguated


def test_rejects_single_model(two_models):
    with pytest.raises(ValueError, match="needs .*2"):
        compare_models(two_models[:1])


def test_ec_codes_union_flattens_list_annotations(two_models):
    a, b = two_models
    a.reactions.get_by_id("r1").annotation["ec-code"] = ["1.1.1.1", "1.1.1.2"]
    b.reactions.get_by_id("r1").annotation["ec-code"] = "1.1.1.1"
    res = compare_models([a, b])
    assert set(res.ec_codes.index) == {"1.1.1.1", "1.1.1.2"}
    assert res.ec_codes.loc["1.1.1.1", "A"] == 1 and res.ec_codes.loc["1.1.1.1", "B"] == 1
    assert res.ec_codes.loc["1.1.1.2", "A"] == 1 and res.ec_codes.loc["1.1.1.2", "B"] == 0


def test_metabolite_names_union(two_models):
    res = compare_models(two_models)
    # names are set to the id stripped of its compartment suffix in _mk.
    assert set(res.metabolite_names.index) == {"A", "B", "C", "D", "E"}
    assert res.metabolite_names.loc["D", "A"] == 1 and res.metabolite_names.loc["D", "B"] == 0


def test_equations_are_direction_and_order_independent():
    """Two reversible reactions that are the same up to metabolite order and which
    side was written as reactants must canonicalize to the same equation string."""
    a = cobra.Model("A")
    x, y = cobra.Metabolite("x_c", compartment="c"), cobra.Metabolite("y_c", compartment="c")
    a.add_metabolites([x, y])
    r = cobra.Reaction("r1", lower_bound=-1000, upper_bound=1000)
    r.add_metabolites({x: -1, y: 1})
    a.add_reactions([r])

    b = cobra.Model("B")
    x2, y2 = cobra.Metabolite("x_c", compartment="c"), cobra.Metabolite("y_c", compartment="c")
    b.add_metabolites([x2, y2])
    r2 = cobra.Reaction("r1", lower_bound=-1000, upper_bound=1000)
    r2.add_metabolites({y2: -1, x2: 1})  # written in the opposite direction
    b.add_reactions([r2])

    res = compare_models([a, b])
    assert res.equations.shape[0] == 1  # both models' r1 collapse to the same row
    assert res.equations.iloc[0]["A"] == 1 and res.equations.iloc[0]["B"] == 1


def test_equations_no_compartments_merges_across_compartments():
    """Same reaction shape in different compartments: distinct under `equations`,
    merged under `equations_no_compartments`."""
    a = cobra.Model("A")
    xc, yc = cobra.Metabolite("x_c", compartment="c"), cobra.Metabolite("y_c", compartment="c")
    a.add_metabolites([xc, yc])
    ra = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    ra.add_metabolites({xc: -1, yc: 1})
    a.add_reactions([ra])

    b = cobra.Model("B")
    xm, ym = cobra.Metabolite("x_m", compartment="m"), cobra.Metabolite("y_m", compartment="m")
    b.add_metabolites([xm, ym])
    rb = cobra.Reaction("r1", lower_bound=0, upper_bound=1000)
    rb.add_metabolites({xm: -1, ym: 1})
    b.add_reactions([rb])

    res = compare_models([a, b])
    assert res.equations.shape[0] == 2  # "x_c => y_c" and "x_m => y_m" stay distinct
    assert res.equations_no_compartments.shape[0] == 1  # both reduce to "x => y"
    assert res.equations_no_compartments.iloc[0]["A"] == 1
    assert res.equations_no_compartments.iloc[0]["B"] == 1
