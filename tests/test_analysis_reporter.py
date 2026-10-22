"""Tests for Reporter Metabolites (analysis/reporter.py)."""
import cobra
import pytest

from raven_toolbox.analysis import ReporterResult, reporter_metabolites


def _met(mid):
    return cobra.Metabolite(mid, name=mid[:-2], compartment="c")


@pytest.fixture
def model():
    """A-r1(g1)-B-r2(g2)-C-r3(g3); rX touches X but has no gene."""
    m = cobra.Model("rep")
    A, B, C, X = _met("A_c"), _met("B_c"), _met("C_c"), _met("X_c")
    m.add_metabolites([A, B, C, X])
    r1 = cobra.Reaction("r1")
    r1.add_metabolites({A: -1, B: 1})
    r2 = cobra.Reaction("r2")
    r2.add_metabolites({B: -1, C: 1})
    r3 = cobra.Reaction("r3")
    r3.add_metabolites({C: -1})
    rX = cobra.Reaction("rX")
    rX.add_metabolites({X: -1})
    m.add_reactions([r1, r2, r3, rX])
    r1.gene_reaction_rule = "g1"
    r2.gene_reaction_rule = "g2"
    r3.gene_reaction_rule = "g3"
    return m


def test_ranks_metabolites_by_surrounding_significance(model):
    # g1, g2 highly significant; g3 not. B (g1,g2) > A (g1) > C (g2,g3).
    (res,) = reporter_metabolites(model, {"g1": 0.001, "g2": 0.001, "g3": 0.5})
    assert isinstance(res, ReporterResult) and res.test == "all"
    assert list(res.table["metabolite"]) == ["B_c", "A_c", "C_c"]
    assert res.table["z_score"].is_monotonic_decreasing
    assert "X_c" not in set(res.table["metabolite"])  # no neighbouring genes -> excluded


def test_neighbour_counts(model):
    (res,) = reporter_metabolites(model, {"g1": 0.01, "g2": 0.01, "g3": 0.01})
    counts = dict(zip(res.table["metabolite"], res.table["n_genes"], strict=True))
    assert counts == {"A_c": 1, "B_c": 2, "C_c": 2}


def test_uniform_pvalues_give_zero_scores(model):
    # All genes identical -> background std 0 -> nothing stands out (corrected z = 0).
    (res,) = reporter_metabolites(model, {"g1": 0.2, "g2": 0.2, "g3": 0.2})
    assert (res.table["z_score"] == 0.0).all()
    assert res.table["p_value"].to_numpy() == pytest.approx(0.5)


def test_p_value_low_for_top_metabolite(model):
    (res,) = reporter_metabolites(model, {"g1": 1e-6, "g2": 1e-6, "g3": 0.9})
    top = res.table.iloc[0]
    assert top["metabolite"] == "B_c"
    assert top["p_value"] < 0.5  # enriched -> significant


def test_fold_change_splits_up_down(model):
    res = reporter_metabolites(
        model,
        {"g1": 0.001, "g2": 0.001, "g3": 0.001},
        gene_fold_changes={"g1": 2.0, "g2": -2.0, "g3": 1.0},
    )
    assert [r.test for r in res] == ["all", "up", "down"]
    # 'up' uses g1,g3 -> A(g1) and C(g3) have neighbours; B needs g2 (down) so its
    # only 'up' neighbour is g1 -> still present. 'down' uses only g2.
    down = next(r for r in res if r.test == "down").table
    assert set(down["metabolite"]) <= {"B_c", "C_c"}  # g2 touches B and C


def test_all_zero_fold_changes_give_only_the_all_test(model):
    res = reporter_metabolites(
        model,
        {"g1": 0.001, "g2": 0.001, "g3": 0.001},
        gene_fold_changes={"g1": 0.0, "g2": 0.0, "g3": 0.0},
    )
    assert [r.test for r in res] == ["all"]


def test_std_z_is_zero_for_a_single_neighbouring_gene(model):
    (res,) = reporter_metabolites(model, {"g1": 0.001, "g2": 0.01, "g3": 0.5})
    std = dict(zip(res.table["metabolite"], res.table["std_z"], strict=True))
    assert std["A_c"] == 0.0
    assert std["B_c"] > 0.0


def test_filters_unknown_and_nan_genes(model):
    # gX not in model, gNaN has NaN p-value -> both ignored; result still computed.
    (res,) = reporter_metabolites(
        model, {"g1": 0.01, "g2": 0.01, "g3": 0.01, "gX": 0.001, "gNaN": float("nan")}
    )
    assert "gX" not in set(model.genes.list_attr("id"))  # sanity
    assert len(res.table) == 3  # A, B, C scored from the three real genes


def test_out_of_range_pvalue_dropped_not_poisoning(model):
    """A p-value outside [0,1] is dropped, not propagated as NaN through all scores."""
    (res,) = reporter_metabolites(model, {"g1": 0.01, "g2": 0.01, "g3": 1.7})  # g3 invalid
    import numpy as np

    assert not np.isnan(res.table["z_score"].to_numpy()).any()  # no NaN poisoning


def test_fold_changes_of_genes_outside_the_model_do_not_count(model):
    res = reporter_metabolites(
        model,
        {"g1": 0.001, "g2": 0.01},
        gene_fold_changes={"g1": 0.0, "g2": 0.0, "gX": 1.5},
    )
    assert [r.test for r in res] == ["all"]


def test_nan_fold_changes_do_not_count(model):
    res = reporter_metabolites(
        model,
        {"g1": 0.001, "g2": 0.01},
        gene_fold_changes={"g1": 0.0, "g2": float("nan")},
    )
    assert [r.test for r in res] == ["all"]
