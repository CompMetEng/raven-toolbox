"""Tests for Reporter Metabolites (analysis/reporter.py)."""
import math

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


@pytest.fixture
def three_gene_pair_model():
    """Metabolites A, B and C, each around one reaction with two genes: {g1, g2}, {g3, g4}, {g5, g6}."""
    m = cobra.Model("pairs")
    rxns = []
    for mid, rule in (("A", "g1 or g2"), ("B", "g3 or g4"), ("C", "g5 or g6")):
        met = cobra.Metabolite(mid, name=mid, compartment="c")
        rxn = cobra.Reaction(f"r{mid}")
        rxn.add_metabolites({met: -1})
        rxns.append((rxn, rule))
    m.add_reactions([r for r, _ in rxns])
    for rxn, rule in rxns:
        rxn.gene_reaction_rule = rule
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


@pytest.mark.parametrize(
    "pvalues",
    [
        [0.2, 0.4, 0.7, 0.3, 0.1, 1 + 2.220446049250313e-16],
        [math.inf, 0.2, 0.4, 0.7, -math.inf, 0.1],
        [0.2, 0.4, 0.7, 0.3, 0.1, -2.220446049250313e-16],
        [0.2, 0.4, 0.7, 0.3, 0.1, -math.inf],
    ],
)
def test_pvalues_outside_zero_one_raise(three_gene_pair_model, pvalues):
    genes = [f"g{i}" for i in range(1, 7)]
    with pytest.raises(ValueError, match="between 0 and 1"):
        reporter_metabolites(three_gene_pair_model, dict(zip(genes, pvalues, strict=True)))


def test_pvalue_error_names_the_count_and_the_first_value(three_gene_pair_model):
    genes = [f"g{i}" for i in range(1, 7)]
    with pytest.raises(ValueError, match=r"2 of them.*1\.5 \(gene g2\)"):
        reporter_metabolites(three_gene_pair_model, dict(zip(genes, [0.2, 1.5, 0.7, 0.3, -0.1, 0.1], strict=True)))


def test_pvalue_outside_zero_one_raises_for_a_gene_outside_the_model(three_gene_pair_model):
    genes = [f"g{i}" for i in range(1, 7)] + ["gX"]
    with pytest.raises(ValueError, match="gene gX"):
        reporter_metabolites(three_gene_pair_model, dict(zip(genes, [0.2, 0.4, 0.7, 0.3, 0.1, 0.5, 2.0], strict=True)))


def test_pvalues_of_zero_and_one_are_valid_and_nan_is_ignored(three_gene_pair_model):
    genes = [f"g{i}" for i in range(1, 7)]
    (res,) = reporter_metabolites(three_gene_pair_model, dict(zip(genes, [0.0, 0.4, 0.7, 0.3, math.nan, 1.0], strict=True)))
    assert len(res.table) == 3
    assert res.table.set_index("metabolite").loc["C", "n_genes"] == 1


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


@pytest.mark.parametrize("n", [6, 7, 11, 22])
def test_identical_p_values_give_zero_scores_whatever_the_rounding(n):
    """With these pool sizes the mean of identical z-scores differs from them by one ulp, so
    the standard deviation is about 5e-17 instead of 0; the spread is tested instead."""
    m = cobra.Model("pool")
    a = _met("A_c")
    m.add_metabolites([a])
    for i in range(n):
        r = cobra.Reaction(f"r{i}")
        r.add_metabolites({a: -1})
        m.add_reactions([r])
        r.gene_reaction_rule = f"h{i}"
    (res,) = reporter_metabolites(m, {f"h{i}": 0.37 for i in range(n)})
    assert (res.table["z_score"] == 0.0).all()
    assert (res.table["p_value"] == 0.5).all()


def test_a_tiny_real_spread_is_kept(model):
    import numpy as np
    from scipy.stats import norm

    # z = 1e-10 * [1, 2, 3] is far above rounding level.
    p = norm.sf(1e-10 * np.array([1.0, 2.0, 3.0]))
    (res,) = reporter_metabolites(model, {"g1": p[0], "g2": p[1], "g3": p[2]})
    assert res.table["z_score"].abs().max() > 0.5


def test_spread_of_a_few_ulps_between_many_identical_zscores_is_resolved():
    """With 20000 genes at p = 0.37 and one shifted by about 1000 ulps, the scores match the
    exact moments of the float z-scores."""
    import math
    from fractions import Fraction

    import numpy as np
    from scipy.stats import norm

    n_genes = 20000
    shifted = 0.37
    for _ in range(1000):
        shifted = math.nextafter(shifted, 1.0)
    pvalues = {f"g{i}": 0.37 for i in range(n_genes)}
    pvalues["g0"] = shifted

    m = cobra.Model("big")
    mets = {k: _met(f"{k}_c") for k in "AB"}
    m.add_metabolites(list(mets.values()))
    for start in range(0, n_genes, 100):
        r = cobra.Reaction(f"bulk{start}")
        r.add_metabolites({cobra.Metabolite(f"M{start}_c", compartment="c"): -1})
        m.add_reactions([r])
        r.gene_reaction_rule = " or ".join(f"g{i}" for i in range(start, start + 100) if i >= 11)
    ra = cobra.Reaction("ra")
    ra.add_metabolites({mets["A"]: -1})
    rb = cobra.Reaction("rb")
    rb.add_metabolites({mets["B"]: -1})
    m.add_reactions([ra, rb])
    ra.gene_reaction_rule = " or ".join(f"g{i}" for i in range(1, 11))
    rb.gene_reaction_rule = "g0"

    z = {g: float(-norm.ppf(p)) for g, p in pvalues.items()}
    exact = {g: Fraction(v) for g, v in z.items()}
    mu = sum(exact.values()) / n_genes
    var = sum((v - mu) ** 2 for v in exact.values()) / n_genes
    sigma = math.sqrt(float(var))

    def expected(genes):
        n = len(genes)
        return float(sum(exact[g] for g in genes) - n * mu) / math.sqrt(n) / sigma

    (res,) = reporter_metabolites(m, pvalues)
    got = res.table.set_index("metabolite")["z_score"]
    assert got["A_c"] == pytest.approx(expected([f"g{i}" for i in range(1, 11)]), rel=1e-6)
    assert got["B_c"] == pytest.approx(expected(["g0"]), rel=1e-6)
    assert np.isfinite(got).all()


def test_scores_match_exact_moments_when_the_first_zscore_is_an_outlier():
    """With one extreme z-score first and the others within a few ulps of each other, the scores
    match the exact moments of the float z-scores."""
    import math
    from fractions import Fraction

    from scipy.stats import norm

    n_genes = 2000
    pvalues = {f"g{i}": 0.37 for i in range(n_genes)}
    pvalues["g0"] = 1e-12
    shifted = 0.37
    for _ in range(300):
        shifted = math.nextafter(shifted, 1.0)
    pvalues["g1"] = shifted

    m = cobra.Model("outlier")
    mets = {k: _met(f"{k}_c") for k in "ABC"}
    m.add_metabolites(list(mets.values()))
    rc = cobra.Reaction("rc")
    rc.add_metabolites({mets["C"]: -1})
    m.add_reactions([rc])
    rc.gene_reaction_rule = "g0"
    for start in range(0, n_genes, 100):
        r = cobra.Reaction(f"bulk{start}")
        r.add_metabolites({cobra.Metabolite(f"M{start}_c", compartment="c"): -1})
        m.add_reactions([r])
        r.gene_reaction_rule = " or ".join(f"g{i}" for i in range(start, start + 100) if i >= 12)
    ra = cobra.Reaction("ra")
    ra.add_metabolites({mets["A"]: -1})
    rb = cobra.Reaction("rb")
    rb.add_metabolites({mets["B"]: -1})
    m.add_reactions([ra, rb])
    ra.gene_reaction_rule = " or ".join(f"g{i}" for i in range(2, 12))
    rb.gene_reaction_rule = "g1"

    z = {g: float(-norm.ppf(p)) for g, p in pvalues.items()}
    exact = {g: Fraction(v) for g, v in z.items()}
    mu = sum(exact.values()) / n_genes
    var = sum((v - mu) ** 2 for v in exact.values()) / n_genes
    sigma = math.sqrt(float(var))

    def expected(genes):
        n = len(genes)
        return float(sum(exact[g] for g in genes) - n * mu) / math.sqrt(n) / sigma

    (res,) = reporter_metabolites(m, pvalues)
    got = res.table.set_index("metabolite")["z_score"]
    assert got["A_c"] == pytest.approx(expected([f"g{i}" for i in range(2, 12)]), rel=1e-13, abs=0)
    assert got["B_c"] == pytest.approx(expected(["g1"]), rel=1e-13, abs=0)


def test_scored_gene_without_a_fold_change_is_in_neither_subset(model):
    res = reporter_metabolites(
        model,
        {"g1": 0.001, "g2": 0.01, "g3": 0.01},
        gene_fold_changes={"g1": 2.0},
    )
    up = next(r for r in res if r.test == "up").table
    down = next(r for r in res if r.test == "down").table
    # g1 alone touches A and B; g2 and g3 (no fold change) do not enter either subset
    assert set(up["metabolite"]) == {"A_c", "B_c"}
    assert down.empty
