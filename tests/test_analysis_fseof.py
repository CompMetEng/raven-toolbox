"""Tests for FSEOF (analysis/fseof.py)."""
import cobra
import pytest

from raven_toolbox.analysis import FSEOFResult, fseof


@pytest.fixture
def model():
    """S -> I, then I branches to product P (via v2) or biomass B (via v3).

    Enforcing product export (EX_P) should amplify the product branch (v1, v2) and
    suppress the biomass branch (v3), which competes for the shared intermediate I.
    """
    m = cobra.Model("cell")
    S, inter, P, B = (cobra.Metabolite(x, compartment="c") for x in ("S", "I", "P", "B"))
    m.add_metabolites([S, inter, P, B])
    sup = cobra.Reaction("sup", lower_bound=0, upper_bound=10)  # -> S (substrate supply)
    sup.add_metabolites({S: 1})
    v1 = cobra.Reaction("v1", lower_bound=0, upper_bound=1000)
    v1.add_metabolites({S: -1, inter: 1})
    v2 = cobra.Reaction("v2", lower_bound=0, upper_bound=1000)
    v2.add_metabolites({inter: -1, P: 1})
    v3 = cobra.Reaction("v3", lower_bound=0, upper_bound=1000)
    v3.add_metabolites({inter: -1, B: 1})
    ex_p = cobra.Reaction("EX_P", lower_bound=0, upper_bound=1000)  # target product export
    ex_p.add_metabolites({P: -1})
    ex_b = cobra.Reaction("EX_B", lower_bound=0, upper_bound=1000)  # biomass
    ex_b.add_metabolites({B: -1})
    m.add_reactions([sup, v1, v2, v3, ex_p, ex_b])
    v1.gene_reaction_rule = "gA"
    v2.gene_reaction_rule = "gB"
    v3.gene_reaction_rule = "gC"
    m.objective = "EX_B"
    return m


def test_returns_result_with_scan(model):
    res = fseof(model, "EX_P", n_steps=8)
    assert isinstance(res, FSEOFResult)
    assert res.scan.shape[1] == len(res.enforced) >= 2
    assert "v2" in res.scan.index  # full scan retained, indexed by reaction


def test_amplification_targets(model):
    res = fseof(model, "EX_P", n_steps=8)
    amp = set(res.amplification["reaction"])
    # the product-forming reaction is amplified as EX_P is enforced upward
    # (v1/sup run at capacity throughout, so they stay constant and aren't flagged).
    assert {"v2", "EX_P"} <= amp
    v2 = res.targets.set_index("reaction").loc["v2"]
    assert v2["slope"] > 0 and v2["correlation"] > 0.9


def test_knockdown_of_competing_branch(model):
    res = fseof(model, "EX_P", n_steps=8)
    # v3 (biomass branch) competes for I -> suppressed toward zero -> knockdown/knockout
    down = set(res.knockout["reaction"])
    assert "v3" in down
    v3 = res.targets.set_index("reaction").loc["v3"]
    assert v3["slope"] < 0
    assert v3["target_type"] in ("knockdown", "knockout")


def test_gene_targets_aggregation(model):
    res = fseof(model, "EX_P", n_steps=8)
    genes = set(res.gene_targets["gene"])
    assert {"gA", "gB", "gC"} & genes  # reaction targets mapped to their genes
    gB = res.gene_targets.set_index("gene").loc["gB"]
    assert "v2" in gB["reactions"]


def test_min_target_overrides_default_floor(model):
    """``min_target`` replaces the default ``target_max / n_steps`` floor,
    letting a caller anchor the scan's low end elsewhere (e.g. at a
    growth-optimal target flux) while the ceiling stays the same."""
    default_res = fseof(model, "EX_P", n_steps=4)
    custom_res = fseof(model, "EX_P", n_steps=4, min_target=0.0)
    assert custom_res.enforced[0] == pytest.approx(0.0)
    assert custom_res.enforced[0] < default_res.enforced[0]
    assert custom_res.enforced[-1] == pytest.approx(default_res.enforced[-1])


def test_unproducible_target_raises(model):
    # A reaction that cannot carry positive flux is not a valid product target.
    dead = cobra.Reaction("dead", lower_bound=0, upper_bound=0)
    dead.add_metabolites({model.metabolites.P: -1})
    model.add_reactions([dead])
    with pytest.raises(ValueError, match="cannot carry positive flux"):
        fseof(model, "dead")


def test_infeasible_model_raises_clear_error(model):
    """An infeasible model (slim_optimize -> NaN) raises the clear guard, not a NaN scan."""
    model.reactions.sup.bounds = (5, 5)  # force uptake while EX_P demands more -> infeasible
    model.reactions.EX_P.bounds = (1000, 1000)
    with pytest.raises(ValueError, match="cannot carry positive flux"):
        fseof(model, "EX_P", n_steps=4)


# --- regression: slope-based labels -----------------------------------

def test_amplify_label_uses_abs_slope_not_endpoint_difference():
    """A reaction whose |flux| trend is upward but whose final value equals
    the initial value (endpoints straddle a peak) must be labelled
    ``amplify`` by the regression-slope rule -- comparing endpoints alone
    would misclassify it as ``knockdown``."""
    import numpy as np
    import pandas as pd

    from raven_toolbox.analysis.fseof import _classify

    # Endpoints equal (0), but the |flux| regression slope is clearly positive
    # over the scan, so this must classify as amplify -- comparing final vs
    # initial value alone would misclassify it as knockdown.
    enforced = np.linspace(0.0, 1.0, 6)
    flux = np.array([0.0, 0.3, 0.6, 0.9, 0.4, 0.0])
    scan = pd.DataFrame([flux], index=["r_test"], columns=enforced)
    m = cobra.Model("synth")
    m.add_reactions([cobra.Reaction("r_test")])
    table = _classify(m, scan, enforced, corr_threshold=0.0)
    assert not table.empty
    assert table.iloc[0]["target_type"] == "amplify"


def test_slope_is_taken_on_absolute_flux():
    """A reversible reaction whose negative flux grows in magnitude has a positive slope."""
    import numpy as np
    import pandas as pd

    from raven_toolbox.analysis.fseof import _classify

    enforced = np.linspace(1.0, 5.0, 5)
    scan = pd.DataFrame([-enforced, enforced], index=["r_neg", "r_pos"], columns=enforced)
    m = cobra.Model("synth")
    m.add_reactions([cobra.Reaction("r_neg"), cobra.Reaction("r_pos")])
    table = _classify(m, scan, enforced, corr_threshold=0.9).set_index("reaction")
    assert table.loc["r_neg", "slope"] == pytest.approx(1.0)
    assert table.loc["r_neg", "target_type"] == "amplify"
    assert table.loc["r_pos", "slope"] == pytest.approx(1.0)


def test_low_flux_track_is_reported_as_in_fseof():
    """A track with fluxes around 1e-7 is an amplify target, as in FSEOF."""
    import numpy as np
    import pandas as pd

    from raven_toolbox.analysis.fseof import _classify

    enforced = np.linspace(1.0, 5.0, 5)
    scan = pd.DataFrame([1e-7 * enforced], index=["r_low"], columns=enforced)
    m = cobra.Model("synth")
    m.add_reactions([cobra.Reaction("r_low")])
    table = _classify(m, scan, enforced, corr_threshold=0.9)
    assert table.loc[0, "target_type"] == "amplify"


def test_knockout_threshold_is_1e_minus_8():
    """A last absolute flux of 5e-7 is a knockdown; one of 5e-9 is a knockout."""
    import numpy as np
    import pandas as pd

    from raven_toolbox.analysis.fseof import _classify

    enforced = np.linspace(1.0, 5.0, 5)
    scan = pd.DataFrame(
        [[4, 3, 2, 1, 5e-7], [4, 3, 2, 1, 5e-9]], index=["r_down", "r_out"], columns=enforced
    )
    m = cobra.Model("synth")
    m.add_reactions([cobra.Reaction("r_down"), cobra.Reaction("r_out")])
    table = _classify(m, scan, enforced, corr_threshold=0.5).set_index("reaction")
    assert table.loc["r_down", "target_type"] == "knockdown"
    assert table.loc["r_out", "target_type"] == "knockout"


def test_pfba_holds_objective_at_matlab_fraction(model, monkeypatch):
    # RAVEN's solveLP minimises the flux sum with the objective fixed at 0.999999 of its optimum,
    # so a solver whose optimum overshoots by its tolerance still has a feasible second step.
    import importlib

    module = importlib.import_module("raven_toolbox.analysis.fseof")

    fractions = []
    real = module.pfba

    def spy(m, **kwargs):
        fractions.append(kwargs.get("fraction_of_optimum"))
        return real(m, **kwargs)

    monkeypatch.setattr(module, "pfba", spy)
    res = fseof(model, "EX_P", n_steps=4)
    assert len(fractions) >= len(res.enforced) >= 2
    assert set(fractions) == {0.999999}


def test_level_with_infeasible_maximisation_ends_the_scan(model):
    # The theoretical maximum of EX_P is 10; levels above it cannot be enforced and are not
    # reported, however the second pFBA step is tolerated.
    res = fseof(model, "EX_P", n_steps=4, max_fraction=2.0)
    assert res.enforced == pytest.approx([5.0, 10.0])


def test_equality_fixes_the_target_flux_at_each_level(model):
    res = fseof(model, "EX_P", n_steps=4, equality=True)
    assert res.scan.loc["EX_P"].to_numpy() == pytest.approx(res.enforced)
    # the default lower bound lets a level exceed the enforced flux when that suits growth;
    # here the product branch is the only sink of the shared intermediate at the top level.
    lower = fseof(model, "EX_P", n_steps=4)
    assert (lower.scan.loc["EX_P"].to_numpy() >= lower.enforced).all()


def test_equality_differs_from_lower_bound_when_the_target_is_coupled_to_growth():
    # S -> B + P is the only route to biomass, so growth alone exports 10 of P; a lower bound
    # below that leaves the flux at 10, while an equality holds it at the level.
    m = cobra.Model("m")
    S, P, B = (cobra.Metabolite(x, compartment="c") for x in ("S", "P", "B"))
    m.add_metabolites([S, P, B])
    sup = cobra.Reaction("sup", lower_bound=0, upper_bound=10)
    sup.add_metabolites({S: 1})
    v1 = cobra.Reaction("v1", lower_bound=0, upper_bound=1000)
    v1.add_metabolites({S: -1, B: 1, P: 1})
    ex_p = cobra.Reaction("EX_P", lower_bound=0, upper_bound=1000)
    ex_p.add_metabolites({P: -1})
    ex_b = cobra.Reaction("EX_B", lower_bound=0, upper_bound=1000)
    ex_b.add_metabolites({B: -1})
    m.add_reactions([sup, v1, ex_p, ex_b])
    m.objective = "EX_B"
    lower = fseof(m, "EX_P", n_steps=4, min_target=2.0)
    fixed = fseof(m, "EX_P", n_steps=4, min_target=2.0, equality=True)
    assert lower.scan.loc["EX_P"].to_numpy() == pytest.approx(10.0)
    assert fixed.scan.loc["EX_P"].to_numpy() == pytest.approx(fixed.enforced)


def test_top_level_is_a_fraction_of_the_pfba_maximum(model):
    # pFBA holds the target at 0.999999 of its maximum while it minimises the flux sum,
    # so the top level is 0.9 of that flux and not of the plain maximum.
    from cobra.flux_analysis import pfba

    with model:
        model.objective = "EX_P"
        plain = model.slim_optimize()
        in_pfba = pfba(model, fraction_of_optimum=0.999999).fluxes["EX_P"]
    assert in_pfba < plain
    res = fseof(model, "EX_P", n_steps=4)
    assert res.enforced[-1] == pytest.approx(0.9 * in_pfba, rel=1e-8)
    assert res.enforced[-1] != pytest.approx(0.9 * plain, rel=1e-8)


def test_equal_scores_keep_model_order():
    # Reactions of three score classes are interleaved in the model; within a class the
    # scores tie exactly, and the table lists each class in model order, highest score first.
    import numpy as np
    import pandas as pd

    from raven_toolbox.analysis.fseof import _classify

    m = cobra.Model("m")
    ids = [f"r{i:03d}" for i in range(600)]
    m.add_reactions([cobra.Reaction(rid) for rid in ids])
    enforced = np.array([1.0, 2.0, 3.0, 4.0])
    scan = pd.DataFrame([enforced * (1 + i % 3) for i in range(600)], index=ids, columns=enforced)
    table = _classify(m, scan, enforced, 0.9)
    expected = [ids[i] for c in (2, 1, 0) for i in range(600) if i % 3 == c]
    assert list(table["reaction"]) == expected
