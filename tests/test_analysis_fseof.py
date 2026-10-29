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
