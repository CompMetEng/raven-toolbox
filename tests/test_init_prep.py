import cobra

from raven_toolbox.init.prep import prep_init_model, rescale_for_init


def _model():
    m = cobra.Model("m")
    for i in "ABC":
        m.add_metabolites([cobra.Metabolite(i)])
    r = cobra.Reaction("R1", lower_bound=-5.0, upper_bound=20.0)
    m.add_reactions([r])
    r.add_metabolites({m.metabolites.A: -100.0, m.metabolites.B: 1.0, m.metabolites.C: 2.0})
    return m


def test_rescale_leaves_bounds():
    m = _model()
    rescale_for_init(m)
    assert m.reactions.R1.bounds == (-5.0, 20.0)
    assert abs(sum(abs(c) for c in m.reactions.R1.metabolites.values()) - 3.0) < 1e-12


def test_prep_init_model_resets_bounds():
    m = cobra.Model("t")
    for i in "ABC":
        m.add_metabolites([cobra.Metabolite(i + "_c", compartment="c")])
    ex = cobra.Reaction("EX_A", lower_bound=0, upper_bound=5)
    r1 = cobra.Reaction("R1", lower_bound=0, upper_bound=50)
    r2 = cobra.Reaction("R2", lower_bound=0, upper_bound=50)
    m.add_reactions([ex, r1, r2])
    ex.add_metabolites({m.metabolites.A_c: 1})
    r1.add_metabolites({m.metabolites.A_c: -1, m.metabolites.B_c: 1})
    r2.add_metabolites({m.metabolites.B_c: -1, m.metabolites.C_c: 1})
    prep = prep_init_model(m, [], ext_comp="e", simplify=False, scale=True)
    for r in prep.min_model.reactions:
        assert r.upper_bound in (0.0, 1000.0)


def test_prep_init_model_resets_reversible_lower_bound():
    m = cobra.Model("t")
    for i in "ABCD":
        m.add_metabolites([cobra.Metabolite(i + "_c", compartment="c")])
    rxns = {
        "EX_A": (-5, 5, {"A_c": 1}),
        "R1": (-50, 50, {"A_c": -1, "B_c": 1}),
        "R2": (-50, 50, {"B_c": -1, "C_c": 1}),
        "R3": (-50, 50, {"B_c": -1, "D_c": 1}),
        "EX_C": (-10, 10, {"C_c": -1}),
        "EX_D": (-10, 10, {"D_c": -1}),
    }
    for rid, (lb, ub, stoich) in rxns.items():
        r = cobra.Reaction(rid, lower_bound=lb, upper_bound=ub)
        m.add_reactions([r])
        r.add_metabolites({m.metabolites.get_by_id(k): v for k, v in stoich.items()})
    prep = prep_init_model(m, [], ext_comp="e", simplify=False, scale=True)
    assert any(r.lower_bound < 0 for r in prep.min_model.reactions)
    for r in prep.min_model.reactions:
        assert r.lower_bound in (0.0, -1000.0)
        assert r.upper_bound in (0.0, 1000.0)


