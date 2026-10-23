"""The staged ftINIT pipeline (prep_init_model + get_init_steps + ftinit).

Oracles: RAVEN tinitTests T0001/T0002 on testModel with the default '1+1' schedule.
"""

from tinit_oracles import (
    TEST_MODEL_FTINIT_NO_TASKS,
    TEST_MODEL_FTINIT_SPONT_R7_R10,
    TEST_MODEL_FTINIT_WITH_TASK,
    TEST_MODEL_SCORES,
    TEST_MODEL_TASK_ESSENTIAL_MERGED,
    expr_for_rxn_score,
    make_test_model,
    make_test_task,
)

from raven_toolbox.init import (
    classify_reactions,
    ftinit,
    get_init_steps,
    prep_init_model,
    score_reactions_from_genes,
)
from raven_toolbox.init.score import gene_scores_from_expression


def _scores(model):
    return score_reactions_from_genes(
        model, gene_scores_from_expression(expr_for_rxn_score(TEST_MODEL_SCORES), 1.0)
    )


# --------------------------------------------------------------------------- #
# classify_reactions (the toIgnore masks) — tinitTests T0001 mask oracle.
# --------------------------------------------------------------------------- #
def test_classify_exchange_and_transport():
    masks = classify_reactions(make_test_model(), ext_comp="s")
    assert masks.exchange == {"R1", "R8"}        # boundary reactions
    assert masks.import_rxns == {"R2"}           # a[s] <=> a[c], no GPR, into ext comp
    assert masks.no_gpr == {"R1", "R2", "R8"}
    assert "R7" not in masks.import_rxns         # R7 has a GPR -> not a transport category


def test_classify_spontaneous():
    masks = classify_reactions(make_test_model(), ext_comp="s", spontaneous=["R7", "R10"])
    assert masks.exchange | masks.spontaneous == {"R1", "R7", "R8", "R10"}


def test_get_init_steps_default():
    steps = get_init_steps("1+1")
    assert len(steps) == 2
    assert steps[0].how_to_use_prev == "ignore"
    assert steps[0].ignore_mask == (1, 1, 1, 1, 1, 1, 1, 0)
    assert steps[1].how_to_use_prev == "essential"
    assert steps[1].ignore_mask == (1, 0, 0, 0, 1, 0, 0, 0)
    full = get_init_steps("full")
    assert len(full) == 1
    assert full[0].milp_runs == ({"mip_gap": 0.0004, "time_limit": 10000, "abs_gap": 10.0},)


def test_get_init_steps_full_ignores_no_metabolites():
    assert get_init_steps("full", mets_to_ignore=["H2O"])[0].mets_to_ignore == ()
    assert get_init_steps("1+1", mets_to_ignore=["H2O"])[0].mets_to_ignore == ("H2O",)


# --------------------------------------------------------------------------- #
# Full '1+1' pipeline — T0001 (no tasks) and T0002 (with task).
# --------------------------------------------------------------------------- #
def test_ftinit_no_tasks_matches_oracle():
    """T0001: testModel, no tasks, '1+1' → {R1,R4,R6,R8,R9,R10}."""
    model = make_test_model()
    prep = prep_init_model(model, ext_comp="s")
    out = ftinit(prep, rxn_scores=_scores(model))
    assert {r.id for r in out.reactions} == set(TEST_MODEL_FTINIT_NO_TASKS)


def test_ftinit_with_spontaneous_matches_oracle():
    """T0001 variant: R7,R10 spontaneous → the path through R2/R7, {R1,R2,R4,R6,R7,R8}."""
    model = make_test_model()
    prep = prep_init_model(model, ext_comp="s", spontaneous=["R7", "R10"])
    out = ftinit(prep, rxn_scores=_scores(model))
    assert {r.id for r in out.reactions} == set(TEST_MODEL_FTINIT_SPONT_R7_R10)


def test_ftinit_with_task_matches_oracle():
    """T0002: task 'make e[s] from a[s]' makes R2,R7 essential → {R1,R2,R4,R6,R7,R8,R9,R10}."""
    model = make_test_model()
    prep = prep_init_model(model, [make_test_task()], ext_comp="s")
    # Essentials map to merged ids {R1, R7} (RAVEN T0002).
    assert prep.essential_rxns == set(TEST_MODEL_TASK_ESSENTIAL_MERGED)
    out = ftinit(prep, rxn_scores=_scores(model))
    assert {r.id for r in out.reactions} == set(TEST_MODEL_FTINIT_WITH_TASK)


def test_full_series_runs():
    """The single-step 'full' series also produces a feasible subnetwork."""
    model = make_test_model()
    prep = prep_init_model(model, ext_comp="s")
    out = ftinit(prep, rxn_scores=_scores(model), series="full")
    assert len(out.reactions) >= 1


def test_pipeline_with_gene_scores_and_tasks_wires_up():
    """ftinit accepts gene_scores (gene pruning) + tasks (gap-fill) without breaking T0002.

    The toy's GPRs are single-gene (nothing to prune) and the task is feasible in the
    extracted model (nothing to gap-fill), so the reaction set is unchanged — this
    confirms the integration wiring (the pruning/gap-fill logic is unit-tested
    separately in test_init_genes / test_init_taskfill).
    """
    model = make_test_model()
    gene_scores = gene_scores_from_expression(expr_for_rxn_score(TEST_MODEL_SCORES), 1.0)
    prep = prep_init_model(model, [make_test_task()], ext_comp="s")
    out = ftinit(prep, rxn_scores=_scores(model), gene_scores=gene_scores)
    assert {r.id for r in out.reactions} == set(TEST_MODEL_FTINIT_WITH_TASK)


def test_orient_forward_reverses_a_reversible_reaction():
    """_orient_forward(rxn, -1) flips stoichiometry and makes it irreversible forward."""
    import cobra

    from raven_toolbox.init.prep import _orient_forward

    m = cobra.Model("o")
    a, b = (cobra.Metabolite(x, compartment="s") for x in "ab")
    m.add_metabolites([a, b])
    r = cobra.Reaction("R", lower_bound=-800, upper_bound=1000)
    r.add_metabolites({a: -1, b: 2})  # a <=> 2 b
    m.add_reactions([r])

    _orient_forward(r, -1)  # forced reverse → becomes forward
    assert r.bounds == (0, 800)  # [-800,1000] → flip [-1000,800] → lb→0
    assert {mt.id: c for mt, c in r.metabolites.items()} == {"a": 1, "b": -2}  # 2 b => a

    fwd = cobra.Reaction("F", lower_bound=-500, upper_bound=900)
    fwd.add_metabolites({a: -1})
    m.add_reactions([fwd])
    _orient_forward(fwd, 1)  # forced forward → just made irreversible
    assert fwd.bounds == (0, 900)


def test_essential_merged_away_is_skipped():
    """An essential reaction whose merge group collapses away imposes no constraint.

    REV sits between two exchanges, so it merges with them into a trivial source→sink
    that is removed; its group has no survivor. prep_init_model must skip it, not crash.
    """
    import cobra

    from raven_toolbox.tasks import Task

    m = cobra.Model("collapse")
    a, b = (cobra.Metabolite(x, name=x, compartment="s") for x in "ab")
    m.add_metabolites([a, b])
    r = cobra.Reaction("REV", lower_bound=-1000, upper_bound=1000)
    r.add_metabolites({a: -1, b: 1})
    r.gene_reaction_rule = "g1"
    exchanges = []
    for met in (a, b):
        ex = cobra.Reaction(f"EX_{met.id}", lower_bound=-1000, upper_bound=1000)
        ex.add_metabolites({met: -1})
        exchanges.append(ex)
    m.add_reactions([r, *exchanges])
    m.objective = "REV"
    task = Task(id="mk_a", inputs=[("b[s]", 0.0, 1000.0)], outputs=[("a[s]", 1.0, 1.0)])

    prep = prep_init_model(m, [task], ext_comp="s")  # must not raise
    assert "REV" not in prep.essential_rxns  # merged into a collapsed group


def test_prep_closes_boundaries_for_task_essentials():
    """prep_init_model must close boundaries for essential discovery, as RAVEN does.

    RAVEN's ``prepINITModel`` runs ``checkTasks`` on ``closeModel(cModel)``
    (prepINITModel.m:81-82), and ``checkTasks`` zeroes every metabolite's balance
    (checkTasks.m:63) before re-opening only what a task declares — so for this step the
    task file *is* the whole boundary. ``find_task_essential_reactions``' own default is
    additive (right for a standalone check on a model with no boundary metabolites), and
    prep must not inherit it: on Human-GEM, whose exchanges all ship open, that collapsed
    the task-essential set from ~206 reactions to 1 and silently removed the task
    constraint from every extraction.

    The task is "make b from a". CONV (a→b) is the only route from the task's declared
    input, but the model also ships an open exchange for c, which ALT (c→b) turns into b
    without touching a. Additively that un-declared exchange satisfies the task, so neither
    route is essential; with the boundary closed, only a is available and CONV becomes
    essential. The toy oracles elsewhere in this file agree under both readings, which is
    why this regression went unnoticed.
    """
    import cobra

    from raven_toolbox.tasks import Task, find_task_essential_reactions

    m = cobra.Model("bypass")
    a, b, c = (cobra.Metabolite(x, name=x, compartment="s") for x in "abc")
    m.add_metabolites([a, b, c])

    conv = cobra.Reaction("CONV", lower_bound=0, upper_bound=1000)   # a -> b
    conv.add_metabolites({a: -1, b: 1})
    conv.gene_reaction_rule = "g1"
    alt = cobra.Reaction("ALT", lower_bound=0, upper_bound=1000)     # c -> b, the bypass
    alt.add_metabolites({c: -1, b: 1})
    alt.gene_reaction_rule = "g2"
    exchanges = []
    for met in (a, b, c):
        ex = cobra.Reaction(f"EX_{met.id}", lower_bound=-1000, upper_bound=1000)
        ex.add_metabolites({met: -1})
        exchanges.append(ex)
    m.add_reactions([conv, alt, *exchanges])

    task = Task(id="mk_b", inputs=[("a[s]", 0.0, 1000.0)], outputs=[("b[s]", 1.0, 1.0)])

    # The discriminating fact, at the level the bug actually lives (no merging involved).
    additive = find_task_essential_reactions(m, [task], close_boundaries=False).reactions
    closed = find_task_essential_reactions(m, [task], close_boundaries=True).reactions
    assert "CONV" not in additive, "the un-declared EX_c bypass hides CONV additively"
    assert "CONV" in closed, "with the boundary closed, CONV is the only route from a"

    # prep must take the closed reading. Essentials are reported on merged ids (CONV can
    # merge with EX_a into one linear group), so assert the set is non-empty rather than
    # naming CONV itself — under the additive reading it would be empty.
    prep = prep_init_model(m, [task], ext_comp="s", simplify=False)
    assert prep.essential_rxns, (
        "prep_init_model found no task-essential reactions; it is using the additive "
        "boundary reading instead of RAVEN's prepINITModel closed one"
    )


# --------------------------------------------------------------------------- #
# prove_abs_gap / resolve_ties (deterministic extraction).
# These are opt-in; the default path stays exact-RAVEN. On the toy oracle they must
# reproduce T0001 (no regression) and be run-to-run identical.
# --------------------------------------------------------------------------- #
def test_ftinit_prove_abs_gap_matches_oracle():
    """prove_abs_gap: one near-proven-optimal solve per step, still gives T0001."""
    model = make_test_model()
    prep = prep_init_model(model, ext_comp="s")
    out = ftinit(prep, rxn_scores=_scores(model), prove_abs_gap=0.05)
    assert {r.id for r in out.reactions} == set(TEST_MODEL_FTINIT_NO_TASKS)


def test_ftinit_resolve_ties_matches_oracle_and_is_stable():
    """resolve_ties (+ prove_abs_gap): preserves T0001, identical across repeated runs."""
    model = make_test_model()
    prep = prep_init_model(model, ext_comp="s")
    out1 = ftinit(prep, rxn_scores=_scores(model), prove_abs_gap=0.05, resolve_ties=True)
    out2 = ftinit(prep, rxn_scores=_scores(model), prove_abs_gap=0.05, resolve_ties=True)
    assert {r.id for r in out1.reactions} == set(TEST_MODEL_FTINIT_NO_TASKS)
    assert {r.id for r in out1.reactions} == {r.id for r in out2.reactions}


# --------------------------------------------------------------------------- #
# metabolomics (production-bonus for detected metabolites) — full pipeline.
# Exercises what run_ftinit's own metabolomics tests cannot: metabolite *name*
# resolution against prep.ref_model and translation through the linear merge.
#
# Reuses testModel rather than a hand-rolled fixture: b[c]/c[c] are produced only by
# R3 (score -1, reversible) and consumed only by R5 (score 0.5, irreversible) -- degree
# exactly 2 each, so prep_init_model's linear merge combines R3+R5 into one reaction
# (their shared metabolites happen to cancel 1:1, collapsing to a plain ac -> ec link;
# verified directly, not assumed). ``b`` is genuinely produced by R3, a *non-survivor*
# member of that merge group whenever R5 is picked as the survivor id -- exactly the
# case _metabolomics_producers' "any member matches" translation exists for. Combined
# score -0.5: outside testModel's own score-optimal internal loop (R4/R6/R9/R10, worth
# 8.0 -- see test_init_ftinit.py), so R3/R5 are excluded from the baseline oracle
# (TEST_MODEL_FTINIT_NO_TASKS) with nothing to do with metabolomics.
# --------------------------------------------------------------------------- #
def _merged_producer_prep():
    model = make_test_model()
    prep = prep_init_model(model, ext_comp="s")
    assert prep.group_of["R3"] == prep.group_of["R5"] != 0  # merge actually happened
    return prep, model


def test_metabolomics_name_resolves_through_the_merge():
    """A detected metabolite produced by a non-survivor merge-group member is still
    correctly resolved and pulls the (badly-scored, merged) reaction in."""
    prep, model = _merged_producer_prep()
    scores = _scores(model)

    baseline = ftinit(prep, rxn_scores=scores, fill_gaps=False)
    assert not ({"R3", "R5"} & {r.id for r in baseline.reactions})
    assert {r.id for r in baseline.reactions} == set(TEST_MODEL_FTINIT_NO_TASKS)

    boosted = ftinit(prep, rxn_scores=scores, fill_gaps=False, metabolomics=["b"], prod_weight=5.0)
    kept_ids = {r.id for r in boosted.reactions}
    assert {"R3", "R5"} <= kept_ids  # merge group survives whole (all-or-nothing)


def test_metabolomics_unknown_name_warns():
    """A name matching no metabolite (typo, or a name from an unrelated model) warns."""
    import pytest

    prep, model = _merged_producer_prep()
    with pytest.warns(UserWarning, match="matched no metabolite"):
        ftinit(prep, rxn_scores=_scores(model), fill_gaps=False,
              metabolomics=["not_a_real_metabolite"])


def test_prep_keeps_the_template_bounds_and_directions_on_ref_model():
    """Tightening reversibility and orienting essentials only affect the merged model; the
    reference model, from which ftinit extracts the output, keeps the template's bounds,
    stoichiometry and objective coefficients."""
    import cobra

    from raven_toolbox.tasks import Task

    m = cobra.Model("keep")
    a, b, c = (cobra.Metabolite(x, name=x, compartment="s") for x in "abc")
    m.add_metabolites([a, b, c])
    ex_a = cobra.Reaction("EX_a", lower_bound=-1000, upper_bound=1000)
    ex_a.add_metabolites({a: -1})
    ex_c = cobra.Reaction("EX_c", lower_bound=0, upper_bound=1000)
    ex_c.add_metabolites({c: -1})
    r1 = cobra.Reaction("R1", lower_bound=-1000, upper_bound=1000)
    r1.add_metabolites({a: -1, b: 1})
    r1.gene_reaction_rule = "g1"
    r2 = cobra.Reaction("R2", lower_bound=-1000, upper_bound=1000)
    r2.add_metabolites({b: -1, c: 1})
    r2.gene_reaction_rule = "g2"
    m.add_reactions([ex_a, ex_c, r1, r2])
    task = Task(id="mk_c", inputs=[("a[s]", 0.0, 1000.0)], outputs=[("c[s]", 1.0, 1000.0)])

    def state(model):
        return {
            r.id: (r.bounds, {mt.id: co for mt, co in r.metabolites.items()}, r.objective_coefficient)
            for r in model.reactions
        }

    template = state(m)
    prep = prep_init_model(m, [task], ext_comp="s")
    assert state(prep.ref_model) == template
    assert state(m) == template  # the template itself is not modified


def _bounds_and_stoichiometry(model):
    return {
        r.id: (r.bounds, {mt.id: co for mt, co in r.metabolites.items()}, r.objective_coefficient)
        for r in model.reactions
    }


def test_prep_restores_an_essential_reaction_that_was_flipped_to_forward():
    """A task forces R1 in its written reverse direction; the orientation is undone on ref_model."""
    import cobra

    from raven_toolbox.tasks import Task, find_task_essential_reactions

    m = cobra.Model("flip")
    a, b, c = (cobra.Metabolite(x, name=x, compartment="s") for x in "abc")
    m.add_metabolites([a, b, c])
    ex_a = cobra.Reaction("EX_a", lower_bound=-1000, upper_bound=1000)
    ex_a.add_metabolites({a: -1})
    ex_c = cobra.Reaction("EX_c", lower_bound=0, upper_bound=1000)
    ex_c.add_metabolites({c: -1})
    r1 = cobra.Reaction("R1", lower_bound=-1000, upper_bound=1000)
    r1.add_metabolites({a: 1, b: -1})  # b <=> a, but the task needs a -> b
    r1.gene_reaction_rule = "g1"
    r2 = cobra.Reaction("R2", lower_bound=-1000, upper_bound=1000)
    r2.add_metabolites({b: -1, c: 1})
    r2.gene_reaction_rule = "g2"
    m.add_reactions([ex_a, ex_c, r1, r2])
    task = Task(id="mk_c", inputs=[("a[s]", 0.0, 1000.0)], outputs=[("c[s]", 1.0, 1000.0)])

    essential = find_task_essential_reactions(m.copy(), [task], close_boundaries=True).reactions
    assert essential["R1"] == -1  # precondition: the forced direction is the reverse one

    template = _bounds_and_stoichiometry(m)
    prep = prep_init_model(m, [task], ext_comp="s")
    assert _bounds_and_stoichiometry(prep.ref_model) == template


def test_prep_restores_a_reversible_reaction_that_can_only_run_backwards():
    """constrain_reversible flips a reversible reaction whose only feasible flux is
    negative; the template's bounds and stoichiometry come back on ref_model."""
    import cobra

    from raven_toolbox.manipulation import simplify_model

    m = cobra.Model("backwards")
    a, b = (cobra.Metabolite(x, name=x, compartment="s") for x in "ab")
    m.add_metabolites([a, b])
    ex_b = cobra.Reaction("EX_b", lower_bound=0, upper_bound=1000)
    ex_b.add_metabolites({b: 1})
    ex_a = cobra.Reaction("EX_a", lower_bound=0, upper_bound=1000)
    ex_a.add_metabolites({a: -1})
    r = cobra.Reaction("R", lower_bound=-1000, upper_bound=1000)
    r.add_metabolites({a: -1, b: 1})  # a <=> b, but only b -> a is feasible
    r.gene_reaction_rule = "g1"
    m.add_reactions([ex_b, ex_a, r])

    tightened = m.copy()
    simplify_model(tightened, constrain_reversible=True)
    assert tightened.reactions.R.lower_bound >= 0  # precondition: the pass changes R

    template = _bounds_and_stoichiometry(m)
    prep = prep_init_model(m, None, ext_comp="s")
    assert _bounds_and_stoichiometry(prep.ref_model) == template
