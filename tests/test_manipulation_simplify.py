"""Tests for simplifyModel reduction modes."""
import cobra
import pytest

from raven_toolbox.manipulation import (
    add_reactions_from_equations,
    constrain_reversible_reactions,
    group_linear_reactions,
    remove_dead_end_reactions,
    remove_duplicate_reactions,
    remove_no_flux_reactions,
    remove_zero_interval_reactions,
    simplify_model,
)

# --- remove_dead_end_reactions --------------------------------------------

def test_dead_end_removed():
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in ("a", "b", "dead")])
    add_reactions_from_equations(
        m,
        [
            {"id": "R_in", "equation": " --> a"},
            {"id": "R1", "equation": "a --> b"},
            {"id": "R_out", "equation": "b --> "},
            {"id": "R_dead", "equation": "a --> dead"},  # 'dead' only produced
        ],
    )
    removed_rxns, removed_mets = remove_dead_end_reactions(m)
    assert "R_dead" in removed_rxns
    assert "dead" in removed_mets
    # the productive path survives
    assert {"R_in", "R1", "R_out"} <= {r.id for r in m.reactions}


def test_dead_end_respects_reserved():
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in ("a", "dead")])
    add_reactions_from_equations(
        m, [{"id": "R_in", "equation": " --> a"}, {"id": "R_dead", "equation": "a --> dead"}]
    )
    removed_rxns, _ = remove_dead_end_reactions(m, reserved=["R_dead"])
    assert "R_dead" not in removed_rxns
    assert "R_dead" in {r.id for r in m.reactions}


def test_dead_end_ignores_bounds_on_its_own_matching_raven():
    """raven-gecko-parity#69: run alone, RAVEN's deleteInaccessible reads dead ends
    off the sign pattern of S only -- a reaction locked at zero flux (lb == ub == 0)
    still counts as a structural producer/consumer, exactly as if unconstrained.
    Detecting reactions that are *actually* blocked by their bounds is
    remove_no_flux_reactions's job (RAVEN deleteMinMax, via FVA); nothing should be
    removed here on bounds grounds alone.
    """
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in ("a", "b", "c")])
    add_reactions_from_equations(
        m,
        [
            {"id": "R_in", "equation": " --> a"},
            {"id": "R1", "equation": "a --> b"},
            # Locked at zero flux -- still a structural producer of c / consumer of
            # b, per RAVEN's bounds-blind sign-pattern check.
            {"id": "R_locked", "equation": "b --> c", "bounds": (0.0, 0.0)},
            {"id": "R_out", "equation": "c --> "},
        ],
    )
    removed_rxns, removed_mets = remove_dead_end_reactions(m)
    assert removed_rxns == []
    assert removed_mets == []
    assert {"R_in", "R1", "R_locked", "R_out"} == {r.id for r in m.reactions}


# --- remove_duplicate_reactions -------------------------------------------

def _two_rxn_model(eq1, b1, eq2, b2, c2=0.0):
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, name=x, compartment="c") for x in ("a", "b")])
    add_reactions_from_equations(
        m,
        [
            {"id": "R1", "equation": eq1, "bounds": b1},
            {"id": "R2", "equation": eq2, "bounds": b2},
        ],
    )
    m.reactions.R2.objective_coefficient = c2
    return m


def test_duplicates_removed():
    m = _two_rxn_model("a --> b", (0, 1000), "a --> b", (0, 1000))
    assert remove_duplicate_reactions(m) == ["R2"]
    assert [r.id for r in m.reactions] == ["R1"]


def test_duplicates_with_different_bounds_merge_to_the_widest():
    m = _two_rxn_model("a <=> b", (-5, 10), "a <=> b", (-10, 20), c2=1.0)
    with pytest.warns(UserWarning) as rec:
        removed = remove_duplicate_reactions(m)
    assert removed == ["R2"]
    assert m.reactions.R1.bounds == (-10, 20)
    assert m.reactions.R1.objective_coefficient == 1.0
    assert len(rec) == 3


def test_duplicates_differing_in_reversibility_are_kept():
    m = _two_rxn_model("a --> b", (0, 10), "a <=> b", (-10, 10))
    assert remove_duplicate_reactions(m) == []


def test_reversible_duplicates_in_opposite_directions_merge():
    m = _two_rxn_model("a <=> b", (-10, 10), "b <=> a", (-10, 10))
    assert remove_duplicate_reactions(m) == ["R2"]


def test_irreversible_opposite_directions_are_kept():
    m = _two_rxn_model("a --> b", (0, 10), "b --> a", (0, 10))
    assert remove_duplicate_reactions(m) == []


def test_duplicates_ignore_reserved_like_contract_model():
    """Reserved reactions are merged away; the survivor takes the widest bounds and all GPRs."""
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in ("a", "b")])
    add_reactions_from_equations(
        m,
        [
            {"id": "R1", "equation": "a --> b", "bounds": (0, 10), "gene_reaction_rule": "g1"},
            {"id": "R2", "equation": "a --> b", "bounds": (0, 1000), "gene_reaction_rule": "g2"},
            {"id": "R3", "equation": "a --> b", "bounds": (0, 10), "gene_reaction_rule": "g3"},
        ],
    )
    with pytest.warns(UserWarning, match="upper bounds"):
        simplify_model(m, delete_duplicates=True, reserved=["R2"])
    assert [r.id for r in m.reactions] == ["R1"]
    assert m.reactions.R1.bounds == (0, 1000)
    assert set(m.reactions.R1.gene_reaction_rule.split(" or ")) == {"g1", "g2", "g3"}


# --- constrain_reversible_reactions ---------------------------------------

def test_forward_only_reversible_constrained():
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in ("a", "b")])
    add_reactions_from_equations(
        m,
        [
            {"id": "R_in", "equation": " --> a", "bounds": (0, 1000)},
            {"id": "R1", "equation": "a <=> b", "bounds": (-1000, 1000)},  # can only go fwd
            {"id": "R_out", "equation": "b --> ", "bounds": (0, 1000)},
        ],
    )
    changed = constrain_reversible_reactions(m)
    assert "R1" in changed
    assert m.reactions.get_by_id("R1").lower_bound == 0  # constrained to forward


def test_truly_reversible_unchanged():
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in ("a", "b")])
    add_reactions_from_equations(
        m,
        [
            {"id": "R_in", "equation": " <=> a", "bounds": (-1000, 1000)},
            {"id": "R1", "equation": "a <=> b", "bounds": (-1000, 1000)},
            {"id": "R_out", "equation": "b <=> ", "bounds": (-1000, 1000)},
        ],
    )
    changed = constrain_reversible_reactions(m)
    assert "R1" not in changed  # can go both ways


# --- group_linear_reactions -----------------------------------------------

def test_linear_chain_merged():
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in ("a", "b", "c")])
    add_reactions_from_equations(
        m,
        [
            {"id": "R1", "equation": "a --> b"},  # b: single producer
            {"id": "R2", "equation": "b --> c"},  # b: single consumer
        ],
    )
    n_before = len(m.reactions)
    group_linear_reactions(m)
    # b is eliminated; R1+R2 merged into one reaction a --> c
    assert "b" not in m.metabolites
    assert len(m.reactions) < n_before


def test_group_linear_discards_genes():
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in ("a", "b", "c")])
    add_reactions_from_equations(
        m,
        [
            {"id": "R1", "equation": "a --> b", "gene_reaction_rule": "G1"},
            {"id": "R2", "equation": "b --> c", "gene_reaction_rule": "G2"},
        ],
    )
    group_linear_reactions(m)
    assert len(m.genes) == 0


# --- regression: incremental merge collapses a long chain -----------------

def test_group_linear_merges_long_chain_in_one_pass():
    """A 5-reaction linear chain collapses to a single reaction in one pass."""
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in "abcdef"])
    add_reactions_from_equations(
        m,
        [
            {"id": "R_in", "equation": " --> a"},
            {"id": "R1", "equation": "a --> b"},
            {"id": "R2", "equation": "b --> c"},
            {"id": "R3", "equation": "c --> d"},
            {"id": "R4", "equation": "d --> e"},
            {"id": "R5", "equation": "e --> f"},
            {"id": "R_out", "equation": "f --> "},
        ],
    )
    group_linear_reactions(m)
    # All the chain's internal metabolites are gone.
    assert {x for x in m.metabolites if x.id in {"b", "c", "d", "e"}} == set()


# --- regression: NaN FVA on infeasible model --------------------------------

def test_constrain_reversible_raises_on_infeasible():
    """An infeasible model produces NaN FVA ranges; a naive abs(NaN) < eps
    check would misread those as 'truly reversible', so this raises instead."""
    m = cobra.Model("t")
    a, b = (cobra.Metabolite(x, compartment="c") for x in ("a", "b"))
    m.add_metabolites([a, b])
    # Force a contradiction: r requires production AND consumption of a, but
    # nothing else produces a.
    r = cobra.Reaction("r", lower_bound=-1, upper_bound=1)
    r.add_metabolites({a: -1, b: 1})
    forced = cobra.Reaction("forced", lower_bound=5, upper_bound=10)  # infeasible
    forced.add_metabolites({a: -1})
    m.add_reactions([r, forced])
    with pytest.raises(RuntimeError, match="infeasible"):
        constrain_reversible_reactions(m)


def _reversible_one_way_model():
    m = cobra.Model("t")
    a = cobra.Metabolite("a", compartment="c")
    m.add_metabolites([a])
    supply = cobra.Reaction("supply", lower_bound=0, upper_bound=5)
    supply.add_metabolites({a: 1})
    r = cobra.Reaction("r", lower_bound=-10, upper_bound=10)
    r.add_metabolites({a: -1})
    m.add_reactions([supply, r])
    return m


class _FailingParams:
    def __setattr__(self, name, value):
        raise ValueError("tolerance out of range")


def test_constrain_reversible_does_not_hide_a_failing_tolerance(monkeypatch):
    """Only a solver without Gurobi parameters is skipped; other errors propagate."""
    from types import SimpleNamespace

    m = _reversible_one_way_model()
    fake = SimpleNamespace(problem=SimpleNamespace(Params=_FailingParams()))
    monkeypatch.setattr(cobra.Model, "solver", property(lambda self: fake))
    with pytest.raises(ValueError, match="tolerance out of range"):
        constrain_reversible_reactions(m)


def test_constrain_reversible_runs_on_a_solver_without_gurobi_parameters():
    m = _reversible_one_way_model()
    m.solver = "glpk"
    assert constrain_reversible_reactions(m) == ["r"]
    assert m.reactions.r.lower_bound == 0.0


# --- order, survivor, reserved, orphans (RAVEN simplifyModel) ---------------

def _chain_model():
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in ("M0", "M1", "M2", "M3")])
    add_reactions_from_equations(
        m,
        [
            {"id": "IN", "equation": " --> M0"},
            {"id": "R1", "equation": "M0 --> M1"},
            {"id": "R2", "equation": "M1 --> M2"},
            {"id": "R3", "equation": "M2 --> M3"},
            {"id": "OUT", "equation": "M3 --> "},
        ],
    )
    return m


def test_group_linear_keeps_the_first_reaction_of_a_chain():
    n = 6
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(f"M{i}", compartment="c") for i in range(n + 1)])
    eqs = [{"id": "IN", "equation": " --> M0"}]
    eqs += [{"id": f"R{i}", "equation": f"M{i - 1} --> M{i}"} for i in range(1, n + 1)]
    eqs += [{"id": "OUT", "equation": f"M{n} --> "}]
    add_reactions_from_equations(m, eqs)
    group_linear_reactions(m, reserved=["IN", "OUT"])
    assert [r.id for r in m.reactions] == ["IN", "R1", "OUT"]
    assert {mm.id for mm in m.reactions.get_by_id("R1").metabolites} == {"M0", f"M{n}"}


def test_group_linear_leaves_no_residue_for_non_representable_ratios():
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in ("A", "B", "C")])
    add_reactions_from_equations(
        m,
        [
            {"id": "IN", "equation": " --> A"},
            {"id": "R1", "equation": "A --> 0.7 B"},
            {"id": "R2", "equation": "0.3 B --> C"},
            {"id": "OUT", "equation": "C --> "},
        ],
    )
    group_linear_reactions(m, reserved=["IN", "OUT"])
    assert {mm.id for mm in m.reactions.get_by_id("R1").metabolites} == {"A", "C"}
    assert "B" not in m.metabolites


def test_constrain_reversible_runs_before_group_linear():
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in ("A", "B")])
    add_reactions_from_equations(
        m,
        [
            {"id": "IN", "equation": " --> A", "bounds": (0, 1000)},
            {"id": "R1", "equation": "A <=> B", "bounds": (-1000, 1000)},
            {"id": "OUT", "equation": "B --> ", "bounds": (0, 1000)},
        ],
    )
    simplify_model(m, constrain_reversible=True, group_linear=True)
    assert len(m.reactions) == 0 and len(m.metabolites) == 0


def test_simplify_model_runs_modes_in_raven_order(monkeypatch):
    from raven_toolbox.manipulation import simplify as mod

    calls = []
    for name in (
        "remove_duplicate_reactions", "remove_zero_interval_reactions",
        "remove_dead_end_reactions", "remove_no_flux_reactions",
        "constrain_reversible_reactions", "group_linear_reactions",
    ):
        monkeypatch.setattr(mod, name, lambda *a, _n=name, **k: calls.append(_n))
    simplify_model(
        cobra.Model("t"), delete_zero_interval=True, delete_dead_end=True, delete_no_flux=True,
        delete_duplicates=True, group_linear=True, constrain_reversible=True,
    )
    assert calls == [
        "remove_duplicate_reactions", "remove_zero_interval_reactions",
        "remove_dead_end_reactions", "remove_no_flux_reactions",
        "constrain_reversible_reactions", "group_linear_reactions",
    ]


def _zero_model():
    m = cobra.Model("t")
    m.add_metabolites([cobra.Metabolite(x, compartment="c") for x in ("A", "B", "ORPH")])
    add_reactions_from_equations(
        m,
        [
            {"id": "R1", "equation": "A --> B", "bounds": (0, 1000)},
            {"id": "R2", "equation": "A --> B", "bounds": (0, 0), "gene_reaction_rule": "g2"},
        ],
    )
    return m


def test_zero_interval_honours_reserved():
    m = _zero_model()
    assert remove_zero_interval_reactions(m, reserved=["R2"]) == []
    assert "R2" in m.reactions and "g2" in m.genes


def test_zero_interval_prunes_every_orphan_metabolite():
    m = _zero_model()
    assert remove_zero_interval_reactions(m) == ["R2"]
    assert "ORPH" not in m.metabolites
    assert "g2" in m.genes


def test_no_flux_honours_reserved_and_prunes_orphans():
    m = _zero_model()
    m.reactions.get_by_id("R2").bounds = (0, 1000)
    # A has no source, so both reactions are blocked
    assert remove_no_flux_reactions(m, reserved=["R2"]) == ["R1"]
    assert "R2" in m.reactions
    assert "ORPH" not in m.metabolites


def test_no_flux_keeps_the_genes_of_removed_reactions():
    m = _zero_model()
    m.reactions.get_by_id("R2").bounds = (0, 1000)
    assert sorted(remove_no_flux_reactions(m)) == ["R1", "R2"]
    assert "g2" in m.genes
