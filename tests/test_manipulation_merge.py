"""Tests for merge_models (mergeModels port)."""
import cobra
import pytest

from raven_toolbox.manipulation import add_reactions_from_equations, merge_models


def _model(mid, mets, reactions):
    m = cobra.Model(mid)
    m.add_metabolites(mets)
    add_reactions_from_equations(m, reactions)
    return m


@pytest.fixture
def model_a():
    return _model(
        "A",
        [
            cobra.Metabolite("glc_c", name="Glucose", compartment="c"),
            cobra.Metabolite("g6p_c", name="G6P", compartment="c"),
        ],
        [{"id": "HEX", "equation": "glc_c --> g6p_c", "gene_reaction_rule": "GA"}],
    )


@pytest.fixture
def model_b():
    # same Glucose[c] compound but a DIFFERENT id
    return _model(
        "B",
        [
            cobra.Metabolite("glucose_c", name="Glucose", compartment="c"),
            cobra.Metabolite("lac_c", name="Lactate", compartment="c"),
        ],
        [{"id": "LDH", "equation": "glucose_c --> lac_c", "gene_reaction_rule": "GB"}],
    )


def test_unifies_metabolites_by_name_comp(model_a, model_b):
    merged = merge_models([model_a, model_b])
    glucoses = [m for m in merged.metabolites if m.name == "Glucose" and m.compartment == "c"]
    assert len(glucoses) == 1  # glc_c and glucose_c unified
    # both reactions reference the same merged Glucose object
    hex_glc = [m for m in merged.reactions.get_by_id("HEX").metabolites if m.name == "Glucose"][0]
    ldh_glc = [m for m in merged.reactions.get_by_id("LDH").metabolites if m.name == "Glucose"][0]
    assert hex_glc is ldh_glc


def test_match_by_id_keeps_distinct(model_a, model_b):
    merged = merge_models([model_a, model_b], match_by="id")
    glucoses = [m for m in merged.metabolites if m.name == "Glucose"]
    assert len(glucoses) == 2  # glc_c and glucose_c are distinct by id


def test_all_reactions_kept(model_a, model_b):
    merged = merge_models([model_a, model_b])
    assert {"HEX", "LDH"} <= {r.id for r in merged.reactions}


def test_reaction_id_collision_renamed(model_a):
    # two models with the same reaction id but different chemistry
    other = _model(
        "B",
        [cobra.Metabolite("glc_c", name="Glucose", compartment="c"),
         cobra.Metabolite("x_c", name="X", compartment="c")],
        [{"id": "HEX", "equation": "glc_c --> x_c"}],
    )
    merged = merge_models([model_a, other])
    assert "HEX" in {r.id for r in merged.reactions}
    assert "HEX_B" in {r.id for r in merged.reactions}  # renamed with source id


def test_genes_merged(model_a, model_b):
    merged = merge_models([model_a, model_b])
    assert {"GA", "GB"} <= {g.id for g in merged.genes}


def test_provenance_recorded(model_a, model_b):
    merged = merge_models([model_a, model_b])
    assert merged.reactions.get_by_id("HEX").notes["origin"] == "A"
    assert merged.reactions.get_by_id("LDH").notes["origin"] == "B"
    assert merged.genes.get_by_id("GA").notes["origin"] == "A"


def test_compartments_preserved(model_a):
    model_a.compartments = {"c": "cytoplasm"}
    merged = merge_models([model_a, model_a.copy()])
    assert merged.compartments.get("c") == "cytoplasm"


def test_single_model_returns_copy(model_a):
    merged = merge_models([model_a])
    assert merged is not model_a
    assert {r.id for r in merged.reactions} == {r.id for r in model_a.reactions}


def test_three_models(model_a, model_b):
    c = _model("C", [cobra.Metabolite("co2_c", name="CO2", compartment="c")],
               [{"id": "SINK", "equation": "co2_c -->"}])
    merged = merge_models([model_a, model_b, c])
    assert {"HEX", "LDH", "SINK"} <= {r.id for r in merged.reactions}


def test_bad_match_by(model_a, model_b):
    with pytest.raises(ValueError, match="match_by"):
        merge_models([model_a, model_b], match_by="oops")


# --- regression: formula/charge conflict ------------------------------------

def test_formula_conflict_warns():
    """Two models sharing a name[comp] but with different formulas warn instead
    of silently keeping the first."""
    a = _model("A",
        [cobra.Metabolite("g1", name="Glucose", formula="C6H12O6", compartment="c")],
        [{"id": "EX_A", "equation": "g1 -->"}])
    b = _model("B",
        [cobra.Metabolite("g2", name="Glucose", formula="C6H12O7", compartment="c")],
        [{"id": "EX_B", "equation": "g2 -->"}])
    with pytest.warns(UserWarning, match="different formulas"):
        merged = merge_models([a, b])
    # The merge still picks the first-seen — the test asserts the warning fired
    # and the model survives.
    assert "EX_A" in merged.reactions and "EX_B" in merged.reactions


def test_charge_conflict_warns():
    a = _model("A",
        [cobra.Metabolite("g1", name="Glucose", formula="C6H12O6", charge=0, compartment="c")],
        [{"id": "EX_A", "equation": "g1 -->"}])
    b = _model("B",
        [cobra.Metabolite("g2", name="Glucose", formula="C6H12O6", charge=-1, compartment="c")],
        [{"id": "EX_B", "equation": "g2 -->"}])
    with pytest.warns(UserWarning, match="different charges"):
        merge_models([a, b])


# --- RAVEN's mergeModels keeps everything of every model ------------------------------

def _with_extras():
    """Model with a metabolite, a gene and a compartment that no reaction uses."""
    m = _model(
        "X",
        [cobra.Metabolite("a_c", name="A", compartment="c"),
         cobra.Metabolite("b_c", name="B", compartment="c")],
        [{"id": "R1", "equation": "a_c --> b_c", "gene_reaction_rule": "G1"}],
    )
    m.add_metabolites([cobra.Metabolite("orph_c", name="Orphan", compartment="c")])
    orphan = cobra.Gene("G_ORPH", name="unused gene")
    m.genes.append(orphan)
    orphan._model = m
    return m


def test_unused_metabolites_and_genes_are_kept(model_b):
    merged = merge_models([_with_extras(), model_b])
    assert "orph_c" in merged.metabolites
    assert "G_ORPH" in merged.genes
    assert merged.genes.get_by_id("G_ORPH").name == "unused gene"
    assert merged.genes.get_by_id("G_ORPH").notes["origin"] == "X"


def test_unused_items_of_a_later_model_are_kept_too(model_a):
    merged = merge_models([model_a, _with_extras()])
    assert "orph_c" in merged.metabolites
    assert "G_ORPH" in merged.genes


def test_objective_of_every_model_is_carried_over(model_a, model_b):
    model_a.objective = "HEX"
    model_b.objective = {model_b.reactions.get_by_id("LDH"): 2.0}
    merged = merge_models([model_a, model_b])
    coefficients = {r.id: r.objective_coefficient for r in merged.reactions if r.objective_coefficient}
    assert coefficients == {"HEX": 1.0, "LDH": 2.0}


def test_objective_follows_a_renamed_reaction(model_a):
    other = _model(
        "B",
        [cobra.Metabolite("glc_c", name="Glucose", compartment="c"),
         cobra.Metabolite("x_c", name="X", compartment="c")],
        [{"id": "HEX", "equation": "glc_c --> x_c"}],
    )
    other.objective = "HEX"
    merged = merge_models([model_a, other])
    assert merged.reactions.get_by_id("HEX_B").objective_coefficient == 1.0
    assert merged.reactions.get_by_id("HEX").objective_coefficient == 0


def test_no_objective_stays_empty(model_a, model_b):
    merged = merge_models([model_a, model_b])
    assert not any(r.objective_coefficient for r in merged.reactions)


def test_objective_direction_comes_from_the_first_model(model_a, model_b):
    model_a.objective = "HEX"
    model_a.objective_direction = "min"
    merged = merge_models([model_a, model_b])
    assert merged.objective_direction == "min"


def test_first_model_names_a_compartment():
    first = _model("A", [cobra.Metabolite("a_c", name="A", compartment="c")], [])
    first.compartments = {"c": "cytoplasm"}
    second = _model("B", [cobra.Metabolite("b_c", name="B", compartment="c")], [])
    second.compartments = {"c": "cytosol", "e": "extracellular"}
    merged = merge_models([first, second])
    assert merged.compartments["c"] == "cytoplasm"
    assert merged._compartments["e"] == "extracellular"    # a compartment without metabolites is kept too


def test_order_follows_the_source_models(model_a, model_b):
    # model_b lists lac_c, then glucose_c: the merged order is model_a's metabolites,
    # then the new ones of model_b in model_b's order (RAVEN appends in model order).
    merged = merge_models([model_a, model_b])
    assert [m.name for m in merged.metabolites] == ["Glucose", "G6P", "Lactate"]
    assert [g.id for g in merged.genes] == ["GA", "GB"]


def test_order_does_not_depend_on_the_reactions():
    # The reaction uses the second metabolite first; the model order is still kept.
    m = cobra.Model("O")
    m.add_metabolites([cobra.Metabolite("p_c", name="P", compartment="c"),
                       cobra.Metabolite("q_c", name="Q", compartment="c")])
    add_reactions_from_equations(m, [{"id": "R", "equation": "q_c --> p_c", "gene_reaction_rule": "G2 or G1"}])
    merged = merge_models([m, m.copy()])
    assert [x.id for x in merged.metabolites][:2] == ["p_c", "q_c"]


def test_merge_is_independent_of_the_hash_seed():
    import os
    import subprocess
    import sys
    code = (
        "import cobra\n"
        "from raven_toolbox.manipulation import merge_models\n"
        "ms = []\n"
        "for i in range(2):\n"
        "    m = cobra.Model(f'M{i}')\n"
        "    m.add_metabolites([cobra.Metabolite(f'm{j}_c', name=f'N{j}', compartment='c') for j in range(6)])\n"
        "    for k in (5, 3, 4, 1, 2, 0):\n"
        "        g = cobra.Gene(f'G{k}'); m.genes.append(g); g._model = m\n"
        "    r = cobra.Reaction(f'R{i}'); m.add_reactions([r])\n"
        "    r.add_metabolites({m.metabolites[0]: -1, m.metabolites[1]: 1})\n"
        "    r.gene_reaction_rule = ' or '.join(f'G{k}' for k in (5, 3, 4, 1, 2, 0))\n"
        "    ms.append(m)\n"
        "g = merge_models(ms)\n"
        "print([x.id for x in g.metabolites], [x.id for x in g.genes])\n"
    )
    outputs = set()
    for seed in ("0", "1", "2"):
        # The seed is set explicitly (not "-I", which ignores PYTHON* variables), and the rest
        # of the environment is kept: Python does not start on Windows without SYSTEMROOT.
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={**os.environ, "PYTHONHASHSEED": seed}, check=True).stdout
        outputs.add(out)
    assert len(outputs) == 1
