import cobra

from raven_toolbox.utils.solver import RAVEN_TOLERANCE, set_raven_tolerance


def test_set_raven_tolerance_changes_the_model_in_place():
    model = cobra.Model("m")
    assert set_raven_tolerance(model) is model
    assert model.tolerance == RAVEN_TOLERANCE == 1e-9


def test_set_raven_tolerance_takes_a_value():
    model = set_raven_tolerance(cobra.Model("m"), 1e-7)
    assert model.tolerance == 1e-7
