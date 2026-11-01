"""Tests for flux-vs-expression change significance (analysis/differential_sampling.py,
getFluxZ/analyzeSampling port)."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import erf
from scipy.stats import t as student_t

from raven_toolbox.analysis import analyze_sampling, get_flux_z


def _const(values: dict[str, float], n: int = 3) -> pd.DataFrame:
    """A samples x reactions frame where every reaction is constant (zero variance)."""
    return pd.DataFrame({rid: [v] * n for rid, v in values.items()})


def test_general_formula():
    a = pd.DataFrame({"R1": [1.0, 2.0, 3.0]})  # mean 2, var 1
    b = pd.DataFrame({"R1": [3.0, 4.0, 5.0]})  # mean 4, var 1
    z = get_flux_z(a, b)
    assert z["R1"] == pytest.approx(2.0 / np.sqrt(2.0))


def test_zero_variance_increase_is_positive():
    a = _const({"R1": 5.0})
    b = _const({"R1": 9.0})
    assert get_flux_z(a, b)["R1"] == 100.0


def test_zero_variance_decrease_is_negative():
    # A flux *decrease* with no variance in either condition scores -100.
    a = _const({"R1": 9.0})
    b = _const({"R1": 5.0})
    assert get_flux_z(a, b)["R1"] == -100.0


def test_equal_means_is_zero_regardless_of_variance():
    a = pd.DataFrame({"R1": [1.0, 2.0, 3.0], "R2": [2.0, 2.0, 2.0]})
    b = pd.DataFrame({"R1": [1.0, 2.0, 3.0], "R2": [2.0, 2.0, 2.0]})
    z = get_flux_z(a, b)
    assert z["R1"] == 0.0
    assert z["R2"] == 0.0


def test_large_z_is_clamped():
    a = pd.DataFrame({"R1": [0.0, 0.0, 1e-6]})
    b = pd.DataFrame({"R1": [1000.0, 1000.0, 1000.0 + 1e-6]})
    assert get_flux_z(a, b)["R1"] == 100.0


def test_mismatched_columns_raise():
    a = pd.DataFrame({"R1": [1.0, 2.0]})
    b = pd.DataFrame({"R2": [1.0, 2.0]})
    with pytest.raises(ValueError, match="same reaction columns"):
        get_flux_z(a, b)


def test_analyze_sampling_concordant_increase():
    a = _const({"R1": 1.0})
    b = _const({"R1": 5.0})
    df = 10
    result = analyze_sampling([3.0], df, a, b)

    p_flux = erf(100.0)
    conf = 2 * student_t.cdf(3.0, df) - 1
    assert result.loc["R1", "concordant"] == pytest.approx(p_flux * conf)
    assert result.loc["R1", "expression_only"] == pytest.approx(conf * (1 - p_flux))
    assert result.loc["R1", "flux_only_or_discordant"] == pytest.approx(p_flux * (1 - conf))


def test_analyze_sampling_discordant_direction():
    a = _const({"R1": 1.0})
    b = _const({"R1": 5.0})  # flux increases (Zf = +100)
    df = 10
    result = analyze_sampling([-3.0], df, a, b)  # expression decreases

    p_flux = erf(100.0)
    conf = 2 * student_t.cdf(3.0, df) - 1
    assert result.loc["R1", "concordant"] == 0.0
    assert result.loc["R1", "expression_only"] == pytest.approx((1 - p_flux) * conf)
    assert result.loc["R1", "flux_only_or_discordant"] == pytest.approx(p_flux)


def test_analyze_sampling_flips_sign_for_more_negative_flux():
    # Both means negative; B is *more* negative than A -- a magnitude increase
    # in reverse flux, which get_flux_z alone scores as a decrease (-100) but
    # analyze_sampling treats as an increase (+100) to compare against
    # expression. Paired with a positive (increased) expression t-score, this
    # is concordant only if the flip actually happens.
    a = _const({"R1": -5.0})
    b = _const({"R1": -9.0})
    assert get_flux_z(a, b)["R1"] == -100.0  # raw Z: unflipped

    df = 10
    result = analyze_sampling([3.0], df, a, b)
    p_flux = erf(100.0)
    conf = 2 * student_t.cdf(3.0, df) - 1
    assert result.loc["R1", "concordant"] == pytest.approx(p_flux * conf)


def test_nan_and_inf_t_scores_are_cleaned_to_zero_with_warning():
    a = _const({"R1": 1.0})
    b = _const({"R1": 5.0})
    df = 10

    with pytest.warns(UserWarning, match="NaN or \\+/- Inf"):
        result_nan = analyze_sampling([np.nan], df, a, b)
    with pytest.warns(UserWarning, match="NaN or \\+/- Inf"):
        result_inf = analyze_sampling([np.inf], df, a, b)
    result_zero = analyze_sampling([0.0], df, a, b)

    pd.testing.assert_frame_equal(result_nan, result_zero)
    pd.testing.assert_frame_equal(result_inf, result_zero)


def test_t_expression_length_mismatch_raises():
    a = _const({"R1": 1.0, "R2": 2.0})
    b = _const({"R1": 5.0, "R2": 6.0})
    with pytest.raises(ValueError, match="one entry per reaction"):
        analyze_sampling([1.0], 10, a, b)


def test_analyze_sampling_mismatched_columns_raise():
    a = pd.DataFrame({"R1": [1.0, 2.0]})
    b = pd.DataFrame({"R2": [1.0, 2.0]})
    with pytest.raises(ValueError, match="same reaction columns"):
        analyze_sampling([1.0], 10, a, b)


def test_result_indexed_by_reaction_id():
    a = _const({"R1": 1.0, "R2": 1.0})
    b = _const({"R1": 5.0, "R2": 5.0})
    result = analyze_sampling([1.0, -1.0], 10, a, b)
    assert list(result.index) == ["R1", "R2"]
    assert list(result.columns) == ["concordant", "expression_only", "flux_only_or_discordant"]
