"""Flux-vs-expression change significance — port of RAVEN's ``getFluxZ`` and
``analyzeSampling``.

Both operate on the output of a random-sampling run comparing two conditions
(e.g. :func:`~raven_toolbox.analysis.random_sampling`): :func:`get_flux_z`
turns two sets of sampled flux vectors into a per-reaction Z-score for
whether flux increased or decreased between them, and :func:`analyze_sampling`
combines that Z-score with a per-reaction gene-expression t-score to say
whether a reaction's flux and expression changed together, only one of them
changed, or they moved in opposite directions.

``analyzeSampling.m`` calls ``getFluxZ.m`` directly and cannot be usefully
ported without it (CompMetEng/raven-gecko-parity#41), so both are ported
together here.

Sample orientation is *samples × reactions* (rows are samples, columns are
reaction ids) — the ``cobra.sampling`` convention
:func:`~raven_toolbox.analysis.random_sampling` already returns, and the
transpose of RAVEN's own reactions × samples arrays. This is a deliberate
adaptation, not a literal port of the axis RAVEN happens to reduce over: it
lets a sampling result be passed straight through with no manual transpose.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import t as _student_t

__all__ = ["get_flux_z", "analyze_sampling"]

_Z_CLAMP = 100.0


def _check_same_reactions(solutions_a: pd.DataFrame, solutions_b: pd.DataFrame) -> list:
    reaction_ids = list(solutions_a.columns)
    if list(solutions_b.columns) != reaction_ids:
        raise ValueError("solutions_a and solutions_b must have the same reaction columns, in the same order.")
    return reaction_ids


def get_flux_z(solutions_a: pd.DataFrame, solutions_b: pd.DataFrame) -> pd.Series:
    """Z-score for how much flux changed between two random-sampling runs.

    Parameters
    ----------
    solutions_a:
        Sampled flux vectors for the reference condition, *samples × reactions*
        (e.g. ``random_sampling(...).samples``).
    solutions_b:
        Sampled flux vectors for the test condition, same reaction columns.

    Returns
    -------
    pd.Series
        Z-score per reaction (indexed by ``solutions_a``'s columns): positive
        when flux increased from A to B, negative when it decreased, 0 when
        the means are equal, clamped to ``[-100, 100]``. A reaction with zero
        variance in both conditions gets exactly ``±100`` rather than a
        division by zero.

    Notes
    -----
    A flux *decrease* with no variance in either condition scores ``-100``,
    the same sign the general formula gives, as in RAVEN's ``getFluxZ``.
    """
    reaction_ids = _check_same_reactions(solutions_a, solutions_b)

    mean_a = solutions_a.mean()
    mean_b = solutions_b.mean()
    var_a = solutions_a.var()
    var_b = solutions_b.var()

    z = pd.Series(0.0, index=reaction_ids)
    changed = mean_a != mean_b
    both_flat = changed & (var_a == 0) & (var_b == 0)
    general = changed & ~both_flat

    z[both_flat] = np.where(mean_b[both_flat] > mean_a[both_flat], _Z_CLAMP, -_Z_CLAMP)
    z[general] = (mean_b[general] - mean_a[general]) / np.sqrt(var_a[general] + var_b[general])

    return z.clip(-_Z_CLAMP, _Z_CLAMP)


def analyze_sampling(
    t_expression: Sequence[float] | np.ndarray | pd.Series,
    df: float,
    solutions_a: pd.DataFrame,
    solutions_b: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the significance of a flux change with a gene-expression change.

    Parameters
    ----------
    t_expression:
        t-score for the change in gene expression, one per reaction, in the
        same order as ``solutions_a``'s columns. Positive means expression
        increased, negative means it decreased. A p-value can be converted
        to this by the inverse error function, choosing the sign by hand
        (the p-value alone does not carry direction).
    df:
        Degrees of freedom for the expression t-test.
    solutions_a, solutions_b:
        Sampled flux vectors for the reference and test condition, *samples ×
        reactions* — see :func:`get_flux_z`.

    Returns
    -------
    pd.DataFrame
        Indexed by reaction id, with columns:

        - ``concordant`` — probability flux and expression both changed
          significantly, in the same direction (RAVEN's ``pR``).
        - ``expression_only`` — probability expression changed significantly
          but flux did not, or they changed in opposite directions (``pH``).
        - ``flux_only_or_discordant`` — probability flux changed
          significantly but expression did not, or they changed in opposite
          directions (``pM``).

    Notes
    -----
    Two simplifications relative to ``analyzeSampling.m``: it computes an
    "opposite flux direction" branch (``pM=erf(|Z|)``, ``pH=(1-pM)*...``,
    ``pR=0``) whose results are unconditionally overwritten by the very next
    branch before ``scores`` is ever read — dead code, verified by tracing
    every assignment, not replicated here. And it bundles its own ``tcdf``
    (a pure-MATLAB fallback for when the Statistics Toolbox isn't installed);
    this uses ``scipy.stats.t.cdf`` directly, which is exact and always
    available.
    """
    reaction_ids = _check_same_reactions(solutions_a, solutions_b)
    tex = np.asarray(t_expression, dtype=float)
    if len(tex) != len(reaction_ids):
        raise ValueError("t_expression must have one entry per reaction in solutions_a/solutions_b.")

    bad = np.isnan(tex) | np.isinf(tex)
    if bad.any():
        warnings.warn(
            "There are t-scores that are NaN or +/- Inf. These values are changed to 0.0",
            stacklevel=2,
        )
    tex = np.where(bad, 0.0, tex)

    mean_a = solutions_a.mean().to_numpy()
    mean_b = solutions_b.mean().to_numpy()
    zf = get_flux_z(solutions_a, solutions_b).to_numpy()

    # A more-negative mean in B than A reads as a flux *decrease* under the
    # general Z-score formula, but when both conditions sit on the same side
    # of zero (or one is exactly zero) and that side is negative, a larger
    # magnitude is really an *increase* in (reverse) flux -- so the sign is
    # flipped to match, exactly as analyzeSampling.m does.
    with np.errstate(divide="ignore", invalid="ignore"):
        same_sign_or_zero = ~((mean_b / mean_a) < 0)
    flip = same_sign_or_zero & (mean_b < 0)
    zf = np.where(flip, -zf, zf)

    confidence_expr = 2 * _student_t.cdf(np.abs(tex), df) - 1
    p_flux = erf(np.abs(zf))

    with np.errstate(divide="ignore", invalid="ignore"):
        discordant = (zf / tex) < 0

    concordant = np.where(discordant, 0.0, p_flux * confidence_expr)
    expression_only = np.where(discordant, (1 - p_flux) * confidence_expr, confidence_expr * (1 - p_flux))
    flux_only_or_discordant = np.where(discordant, p_flux, p_flux * (1 - confidence_expr))

    return pd.DataFrame(
        {
            "concordant": concordant,
            "expression_only": expression_only,
            "flux_only_or_discordant": flux_only_or_discordant,
        },
        index=reaction_ids,
    )
