"""Markov-chain Monte Carlo flux sampling with CHRR and ACHR.

Two near-uniform samplers of the flux polytope ``{v : S v = 0, lb <= v <= ub}``,
reached through the unified :func:`~raven_toolbox.analysis.sampling.random_sampling`
entry point (``method="achr"`` default, or ``method="chrr"``):

* **CHRR**: Coordinate Hit-and-Run with Rounding (Haraldsdóttir et al. 2017,
  Bioinformatics 33:1741). The polytope is reduced to a full-dimensional body via
  the nullspace of ``S``, rounded with the maximum-volume inscribed ellipsoid
  (MVE, Zhang & Gao 2003), and then walked with coordinate hit-and-run. Rounding
  makes the mixing time independent of how elongated the original polytope is.
  Enzyme-constrained (ecModel + proteomics) and flux-measured models are elongated,
  because the feasible set is a thin, badly-conditioned slab. cobrapy has no CHRR.

* **ACHR**: Artificially Centered Hit-and-Run (Kaufman & Smith 1998). Walks the
  polytope directly using FVA warmup directions mixed through a running centre,
  without rounding. This module wraps cobrapy's ACHR
  (:class:`cobra.sampling.ACHRSampler`) so that both methods are reachable through the
  same RAVEN-style API and return the same :class:`FluxSamplingResult`. ACHR needs no
  MVE solve. On elongated polytopes its chains mix slowly and can look converged while
  having explored only the long axes; CHRR rounds the polytope first and does not have
  this limitation.

Both return a ``samples`` DataFrame shaped n_samples × n_reactions (the
``cobra.sampling`` layout), matching :class:`~raven_toolbox.analysis.sampling.RandomSamplingResult`.

:func:`~raven_toolbox.analysis.sampling.random_sampling` (Bordel 2010) draws diverse
vertices of the polytope; CHRR and ACHR draw the (near-)uniform interior distribution.

Relation to RAVEN MATLAB
------------------------
RAVEN's ``randomSampling`` dispatches on ``method`` (``'achr'``, ``'chrr'`` or
``'randomObjective'``). The CHRR and MVE numerics follow COBRA's ``chrrSampler``;
``tests/test_analysis_flux_sampling.py`` checks the MVE solver against boxes and a triangle
with known inscribed ellipsoids.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import flux_variability_analysis
from scipy.linalg import null_space
from scipy.optimize import linprog

logger = logging.getLogger(__name__)

__all__ = [
    "FluxSamplingResult",
    "max_volume_ellipsoid",
]


@dataclass
class FluxSamplingResult:
    """Output of :func:`~raven_toolbox.analysis.sampling.random_sampling`.

    Attributes
    ----------
    samples:
        Flux vectors shaped n_samples × n_reactions (one sample per row, reaction
        ids as columns; the ``cobra.sampling`` layout).
    method:
        ``"achr"``, ``"chrr"``, or ``"random_objective"``.
    good_reactions:
        ``random_objective`` only: reaction ids eligible as random objectives;
        ``None`` for the MCMC methods.
    n_dimensions:
        ``chrr`` only: dimension of the full-dimensional flux polytope sampled
        (degrees of freedom after fixing implicitly-determined reactions).
    mve_converged:
        ``chrr`` only: whether the MVE rounding solver reached its tolerance. With
        ``False``, the last ellipsoid iterate is used as the rounding; elongated
        polytopes may need more thinning.
    n_warmup:
        ``achr`` only: number of FVA warmup directions.
    fixed_reactions:
        ``chrr`` only: reactions added to the equality system as fixed (FVA width below
        ``fixed_width_tol``).
    """

    samples: pd.DataFrame
    method: str
    good_reactions: list[str] | None = None
    n_dimensions: int | None = None
    mve_converged: bool | None = None
    n_warmup: int | None = None
    fixed_reactions: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Maximum-volume inscribed ellipsoid (Zhang & Gao 2003 primal-dual interior point)
# --------------------------------------------------------------------------- #
def _nearest_spd(M: np.ndarray) -> np.ndarray:
    """Return the nearest symmetric positive-definite matrix to ``M``.

    A light-weight variant of Higham's algorithm: symmetrise, then add a growing
    multiple of the identity to the diagonal until a Cholesky factorisation succeeds. ``M``
    here is an ellipsoid matrix ``E2`` that is theoretically SPD but may acquire small
    asymmetries from the matrix inverse.
    """
    S = (M + M.T) / 2.0
    try:
        np.linalg.cholesky(S)
        return S
    except np.linalg.LinAlgError:
        pass
    eigvals = np.linalg.eigvalsh(S)
    jitter = max(-eigvals.min(), 0.0) + 1e-12
    n = S.shape[0]
    for _ in range(30):
        try:
            np.linalg.cholesky(S + jitter * np.eye(n))
            return S + jitter * np.eye(n)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return S + jitter * np.eye(n)


def max_volume_ellipsoid(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    *,
    maxiter: int = 150,
    tol: float = 1e-6,
    reg: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, bool]:
    r"""Maximum-volume ellipsoid inscribed in the polytope ``{z : A z <= b}``.

    Solves ``max log det E`` over centre ``x`` and SPD ``E`` such that the ellipsoid
    ``{x + E s : ||s||_2 <= 1}`` is contained in ``{z : A z <= b}``, using the
    Zhang & Gao (2003) primal-dual interior-point method (the regularised variant
    shipped in COBRA's ``chrrSampler``).

    Parameters
    ----------
    A, b:
        Polytope ``A z <= b`` with ``A`` of shape ``(m, n)``, ``m >= n + 1``, and a
        non-empty interior.
    x0:
        A strictly interior point (``A x0 < b``). If ``None`` a Chebyshev centre is
        computed by LP.
    maxiter, tol, reg:
        Iteration cap, convergence tolerance, and the diagonal (Levenberg)
        regularisation added to the two Newton solves.

    Returns
    -------
    x:
        Ellipsoid centre, shape ``(n,)``.
    E:
        Lower-triangular rounding transform with ``E @ E.T == E2`` (the SPD
        ellipsoid matrix). The rounding substitution is ``z = x + E y``.
    converged:
        Whether ``tol`` was reached within ``maxiter``.

    Notes
    -----
    Validated against analytic cases: a box maps to the unit ball (``E = I``); a
    scaled or sheared box ``{-1 <= M z <= 1}`` gives ``E E^T = M^{-1} M^{-T}``; the
    standard simplex gives its inscribed ball.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    # bnrm is the norm of the ORIGINAL b (before the row-normalisation below sets
    # b <- ones). This matches COBRA's chrrSampler reference; do not "correct" it
    # to sqrt(m), or convergence scaling diverges from the reference.
    bnrm = np.linalg.norm(b)
    minmu = 1e-8
    tau0 = 0.75

    if x0 is None:
        x0 = _chebyshev_center(A, b)
    x0 = np.asarray(x0, dtype=float).ravel()

    bmAx0 = b - A @ x0
    if np.any(bmAx0 <= 0):
        raise ValueError("max_volume_ellipsoid: x0 is not strictly interior (A x0 < b).")

    # Row-normalise so the constraint RHS is all ones; solve in shifted coords.
    A = A / bmAx0[:, None]
    b = np.ones(m)

    x = np.zeros(n)
    y = np.ones(m)
    bmAx = b.copy()
    z = np.empty(m)
    astep = 0.0
    Adx = np.zeros(n)
    converged = False
    E2: np.ndarray = np.eye(n)

    for it in range(1, maxiter + 1):
        if it > 1:
            bmAx = bmAx - astep * Adx

        AtYA = A.T @ (A * y[:, None])          # A' diag(y) A   (n x n)
        E2 = np.linalg.inv(AtYA)
        Q = A @ E2 @ A.T                       # m x m
        h = np.sqrt(np.maximum(np.diag(Q), 0.0))

        if it == 1:
            t = np.min(bmAx / h)
            y = y / t**2
            h = t * h
            z = np.maximum(1e-1, bmAx - h)
            Q = t**2 * Q

        yz = y * z
        yh = y * h
        gap = np.sum(yz) / m
        rmu = max(min(0.5, gap) * gap, minmu)

        R1 = -A.T @ yh                         # dual residual      (n,)
        R2 = bmAx - h - z                      # primal/slack       (m,)
        R3 = rmu - yz                          # complementarity    (m,)
        res = max(
            np.max(np.abs(R1)), np.max(np.abs(R2)), np.max(np.abs(R3))
        )

        if res < tol * (1 + bnrm) and rmu <= minmu:
            x = x + x0
            converged = True
            break

        YQ = y[:, None] * Q                    # diag(y) Q
        YQQY = YQ * YQ.T                        # Hadamard product
        y2h = 2.0 * yh
        G = YQQY + np.diag(np.maximum(reg, y2h * z))
        YA = y[:, None] * A                     # diag(y) A
        T = np.linalg.solve(G, (h + z)[:, None] * YA)          # m x n
        ATP = (y2h[:, None] * T - YA).T                        # n x m
        R3Dy = R3 / y
        R23 = R2 - R3Dy
        ATP_A = ATP @ A + reg * np.eye(n)
        dx = np.linalg.solve(ATP_A, R1 + ATP @ R23)
        Adx = A @ dx
        dyDy = np.linalg.solve(G, y2h * (Adx - R23))
        dy = y * dyDy
        dz = R3Dy - z * dyDy

        ax = -1.0 / min(np.min(-Adx / bmAx), -0.5)
        ay = -1.0 / min(np.min(dyDy), -0.5)
        az = -1.0 / min(np.min(dz / z), -0.5)
        tau = max(tau0, 1.0 - res)
        astep = tau * min(1.0, ax, ay, az)

        x = x + astep * dx
        y = y + astep * dy
        z = z + astep * dz

    if not converged:
        x = x + x0

    E = np.linalg.cholesky(_nearest_spd(E2))   # lower-tri, E @ E.T == E2
    return x, E, converged


def _chebyshev_center(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Centre of the largest ball inscribed in ``{z : A z <= b}`` (an LP).

    ``max r`` s.t. ``a_i^T z + ||a_i|| r <= b_i``, ``r >= 0``. Provides a strictly
    interior starting point for the MVE solver.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    norms = np.linalg.norm(A, axis=1)
    # variables [z (n, free); r (1, >=0)]; minimise -r
    c = np.concatenate([np.zeros(n), [-1.0]])
    A_ub = np.hstack([A, norms[:, None]])
    bounds = [(None, None)] * n + [(0.0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b, bounds=bounds, method="highs")
    if not res.success or res.x[-1] <= 1e-12:
        raise ValueError(
            "Could not find an interior point: the flux polytope has empty "
            "interior (it may be a single point or lower-dimensional than detected)."
        )
    return res.x[:n]


# --------------------------------------------------------------------------- #
# CHRR
# --------------------------------------------------------------------------- #
def _model_arrays(
    model: cobra.Model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    S = cobra.util.create_stoichiometric_matrix(model, array_type="dense")
    lb = np.array([r.lower_bound for r in model.reactions], dtype=float)
    ub = np.array([r.upper_bound for r in model.reactions], dtype=float)
    rxn_ids = [r.id for r in model.reactions]
    return S, lb, ub, rxn_ids


def _sample_chrr(
    model: cobra.Model,
    *,
    n_samples: int,
    thinning: int,
    warmup: int,
    seed: int | None,
    tol: float,
    fixed_width_tol: float,
) -> FluxSamplingResult:
    rng = np.random.default_rng(seed)
    S, lb, ub, rxn_ids = _model_arrays(model)
    n_rxns = len(rxn_ids)

    # ---- Fold implicitly-fixed reactions into equalities -------------------
    # A reaction whose FVA width is ~0 is determined by stoichiometry; keeping it
    # as a box constraint would make the reduced polytope lower-dimensional and the
    # MVE solve singular. Fix it at its (unique) value instead.
    fva = flux_variability_analysis(model, fraction_of_optimum=0.0)
    width = (fva["maximum"] - fva["minimum"]).reindex(rxn_ids).to_numpy()
    fixed = width < fixed_width_tol
    fixed_vals = ((fva["maximum"] + fva["minimum"]) / 2.0).reindex(rxn_ids).to_numpy()

    # Equality system: S v = 0, plus v_i = fixed_vals[i] for fixed reactions.
    eq_rows = [S]
    eq_rhs = [np.zeros(S.shape[0])]
    fixed_idx = np.flatnonzero(fixed)
    if fixed_idx.size:
        fix_block = np.zeros((fixed_idx.size, n_rxns))
        fix_block[np.arange(fixed_idx.size), fixed_idx] = 1.0
        eq_rows.append(fix_block)
        eq_rhs.append(fixed_vals[fixed_idx])
    A_eq = np.vstack(eq_rows)
    b_eq = np.concatenate(eq_rhs)

    # Particular solution v0 and nullspace basis N (v = v0 + N x).
    v0, *_ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
    N = null_space(A_eq)
    d = N.shape[1]
    if d == 0:
        # Flux space is a single point — nothing to sample.
        samples = np.tile(v0, (n_samples, 1))
        return FluxSamplingResult(
            samples=pd.DataFrame(samples, columns=rxn_ids),
            method="chrr",
            n_dimensions=0,
            mve_converged=True,
            fixed_reactions=[rxn_ids[i] for i in fixed_idx],
        )

    # ---- Build the full-dimensional polytope in x (free reactions only) -----
    free = ~fixed
    Nf = N[free]
    A_full = np.vstack([Nf, -Nf])
    b_full = np.concatenate([ub[free] - v0[free], v0[free] - lb[free]])
    # Drop rows whose direction vanished in the nullspace (constraint is constant).
    keep = np.linalg.norm(A_full, axis=1) > 1e-10
    A_full = A_full[keep]
    b_full = b_full[keep]

    # ---- Round with the MVE -------------------------------------------------
    x0 = _chebyshev_center(A_full, b_full)
    center, E, converged = max_volume_ellipsoid(A_full, b_full, x0)
    if not converged:
        logger.info(
            "CHRR: MVE rounding did not fully converge (dim=%d); using last "
            "iterate. Increase thinning if samples look correlated.", d
        )

    # Rounded polytope {y : A_r y <= b_r}, which contains the unit ball.
    A_r = A_full @ E
    b_r = b_full - A_full @ center

    # ---- Coordinate hit-and-run on the rounded polytope ---------------------
    y = np.zeros(d)
    s = b_r - A_r @ y                          # maintained slack, > 0 at start
    samples = np.zeros((n_samples, n_rxns))
    rec = 0
    total = warmup + thinning * n_samples
    col_tol = 1e-12

    for stepno in range(1, total + 1):
        j = int(rng.integers(d))
        col = A_r[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = s / col
        pos = col > col_tol
        neg = col < -col_tol
        t_hi = ratios[pos].min() if pos.any() else np.inf
        t_lo = ratios[neg].max() if neg.any() else -np.inf
        if not np.isfinite(t_hi) or not np.isfinite(t_lo):
            # Unbounded coordinate (e.g. an unconstrained loop direction); skip.
            continue
        if t_hi <= t_lo:
            continue
        t = rng.uniform(t_lo, t_hi)
        y[j] += t
        s -= col * t

        if stepno > warmup and (stepno - warmup) % thinning == 0:
            x = center + E @ y
            samples[rec, :] = v0 + N @ x
            rec += 1

    if rec < n_samples:                        # only if many steps were skipped
        samples[rec:, :] = samples[rec - 1, :] if rec else v0

    return FluxSamplingResult(
        samples=pd.DataFrame(samples, columns=rxn_ids),
        method="chrr",
        n_dimensions=d,
        mve_converged=converged,
        fixed_reactions=[rxn_ids[i] for i in fixed_idx],
    )


# --------------------------------------------------------------------------- #
# ACHR (thin wrapper over cobrapy's mature implementation)
# --------------------------------------------------------------------------- #
def _sample_achr(
    model: cobra.Model,
    *,
    n_samples: int,
    thinning: int,
    seed: int | None,
) -> FluxSamplingResult:
    from cobra.sampling import ACHRSampler

    sampler = ACHRSampler(model, thinning=thinning, seed=seed)
    samples = sampler.sample(n_samples)
    samples.index = range(n_samples)
    return FluxSamplingResult(
        samples=samples,
        method="achr",
        n_warmup=int(sampler.n_warmup),
    )


# The public entry point that dispatches to these samplers is
# :func:`raven_toolbox.analysis.sampling.random_sampling` (method="achr"|"chrr").
