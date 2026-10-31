"""Flux sampling: RAVEN's ``randomSampling``.

:func:`random_sampling` is the entry point for every sampling method and, like
RAVEN's ``randomSampling``, dispatches on ``method``:

* ``"achr"`` (default) and ``"chrr"`` draw a (near-)uniform Markov-chain sample
  of the polytope interior; they are implemented in
  :mod:`raven_toolbox.analysis.flux_sampling`.
* ``"random_objective"`` is the random-objective method of Bordel et al. (2010,
  PLoS Comput Biol, doi:10.1371/journal.pcbi.1000859), implemented here: each
  sample maximises a small random linear combination of reactions, so every
  sample is an extreme point (vertex) of the flux polytope. cobrapy has no
  equivalent. The method samples diverse optimal states; the uniform flux
  distribution is sampled by ``"achr"`` and ``"chrr"``.

Differences from RAVEN's random-objective method:

* **Reactions usable as random objectives** (``good_reactions``) come from one
  FVA pass, loopless by default (:func:`find_good_reactions`), which drops
  reactions that carry flux only through a loop. RAVEN runs
  ``getAllowedBounds`` and drops reactions that reach 0.999 × the model's
  largest bound, and on GECKO ecModels also the ``usage_prot_`` reactions.
* **``replace_max_bound``** defaults to False; RAVEN's ``replaceBoundsWithInf``
  defaults to true.
* **Output**: a ``samples`` DataFrame shaped samples × reactions (matching
  ``cobra.sampling``), plus the reusable ``good_reactions`` list. RAVEN returns a
  reactions × samples matrix and a vector of reaction indexes.

``seed`` and ``n_objectives`` (default 2) behave as RAVEN's ``seed`` and
``nObjectives``.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

import cobra
import numpy as np
import pandas as pd
from cobra.exceptions import OptimizationError
from cobra.flux_analysis import flux_variability_analysis, pfba
from cobra.util import ProcessPool
from optlang.symbolics import add

from raven_toolbox.analysis.flux_sampling import (
    FluxSamplingResult,
    _sample_achr,
    _sample_chrr,
)

logger = logging.getLogger(__name__)

# Alias for the FluxSamplingResult returned by every sampling method; its
# ``good_reactions`` field is populated for method='random_objective' only.
RandomSamplingResult = FluxSamplingResult


def random_sampling(
    model: cobra.Model,
    n_samples: int = 1000,
    *,
    method: str = "achr",
    seed: int | None = None,
    thinning: int = 100,
    warmup: int = 1000,
    fixed_width_tol: float = 1e-7,
    n_objectives: int = 2,
    good_reactions: Iterable[str] | None = None,
    replace_max_bound: bool = False,
    min_flux: bool = False,
    loopless_good_reactions: bool = True,
    exclude_reactions: Iterable[str] | None = None,
    max_attempts: int = 100,
    suppress_errors: bool = False,
    n_proc: int | None = None,
) -> FluxSamplingResult:
    """Sample ``model``'s flux space with any of the sampling methods.

    Dispatches on ``method``:

    * ``"achr"`` (default): Artificially Centered Hit-and-Run; near-uniform MCMC
      sampling of the polytope interior (wraps :class:`cobra.sampling.ACHRSampler`).
    * ``"chrr"``: Coordinate Hit-and-Run with Rounding; near-uniform MCMC with
      maximum-volume-ellipsoid rounding, for better mixing on thin or ill-conditioned
      polytopes such as enzyme-constrained models.
    * ``"random_objective"``: the random-objective vertex method of Bordel et al.
      (2010): each sample maximises a small random objective, returning a polytope
      vertex.

    The ``"achr"`` and ``"chrr"`` methods draw the (near-)uniform interior distribution;
    ``"random_objective"`` draws diverse vertices. The samples are conditioned on the
    constraints set on the model before the call (for example a biomass lower bound, measured
    fluxes, enzyme-usage bounds).

    Parameters
    ----------
    model : cobra.Model
        Model to sample. Its current bounds and constraints define the sampled space.
    n_samples:
        Number of flux vectors to return.
    method:
        ``"achr"`` (default), ``"chrr"``, or ``"random_objective"``.
    seed:
        Seed for reproducible chains and draws.
    thinning:
        ``achr`` and ``chrr``: Markov-chain steps between recorded samples (default 100).
        The default is calibrated for small models (~200 to 500 reactions). On genome-scale
        models (>2000 reactions) lag-1 autocorrelation remains high even at thinning=100
        (measured: 0.926 on yeast-GEM with 4102 reactions), giving about 12 effective
        samples from 300 stored samples (ESS ≈ n × (1−ρ)/(1+ρ)). Independent chains at
        these default settings also fail to agree with each other: on
        yeast-GEM, 67.5% of reactions fail the Gelman-Rubin R-hat>1.1 convergence
        threshold across 4 independent chains (96.5% fail the stricter R-hat>1.01).
        Default-settings genome-scale output is therefore unconverged for most
        reactions. Options for genome-scale analyses are increasing thinning
        substantially (≥1000), increasing n_samples to compensate, using
        ``method='chrr'`` (raven-toolbox's rounding-based alternative to ACHR), or
        assessing sample quality with an ESS or R-hat diagnostic (see the
        `sampling convergence calibration study
        <https://github.com/CompMetEng/raven-docs/blob/main/docs/parameter-tuning/studies/sampling-convergence-calibration.md>`_
        on raven-docs).
    warmup:
        ``chrr``: burn-in steps discarded before the first recorded sample.
    fixed_width_tol:
        ``chrr``: a reaction whose FVA range is narrower than this is added to
        the equality system as fixed, which keeps the reduced polytope full-dimensional.
    n_objectives, good_reactions, loopless_good_reactions, exclude_reactions, max_attempts, suppress_errors:
        ``random_objective`` only: see the method's parameters; ``good_reactions``
        can be passed back from a previous result to skip the one-off FVA.
    replace_max_bound:
        ``random_objective`` only: replace the largest upper bound with ``+inf`` and the
        smallest negative lower bound with ``-inf`` before sampling, so a reaction whose
        biological maximum exceeds the model's arbitrary cap is not pinned there. Off by default.
    min_flux:
        ``random_objective`` only: re-solve each sample parsimoniously to
        minimise total flux at the optimum, which removes residual loop flux.
    n_proc:
        ``random_objective`` only: worker processes for drawing samples in
        parallel (each sample is an independent LP solve). Defaults to
        ``cobra.Configuration().processes``; 1 draws the samples serially in this process.
        The result is reproducible for a given ``seed`` regardless of ``n_proc``: each
        sample draws from its own independent RNG stream spawned from ``seed``.

    Returns
    -------
    FluxSamplingResult
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    chosen = method.lower()
    if chosen == "achr":
        return _sample_achr(model, n_samples=n_samples, thinning=thinning, seed=seed)
    if chosen == "chrr":
        return _sample_chrr(
            model,
            n_samples=n_samples,
            thinning=thinning,
            warmup=warmup,
            seed=seed,
            tol=1e-9,
            fixed_width_tol=fixed_width_tol,
        )
    if chosen in ("random_objective", "objective"):
        return _random_objective(
            model,
            n_samples,
            n_objectives=n_objectives,
            good_reactions=good_reactions,
            replace_max_bound=replace_max_bound,
            min_flux=min_flux,
            loopless_good_reactions=loopless_good_reactions,
            exclude_reactions=exclude_reactions,
            max_attempts=max_attempts,
            suppress_errors=suppress_errors,
            seed=seed,
            n_proc=n_proc,
        )
    raise ValueError(
        f"Unknown method {method!r}; expected 'achr', 'chrr', or 'random_objective'."
    )


def find_good_reactions(
    model: cobra.Model,
    *,
    flux_tol: float = 1e-9,
    loopless: bool = True,
    exclude_reactions: Iterable[str] | None = None,
) -> list[str]:
    """Reactions usable as random objectives: those that carry non-loop flux.

    A reaction is kept if the larger of ``|minimum|`` and ``|maximum|`` of its FVA range
    exceeds ``flux_tol``. With ``loopless`` (default) the FVA is loopless
    (``cycleFreeFlux``), so reactions that can carry flux only through a
    thermodynamically infeasible cycle have a ~0 loopless range and are dropped. A fixed
    bound threshold would also drop legitimate reactions that reach the model's default
    (for example 1000) bound. With ``loopless=False`` the FVA is faster and keeps any
    flux-carrying reaction (loops included).

    Parameters
    ----------
    model : cobra.Model
        Model whose reactions are tested.
    flux_tol : float
        A reaction is kept when the larger of ``|minimum|`` and ``|maximum|`` of its FVA range
        exceeds this value. The default is 1e-9.
    loopless : bool
        Whether the FVA is loopless (``cycleFreeFlux``). The default is ``True``.
    exclude_reactions : Iterable[str] or None
        Identifiers of reactions dropped from the result. The default ``None`` excludes none.
    """
    fva = flux_variability_analysis(
        model, fraction_of_optimum=0.0,
        loopless="cycleFreeFlux" if loopless else None,
    )
    excluded = set(exclude_reactions or ())
    return [
        rxn_id
        for rxn_id, lo, hi in zip(fva.index, fva["minimum"], fva["maximum"], strict=True)
        if rxn_id not in excluded and max(abs(lo), abs(hi)) > flux_tol
    ]


def _draw_one_sample(
    model: cobra.Model,
    good_rxn_objs: list,
    n_objectives: int,
    min_flux: bool,
    max_attempts: int,
    suppress_errors: bool,
    reaction_ids: list[str],
    seed_seq: np.random.SeedSequence,
    sample_index: int,
) -> np.ndarray:
    """Draw one random-objective sample, retrying on a degenerate objective.

    ``model``'s objective is modified and re-solved in place, so each concurrent
    caller needs its own model copy (``ProcessPool`` provides one for the parallel
    path; the serial path reuses one copy across samples).
    """
    rng = np.random.default_rng(seed_seq)
    for attempt in range(1, max_attempts + 1):
        chosen = rng.choice(len(good_rxn_objs), size=n_objectives, replace=False)
        signs = rng.choice((-1.0, 1.0), size=n_objectives)
        weights = rng.random(n_objectives) * signs
        terms = [w * good_rxn_objs[j].flux_expression
                 for j, w in zip(chosen, weights, strict=True)]
        # add() (not sum()) builds the symbolic objective in one pass; sum()
        # re-canonicalises the optlang expression on every term (O(n^2)).
        model.objective = model.problem.Objective(add(terms), direction="max")
        sol = model.optimize()
        if sol.status == "optimal" and abs(sol.objective_value) > 1e-8:
            fluxes = (pfba(model) if min_flux else sol).fluxes.reindex(reaction_ids)
            if fluxes.isna().any():  # solver returned an unexpected reaction set
                missing = fluxes.index[fluxes.isna()].tolist()
                raise OptimizationError(
                    "solver returned fluxes missing reaction(s) "
                    f"{missing[:5]}; cannot assemble a NaN-free sample matrix."
                )
            return fluxes.to_numpy()
        if attempt == max_attempts:
            if not suppress_errors:
                raise OptimizationError(
                    "Could not find a non-zero, loop-free solution after "
                    f"{max_attempts} attempts for sample {sample_index}. Review the "
                    "model's constraints, or set suppress_errors=True."
                )
            logger.warning("Sample %d: kept a degenerate solution after %d attempts.",
                           sample_index, max_attempts)
            return sol.fluxes.reindex(reaction_ids).to_numpy()


# --------------------------------------------------------------------------- #
# Worker globals for the parallel path (n_proc > 1): populated once per
# worker process by _init_worker and reused for every sample it draws.
# --------------------------------------------------------------------------- #

_WORKER_MODEL: cobra.Model | None = None
_WORKER_GOOD_RXN_IDS: list[str] | None = None
_WORKER_REACTION_IDS: list[str] | None = None
_WORKER_N_OBJECTIVES: int | None = None
_WORKER_MIN_FLUX: bool | None = None
_WORKER_MAX_ATTEMPTS: int | None = None
_WORKER_SUPPRESS_ERRORS: bool | None = None


def _init_worker(
    model: cobra.Model,
    good_rxn_ids: list[str],
    reaction_ids: list[str],
    n_objectives: int,
    min_flux: bool,
    max_attempts: int,
    suppress_errors: bool,
) -> None:
    """Pool initialiser: store this worker's own model copy (deserialised by
    ``ProcessPool``) and everything else needed to draw a sample."""
    global _WORKER_MODEL, _WORKER_GOOD_RXN_IDS, _WORKER_REACTION_IDS
    global _WORKER_N_OBJECTIVES, _WORKER_MIN_FLUX, _WORKER_MAX_ATTEMPTS, _WORKER_SUPPRESS_ERRORS
    _WORKER_MODEL = model
    _WORKER_GOOD_RXN_IDS = good_rxn_ids
    _WORKER_REACTION_IDS = reaction_ids
    _WORKER_N_OBJECTIVES = n_objectives
    _WORKER_MIN_FLUX = min_flux
    _WORKER_MAX_ATTEMPTS = max_attempts
    _WORKER_SUPPRESS_ERRORS = suppress_errors


def _sample_worker(task: tuple[int, np.random.SeedSequence]) -> np.ndarray:
    assert _WORKER_MODEL is not None, "_sample_worker called before _init_worker"
    assert _WORKER_GOOD_RXN_IDS is not None
    assert _WORKER_REACTION_IDS is not None
    assert _WORKER_N_OBJECTIVES is not None
    assert _WORKER_MIN_FLUX is not None
    assert _WORKER_MAX_ATTEMPTS is not None
    assert _WORKER_SUPPRESS_ERRORS is not None
    sample_index, seed_seq = task
    good_rxn_objs = [_WORKER_MODEL.reactions.get_by_id(r) for r in _WORKER_GOOD_RXN_IDS]
    return _draw_one_sample(
        _WORKER_MODEL, good_rxn_objs, _WORKER_N_OBJECTIVES, _WORKER_MIN_FLUX,
        _WORKER_MAX_ATTEMPTS, _WORKER_SUPPRESS_ERRORS, _WORKER_REACTION_IDS,
        seed_seq, sample_index,
    )


def _random_objective(
    model: cobra.Model,
    n_samples: int = 1000,
    *,
    n_objectives: int = 2,
    good_reactions: Iterable[str] | None = None,
    replace_max_bound: bool = False,
    min_flux: bool = False,
    loopless_good_reactions: bool = True,
    exclude_reactions: Iterable[str] | None = None,
    max_attempts: int = 100,
    suppress_errors: bool = False,
    seed: int | None = None,
    n_proc: int | None = None,
) -> FluxSamplingResult:
    """Random-objective sampling of ``model``'s flux space (Bordel et al. 2010).

    Each sample maximises ``sum(w_i * v_i)`` over ``n_objectives`` reactions drawn at
    random from ``good_reactions``, with weights ``w_i = U(0,1) * (±1)`` (a random
    sign per reaction, as in RAVEN). The resulting flux vector is one sample.

    Parameters
    ----------
    n_samples
        Number of flux vectors to return.
    n_objectives
        Reactions combined into each random objective (default 2, as RAVEN's
        ``nObjectives``).
    good_reactions
        Reaction ids eligible as objectives. If ``None`` they are computed once with
        :func:`find_good_reactions` and returned for reuse.
    replace_max_bound
        RAVEN's ``replaceBoundsWithInf``: replace the largest upper bound with
        ``+inf`` (and the smallest negative lower bound with ``-inf``) before
        sampling, so a reaction whose biological maximum exceeds the model's
        arbitrary cap is not pinned at it. Off by default (RAVEN's default is true). It
        applies only to the sampling phase (``good_reactions`` is always found on
        the finite bounds), and it can open unbounded directions through loops
        that show up as large fluxes in non-objective reactions; with ``min_flux`` the
        loop flux is removed.
    min_flux
        After maximising the random objective, re-solve parsimoniously
        (:func:`cobra.flux_analysis.pfba`) to minimise total flux at that optimum,
        which removes residual loops from each sample.
    loopless_good_reactions, exclude_reactions
        Passed to :func:`find_good_reactions` when ``good_reactions`` is ``None``
        (loopless loop detection is on by default).
    max_attempts, suppress_errors
        A sample is retried if the random objective is degenerate (zero flux). After
        ``max_attempts`` failures this raises, unless ``suppress_errors`` (then the
        degenerate solution is kept with a warning).
    seed
        Seed for reproducible objective draws. Sample ``i`` always draws from
        the ``i``-th child stream of ``numpy.random.SeedSequence(seed)``,
        independent of every other sample and of ``n_proc``; a given seed
        therefore reproduces the same samples whether drawn serially or in parallel.
    n_proc
        Worker processes for drawing samples in parallel. Defaults to
        ``cobra.Configuration().processes``; set to 1 to sample serially in
        this process. Each sample is an independent LP solve (a fresh random
        objective, then optimise), so this parallelises via
        ``cobra.util.ProcessPool``, the same pool that cobrapy's own
        ``flux_variability_analysis`` and ``single_reaction_deletion`` use.

    Returns
    -------
    RandomSamplingResult
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    model = model.copy()

    if model.slim_optimize(error_value=None) is None:
        raise ValueError(
            "The model has no feasible solution, likely due to incompatible constraints."
        )

    # good_reactions must be found on the finite bounds (FVA cannot handle inf),
    # before any bound replacement.
    if good_reactions is None:
        good_reactions = find_good_reactions(
            model, loopless=loopless_good_reactions,
            exclude_reactions=exclude_reactions,
        )
    good_reactions = list(good_reactions)

    if replace_max_bound:
        max_ub = max(r.upper_bound for r in model.reactions)
        min_lb = min(r.lower_bound for r in model.reactions)
        for r in model.reactions:
            if r.upper_bound == max_ub:
                r.upper_bound = float("inf")
            if min_lb < 0 and r.lower_bound == min_lb:
                r.lower_bound = float("-inf")

    if len(good_reactions) < n_objectives:
        raise ValueError(
            f"Only {len(good_reactions)} usable reactions found, need at least "
            f"n_objectives={n_objectives}. Check the model's constraints."
        )

    reaction_ids = [r.id for r in model.reactions]
    samples = np.zeros((n_samples, len(reaction_ids)))

    # One independent, reproducible RNG stream per sample -- spawned once up
    # front so sample i's draws never depend on execution order, only on i.
    # That's what lets serial (n_proc=1) and parallel (n_proc>1) produce
    # identical output for the same seed.
    tasks = list(enumerate(np.random.SeedSequence(seed).spawn(n_samples)))

    if n_proc is None:
        n_proc = cobra.Configuration().processes
    n_proc = max(1, int(n_proc))

    if n_proc == 1:
        good_rxn_objs = [model.reactions.get_by_id(r) for r in good_reactions]
        for i, seed_seq in tasks:
            samples[i, :] = _draw_one_sample(
                model, good_rxn_objs, n_objectives, min_flux, max_attempts,
                suppress_errors, reaction_ids, seed_seq, i,
            )
    else:
        # ProcessPool (cobra.util.process_pool) handles serialising the
        # model to each worker -- including a Windows-specific performance
        # workaround -- so `model` is passed as-is, not pre-pickled.
        chunk = max(1, n_samples // (n_proc * 4))
        with ProcessPool(
            n_proc, initializer=_init_worker,
            initargs=(
                model, good_reactions, reaction_ids, n_objectives, min_flux,
                max_attempts, suppress_errors,
            ),
        ) as pool:
            for i, fluxes in enumerate(pool.map(_sample_worker, tasks, chunksize=chunk)):
                samples[i, :] = fluxes

    return FluxSamplingResult(
        samples=pd.DataFrame(samples, columns=reaction_ids),
        method="random_objective",
        good_reactions=good_reactions,
    )
