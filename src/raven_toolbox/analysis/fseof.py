"""Flux Scanning based on Enforced Objective Flux (FSEOF).

FSEOF (Choi et al., Appl Environ Microbiol 2010) finds metabolic-engineering targets
for over-producing a metabolite: enforce an increasing flux toward the target product
while optimising growth, and watch how each reaction's flux responds. This follows
RAVEN's ``FSEOF`` and returns tables, where RAVEN prints a list:

* **Trend of the absolute flux.** Each reaction's absolute flux is regressed against the
  enforced product flux across the scan; the **slope** is the response and the
  **correlation** (|r|) is a quality score. A reaction is a target when its flux across the
  scan is not constant and |r| >= ``correlation_threshold``. Every step is solved with pFBA.
* **Direction classification.** Targets are labelled ``amplify`` (the absolute flux
  rises with the product), ``knockdown`` (it falls) or ``knockout`` (it falls to at most
  1e-8), like the Amplify, Knockdown and Knockout labels of RAVEN's ``FSEOF``.
* **Gene-level view** via :attr:`FSEOFResult.gene_targets`, and the full flux scan in
  :attr:`FSEOFResult.scan`, both as DataFrames.
"""
from __future__ import annotations

from dataclasses import dataclass

import cobra
import numpy as np
import pandas as pd
from cobra.exceptions import OptimizationError
from cobra.flux_analysis import pfba
from scipy.stats import linregress

# pFBA keeps the objective at this fraction of the optimum found at the same level, as
# RAVEN's solveLP does when it minimises the flux sum (``solution.f*0.999999``). A solver
# reports its optimum to within its feasibility tolerance, so a second solve that demands
# the full value can be infeasible on that solver (HiGHS at the top level of ecYeastGEM).
_PFBA_FRACTION = 0.999999


@dataclass
class FSEOFResult:
    """FSEOF output.

    ``scan`` is reactions × enforced-flux-levels (the full flux scan); ``enforced`` are
    the enforced target fluxes; ``targets`` is the classified per-reaction table
    (sorted by score). :attr:`gene_targets` aggregates targets to genes.
    """

    scan: pd.DataFrame
    enforced: list[float]
    targets: pd.DataFrame

    @property
    def amplification(self) -> pd.DataFrame:
        """Targets classified ``"amplify"`` (flux rises with the enforced product)."""
        return self.targets[self.targets["target_type"] == "amplify"].reset_index(drop=True)

    @property
    def knockout(self) -> pd.DataFrame:
        """Targets classified ``"knockout"`` or ``"knockdown"`` (flux falls toward
        or to zero as the enforced product flux rises)."""
        mask = self.targets["target_type"].isin(["knockout", "knockdown"])
        return self.targets[mask].reset_index(drop=True)

    @property
    def gene_targets(self) -> pd.DataFrame:
        """Per-gene aggregation: the target reactions each gene is associated with."""
        rows = []
        for _, t in self.targets.iterrows():
            for gene in t["genes"]:
                rows.append({"gene": gene, "reaction": t["reaction"],
                             "target_type": t["target_type"], "slope": t["slope"]})
        if not rows:
            return pd.DataFrame(columns=["gene", "target_type", "reactions", "max_abs_slope"])
        df = pd.DataFrame(rows)
        agg = df.groupby("gene").agg(
            target_type=("target_type", lambda s: ";".join(sorted(set(s)))),
            reactions=("reaction", lambda s: ";".join(sorted(set(s)))),
            max_abs_slope=("slope", lambda s: float(np.max(np.abs(s)))),
        ).reset_index()
        return agg.sort_values("max_abs_slope", ascending=False, ignore_index=True)


def fseof(
    model: cobra.Model,
    target_rxn: str,
    *,
    biomass_rxn: str | None = None,
    n_steps: int = 10,
    max_fraction: float = 0.9,
    correlation_threshold: float = 0.9,
    flux_eps: float = 1e-6,
    min_target: float | None = None,
    equality: bool = False,
) -> FSEOFResult:
    """Run FSEOF for over-production of ``target_rxn``'s product.

    Enforces the target flux in ``n_steps`` evenly spaced levels, from
    ``max_fraction`` / ``n_steps`` times the theoretical maximum up to ``max_fraction`` times
    the theoretical maximum (the target flux of the pFBA solution that maximises it),
    maximising growth (``biomass_rxn`` or the model's current objective) with pFBA at each
    step. Returns an :class:`FSEOFResult`.

    Parameters
    ----------
    model : cobra.Model
        Model to scan. It is restored after the call.
    target_rxn : str
        Identifier of the reaction whose flux is enforced. The theoretical maximum is found by
        maximising this reaction; a ``ValueError`` is raised when that maximum is not above
        ``flux_eps``.
    biomass_rxn : str or None
        Identifier of the reaction maximised at each enforced level. The default ``None``
        keeps the model's current objective.
    n_steps : int
        Number of enforced flux levels. The default is 10. Levels are evenly spaced; the scan
        raises ``RuntimeError`` when fewer than two levels are feasible.
    max_fraction : float
        Fraction of the theoretical maximum target flux at which the highest level is set.
        The default is 0.9.
    correlation_threshold:
        Minimum absolute Pearson correlation between the absolute flux of a reaction and
        the enforced product flux for the reaction to be reported as a target; a higher
        value keeps fewer reactions, those with the most consistent trend.
    flux_eps:
        Tolerance below which the theoretical maximum of the target flux is treated as zero
        (the scan then raises). It gates only that check.
    min_target:
        Lowest enforced flux. The ``n_steps`` levels are evenly spaced from ``min_target``
        up to the scan's ceiling (``max_fraction`` times the theoretical maximum).
        Defaults to ``None``, in which case the lowest level is the ceiling divided by
        ``n_steps``. Passing the flux of the target reaction at the growth optimum starts
        the scan at that flux.
    equality:
        How the enforced flux is imposed. ``False`` (the default) sets it as the lower bound
        of the target reaction, as RAVEN's ``FSEOF`` does; ``True`` fixes the target flux at
        the level (lower and upper bound), as GECKO's ``ecFSEOF`` does.

    Notes
    -----
    Reaction fluxes are classified as in RAVEN's ``FSEOF``: a reaction is a target when some
    absolute flux exceeds 1e-10, the standard deviation of its absolute flux is above 0 and
    the correlation reaches ``correlation_threshold``; it is ``knockout`` when the slope is
    negative and the last absolute flux is at most 1e-8, ``knockdown`` for a negative slope
    otherwise.

    Each level is solved in two steps as in RAVEN: the objective is maximised, then the
    flux sum is minimised with the objective held at 0.999999 of its optimum at that
    level. A level at which the first step is infeasible ends the scan.

    The top of the scan is ``max_fraction`` of the target flux in the pFBA solution that
    maximises the target, as in RAVEN's ``FSEOF`` and GECKO's ``ecFSEOF``.

    Differences from ``FSEOF.m``: the scan stops at the first infeasible enforced level,
    so the last absolute flux used for the knockout label is that of the last feasible
    level; ``targets`` is sorted by ``|slope| * |r|`` (equal scores in model order) and is
    one table, whereas ``FSEOF.m`` prints the targets grouped by the labels Amplify,
    Knockdown and Knockout; it is not split into reaction and transport targets.
    """
    with model:  # find the theoretical maximum target flux
        model.objective = target_rxn
        target_opt = model.slim_optimize()
    # slim_optimize returns NaN on an infeasible model; np.isfinite catches that too.
    if target_opt is None or not np.isfinite(target_opt) or target_opt <= flux_eps:
        raise ValueError(f"{target_rxn!r} cannot carry positive flux; nothing to scan.")
    with model:  # the maximum is read from the pFBA solution, as RAVEN's solveLP(model, 1) gives it
        model.objective = target_rxn
        target_pfba = pfba(model, fraction_of_optimum=_PFBA_FRACTION).fluxes[target_rxn]
    target_max = float(target_pfba) * max_fraction
    if min_target is None:
        levels = [target_max * (i + 1) / n_steps for i in range(n_steps)]
    else:
        levels = list(np.linspace(min_target, target_max, n_steps))

    columns: dict[float, pd.Series] = {}
    enforced: list[float] = []
    for level in levels:
        with model:
            if biomass_rxn is not None:
                model.objective = biomass_rxn
            target = model.reactions.get_by_id(target_rxn)
            if equality:
                target.bounds = (level, level)
            else:
                target.lower_bound = level
            try:
                columns[level] = pfba(model, fraction_of_optimum=_PFBA_FRACTION).fluxes
            except OptimizationError:
                break  # enforced flux became infeasible: stop scanning
            enforced.append(level)
    if len(enforced) < 2:
        raise RuntimeError("FSEOF needs at least two feasible enforced-flux levels.")

    scan = pd.DataFrame(columns)
    targets = _classify(model, scan, np.asarray(enforced), correlation_threshold)
    return FSEOFResult(scan=scan, enforced=enforced, targets=targets)


_TRACK_EPS = 1e-10  # a track must exceed this somewhere to be fitted, as in FSEOF.m
_KNOCKOUT_EPS = 1e-8  # last absolute flux at or below this is a knockout, as in FSEOF.m


def _classify(model, scan, enforced, corr_threshold) -> pd.DataFrame:
    rows = []
    for rxn in model.reactions:
        flux = scan.loc[rxn.id, enforced.tolist() if hasattr(enforced, "tolist") else enforced]
        flux = flux.to_numpy(dtype=float)
        initial, final = flux[0], flux[-1]
        # Slope and correlation are taken on absolute flux, as in RAVEN's FSEOF, so a
        # reversible reaction running backwards counts by its magnitude.
        magnitude = np.abs(flux)
        if not (magnitude > _TRACK_EPS).any() or magnitude.std() <= 0:
            continue
        fit = linregress(enforced, magnitude)
        slope, corr = float(fit.slope), float(fit.rvalue)
        if abs(corr) < corr_threshold or slope == 0:
            continue
        if slope > 0:
            ttype = "amplify"
        elif abs(final) <= _KNOCKOUT_EPS:
            ttype = "knockout"
        else:
            ttype = "knockdown"
        rows.append({
            "reaction": rxn.id,
            "name": rxn.name,
            "subsystem": rxn.subsystem,
            "gene_reaction_rule": rxn.gene_reaction_rule,
            "genes": sorted(g.id for g in rxn.genes),
            "target_type": ttype,
            "slope": slope,
            "correlation": corr,
            "initial_flux": initial,
            "final_flux": final,
            "score": abs(slope) * abs(corr),
        })
    table = pd.DataFrame(rows, columns=[
        "reaction", "name", "subsystem", "gene_reaction_rule", "genes",
        "target_type", "slope", "correlation", "initial_flux", "final_flux", "score",
    ])
    # A stable sort keeps reactions with equal scores in model order, as MATLAB's sort does.
    return table.sort_values("score", ascending=False, ignore_index=True, kind="stable")
