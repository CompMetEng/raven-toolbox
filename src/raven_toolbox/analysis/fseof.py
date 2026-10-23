"""Flux Scanning based on Enforced Objective Flux — FSEOF (port + redesign).

FSEOF (Choi et al., Appl Environ Microbiol 2010) finds metabolic-engineering targets
for over-producing a metabolite: enforce an increasing flux toward the target product
while optimising growth, and watch how each reaction's flux responds. This is a port
of RAVEN's ``FSEOF`` with a substantially richer, more robust output (RAVEN's
weaknesses are noted in IMPROVEMENTS, FS1–FS4):

* **Robust trend, not strict monotonicity.** Each reaction's flux is regressed against
  the enforced product flux across the scan; the **slope** is the response and the
  **correlation** (|r|) is a quality score. A reaction is a target if it tracks the
  product cleanly (|r| ≥ ``correlation_threshold``), so one noisy step from LP
  alternative optima does not discard it (and pFBA per step keeps the scan stable).
* **Direction classification.** Targets are labelled ``amplify`` (|flux|
  rises with the product → over-express), ``knockdown`` (|flux| falls), or ``knockout``
  (|flux| → ~0 → delete), like the Amplify, Knockdown and Knockout labels of RAVEN's
  ``FSEOF``.
* **Gene-level view** via :attr:`FSEOFResult.gene_targets`, and the full flux scan is
  retained in :attr:`FSEOFResult.scan` — all as DataFrames, not a printed TSV.
"""
from __future__ import annotations

from dataclasses import dataclass

import cobra
import numpy as np
import pandas as pd
from cobra.exceptions import OptimizationError
from cobra.flux_analysis import pfba
from scipy.stats import linregress


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
) -> FSEOFResult:
    """Run FSEOF for over-production of ``target_rxn``'s product.

    Enforces target flux from ``max_fraction/n_steps`` up to ``max_fraction`` of the
    theoretical maximum in ``n_steps`` steps, maximising growth (``biomass_rxn`` or the
    model's current objective) with pFBA at each step. Returns an :class:`FSEOFResult`.

    Parameters
    ----------
    correlation_threshold:
        Minimum absolute correlation between a reaction's flux and the enforced
        product flux for it to be reported as a target; raise it to keep only
        the most cleanly-trending reactions.
    flux_eps:
        Tolerance below which the theoretical maximum of the target flux is treated as zero
        (the scan then raises). Reaction fluxes are classified as in RAVEN's ``FSEOF``: a
        reaction is a target when some absolute flux exceeds 1e-10, the standard deviation of
        its absolute flux is above 0 and the correlation reaches ``correlation_threshold``;
        it is ``knockout`` when the slope is negative and the last absolute flux is at most
        1e-8, ``knockdown`` for a negative slope otherwise.
    min_target:
        Floor of the enforced-flux range, evenly spaced up to the scan's ceiling
        (``max_fraction`` of the theoretical maximum) in ``n_steps`` steps.
        Defaults to ``None``, which floors the range at
        ``max_fraction of the theoretical maximum / n_steps`` instead. Pass e.g.
        the target reaction's flux at growth-optimum to scan from there.
    """
    with model:  # find the theoretical maximum target flux
        model.objective = target_rxn
        target_opt = model.slim_optimize()
    # slim_optimize returns NaN on an infeasible model; np.isfinite catches that too.
    if target_opt is None or not np.isfinite(target_opt) or target_opt <= flux_eps:
        raise ValueError(f"{target_rxn!r} cannot carry positive flux; nothing to scan.")
    target_max = target_opt * max_fraction
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
            model.reactions.get_by_id(target_rxn).lower_bound = level
            try:
                columns[level] = pfba(model).fluxes
            except OptimizationError:
                break  # enforced flux became infeasible — stop scanning
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
    return table.sort_values("score", ascending=False, ignore_index=True)
