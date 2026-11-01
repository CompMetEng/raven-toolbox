"""Gap-filling against template models.

Four strategies are provided:

- :func:`connect_blocked_reactions` (connectivity MILP): minimum-penalty template
  reactions so that blocked draft reactions can carry flux.
- :func:`fill_gaps_fast_lp` (LP): L1-norm LP per blocked reaction; needs no MILP solver
  (fastGapFill and swiftGapFill).
- :func:`analyse_topology` (no solver): BFS metabolite-producibility pre-screening;
  identifies unreachable metabolites and prunes the candidate reaction pool.
- :func:`fill_gaps_milp` (objective MILP): fewest template reactions that let the
  objective reach a growth floor (Kumar et al. 2007); ``cobra.flux_analysis.gapfill``
  solves a similar problem.

:func:`find_leak_metabolite` tests whether the model itself contains a
stoichiometric leak (a metabolite it can produce or consume for free); it does not fill
a gap against a template. :func:`gap_report` and :func:`print_gap_report`
run a set of connectivity checks and summarise the result.
"""
from raven_toolbox.gapfilling.fast_lp import FastLPResult, fill_gaps_fast_lp
from raven_toolbox.gapfilling.fill import GapFillResult, connect_blocked_reactions
from raven_toolbox.gapfilling.leak import LeakMetaboliteResult, find_leak_metabolite
from raven_toolbox.gapfilling.milp import MILPGapFillResult, fill_gaps_milp
from raven_toolbox.gapfilling.report import (
    GapReportResult,
    MinToConnect,
    gap_report,
    print_gap_report,
)
from raven_toolbox.gapfilling.topological import TopologicalAnalysisResult, analyse_topology

__all__ = [
    "GapFillResult",
    "connect_blocked_reactions",
    "FastLPResult",
    "fill_gaps_fast_lp",
    "TopologicalAnalysisResult",
    "analyse_topology",
    "MILPGapFillResult",
    "fill_gaps_milp",
    "LeakMetaboliteResult",
    "find_leak_metabolite",
    "GapReportResult",
    "MinToConnect",
    "gap_report",
    "print_gap_report",
]
