"""ftINIT step schedule.

ftINIT runs a short sequence of MILP steps. Each step (:class:`InitStep`) chooses which
reaction categories to hold out of the problem (``ignore_mask``, an 8-bit pattern over
:class:`raven_toolbox.init.ReactionMasks`), whether to drop positive reversibles and allow
metabolite secretion, and how to treat the reactions turned on by previous steps
(``'ignore'`` for the first step, ``'essential'`` to fix them on).
:func:`get_init_steps` builds the standard schedules.

The default ``'1+1'`` has two steps. Step 1 holds out the exchange, import, transport,
spontaneous, extracellular and custom categories, so it decides the GPR-associated reactions
and the remaining GPR-less reactions. Step 2 holds out only the exchange and spontaneous
reactions, with the step-1 reactions fixed as essential. ``'full'`` is the single-MILP
classic-tINIT variant (nothing held out).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

# 8-bit ignore patterns (exchange, import, simple-transp, adv-transp, spontaneous,
# extracellular, custom, no-GPR) — see ReactionMasks.
_ALL_NO_GPR_KEPT = (1, 1, 1, 1, 1, 1, 1, 0)  # hold out every GPR-less category but "all no-GPR"
_EXCH_SPONT = (1, 0, 0, 0, 1, 0, 0, 0)        # hold out only exchange + spontaneous
_NONE = (0, 0, 0, 0, 0, 0, 0, 0)

# Per-step MILP run schedule (RAVEN INITStepDesc MILPParams + AbsMIPGaps). Each run is
# (relative mip_gap, time_limit seconds, absolute mip gap). RAVEN runs a step up to this
# many times: run 1 at the flat relative gap, later runs at max(mip_gap, abs_gap/|obj|),
# breaking as soon as the achieved gap already meets the next target. The final step's
# quick 5 s first run just estimates the (near-zero) objective so the absolute gap can be
# converted to a sane relative one for the escalation runs.
_STEP_RUNS = (
    {"mip_gap": 0.0004, "time_limit": 120, "abs_gap": 10.0},
    {"mip_gap": 0.0030, "time_limit": 5000, "abs_gap": 20.0},
)
_FINAL_STEP_RUNS = (
    {"mip_gap": 0.0004, "time_limit": 5, "abs_gap": 10.0},
    {"mip_gap": 0.0004, "time_limit": 120, "abs_gap": 10.0},
    {"mip_gap": 0.0030, "time_limit": 5000, "abs_gap": 20.0},
)
# RAVEN 'full' series: one run (INITStepDesc with MIPGap 0.0004, TimeLimit 10000).
_FULL_RUNS = ({"mip_gap": 0.0004, "time_limit": 10000, "abs_gap": 10.0},)


@dataclass
class InitStep:
    """One ftINIT MILP step."""

    how_to_use_prev: str = "essential"          # 'ignore' | 'essential'
    ignore_mask: tuple[int, ...] = _ALL_NO_GPR_KEPT
    pos_rev_off: bool = False                    # drop positive reversibles from the problem
    allow_met_secr: bool = False                 # relax S·v = 0 to ≥ 0
    mets_to_ignore: Sequence[str] = field(default_factory=tuple)  # met names zeroed from S (e.g. H2O)
    # RAVEN's per-step MILPParams + AbsMIPGaps escalation schedule (see _STEP_RUNS). Empty
    # → a single solve at the caller's mip_gap / mip_gap_abs (ad-hoc steps).
    milp_runs: tuple[dict, ...] = ()


def get_init_steps(series: str = "1+1", *, mets_to_ignore: Sequence[str] = ()) -> list[InitStep]:
    """Return the step schedule for a named ftINIT ``series`` (RAVEN ``getINITSteps``).

    ``'1+1'`` (default) has two steps; ``'2+1'`` has three steps, the first with positive
    reversibles dropped and metabolite secretion allowed; ``'1+0'`` and ``'2+0'`` omit the
    final step, which brings in the held-out GPR-less reactions; ``'full'`` is a single MILP.
    ``mets_to_ignore`` are metabolite names removed from the stoichiometry in each step (for
    example H2O and H+); the ``'full'`` series ignores none, as RAVEN's ``getINITSteps`` does.
    An unknown ``series`` raises :class:`ValueError`.

    Parameters
    ----------
    series : str
        Name of the schedule, default ``"1+1"``. One of ``"1+1"``, ``"2+1"``, ``"1+0"``,
        ``"2+0"`` and ``"full"``; any other value raises :class:`ValueError`.
    mets_to_ignore : Sequence[str]
        Keyword-only; default empty. Names of the metabolites removed from the stoichiometry
        in each step. The ``"full"`` series ignores none, whatever the value.
    """
    m = tuple(mets_to_ignore)
    s1 = InitStep("ignore", _ALL_NO_GPR_KEPT, mets_to_ignore=m, milp_runs=_STEP_RUNS)
    s1_posrev = InitStep("ignore", _ALL_NO_GPR_KEPT, pos_rev_off=True, allow_met_secr=True,
                         mets_to_ignore=m, milp_runs=_STEP_RUNS)
    s2_all = InitStep("essential", _ALL_NO_GPR_KEPT, mets_to_ignore=m, milp_runs=_STEP_RUNS)
    s_final = InitStep("essential", _EXCH_SPONT, mets_to_ignore=m, milp_runs=_FINAL_STEP_RUNS)

    if series == "1+1":
        return [s1, s_final]
    if series == "2+1":
        return [s1_posrev, s2_all, s_final]
    if series == "1+0":
        return [s1]
    if series == "2+0":
        return [s1_posrev, s2_all]
    if series == "full":
        return [InitStep("ignore", _NONE, mets_to_ignore=(), milp_runs=_FULL_RUNS)]
    raise ValueError(f"Unknown ftINIT series {series!r}; expected 1+1, 2+1, 1+0, 2+0, full.")
