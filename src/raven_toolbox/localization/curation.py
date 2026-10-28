"""Curation-priority scoring: which compartment assignments / transports to review by hand.

:func:`curation_priority` attaches a 0-1 "please-check-me" score to every reaction placement and every
added transport in an :class:`~raven_toolbox.localization.AssignmentProposal`, so a curator can sort
thousands of calls and review only the highest-value few percent. It is a post-processor — no new solve
except an optional gated essentiality FBA — combining several cheap signals that each flag a *distinct*
way a placement can be wrong. See the `curation-priority signals study
<https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/localization/curation_priority_signals.md>`_ on raven-gecko-parity for the full catalogue;
this ships the recommended first composite:

* **override** — the reaction was placed *against* its gene's top DeepLoc compartment, and that top
  compartment is not among the reaction's placements (functionality overrode confident evidence — the
  method's headline risk). Graded by ``score(top) - score(assigned)``, robust to organelle
  sub-compartment splits (e.g. mito matrix vs membrane) that a naive top1-top2 margin would mask;
* **sandwich** — the reaction's substrates are imported *into* its compartment and its products
  exported *out*, so it may belong in the neighbouring compartment (the seed motif). Graded by
  ``min(fraction of substrates imported, fraction of products exported)`` so a pure pass-through scores
  high and an incidental single shared transport scores low;
* **no_evidence** — no scored gene; compartment chosen by connectivity alone;
* **neighbour_outlier** — most of the reaction's metabolic neighbours sit in a *different* compartment
  (needs >=3 neighbours; ramped from a bare majority so a 51 % split is not treated like a 100 % one);
* **gapfill** — a reaction pulled from the universal model (inherently uncertain);
* **impermeant** (transports) — a transport of a species cells cannot freely move across a membrane:
  CoA-thioesters, acyl-ACP, tRNA-bound cargo, and free dinucleotide/flavin cofactors (NAD(P)(H),
  FAD/FMN) that in reality require a shuttle;
* **essential** (gated) — removing the reaction drops the *materialised* model's growth to near zero,
  *amplifying* an already-uncertain call.

Signals combine by **noisy-OR** so corroboration ranks a doubly-flagged call above either alone, with
essentiality as a multiplicative high-stakes gate. Currency metabolites (ATP/NAD(P)H/H+/H2O/CoA/Pi/...)
are excluded from the topology signals — without that the sandwich signal fires on nearly everything.
Currency/cofactor matching normalises the metabolite base key (strips an ``M_`` prefix and a bracketed
``[compartment]`` tag) so the exclusion also holds on id-keyed models, not only clean-name ones.

The two **topology** signals (sandwich, neighbour_outlier) are additionally multiplied by evidence
*weakness* ``1 - support(assigned)``: in this assignment method, importing substrates and exporting
products is the *normal* consequence of moving a reaction into an organelle, so those motifs describe
most placements and carry information only when the placement is *also* evidentially unsupported. This
gate is what makes the ranking selective (on genome-scale yeast it takes the flagged set from ~70 % of
reactions to ~20 %, with a clean separable top few percent) — surfacing placements that are both
structurally anomalous and unsupported by localisation evidence, which is the actual signature of a
questionable call. ``override`` is already evidence-based; ``no_evidence``/``gapfill``/``impermeant``
are independent of it.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from typing import Any

import cobra

from raven_toolbox.localization.assign import (
    AssignmentProposal,
    _base_met,
    _check_unique_keys,
    _rxn_compartment,
)
from raven_toolbox.localization.scores import LocalizationScores

__all__ = ["curation_priority"]

# Currency / cofactor metabolites, matched case-insensitively against the *normalised* base key. These
# are legitimately transported and near-ubiquitous, so they must not drive the topology signals.
_CURRENCY = {
    "atp", "adp", "amp", "gtp", "gdp", "gmp", "utp", "udp", "ctp", "cdp", "itp", "idp",
    "nad", "nad+", "nadh", "nadp", "nadp+", "nadph", "fad", "fadh2", "fmn", "fmnh2",
    "h", "h+", "proton", "h2o", "water", "co2", "o2", "hco3", "h2o2",
    "pi", "phosphate", "ppi", "diphosphate", "pppi", "coa", "coenzyme a",
    "nh3", "nh4", "nh4+", "ammonium", "ammonia", "so3", "so4", "hs", "h2s",
    "na+", "k+", "cl-", "mg2+", "ca2+", "fe2+", "fe3+", "zn2+", "cu2+", "mn2+",
}
# Reduced dinucleotide / flavin cofactors: currency for topology, but ALSO impermeant as free molecules
# (they cross membranes only via a shuttle), so they are flagged when transported. Matched on the
# normalised base key (exact), independent of the currency exclusion above.
_IMPERMEANT_COFACTORS = {"nad", "nadh", "nadp", "nadph", "fad", "fadh2", "fmn", "fmnh2"}
# substrings marking other impermeant cargo (thioesters, carrier-bound acyls, tRNA-bound species)
_IMPERMEANT_TOKENS = ("coa", "-coa", "acp", "acyl-carrier", "trna", "-trna", "thioester")

_WEIGHTS = {"override": 0.9, "sandwich": 0.85, "impermeant": 0.85, "gapfill": 0.8,
            "no_evidence": 0.7, "neighbour_outlier": 0.6}
_BETA_ESSENTIAL = 0.6
_MIN_NEIGHBOURS = 3            # a neighbour-outlier verdict needs at least this many neighbours
_AFFECTED_CAP = 12            # max coupled ids listed in the `affected` column (n_affected has the full count)
# flags that describe *what* is wrong (drive the action text); "essential" is a stakes multiplier only
_ACTION_KEYS = frozenset(_WEIGHTS)


def _norm_key(k: str) -> str:
    """Normalise a metabolite base key for currency/cofactor matching: lowercase, strip a leading
    ``M_`` (SBML-retained ids) and any bracketed ``[compartment]`` tag, so ``M_atp_c`` and ``atp[c]``
    both reduce to ``atp``."""
    k = str(k).strip().lower()
    if k.startswith("m_"):
        k = k[2:]
    return re.sub(r"\[.*?\]", "", k).strip()


def _is_currency(base_key: str, extra: set[str]) -> bool:
    k = _norm_key(base_key)
    return k in _CURRENCY or k in extra


def _is_impermeant(base_key: str, name: str) -> bool:
    k = _norm_key(base_key)
    if k in _IMPERMEANT_COFACTORS:
        return True
    nm = str(name).lower()
    return any(tok in nm for tok in _IMPERMEANT_TOKENS) or any(tok in k for tok in _IMPERMEANT_TOKENS)


def _score(df, g: str, c: str) -> float:
    if g not in df.index or c not in df.columns:
        return 0.0
    v = df.at[g, c]
    return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _noisy_or(flags: dict[str, float]) -> float:
    prod = 1.0
    for name, s in flags.items():
        prod *= 1.0 - _WEIGHTS.get(name, 0.7) * max(0.0, min(1.0, s))
    return 1.0 - prod


def curation_priority(
    model: cobra.Model,
    proposal: AssignmentProposal,
    scores: LocalizationScores,
    *,
    base_metabolite: Callable[[cobra.Metabolite], str] | None = None,
    default_compartment: str = "c",
    universal: cobra.Model | None = None,
    currency_metabolites: Iterable[str] | None = None,
    check_essential: bool = True,
    essential_candidate_threshold: float = 0.2,
    min_growth: float | None = None,
    include_curated: bool = False,
):
    """Rank reaction placements and added transports by how likely each needs manual curation.

    Parameters mirror :func:`assign_compartments` where they overlap (``model`` is the draft
    that was placed; pass the same ``base_metabolite``/``universal``). Returns a
    :class:`pandas.DataFrame` sorted by ``priority`` (descending), one row per *(reaction, compartment)*
    placement and per flagged added transport, with columns ``target`` (id), ``type``
    (``"placement"``/``"gapfill"``/``"transport"``), ``compartment``, ``priority`` (0-1), ``flags`` (the
    firing signal names, strongest first), ``action`` (the concrete thing to check), ``n_affected`` and
    ``affected``. Dual-localised reactions yield one row per compartment. Surface the top few percent.

    ``n_affected``/``affected`` give the **one-hop coupling** of each row — the other placed reactions
    that curating this one would touch, because they share a non-currency metabolite with it at the same
    compartment (for a placement) or depend on the transported species at either end of the bridge (for
    a transport). ``n_affected`` is the full count (sort by it to find high-impact curations that
    warrant care or a batched review); ``affected`` lists up to a dozen ids, the ones also flagged in
    this table first. The coupling is not transitive — each coupled reaction has its own blast radius.

    ``check_essential`` runs a single-reaction-deletion essentiality check (an FBA per candidate) on the
    already-flagged placements only (base priority ``>= essential_candidate_threshold``) and multiplies
    their score by ``1 + 0.6`` when removing them drops the *materialised* model's growth to near zero.
    Essentiality is measured against the applied model's own achievable growth, not the assignment's
    heuristic growth floor, and is skipped entirely if the materialised model cannot beat that floor
    (where the test could not discriminate) — pass ``min_growth`` to override the floor used for this
    comparison (default: the proposal's own floor). Set ``check_essential=False`` to skip the only
    non-cheap signal.

    ``currency_metabolites`` extends the built-in currency/cofactor set (ATP, NAD(P)H, H+, H2O, CoA, Pi,
    ...) that is excluded from the topology signals — pass ids or names for any organism-specific pool
    that should also count as freely available.

    A reaction whose ``localization`` confidence has been marked ``curated`` (via
    :func:`~raven_toolbox.confidence.mark_curated`, e.g. after a curator settled its placement) is
    dropped from the queue, closing the score -> review -> curate loop so a settled reaction stops
    resurfacing. Pass ``include_curated=True`` to score it anyway.
    """
    import pandas as pd

    base = base_metabolite if base_metabolite is not None else _base_met
    _check_unique_keys(model.metabolites, base)
    df = scores.df
    extra_currency = {_norm_key(x) for x in (currency_metabolites or ())}
    unplaced = set(proposal.unplaced_reactions)
    # gap-fills that are also (somehow) in placements are scored as placements, not duplicated as gapfill
    added_reactions = [r for r in proposal.added_reactions if r not in proposal.placements]

    # Reactions a curator has settled (localization facet marked "curated"): drop them from the queue so
    # they stop resurfacing, closing the score -> review -> curate loop. include_curated keeps them.
    curated_localization: set[str] = set()
    if not include_curated:
        from raven_toolbox.confidence import get_confidence
        for rid in {*proposal.placements, *added_reactions}:
            if rid not in model.reactions:
                continue
            entry = get_confidence(model.reactions.get_by_id(rid)).facets.get("localization")
            if entry is not None and entry.level == "curated":
                curated_localization.add(rid)

    def _cur(b):
        return _is_currency(b, extra_currency)

    # full placement lists (ALL compartments, not just the first) for every movable reaction
    placements_of = {rid: list(cs) for rid, cs in proposal.placements.items() if cs}
    # rid -> set of compartments it occupies (movable from the proposal, pinned from the model), so the
    # neighbour signal sees the whole materialised layout including dual-localised reactions.
    comps_of: dict[str, set[str]] = {rid: set(cs) for rid, cs in placements_of.items()}
    for r in model.reactions:
        if r.id not in comps_of and not r.boundary:
            c = _rxn_compartment(r)
            if c is not None:
                comps_of[r.id] = {c}

    # metabolite base index: base key -> representative name (impermeant matching) and the placed
    # reactions touching it. Indexes ALL bases (the neighbour signal only ever looks up non-currency
    # ones, but the "affected" coupling below must resolve any transported species, cofactors included).
    base_name: dict[str, str] = {}
    touch: dict[str, set[str]] = {}
    for r in model.reactions:
        if r.boundary:
            continue
        for m in r.metabolites:
            b = base(m)
            base_name.setdefault(b, m.name or m.id)
            if r.id in comps_of:
                touch.setdefault(b, set()).add(r.id)

    transported_at = set(proposal.added_transports)  # {(base, compartment)}
    nonbases_of: dict[str, list[str]] = {}  # movable rid -> its non-currency metabolite base keys

    def _gene_consensus(rid):
        r = model.reactions.get_by_id(rid)
        genes = [g.id for g in r.genes if g.id in df.index]
        if not genes:
            return None
        return {c: max(_score(df, g, c) for g in genes) for c in df.columns}

    rows: list[dict[str, Any]] = []

    # ---- placement signals: one row per (reaction, compartment) ----
    for rid, cs in placements_of.items():
        if rid in curated_localization:
            continue
        r = model.reactions.get_by_id(rid)
        noncur = [(base(m), coeff) for m, coeff in r.metabolites.items() if not _cur(base(m))]
        subs = [b for b, coeff in noncur if coeff < 0]
        prods = [b for b, coeff in noncur if coeff > 0]
        nonbases_of[rid] = [b for b, _c in noncur]
        per = _gene_consensus(rid)
        top_c = max(per, key=per.get) if per else None
        placement_set = set(cs)
        # neighbours are compartment-independent; compute once per reaction
        neigh = set()
        for b, _c in noncur:
            neigh |= touch.get(b, set())
        neigh.discard(rid)

        for comp in cs:
            flags: dict[str, float] = {}
            # Evidence *weakness* at the assigned compartment: 0 when a gene confidently predicts comp,
            # 1 when nothing supports it. The topology signals below are multiplied by it, because in
            # this method importing substrates / exporting products is the NORMAL consequence of moving
            # a reaction into an organelle -- it is only *suspicious* when the placement is also
            # evidentially unsupported. Without this gate the sandwich/neighbour signals fire on the
            # ~86% of placements DeepLoc actively supports and the ranking is uninformative.
            weak = 1.0 - (per.get(comp, 0.0) if per else 0.0)
            # B4 no evidence
            if rid in unplaced:
                flags["no_evidence"] = 1.0
            # B1 override: fire only if the gene's top compartment is NOT one of this reaction's
            # placements (so genuine dual-localisation does not false-fire), graded by top-vs-assigned.
            if per and top_c is not None and top_c not in placement_set and per[top_c] > 0:
                ov = max(0.0, min(1.0, per[top_c] - per.get(comp, 0.0)))
                if ov > 0:
                    flags["override"] = ov
            # C1 sandwich (graded x evidence-weak): substrates imported into comp AND products exported
            if subs and prods and weak > 0:
                fs = sum((b, comp) in transported_at for b in subs) / len(subs)
                fp = sum((b, comp) in transported_at for b in prods) / len(prods)
                sw = min(fs, fp) * weak
                if sw > 0:
                    flags["sandwich"] = sw
            # D1 neighbour / pathway outlier (>=3 neighbours; ramp from a bare majority x evidence-weak)
            if len(neigh) >= _MIN_NEIGHBOURS and weak > 0:
                other = sum(1 for n in neigh if comp not in comps_of.get(n, set()))
                frac_other = other / len(neigh)
                if frac_other > 0.5:
                    flags["neighbour_outlier"] = (frac_other - 0.5) / 0.5 * weak

            if flags:
                rows.append({"target": rid, "type": "placement", "compartment": comp,
                             "_base": _noisy_or(flags), "flags": dict(flags)})

    # ---- H3 gap-fill: reactions pulled from the universal model (not in the draft's placements) ----
    for rid in added_reactions:
        if rid in curated_localization:
            continue
        comp = default_compartment
        if universal is not None and rid in universal.reactions:
            csx = {mm.compartment for mm in universal.reactions.get_by_id(rid).metabolites}
            comp = next(iter(csx)) if len(csx) == 1 else default_compartment
        rows.append({"target": rid, "type": "gapfill", "compartment": comp,
                     "_base": _noisy_or({"gapfill": 1.0}), "flags": {"gapfill": 1.0}})

    # ---- transport signals (E1 impermeant cargo). Target id is namespaced so it cannot collide with a
    # reaction id (which would duplicate a row and, worse, inherit the essential flag below). ----
    for b, c in proposal.added_transports:
        if _is_impermeant(b, base_name.get(b, b)):
            rows.append({"target": f"transport::{b}->{c}", "type": "transport", "compartment": c,
                         "_base": _noisy_or({"impermeant": 1.0}), "flags": {"impermeant": 1.0},
                         "_metabolite": base_name.get(b, b), "_bkey": b})

    if not rows:
        return pd.DataFrame({"target": [], "type": [], "compartment": [],
                             "priority": pd.Series([], dtype="float64"), "flags": [], "action": [],
                             "n_affected": pd.Series([], dtype="int64"), "affected": []})

    # ---- essentiality gate (the only non-cheap signal): amplify already-flagged PLACEMENTS ----
    floor = min_growth if min_growth is not None else proposal.min_growth
    if check_essential and floor:
        cand = {row["target"] for row in rows
                if row["type"] == "placement" and row["_base"] >= essential_candidate_threshold}
        essential = _essential_set(model, proposal, base, default_compartment, universal, floor, cand)
        for row in rows:
            if row["type"] == "placement" and row["target"] in essential:
                row["flags"]["essential"] = 1.0

    for row in rows:
        stakes = 1.0 + _BETA_ESSENTIAL if "essential" in row["flags"] else 1.0
        row["priority"] = round(min(1.0, row["_base"] * stakes), 4)
        row["action"] = _action(row)

    # ---- coupling: which other placed reactions a curation of this row would touch ----
    # A reaction R at C is coupled to every reaction that shares a non-currency metabolite with it AND
    # is also at C (moving/removing R changes their local balance); a transport of species b at C is
    # coupled to the reactions touching b at either end of the bridge (C or the default compartment).
    # This is the one-hop blast radius; it is not transitive (curating a coupled reaction has its own).
    flagged_rxn = {row["target"] for row in rows if row["type"] in ("placement", "gapfill")}
    for row in rows:
        comp = row["compartment"]
        if row["type"] == "transport":
            bkey = row["_bkey"]
            aff = {s for s in touch.get(bkey, ()) if comps_of.get(s, set()) & {comp, default_compartment}}
        else:  # placement / gapfill
            rid = row["target"]
            bases = nonbases_of.get(rid)
            if bases is None and universal is not None and rid in universal.reactions:
                bases = [base(m) for m in universal.reactions.get_by_id(rid).metabolites if not _cur(base(m))]
            aff = set()
            for b in bases or ():
                aff |= {s for s in touch.get(b, ()) if comp in comps_of.get(s, set())}
            aff.discard(rid)
        row["n_affected"] = len(aff)
        # display flagged (curator will also see them) first, then the rest, capped
        ordered = sorted(aff, key=lambda s: (s not in flagged_rxn, s))
        row["affected"] = ", ".join(ordered[:_AFFECTED_CAP]) + (
            f" +{len(ordered) - _AFFECTED_CAP}" if len(ordered) > _AFFECTED_CAP else "")

    cols = ("target", "type", "compartment", "priority", "flags", "action", "n_affected", "affected")
    out = pd.DataFrame([{k: row[k] for k in cols} for row in rows])
    out["flags"] = out["flags"].apply(lambda d: ", ".join(sorted(d, key=lambda k: -d[k])))
    return out.sort_values("priority", ascending=False, ignore_index=True)


def _essential_set(model, proposal, base, default_compartment, universal, floor, candidates):
    """Placements whose removal drops the *materialised* model's growth to near zero (single-deletion).

    Essentiality is measured against the applied model's own achievable growth (``base_g``), not the
    assignment's heuristic ``min_growth`` floor: the floor can be higher than the materialised model can
    actually reach (the documented growth-floor gap), in which case comparing knockout growth to it
    would flag every candidate and collapse the ranking. If the materialised model cannot beat the
    floor, essentiality cannot discriminate, so return empty.
    """
    from raven_toolbox.localization.assign import apply_assignment
    if not candidates:
        return set()
    applied = apply_assignment(model, proposal, default_compartment=default_compartment,
                               base_metabolite=base, universal=universal)
    base_g = applied.slim_optimize(error_value=0.0) or 0.0
    if base_g <= max(floor, 1e-9):
        return set()
    lethal = 0.01 * base_g
    essential = set()
    for rid in candidates:
        if rid not in applied.reactions:
            continue
        with applied:
            applied.reactions.get_by_id(rid).knock_out()
            g = applied.slim_optimize(error_value=0.0) or 0.0
        if g < lethal:
            essential.add(rid)
    return essential


def _action(row) -> str:
    f = {k: v for k, v in row["flags"].items() if k in _ACTION_KEYS}
    if not f:
        return "review the placement"
    top = max(f, key=lambda k: _WEIGHTS.get(k, 0.7) * f[k])
    c = row["compartment"]
    if top == "override":
        return (f"placed in {c} against the gene's top DeepLoc compartment; confirm the relocation is "
                f"real (dual-targeting) or a neighbour/gap-fill forced it")
    if top == "sandwich":
        return (f"substrates imported into {c} and products exported; try re-placing to the neighbouring "
                f"compartment and check if both transports vanish and growth holds")
    if top == "no_evidence":
        return "no scored gene; compartment chosen by connectivity only, assign from any evidence"
    if top == "gapfill":
        return "gap-fill reaction from the universal model; verify it belongs and in this compartment"
    if top == "neighbour_outlier":
        return f"metabolic neighbours sit mostly outside {c}; check for a mislocated pathway step"
    if top == "impermeant":
        return (f"transport of {row.get('_metabolite', 'this species')} into {c} is biologically "
                f"implausible (impermeant cargo); a placement should likely move instead")
    return "review the placement"
