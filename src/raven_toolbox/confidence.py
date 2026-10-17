"""Per-reaction, multi-facet confidence — persisted in the model, ignored by plain cobra.

Attaches a small structured record to a reaction scoring how well-supported each *facet* of it is:
``localization``, ``equation`` and ``gene_association``. Each facet is a :class:`ConfidenceEntry` — a
continuous 0-1 ``score`` plus optional provenance (a categorical ``level``, the ``basis`` evidence,
``method``/``source``/``note``). A reaction carries a :class:`ReactionConfidence` (facet → entry) whose
``overall`` is the weakest facet.

**Two rules govern every score**, because ``overall = min(facets)`` and :func:`_write` drops the record
when no facet remains:

1. **Abstain rather than guess.** A facet that *does not apply* to a reaction is not written at all — an
   absent facet is neutral under ``min``, whereas ``1.0`` would claim the reaction was checked and
   passed. An exchange reaction is imbalanced by construction; it gets no ``equation`` facet, not a
   perfect one. Use :func:`equation_exempt` / :func:`gene_association_exempt` to ask why.
2. **A zero is a measurement, never ignorance.** ``score == 0.0`` means evidence *contradicts* the model
   (a proven mass imbalance; zero localisation support at the assigned compartment). Where the evidence
   is merely missing or uninterpretable, the score is low but positive. Hence the invariant enforced in
   the tests: ``max(defect scores) < min(ignorance scores)`` across all facets, so a reaction with a
   proven defect always outranks one that is merely unverifiable.

**Storage.** The record lives as one JSON blob under ``reaction.notes["raven_confidence"]``. cobra
round-trips ``notes`` losslessly through YAML/JSON and, as an HTML-escaped string, through SBML; the
helpers here write ``json.dumps`` and read ``json.loads(html.unescape(...))``, so the same code works for
either format. Plain cobra ignores the key entirely — a confidence-annotated model still loads and
solves unchanged (a test invariant).

**SBO precondition.** The exemptions read ``reaction.annotation["sbo"]``. On a model carrying no reaction
SBO terms the scorers warn, because they cannot then tell a biomass pseudo-reaction from a chemistry
defect. Detecting biomass by name instead is deliberately *not* done: ``\\bgrowth\\b`` matches
"non-growth associated maintenance reaction", and a name regex must never silence a chemistry check.

The design and the measured yeast-GEM distributions are in the `confidence tracking study
<https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/localization/confidence_tracking.md>`_ on raven-gecko-parity; the
facet set above is closed. Wire it in by calling :func:`score_localization_confidence` on an
:class:`~raven_toolbox.localization.AssignmentProposal`, :func:`score_equation_confidence` and
:func:`score_gene_association_confidence` on any model, and :func:`mark_curated` when a curator firmly
fixes a facet (e.g. after :func:`~raven_toolbox.localization.relocate_reactions`).
"""
from __future__ import annotations

import ast
import html
import json
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import cobra
from cobra.core.formula import elements_and_molecular_weights

from raven_toolbox.utils.balance import get_elemental_balance

__all__ = [
    "ConfidenceEntry",
    "ReactionConfidence",
    "annotate_confidence",
    "clear_confidence",
    "confidence_report",
    "equation_exempt",
    "facet_summary",
    "gene_association_exempt",
    "get_confidence",
    "mark_curated",
    "read_confidence",
    "score_equation_confidence",
    "score_gene_association_confidence",
    "score_localization_confidence",
    "set_confidence",
    "thiele_palsson_score",
]

_KEY = "raven_confidence"
_SCHEMA_VERSION = 1
# reserved top-level keys inside the stored blob that are not facets
_RESERVED = frozenset({"schema_version"})

# SBO terms cobra reads from SBML and writes via Model.add_boundary().
_SBO_BIOMASS = "SBO:0000629"
_SBO_PSEUDOREACTION = "SBO:0000395"  # "encapsulating process" — SLIME, pool, and lumped reactions
_SBO_ATP_MAINTENANCE = "SBO:0000630"
_SBO_SPONTANEOUS = "SBO:0000672"

#: Symbols cobra's formula parser accepts that are not real elements (R groups, unspecified residues).
#: A residual reported in one of these is uninterpretable, not a proven imbalance.
_PERIODIC = frozenset(elements_and_molecular_weights)


@dataclass
class ConfidenceEntry:
    """One facet's confidence: a continuous ``score`` in [0, 1] plus optional provenance.

    ``level`` is an optional categorical band (``"curated"`` / ``"strong"`` / ``"moderate"`` / ``"weak"``
    / ``"none"``); ``basis`` names the evidence (``"deeploc"``, ``"deeploc+fba-certified"``, ``"balanced"``,
    ``"mass-imbalanced"``, ``"gpr+literature"``, ``"curator"``, ...); ``source`` distinguishes ``"auto"``
    from ``"curator:<id>"``.
    """

    score: float
    basis: str = ""
    level: str | None = None
    method: str | None = None
    source: str | None = None
    note: str | None = None
    updated: str | None = None  # ISO date, passed in by the caller (not generated here, to stay pure)

    def __post_init__(self) -> None:
        self.score = float(max(0.0, min(1.0, self.score)))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"score": round(self.score, 4)}
        for k in ("level", "basis", "method", "source", "note", "updated"):
            v = getattr(self, k)
            if v:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConfidenceEntry:
        return cls(score=float(d.get("score", 0.0)), basis=d.get("basis", ""), level=d.get("level"),
                   method=d.get("method"), source=d.get("source"), note=d.get("note"),
                   updated=d.get("updated"))


@dataclass
class ReactionConfidence:
    """A reaction's confidence across facets (``{facet_name: ConfidenceEntry}``)."""

    facets: dict[str, ConfidenceEntry] = field(default_factory=dict)

    @property
    def overall(self) -> float | None:
        """The weakest facet's score (the honest single number), or ``None`` when there are no facets."""
        return min((e.score for e in self.facets.values()), default=None)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"schema_version": _SCHEMA_VERSION}
        for name, entry in self.facets.items():
            d[name] = entry.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReactionConfidence:
        facets = {k: ConfidenceEntry.from_dict(v) for k, v in d.items()
                  if k not in _RESERVED and isinstance(v, dict)}
        return cls(facets=facets)


# --------------------------------------------------------------------------- storage (notes JSON blob)

def get_confidence(reaction: cobra.Reaction) -> ReactionConfidence:
    """Read a reaction's :class:`ReactionConfidence` from its notes (empty if none / unreadable).

    Handles both a native dict (YAML/JSON models) and a JSON string (SBML, possibly HTML-escaped).
    """
    raw = reaction.notes.get(_KEY)
    if raw is None:
        return ReactionConfidence()
    if isinstance(raw, str):
        try:
            raw = json.loads(html.unescape(raw))
        except (ValueError, TypeError):
            return ReactionConfidence()
    if not isinstance(raw, dict):
        return ReactionConfidence()
    return ReactionConfidence.from_dict(raw)


def _write(reaction: cobra.Reaction, record: ReactionConfidence) -> None:
    if record.facets:
        reaction.notes[_KEY] = json.dumps(record.to_dict())
    else:
        reaction.notes.pop(_KEY, None)


def set_confidence(reaction: cobra.Reaction, facet: str, entry: ConfidenceEntry) -> None:
    """Merge one facet's :class:`ConfidenceEntry` into the reaction's record and persist it to notes."""
    record = get_confidence(reaction)
    record.facets[facet] = entry
    _write(reaction, record)


def clear_confidence(reaction: cobra.Reaction, facet: str | None = None) -> None:
    """Drop one facet (``facet=...``) or the whole record (``facet=None``) from the reaction's notes."""
    if facet is None:
        reaction.notes.pop(_KEY, None)
        return
    record = get_confidence(reaction)
    record.facets.pop(facet, None)
    _write(reaction, record)


def read_confidence(model: cobra.Model) -> dict[str, ReactionConfidence]:
    """Every annotated reaction id -> its :class:`ReactionConfidence` (reactions with no record omitted)."""
    out: dict[str, ReactionConfidence] = {}
    for r in model.reactions:
        rc = get_confidence(r)
        if rc.facets:
            out[r.id] = rc
    return out


def confidence_report(model: cobra.Model):
    """A :class:`pandas.DataFrame` of confidence: one row per annotated reaction, one column per facet
    score, plus ``overall`` (the weakest facet). Sorted lowest-confidence first — the review queue."""
    import pandas as pd

    records = read_confidence(model)
    facet_names: list[str] = []
    for rc in records.values():
        for name in rc.facets:
            if name not in facet_names:
                facet_names.append(name)
    rows = []
    for rid, rc in records.items():
        row: dict[str, Any] = {"reaction": rid}
        for name in facet_names:
            entry = rc.facets.get(name)
            row[name] = entry.score if entry else None
        row["overall"] = rc.overall
        rows.append(row)
    cols = ["reaction", *sorted(facet_names), "overall"]
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.sort_values("overall", ascending=True, na_position="last", ignore_index=True)
    return out


def facet_summary(model: cobra.Model):
    """A :class:`pandas.DataFrame` of ``facet`` × ``basis`` × ``score`` counts over the model.

    The audit trail abstention leaves behind: it separates *scored* reactions from the ones no facet
    applies to, so "the fraction of this model whose chemistry is verified balanced" is a number you can
    read off rather than infer from a score column."""
    import pandas as pd

    rows = [{"facet": facet, "basis": entry.basis, "score": entry.score, "level": entry.level}
            for rc in read_confidence(model).values() for facet, entry in rc.facets.items()]
    if not rows:
        return pd.DataFrame(columns=["facet", "basis", "score", "level", "n"])
    out = (pd.DataFrame(rows).groupby(["facet", "basis", "score", "level"], dropna=False)
           .size().reset_index(name="n"))
    return out.sort_values(["facet", "score"], ignore_index=True)


# --------------------------------------------------------------------------- localization scorer (P1)

def _level(score: float, *, curated: bool = False) -> str:
    if curated:
        return "curated"
    if score >= 0.8:
        return "strong"
    if score >= 0.5:
        return "moderate"
    if score > 0.0:
        return "weak"
    return "none"


def mark_curated(reaction: cobra.Reaction, *, facet: str = "localization", source: str = "curator",
                 note: str | None = None, updated: str | None = None) -> None:
    """Stamp a facet as **curator-verified**: confidence 1.0, ``level="curated"``.

    Call after a curator firmly fixes a reaction (e.g. its compartment, via
    :func:`~raven_toolbox.localization.relocate_reactions`), so the decision persists in the model and
    the automated scorer for that ``facet`` leaves it untouched on a later pass.

    ``1.0`` is reserved for this and for a proven check (a reaction whose chemistry is verified
    balanced); the evidence-derived bands cap at 0.9, so a curated call always outranks an inferred one.
    """
    set_confidence(reaction, facet,
                   ConfidenceEntry(score=1.0, level="curated", basis="curator", source=source,
                                   note=note, updated=updated))


def _curated(reaction: cobra.Reaction, facet: str, overwrite: bool) -> bool:
    """True when ``facet`` carries a curator's call that an automated pass must not overwrite."""
    entry = get_confidence(reaction).facets.get(facet)
    return entry is not None and entry.level == "curated" and not overwrite


def _abstain(reaction: cobra.Reaction, facet: str) -> None:
    """Write no ``facet``, dropping a stale score from an earlier pass — but never a curator's call.

    ``overwrite_curated=True`` licenses *recomputing* over a curated entry, not deleting one: a scorer
    that cannot measure a reaction has learned nothing that invalidates a human's assertion about it.
    Only an explicit :func:`clear_confidence` removes that.
    """
    entry = get_confidence(reaction).facets.get(facet)
    if entry is not None and entry.level == "curated":
        return
    clear_confidence(reaction, facet)


def score_localization_confidence(model, proposal, scores, *,
                                  overwrite_curated: bool = False, updated: str | None = None) -> int:
    """Attach a ``localization`` confidence to the placements in ``proposal`` and return the count scored.

    The score is the DeepLoc support for the assigned compartment (the strongest scored gene's score
    there), so ``0.0`` means the evidence actively puts the reaction *elsewhere*. The ``basis`` records
    ``deeploc`` and, when the proposal is certified, ``fba-certified``.

    A reaction whose genes are absent from ``scores`` (placed by connectivity alone) is **not scored**:
    no measurement was possible, and writing ``0.0`` would veto the reaction's ``overall`` on the
    strength of a missing input rather than of evidence. Such reactions surface through their
    ``gene_association`` facet and through
    :func:`~raven_toolbox.localization.curation_priority`'s ``no_evidence`` signal instead. Placements
    already marked ``curated`` are left alone unless ``overwrite_curated``.

    This is the inverse of the localization signal in ``curation_priority``: high confidence here ==
    low review priority there. ``model`` is the draft the proposal was built on.
    """
    df = scores.df
    certified = bool(getattr(proposal, "certified", False))
    n = 0
    for rid, comps in proposal.placements.items():
        if not comps or rid not in model.reactions:
            continue
        reaction = model.reactions.get_by_id(rid)
        if _curated(reaction, "localization", overwrite_curated):
            continue
        comp = comps[0]
        genes = [g.id for g in reaction.genes if g.id in df.index]
        if not genes or comp not in df.columns:
            # Unmeasurable, not unsupported: abstain, and drop a stale score from an earlier pass.
            _abstain(reaction, "localization")
            continue
        support = max((_score(df, g, comp) for g in genes), default=0.0)
        set_confidence(reaction, "localization",
                       ConfidenceEntry(score=support, level=_level(support),
                                       basis="deeploc+fba-certified" if certified else "deeploc",
                                       method="score_localization_confidence", source="auto",
                                       updated=updated))
        n += 1
    return n


# ------------------------------------------------------------ exemptions (which facets even apply)

def _sbo(reaction: cobra.Reaction) -> str | None:
    v = (reaction.annotation or {}).get("sbo")
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)) and v:
        return str(v[0])
    return None


def equation_exempt(reaction: cobra.Reaction) -> str | None:
    """Why ``equation`` does not apply to ``reaction`` (a reason string), or ``None`` if it does.

    Exchange/demand/sink reactions (``reaction.boundary``, i.e. any single-metabolite reaction) exchange
    mass with the environment; biomass (``SBO:0000629``) and pool/SLIME pseudo-reactions
    (``SBO:0000395``) are lumped by construction. All are imbalanced *by design*, so their chemistry is
    not a defect and must not be scored.

    ATP maintenance (``SBO:0000630``) is deliberately **not** exempt: it is real chemistry
    (``ATP + H2O -> ADP + Pi + H``) and must balance. It is exempt from ``gene_association`` instead —
    real chemistry, no catalyst. See :func:`gene_association_exempt`.
    """
    if reaction.boundary:
        return "boundary"
    sbo = _sbo(reaction)
    if sbo == _SBO_BIOMASS:
        return "biomass"
    if sbo == _SBO_PSEUDOREACTION:
        return "pseudoreaction"
    return None


def gene_association_exempt(reaction: cobra.Reaction) -> str | None:
    """Why ``gene_association`` does not apply to ``reaction`` (a reason string), or ``None`` if it does.

    Everything :func:`equation_exempt` excludes, plus the reactions that legitimately have no catalyst:
    ATP maintenance (``SBO:0000630``) and spontaneous reactions (``SBO:0000672``, defined as "reaction
    with no catalyst ... is needed to proceed"). Transport (``SBO:0000655``) is **not** exempt: the term
    is defined as movement "mediated by a transporter protein", so a transport reaction with no
    transporter gene is a genuine curation gap, not a modelling convention.
    """
    reason = equation_exempt(reaction)
    if reason is not None:
        return reason
    sbo = _sbo(reaction)
    if sbo == _SBO_ATP_MAINTENANCE:
        return "maintenance"
    if sbo == _SBO_SPONTANEOUS:
        return "spontaneous"
    return None


def _warn_if_no_sbo(model: cobra.Model) -> None:
    if model.reactions and not any(_sbo(r) for r in model.reactions):
        warnings.warn(
            "no reaction carries an SBO term, so biomass and pool pseudo-reactions cannot be told from "
            "chemistry defects and will be scored as defects. Annotate SBO terms first with "
            "raven_toolbox.annotation.add_sbo_terms(model).",
            stacklevel=3,
        )


# --------------------------------------------------------------------------- equation facet (P2)

# Defects (evidence contradicts the model) sort strictly below ignorance (evidence is missing).
_EQ_MASS_IMBALANCED = 0.0     # proven: atoms do not conserve
_EQ_CHARGE_IMBALANCED = 0.1   # proven, but often a protonation-state convention rather than an error
_EQ_FORMULA_UNKNOWN = 0.3     # a formula is missing, unparseable, or generic -- verdict impossible
_EQ_CHARGE_UNKNOWN = 0.6      # mass proven balanced, charge unverifiable
_EQ_BALANCED = 1.0            # proven: mass and charge conserve

_CHARGE_TOL = 1e-6


def _charge_residual(reaction: cobra.Reaction) -> float | None:
    """Net charge (products − reactants), or ``None`` if any metabolite's charge is unset.

    Recomputed rather than read from ``check_mass_balance()["charge"]``, which cobra accumulates over
    only the *non-``None``* metabolites — fabricating a residual from a partial sum.
    """
    total = 0.0
    for met, coeff in reaction.metabolites.items():
        if met.charge is None:
            return None
        total += coeff * met.charge
    return total


def _generic_symbols(reaction: cobra.Reaction) -> list[str]:
    """Non-periodic symbols (R groups, residues) appearing in this reaction's formulas."""
    out: set[str] = set()
    for met in reaction.metabolites:
        for symbol in met.elements or {}:
            if symbol not in _PERIODIC:
                out.add(symbol)
    return sorted(out)


def _equation_entry(reaction: cobra.Reaction, balance, tolerance: float) -> ConfidenceEntry:
    if not reaction.metabolites:
        # Not `boundary` (that is exactly one metabolite), so it reaches the scorer. Nothing to balance.
        return ConfidenceEntry(_EQ_FORMULA_UNKNOWN, basis="no-stoichiometry",
                               note="reaction has no metabolites")

    if balance.status == "unknown":
        missing = sorted(m.id for m in reaction.metabolites if not m.formula)
        if missing:
            shown = ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else "")
            return ConfidenceEntry(_EQ_FORMULA_UNKNOWN, basis="formula-missing",
                                   note=f"no formula: {shown}")
        return ConfidenceEntry(_EQ_FORMULA_UNKNOWN, basis="formula-unparseable",
                               note="a formula cobra cannot parse (e.g. a parenthesised polymer)")

    if balance.status == "unbalanced":
        generic = sorted(set(balance.imbalance) - _PERIODIC)
        if generic:
            # The residual lands in an R group: the verdict is uninterpretable, not a proven imbalance.
            return ConfidenceEntry(_EQ_FORMULA_UNKNOWN, basis="formula-generic",
                                   note=f"residual in non-element symbols: {', '.join(generic)}")
        residual = ", ".join(f"{el}{amount:+g}" for el, amount in sorted(balance.imbalance.items()))
        return ConfidenceEntry(_EQ_MASS_IMBALANCED, basis="mass-imbalanced", note=residual)

    # Mass balances. Note any R groups, but do not let them move a score they demonstrably cancel out of.
    generic = _generic_symbols(reaction)
    note = f"formulas contain non-element symbols: {', '.join(generic)}" if generic else None

    charge = _charge_residual(reaction)
    if charge is None:
        return ConfidenceEntry(_EQ_CHARGE_UNKNOWN, basis="charge-unknown",
                               note=note or "a metabolite has no charge")
    if abs(charge) > tolerance:
        return ConfidenceEntry(_EQ_CHARGE_IMBALANCED, basis="charge-imbalanced",
                               note=f"net charge {charge:+g}")
    return ConfidenceEntry(_EQ_BALANCED, basis="balanced", note=note)


def score_equation_confidence(model, *, overwrite_curated: bool = False, updated: str | None = None,
                              tolerance: float = _CHARGE_TOL) -> int:
    """Attach an ``equation`` confidence (mass & charge balance) to every applicable reaction.

    Returns the number of reactions scored. Reactions :func:`equation_exempt` excludes are **not**
    scored — an exchange reaction is imbalanced by construction, and writing ``1.0`` there would make
    469 never-checked yeast-GEM reactions indistinguishable from the 3617 verified ones.

    Bands, lowest first: a proven mass imbalance (``0.0``); a proven charge imbalance (``0.1``, a rung
    up because it is usually a protonation convention); a formula missing, unparseable, or generic
    (``0.3`` — unverifiable, not wrong); mass proven but charge unset (``0.6``); balanced (``1.0``).

    ``tolerance`` is the absolute net-charge threshold below which a reaction counts as
    charge-balanced rather than charge-imbalanced.
    """
    _warn_if_no_sbo(model)
    balances = {b.reaction_id: b for b in get_elemental_balance(model)}
    n = 0
    for reaction in model.reactions:
        if _curated(reaction, "equation", overwrite_curated):
            continue
        if equation_exempt(reaction) is not None:
            _abstain(reaction, "equation")
            continue
        entry = _equation_entry(reaction, balances[reaction.id], tolerance)
        entry.level = _level(entry.score)
        entry.method = "score_equation_confidence"
        entry.source = "auto"
        entry.updated = updated
        set_confidence(reaction, "equation", entry)
        n += 1
    return n


# --------------------------------------------------------------------- gene_association facet (P2)

_GA_NO_GPR = 0.2             # a reaction that should have a catalyst and does not
_GA_GPR = 0.6                # genetic evidence only            (~ Thiele-Palsson 2)
_GA_GPR_LITERATURE = 0.9     # genetic + literature evidence    (~ Thiele-Palsson 3)


def _gpr_shape(reaction: cobra.Reaction) -> tuple[int, int]:
    """``(alternative gene groups, largest group)`` — isozyme count and complex size.

    Read off the rule's *top-level* operator, so a non-DNF rule such as ``(g1 or g2) and (g3 or g4)``
    is summarised loosely as one group of four. This only ever feeds ``note``; nothing is scored from it.
    """
    body = reaction.gpr.body
    if body is None:
        return 0, 0

    def leaves(node: ast.AST) -> int:
        if isinstance(node, ast.Name):
            return 1
        if isinstance(node, ast.BoolOp):
            return sum(leaves(v) for v in node.values)
        return 0

    if isinstance(body, ast.BoolOp) and isinstance(body.op, ast.Or):
        return len(body.values), max(leaves(v) for v in body.values)
    return 1, leaves(body)


def _recorded_confidence(reaction: cobra.Reaction) -> int | None:
    """The model's own Thiele-Palsson ``Confidence Level`` note (0-4), if it carries one.

    Via ``float`` so that a level stored as ``3.0`` or ``"3.0"`` — which a YAML round-trip readily
    produces — reads as 3 rather than silently as "absent".
    """
    raw = (reaction.notes or {}).get("Confidence Level")
    try:
        return int(float(str(raw)))
    except (TypeError, ValueError):
        return None


def _gene_entry(reaction: cobra.Reaction) -> ConfidenceEntry:
    recorded = _recorded_confidence(reaction)
    if not reaction.gene_reaction_rule.strip():
        # Flagged, never scored from: keeping our rubric independent of the model's recorded score is
        # what lets the recorded score serve as an *independent* check of it (see the study doc).
        note = (f"recorded Confidence Level {recorded} but no GPR" if recorded is not None and
                recorded >= 2 else None)
        return ConfidenceEntry(_GA_NO_GPR, basis="no-gpr", note=note)

    isozymes, complex_size = _gpr_shape(reaction)
    bits = []
    if isozymes > 1:
        bits.append(f"{isozymes} isozymes")
    if complex_size > 1:
        bits.append(f"complex of {complex_size}")
    note = "; ".join(bits) or None

    if "pubmed" in (reaction.annotation or {}):
        return ConfidenceEntry(_GA_GPR_LITERATURE, basis="gpr+literature", note=note)
    return ConfidenceEntry(_GA_GPR, basis="gpr", note=note)


def score_gene_association_confidence(model, *, overwrite_curated: bool = False,
                                      updated: str | None = None) -> int:
    """Attach a ``gene_association`` confidence to every applicable reaction; return the count scored.

    Bands: no GPR (``0.2``); a GPR (``0.6``); a GPR plus a literature citation, i.e. a ``pubmed``
    annotation (``0.9``). These mirror the Thiele-Palsson levels 0/1, 2 and 3 — and because the rubric
    reads only the GPR and the citation, a curated model's *own* recorded ``Confidence Level`` note is
    left free to serve as an independent check of it rather than as an input.

    ``1.0`` is reserved for :func:`mark_curated`. Reactions :func:`gene_association_exempt` excludes
    (boundary, biomass, pseudo, maintenance, spontaneous) are not scored; transport reactions are.
    """
    _warn_if_no_sbo(model)
    n = 0
    for reaction in model.reactions:
        if _curated(reaction, "gene_association", overwrite_curated):
            continue
        if gene_association_exempt(reaction) is not None:
            _abstain(reaction, "gene_association")
            continue
        entry = _gene_entry(reaction)
        entry.level = _level(entry.score)
        entry.method = "score_gene_association_confidence"
        entry.source = "auto"
        entry.updated = updated
        set_confidence(reaction, "gene_association", entry)
        n += 1
    return n


#: ``gene_association`` facet ``basis`` -> Thiele & Palsson reconstruction confidence score (0-4). A
#: homology-derived GPR is sequence evidence (2); a GPR with a literature citation is
#: experimental/genetic (3); a reaction that should have a catalyst but has none is a modelling
#: inference (1). ``curated`` is deliberately absent: a curator's assertion does not, on its own, name
#: the *evidence class* it rests on (the score's ``basis`` is ``"curator"``), so its Thiele-Palsson
#: class must be set from the evidence the curator used, not inferred here (see the study doc §9).
_THIELE_PALSSON_FROM_GA_BASIS = {"gpr+literature": 3, "gpr": 2, "no-gpr": 1}


def thiele_palsson_score(reaction: cobra.Reaction) -> int | None:
    """The reaction's Thiele & Palsson reconstruction confidence score (0-4), or ``None``.

    Derived from the ``gene_association`` facet's ``basis`` (:func:`score_gene_association_confidence`),
    the facet that captures reaction-inclusion evidence: ``gpr+literature`` -> 3 (experimental/genetic),
    ``gpr`` -> 2 (sequence), ``no-gpr`` -> 1 (modelling). Returns ``None`` when the facet is absent (the
    reaction was not scored, or ``gene_association`` does not apply to it) or ``curated`` (whose evidence
    class the curator must name — see the module note and the study doc §9). The ``localization`` and
    ``equation`` facets are quality checks, not Thiele-Palsson evidence classes, and are not consulted.
    """
    entry = get_confidence(reaction).facets.get("gene_association")
    if entry is None:
        return None
    return _THIELE_PALSSON_FROM_GA_BASIS.get(entry.basis)


def annotate_confidence(
    model: cobra.Model,
    *,
    proposal: Any = None,
    scores: Any = None,
    facets: Iterable[str] | None = None,
    overwrite_curated: bool = False,
    updated: str | None = None,
) -> dict[str, int]:
    """Run every applicable confidence scorer in one call; return ``{facet: reactions_scored}``.

    ``equation`` and ``gene_association`` need only the model and always run; ``localization`` runs
    only when both ``proposal`` (an :class:`~raven_toolbox.localization.AssignmentProposal`) and
    ``scores`` (a :class:`~raven_toolbox.localization.LocalizationScores`) are given, and is otherwise
    **skipped rather than failing** — the same abstain-rather-than-guess rule the scores follow, so a
    caller without a localisation proposal still gets the other two facets. ``facets=[...]`` restricts
    to a subset (names outside the facet set are ignored). ``overwrite_curated`` and ``updated`` pass
    through to each scorer.
    """
    requested = {"localization", "equation", "gene_association"} if facets is None else set(facets)
    counts: dict[str, int] = {}
    if "localization" in requested and proposal is not None and scores is not None:
        counts["localization"] = score_localization_confidence(
            model, proposal, scores, overwrite_curated=overwrite_curated, updated=updated
        )
    if "equation" in requested:
        counts["equation"] = score_equation_confidence(
            model, overwrite_curated=overwrite_curated, updated=updated
        )
    if "gene_association" in requested:
        counts["gene_association"] = score_gene_association_confidence(
            model, overwrite_curated=overwrite_curated, updated=updated
        )
    return counts


def _score(df, g: str, c: str) -> float:
    if g not in df.index or c not in df.columns:
        return 0.0
    v = df.at[g, c]
    return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
