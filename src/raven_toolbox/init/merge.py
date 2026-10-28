"""Linear reaction merging for ftINIT.

ftINIT shrinks the MILP losslessly by **contracting linear reaction chains**: a
metabolite that appears in exactly two reactions (one net producer, one net consumer)
links them into a single combined reaction. Iterating this collapses unbranched
pathways — on Human-GEM ~12k → ~8k reactions, a ~⅓ smaller MILP — without changing
the feasible flux space. Reversible reactions may merge too (unlike
``simplifyModel``'s merge), which is why ftINIT ships its own.

:func:`merge_linear` returns the reduced model plus the bookkeeping needed to map
scores and results back to the original reactions:

* ``group_ids`` — one integer per original reaction; ``0`` = not merged, equal
  non-zero integers = merged into the same combined reaction (which keeps one
  member's id). The groups are numbered 1, 2, ... in the order of their first
  reaction in the original model, as RAVEN's ``mergeLinear`` numbers them.
* ``reversed_rxns`` — which originals were flipped (their stored direction negated)
  when oriented for merging; needed to map fluxes/directions back.

:func:`group_rxn_scores` then sums the original per-reaction scores over each group,
with RAVEN's zero-handling (see its docstring): genuine 0 → 0.01, ignore-masked → 0,
a group cancelling to 0 with non-zero members → 0.01 — all so the MILP never sees an
exactly-zero score (whose on/off would be arbitrary).
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

import cobra

_TOL = 1e-12


class _Rxn:
    """Mutable working reaction during the merge."""

    __slots__ = ("id", "name", "coeffs", "lb", "ub")

    def __init__(self, rid, name, coeffs, lb, ub):
        self.id, self.name, self.coeffs, self.lb, self.ub = rid, name, coeffs, lb, ub

    @property
    def reversible(self) -> bool:  # RAVEN's rev flag ≡ a negative lower bound
        return self.lb < 0


def merge_linear(
    model: cobra.Model, no_merge: Iterable[str] = ()
) -> tuple[cobra.Model, list[str], list[int], list[bool]]:
    """Merge linearly-dependent reactions; return ``(reduced, orig_ids, group_ids, reversed)``.

    ``no_merge`` reaction ids are never merged. The reduced model carries no genes
    (merging makes GPRs meaningless); scores are remapped with
    :func:`group_rxn_scores`.
    """
    banned = set(no_merge)
    orig_ids = [r.id for r in model.reactions]
    group_of: dict[str, int] = {rid: 0 for rid in orig_ids}
    reversed_of: dict[str, bool] = {rid: False for rid in orig_ids}
    group_members: dict[int, set[str]] = {}  # non-zero group -> its original reaction ids
    next_group = 1

    rxns = [
        _Rxn(r.id, r.name, {m.id: c for m, c in r.metabolites.items()},
             r.lower_bound, r.upper_bound)
        for r in model.reactions
    ]

    def flip(rx: _Rxn) -> None:
        rx.coeffs = {m: -c for m, c in rx.coeffs.items()}
        rx.lb, rx.ub = -rx.ub, -rx.lb
        grp = group_of[rx.id]
        targets = group_members[grp] if grp else (rx.id,)
        for o in targets:
            reversed_of[o] = not reversed_of[o]

    def relabel(rx: _Rxn, grp: int) -> None:
        old = group_of[rx.id]
        if old == grp:
            return
        if old == 0:
            group_of[rx.id] = grp
            group_members.setdefault(grp, set()).add(rx.id)
        else:
            members = group_members.pop(old)
            for o in members:
                group_of[o] = grp
            group_members.setdefault(grp, set()).update(members)

    while True:
        incidence: dict[str, list[int]] = defaultdict(list)
        for i, rx in enumerate(rxns):
            for m in rx.coeffs:
                incidence[m].append(i)
        degree2 = [m for m, ii in incidence.items() if len(ii) == 2]

        merged_some = False
        for met in degree2:
            involved = [i for i in incidence[met] if met in rxns[i].coeffs]
            if len(involved) != 2:
                continue  # one side already merged away this pass
            a, b = involved
            if rxns[a].id in banned or rxns[b].id in banned:
                continue
            ca, cb = rxns[a].coeffs[met], rxns[b].coeffs[met]
            ra, rb = rxns[a].reversible, rxns[b].reversible
            pos = (ca > 0 or ra) + (cb > 0 or rb)
            neg = (ca < 0 or ra) + (cb < 0 or rb)
            if pos < 1 or neg < 1:
                continue  # need one producer and one consumer

            r1, r2 = a, b
            # Special case: rev producer first, irrev producer second → swap (RAVEN l.74).
            if rxns[r1].reversible and not rxns[r2].reversible \
                    and rxns[r1].coeffs[met] > 0 and rxns[r2].coeffs[met] > 0:
                r1, r2 = r2, r1
            # Make r1 the producer of `met`.
            if rxns[r1].coeffs[met] < 0:
                if rxns[r2].coeffs[met] > 0:
                    r1, r2 = r2, r1
                elif rxns[r1].reversible:
                    flip(rxns[r1])
                elif rxns[r2].reversible:
                    flip(rxns[r2])
                    r1, r2 = r2, r1
                else:
                    raise RuntimeError("mergeLinear: no producer orientation possible.")
            # Make r2 the consumer.
            if rxns[r2].coeffs[met] > 0:
                if rxns[r2].reversible:
                    flip(rxns[r2])
                else:
                    raise RuntimeError("mergeLinear: no consumer orientation possible.")

            ratio = abs(rxns[r1].coeffs[met] / rxns[r2].coeffs[met])
            merged = defaultdict(float, rxns[r1].coeffs)
            for m, c in rxns[r2].coeffs.items():
                merged[m] += c * ratio
            merged[met] = 0.0
            rxns[r1].coeffs = {m: c for m, c in merged.items() if abs(c) > _TOL}

            # Most-constraining bounds win (RAVEN scales r2's bounds by the ratio).
            if not math.isinf(rxns[r2].lb):
                rxns[r1].lb = max(rxns[r1].lb, rxns[r2].lb / ratio)
            if not math.isinf(rxns[r2].ub):
                rxns[r1].ub = min(rxns[r1].ub, rxns[r2].ub / ratio)
            rxns[r2].coeffs = {}  # cleared → removed after the pass

            grp = max(group_of[rxns[r1].id], group_of[rxns[r2].id]) or next_group
            if grp == next_group:
                next_group += 1
            relabel(rxns[r1], grp)
            relabel(rxns[r2], grp)
            merged_some = True

        if not merged_some:
            break
        rxns = [rx for rx in rxns if rx.coeffs]

    # Number the groups in the order of their first original reaction, so the numbers do not
    # depend on the order in which the reactions were merged.
    number: dict[int, int] = {}
    for o in orig_ids:
        if group_of[o] and group_of[o] not in number:
            number[group_of[o]] = len(number) + 1
    group_ids = [number.get(group_of[o], 0) for o in orig_ids]
    return _build_model(model, rxns), orig_ids, group_ids, [reversed_of[o] for o in orig_ids]


def _build_model(template: cobra.Model, rxns: list[_Rxn]) -> cobra.Model:
    """Assemble the reduced cobra model (gene-free) from the merged working reactions."""
    reduced = cobra.Model(template.id)
    reduced.solver = template.problem  # else cobra's global default, not the template's
    used = {m for rx in rxns for m in rx.coeffs}
    reduced.add_metabolites([
        cobra.Metabolite(m.id, name=m.name, compartment=m.compartment, formula=m.formula)
        for m in template.metabolites if m.id in used  # template order preserved
    ])
    new_rxns = []
    for rx in rxns:
        r = cobra.Reaction(rx.id, name=rx.name, lower_bound=rx.lb, upper_bound=rx.ub)
        new_rxns.append(r)
    reduced.add_reactions(new_rxns)
    for rx, r in zip(rxns, new_rxns, strict=True):
        r.add_metabolites({reduced.metabolites.get_by_id(m): c for m, c in rx.coeffs.items()})
    return reduced


def group_rxn_scores(
    reduced_model: cobra.Model,
    orig_scores: Mapping[str, float],
    orig_rxn_ids: list[str],
    group_ids: list[int],
    to_zero: Iterable[str] = (),
) -> dict[str, float]:
    """Sum original reaction scores over merged groups (RAVEN ``groupRxnScores``).

    ``orig_scores`` maps original reaction id → score; ``to_zero`` are reactions to
    drop from the problem (the ``toIgnore`` masks) — their score becomes 0. Genuine
    zeros and groups cancelling to zero become 0.01 so the MILP never sees an exactly
    zero score. Returns ``{reduced_reaction_id: score}``.
    """
    zero = set(to_zero)
    group_of = dict(zip(orig_rxn_ids, group_ids, strict=True))
    # Per-original adjusted score: genuine 0 → 0.01, then ignore-masked → 0.
    adj: dict[str, float] = {}
    for rid in orig_rxn_ids:
        s = float(orig_scores.get(rid, 0.0))
        s = 0.01 if s == 0.0 else s
        adj[rid] = 0.0 if rid in zero else s
    members: dict[int, list[str]] = defaultdict(list)
    for rid in orig_rxn_ids:
        if group_of[rid] != 0:  # only merged groups need member lists
            members[group_of[rid]].append(rid)

    scores: dict[str, float] = {}
    for r in reduced_model.reactions:
        grp = group_of[r.id]
        if grp == 0:  # unmerged: keep the reaction's own (adjusted) score
            scores[r.id] = adj[r.id]
        else:
            group = members[grp]
            total = sum(adj[m] for m in group)
            if total == 0.0 and any(adj[m] != 0.0 for m in group):
                total = 0.01  # cancelled to zero but had non-zero members
            scores[r.id] = total
    return scores
