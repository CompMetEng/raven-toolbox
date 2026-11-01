"""ChEBI-ontology substrate matching, the substrate-specific layer of transport scoring.

Two curated toolbox-assets artefacts underlie this layer (built by
``scripts/build_transporter_data.py``):

* ``tcdb_substrates.tsv``: the curated TCDB ``TC-ID -> substrate ChEBI(s)`` table;
* ``chebi_relations.tsv.gz``: the ChEBI ``is_a`` graph plus the protonation and tautomer bridges.

Together they match the ChEBI of a model metabolite to the specific substrate that a transporter is
curated to carry. The match is graded and decays with graph distance, and it refines the coarse
family-to-class match. A metabolite that is the curated substrate scores 1.0. A near subtype, or a
protonation or tautomer variant, scores less. A distant, generic relative scores 0 and the coarse
class layer applies instead: almost every metabolite reaches a generic ancestor such as "organic
anion" within a few ``is_a`` steps, so a relative that is only linked through such an ancestor lies
outside the hop budget. The graded match is stricter than the coarse class match, which pairs a
transporter with every metabolite of the same broad class: it gives evidence only to metabolites
that are close to a curated substrate. The layer increases recall: metabolites that the
name-keyword classifier (:func:`~.transport_evidence.default_substrate_of`) misses receive evidence.
"""
from __future__ import annotations

import gzip
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from raven_toolbox.data import ensure_data_file

__all__ = ["SubstrateOntology", "load_tc_substrates"]

# ChEBI relations traversed when rolling a metabolite up toward a transporter substrate. ``is_a`` is
# directed (specific -> general); the protonation/tautomer relations are chemical equivalences, so they
# are followed in both directions (a model anion <-> the TCDB acid/base form).
_SYMMETRIC = frozenset({"is_conjugate_base_of", "is_conjugate_acid_of", "is_tautomer_of"})
# Evidence weight by hop distance to the nearest curated substrate; beyond the last entry the relative
# is too generic to trust and the match is dropped (coarse class takes over). Tunable.
_WEIGHT_BY_HOP: tuple[float, ...] = (1.0, 0.9, 0.8, 0.65, 0.5)
_MAX_HOPS = len(_WEIGHT_BY_HOP) - 1


def _sibling_decay(dist: int) -> float:
    """Weight for two ChEBIs meeting ``dist`` hops apart through their nearest common ancestor.

    Used for the optional sibling credit only. A direct roll-up ends at a substrate that is an
    ancestor of the metabolite. A sibling path climbs to a shared ancestor and back down, so the
    weight decays over the combined distance (minimum 2, for two children of one parent at 1 hop
    each)."""
    return max(0.0, 1.0 - (dist - 1) / (2 * _MAX_HOPS))


def load_tc_substrates(path: str | Path | None = None) -> dict[str, frozenset[str]]:
    """Load ``tcdb_substrates.tsv`` as ``{TC-ID: frozenset(ChEBI)}``.

    The file is downloaded if ``path`` is None."""
    path = ensure_data_file("transporters", "tcdb_substrates.tsv") if path is None else Path(path)
    out: dict[str, frozenset[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        tc, _, chebis = line.partition("\t")
        if chebis:
            out[tc] = frozenset(chebis.split(";"))
    return out


class SubstrateOntology:
    """The ChEBI graph and the TCDB substrate table, with a graded metabolite-to-substrate match.

    Build an instance with :meth:`load` (downloads and caches both artefacts) and pass it to
    :func:`~.transport_evidence.evidence_aware_transport_cost`. Reachability closures are memoised,
    so scoring a whole model reuses the roll-up of each metabolite ChEBI.
    """

    def __init__(self, edges: dict[str, set[str]], tc_substrates: dict[str, frozenset[str]],
                 alt: dict[str, str] | None = None):
        self._edges = edges
        self._tc = tc_substrates
        self._alt = alt or {}  # secondary/deprecated ChEBI id -> connected primary id
        self._reach_cache: dict[str, dict[str, int]] = {}

    @classmethod
    def load(
        cls,
        *,
        relations_path: str | Path | None = None,
        substrates_path: str | Path | None = None,
    ) -> SubstrateOntology:
        """Build a :class:`SubstrateOntology` from the ChEBI relations and TCDB substrates.

        Each toolbox-assets artefact is downloaded and cached when its path is ``None``.
        """
        relations_path = (relations_path if relations_path is not None
                          else ensure_data_file("transporters", "chebi_relations.tsv.gz"))
        edges: dict[str, set[str]] = defaultdict(set)
        alt: dict[str, str] = {}
        with gzip.open(relations_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                child, rel, parent = line.rstrip("\n").split("\t")
                if rel == "alt_id":
                    alt[child] = parent  # secondary -> primary; applied before any graph walk
                    continue
                # Only subtype (is_a) and the chemical-equivalence bridges are roll-up edges. Relations
                # like has_functional_parent are "derived-from" links (a phosphorylated/acylated
                # derivative points at its parent), NOT subtypes, so rolling a derivative up to its
                # parent's specific carrier would score it wrongly cheap — skip them.
                if rel == "is_a":
                    edges[child].add(parent)
                elif rel in _SYMMETRIC:
                    edges[child].add(parent)
                    edges[parent].add(child)
        return cls(dict(edges), load_tc_substrates(substrates_path), alt)

    def substrates_of(self, tc_ids: Iterable[str]) -> frozenset[str]:
        """Union of the curated substrate ChEBIs over a set of full (5-level) TCDB TC-IDs."""
        out: set[str] = set()
        for tc in tc_ids:
            out |= self._tc.get(tc, frozenset())
        return frozenset(out)

    def _reach(self, chebi: str) -> dict[str, int]:
        """``{reachable ChEBI: minimum hop}`` within the hop budget (the ChEBI itself at hop 0).

        The result is memoised."""
        chebi = self._alt.get(chebi, chebi)  # normalise a secondary id onto its connected primary
        cached = self._reach_cache.get(chebi)
        if cached is not None:
            return cached
        reach = {chebi: 0}
        frontier = {chebi}
        for hop in range(1, _MAX_HOPS + 1):
            nxt = set().union(*(self._edges.get(x, ()) for x in frontier)) - reach.keys()
            if not nxt:
                break
            for node in nxt:
                reach[node] = hop
            frontier = nxt
        self._reach_cache[chebi] = reach
        return reach

    def match(self, metabolite_chebis: Iterable[str], substrate_chebis: Iterable[str],
              *, sibling_weight: float = 0.0) -> float:
        """Graded [0, 1] match from the ChEBI(s) of a metabolite to the curated substrate ChEBIs
        of a transporter.

        The direct score rolls the metabolite up its ``is_a`` and protonation ancestry to a curated
        substrate (or the substrate itself): 1.0 for an identical ChEBI, decaying with hop distance.
        With ``sibling_weight > 0``, a metabolite that is below no substrate but shares a near
        common ancestor with one (a chemical relative of the cargo, such as fructose for a
        glucose-only carrier) also scores ``sibling_weight`` multiplied by a decay in the meeting
        distance. The sibling credit increases recall and lowers specificity, and is therefore
        opt-in. The result is 0 when nothing connects within the hop budget.
        """
        subs = frozenset(self._alt.get(s, s) for s in substrate_chebis)  # normalise secondary ids
        if not subs:
            return 0.0
        best = 0.0
        for cm in metabolite_chebis:
            reach_m = self._reach(cm)
            for s in subs:  # direct: the substrate is the metabolite itself or one of its ancestors
                hop = reach_m.get(s)
                if hop is not None and _WEIGHT_BY_HOP[hop] > best:
                    best = _WEIGHT_BY_HOP[hop]
            if best >= 1.0:
                return 1.0
            if sibling_weight <= 0.0:
                continue
            for s in subs:  # sibling: metabolite and substrate meet at a shared ancestor
                if s in reach_m:
                    continue  # already covered by the (stronger) direct score
                reach_s = self._reach(s)
                if cm in reach_s:
                    continue  # metabolite is a *generic ancestor* of the substrate, not a sibling: a
                    # broad metabolite must not earn near-full credit for a specific-substrate carrier
                common = reach_m.keys() & reach_s.keys()
                if common:
                    dist = min(reach_m[a] + reach_s[a] for a in common)
                    score = sibling_weight * _sibling_decay(dist)
                    if score > best:
                        best = score
        return best
