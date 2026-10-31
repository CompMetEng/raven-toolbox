"""Parse the KEGG ``taxonomy`` file into lineages, domains and phylogenetic distances.

Ports RAVEN ``getPhylDist``: the file-reading half (per-organism category lineages and
the domain split) and the distance-matrix half (:func:`phyl_dist`). The ``taxonomy``
file is an indented tree: ``#``-prefixed lines name a category, the number of leading
``#`` giving its depth; organism lines are tab-separated
``T-number<tab>org_id<tab>T-number<tab>name``. Each organism inherits the stack of
categories above it, the first of which is its domain (``Prokaryotes`` or ``Eukaryotes``).

:mod:`.hmm` uses it to split genes into the prokaryote and eukaryote HMM libraries,
:mod:`.organism` uses it in domain mode, and :func:`phyl_dist` uses it to regenerate the KEGG
phylogenetic distance matrix (RAVEN's ``keggPhylDist``) that GECKO uses to select the
closest organism for kcat assignment. This needs no MATLAB ``.mat`` file, only the
published ``taxonomy`` artefact.
"""
from __future__ import annotations

import gzip
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _open_text(path: str | Path):
    """Open ``path`` as UTF-8 text, transparently decompressing a ``.gz`` artefact."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def parse_taxonomy_records(path: str | Path) -> list[tuple[str, str, list[str]]]:
    """Return ``[(organism_id, name, [category, ...]), ...]`` in file order.

    ``name`` is the organism's scientific name with any trailing KEGG parenthetical
    kept (for example ``"Homo sapiens (human)"``), matching RAVEN ``getPhylDist``; the category
    list is the lineage from outermost (domain) to innermost. Order is preserved so it
    aligns with the rows and columns of :func:`phyl_dist`.
    """
    records: list[tuple[str, str, list[str]]] = []
    stack: list[str] = []
    skipped_level_warned = False
    with _open_text(path) as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                depth = len(line) - len(line.lstrip("#"))
                name = line[depth:].strip()
                if depth - 1 > len(stack):
                    # Depth-skip (e.g. ## then ####): pad the missing levels with blanks
                    # so downstream slices stay aligned; warn once.
                    if not skipped_level_warned:
                        warnings.warn(
                            f"{path}: taxonomy depth skips a level near line {line_no} "
                            f"({'#' * depth} {name!r} appeared with stack {stack!r}); "
                            "padding the missing levels with '' (later occurrences silenced).",
                            stacklevel=2,
                        )
                        skipped_level_warned = True
                    stack = stack + [""] * (depth - 1 - len(stack))
                else:
                    stack = stack[: depth - 1]
                stack.append(name)
            else:
                # Organism line. RAVEN reads the id between the 1st/2nd whitespace and the
                # name after the 3rd; with the tab-delimited file that is fields[1]/fields[3].
                if "\t" in line:
                    fields = line.split("\t")
                    org_id = fields[1].strip() if len(fields) > 1 else ""
                    org_name = fields[3].strip() if len(fields) > 3 else ""
                else:
                    fields = line.split()
                    org_id = fields[1] if len(fields) > 1 else ""
                    org_name = " ".join(fields[3:]) if len(fields) > 3 else ""
                if not org_id:
                    continue
                records.append((org_id, org_name, list(stack)))
    return records


def parse_taxonomy(path: str | Path) -> dict[str, list[str]]:
    """Return ``{organism_id: [category, ...]}`` from outermost to innermost."""
    return {org_id: lineage for org_id, _name, lineage in parse_taxonomy_records(path)}


def organism_domains(path: str | Path) -> dict[str, str]:
    """Return ``{organism_id: domain}`` (the outermost category)."""
    return {org: cats[0] for org, cats in parse_taxonomy(path).items() if cats}


def organisms_in_domain(path: str | Path, domain: str) -> set[str]:
    """Organism ids whose outermost category matches ``domain`` (case-insensitive).

    Accepts a prefix, so ``"prok"`` matches ``"Prokaryotes"``.
    """
    needle = domain.lower()
    return {
        org
        for org, dom in organism_domains(path).items()
        if dom.lower().startswith(needle) or needle.startswith(dom.lower())
    }


@dataclass
class PhylDist:
    """KEGG phylogenetic distance structure of RAVEN ``getPhylDist`` and ``keggPhylDist``.

    ``ids`` and ``names`` are the KEGG organism ids and scientific names (names keep the
    trailing parenthetical of RAVEN; consumers that need them without it, for example
    geckopy, strip it). ``dist_matrix[i, j]`` is RAVEN's distance from ``ids[i]`` to
    ``ids[j]``. It reproduces RAVEN's metric, which is asymmetric and may be negative;
    GECKO uses it only to select the closest (minimum-distance) organism, so the exact
    values matter for parity with MATLAB, and the sign and asymmetry do not affect that
    selection.
    """

    ids: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    dist_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.int64))


def phyl_dist(path: str | Path, *, only_in_kingdom: bool = False) -> PhylDist:
    """Compute the KEGG phylogenetic distance matrix from the ``taxonomy`` file.

    Port of RAVEN ``getPhylDist``. For organisms ``i`` and ``j`` with category
    lineages of lengths ``Li, Lj``, the distance is ``(Li - Lj) + min(Li, Lj) - k`` where
    ``k`` is the deepest level (within the shorter lineage, comparing position by position
    from the root) at which the two categories coincide, or ``1`` if none coincide. This
    reproduces the values of RAVEN, including their asymmetry and negative values, so the
    output matches MATLAB ``keggPhylDist``, which GECKO uses.

    ``only_in_kingdom`` sets the distance to ``+inf`` between organisms in different
    domains (RAVEN's ``onlyInKingdom`` variant); the matrix is then returned as ``float``.

    Time and memory use are ``O(N²)``, with ``N`` the number of KEGG organisms (~10⁴).
    """
    records = parse_taxonomy_records(path)
    ids = [r[0] for r in records]
    names = [r[1] for r in records]
    lineages = [r[2] for r in records]
    n = len(records)
    if n == 0:
        return PhylDist(ids, names, np.empty((0, 0), dtype=np.int64))

    lengths = np.array([len(lin) for lin in lineages], dtype=np.int32)
    max_depth = int(lengths.max())

    # deepest[i, j] = deepest level d (1-based) at which lineage_i[d-1] == lineage_j[d-1],
    # counting only levels present in both (d <= min(Li, Lj)); 0 if none coincide.
    deepest = np.zeros((n, n), dtype=np.int32)
    for d in range(1, max_depth + 1):
        lookup: dict[str, int] = {}
        lab = np.full(n, -1, dtype=np.int64)
        for i, lin in enumerate(lineages):
            if len(lin) >= d:
                lab[i] = lookup.setdefault(lin[d - 1], len(lookup))
        eq = (lab[:, None] == lab[None, :]) & (lab[:, None] != -1)
        # Ascending d, so the last (deepest) coinciding level wins -> max coinciding depth.
        deepest[eq] = d
    k = np.where(deepest == 0, 1, deepest).astype(np.int32)

    # Values are tiny (|dist| <= max lineage depth); int32 throughout keeps the N x N
    # matrix (and its intermediates) half the size of the int64 default.
    li = lengths[:, None]
    lj = lengths[None, :]
    dist = (li - lj) + np.minimum(li, lj) - k

    if only_in_kingdom:
        domains = np.array([lin[0] if lin else "" for lin in lineages])
        dist = dist.astype(float)
        dist[domains[:, None] != domains[None, :]] = np.inf

    return PhylDist(ids=ids, names=names, dist_matrix=dist)
