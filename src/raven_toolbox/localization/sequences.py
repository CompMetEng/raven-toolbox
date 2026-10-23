"""Prepare input for sequence-based localisation predictors (DeepLoc 2.1, MULocDeep, …).

These predictors take protein sequences in **FASTA** and emit a per-protein localisation table you
then load with :func:`load_deeploc` / :func:`load_mulocdeep`. This module fetches protein sequences
for a model's genes from UniProtKB and writes a FASTA whose headers are the model's **gene ids**, so
the predictor's output lines up with the model — and with the loaders here — directly:

    fetch + write FASTA ─▶ run DeepLoc 2.1 (you) ─▶ load_deeploc(output) ─▶ predict_localization

**There is no public batch API for DeepLoc 2.1.** Run it yourself on the FASTA this writes, either on
the web server (https://services.healthtech.dtu.dk/services/DeepLoc-2.1/, **max 500 sequences per
submission** — :func:`prepare_deeploc_input` chunks for you) or with the downloadable standalone
(no per-submission limit). DeepLoc requires sequences of **≥ 10 amino acids**.

Sequences come from UniProtKB by the same identifier the localisation loaders key on
(``gene_oln`` = the ORF / ordered-locus name like ``YNR001C`` for yeast-GEM). Genes with no reviewed
UniProt entry are reported back to you rather than silently dropped.
"""
from __future__ import annotations

import io
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import cobra
import pandas as pd

__all__ = [
    "fetch_protein_sequences",
    "write_fasta",
    "prepare_deeploc_input",
    "PreparedFasta",
]

#: UniProt field → its TSV column header (matches :mod:`raven_toolbox.localization.scores`).
_ID_COLUMN = {
    "accession": "Entry",
    "gene_primary": "Gene Names (primary)",
    "gene_oln": "Gene Names (ordered locus)",
}


def fetch_protein_sequences(
    organism: int | str,
    *,
    genes: Iterable[str] | None = None,
    id_field: str = "gene_oln",
    reviewed: bool = True,
    extra_query: str | None = None,
    min_length: int = 1,
    timeout: float = 120.0,
) -> dict[str, str]:
    """Fetch ``{gene_id: protein_sequence}`` from the UniProtKB REST API for an organism.

    ``organism`` is a UniProt organism/taxon id (e.g. ``559292`` for *S. cerevisiae* S288C).
    ``id_field`` picks which identifier becomes the key — ``"gene_oln"`` (ordered locus, the ORF name
    like ``YNR001C``) matches yeast-GEM gene ids; also ``"accession"`` or ``"gene_primary"``. A
    UniProt entry listing several ids for the field maps each of them to its sequence.

    ``genes`` restricts the result to those gene ids (others are still fetched but dropped); ``None``
    returns every entry. ``reviewed=True`` keeps only curated Swiss-Prot entries; ``extra_query`` is
    ANDed into the UniProt query. ``min_length`` drops sequences shorter than it (DeepLoc 2.1 needs
    ≥ 10 aa). The query mirrors :func:`fetch_uniprot_localization`.
    """
    if id_field not in _ID_COLUMN:
        raise ValueError(f"id_field must be one of {list(_ID_COLUMN)}")
    query = f"organism_id:{organism}"
    if reviewed:
        query += " AND reviewed:true"
    if extra_query:
        query += f" AND ({extra_query})"
    params = {"query": query, "format": "tsv",
              "fields": "accession,gene_primary,gene_oln,sequence"}
    url = "https://rest.uniprot.org/uniprotkb/stream?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "raven-toolbox"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed https host
        text = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str).fillna("")
    return _sequences_from_frame(df, id_column=_ID_COLUMN[id_field],
                                 wanted=set(genes) if genes is not None else None,
                                 min_length=min_length)


def _sequences_from_frame(df: pd.DataFrame, *, id_column: str,
                          wanted: set[str] | None, min_length: int) -> dict[str, str]:
    """Map each id token in ``id_column`` to its (longest, if duplicated) ``Sequence``."""
    if id_column not in df.columns or "Sequence" not in df.columns:
        raise ValueError(f"expected '{id_column}' and 'Sequence' columns, got {list(df.columns)}")
    out: dict[str, str] = {}
    for ids, seq in zip(df[id_column].astype(str), df["Sequence"].astype(str), strict=True):
        seq = seq.strip()
        if len(seq) < min_length:
            continue
        for gid in ids.split():                      # an entry may list several ORF names
            if wanted is not None and gid not in wanted:
                continue
            # keep the longest sequence if an id appears twice (canonical over fragment)
            if gid not in out or len(seq) > len(out[gid]):
                out[gid] = seq
    return out


@dataclass
class PreparedFasta:
    """Outcome of :func:`prepare_deeploc_input`."""

    paths: list[Path]                      # FASTA file(s) written (chunked if needed)
    n_requested: int                       # distinct genes asked for
    n_written: int                         # genes with a sequence written
    missing: list[str] = field(default_factory=list)  # requested genes with no reviewed sequence

    def __str__(self) -> str:
        files = ", ".join(p.name for p in self.paths)
        return (f"{self.n_written}/{self.n_requested} sequences -> {files}"
                f" ({len(self.missing)} missing)")


def write_fasta(sequences: Mapping[str, str], path: str | Path, *,
                max_records_per_file: int | None = None, wrap: int = 60) -> list[Path]:
    """Write ``{id: sequence}`` to FASTA and return the file path(s).

    Headers are the dict keys verbatim (no leading ``>``). FASTA truncates a header at the first
    space and many tools key results on that token, so ids **must not contain whitespace** — a
    ``ValueError`` is raised otherwise (it would silently corrupt the id↔result mapping).

    With ``max_records_per_file`` set (e.g. ``500`` for the DeepLoc 2.1 web server), records are
    split across ``<stem>_001<suffix>``, ``<stem>_002<suffix>``, … and every path is returned;
    otherwise a single file is written at ``path``. ``wrap`` sets the residues-per-line (≤ 0 = one
    line per sequence). An empty sequence gives the header line only.
    """
    items = list(sequences.items())
    bad = [k for k, _ in items if k != k.strip() or any(ch.isspace() for ch in k)]
    if bad:
        raise ValueError(f"sequence ids must not contain whitespace (FASTA headers): {bad[:5]}")

    path = Path(path)
    if max_records_per_file and len(items) > max_records_per_file:
        chunks = [items[i:i + max_records_per_file]
                  for i in range(0, len(items), max_records_per_file)]
    else:
        chunks = [items]

    written: list[Path] = []
    multi = len(chunks) > 1
    for n, chunk in enumerate(chunks, start=1):
        target = path.with_name(f"{path.stem}_{n:03d}{path.suffix}") if multi else path
        target.write_text("".join(_fasta_record(gid, seq, wrap) for gid, seq in chunk),
                          encoding="utf-8", newline="")   # LF on every platform (FASTA convention)
        written.append(target)
    return written


def _fasta_record(gid: str, seq: str, wrap: int) -> str:
    if not seq:
        return f">{gid}\n"
    body = seq if wrap <= 0 else "\n".join(seq[i:i + wrap] for i in range(0, len(seq), wrap))
    return f">{gid}\n{body}\n"


def prepare_deeploc_input(
    genes: Iterable[str] | cobra.Model,
    organism: int | str,
    path: str | Path,
    *,
    max_records_per_file: int | None = 500,
    min_length: int = 10,
    **fetch_kwargs,
) -> PreparedFasta:
    """Write a DeepLoc-2.1-ready FASTA of protein sequences for ``genes``.

    ``genes`` is an iterable of gene ids, **or** a cobra model (its ``.genes`` ids are used) — for
    yeast-GEM those are the ORF names that line up with ``id_field="gene_oln"``. Sequences are
    fetched from UniProtKB (see :func:`fetch_protein_sequences`; pass e.g. ``reviewed=``,
    ``id_field=`` through ``fetch_kwargs``). The FASTA is chunked at ``max_records_per_file``
    (default ``500`` — the DeepLoc 2.1 web-server limit; pass ``None`` for the standalone tool which
    has no limit). ``min_length`` drops sequences below DeepLoc's 10-aa minimum.

    Returns a :class:`PreparedFasta` with the file path(s) and the genes that had no reviewed
    sequence, so you can chase those separately. Run DeepLoc 2.1 on the file(s), then read its CSV
    with :func:`load_deeploc` — the ``Protein_ID`` column will be these gene ids.
    """
    gene_ids = [g.id for g in genes.genes] if isinstance(genes, cobra.Model) else list(genes)
    requested = list(dict.fromkeys(gene_ids))        # de-dupe, keep order
    found = fetch_protein_sequences(organism, genes=requested, min_length=min_length,
                                    **fetch_kwargs)
    ordered = {g: found[g] for g in requested if g in found}   # model order, only those found
    missing = [g for g in requested if g not in found]
    paths = write_fasta(ordered, path, max_records_per_file=max_records_per_file)
    return PreparedFasta(paths=paths, n_requested=len(requested),
                         n_written=len(ordered), missing=missing)
