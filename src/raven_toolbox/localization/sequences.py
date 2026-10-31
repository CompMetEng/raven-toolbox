"""Prepare input for sequence-based localisation predictors (DeepLoc 2.1, MULocDeep, …).

These predictors take protein sequences in FASTA and write a per-protein localisation table that
:func:`load_deeploc` or :func:`load_mulocdeep` reads. This module fetches protein sequences
for the genes of a model from UniProtKB and writes a FASTA whose headers are the gene ids of the
model, so the output of the predictor matches the model and the loaders here:

    fetch + write FASTA ─▶ run DeepLoc 2.1 (external) ─▶ load_deeploc(output) ─▶
    predict_localization

DeepLoc 2.1 has no public batch API. It runs on the FASTA that this module writes, either on the web
server (https://services.healthtech.dtu.dk/services/DeepLoc-2.1/, **max 500 sequences per
submission**; :func:`prepare_deeploc_input` splits the output into files of that size) or with the
downloadable standalone (no per-submission limit). DeepLoc requires sequences of **≥ 10 amino
acids**.

Sequences come from UniProtKB by the same identifier that the localisation loaders use
(``gene_oln`` = the ORF / ordered-locus name like ``YNR001C`` for yeast-GEM). Genes without a
sequence are listed in :attr:`PreparedFasta.missing`; with the default ``reviewed=True`` a gene is
missing when UniProtKB has no reviewed (Swiss-Prot) sequence for it.
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

    A UniProt entry listing several ids for the field maps each of them to its sequence. The query
    is the same as in :func:`fetch_uniprot_localization`. Raises :class:`ValueError` when
    ``id_field`` is not one of the accepted values.

    Parameters
    ----------
    organism : int or str
        UniProt organism or taxon id (for example ``559292`` for *S. cerevisiae* S288C).
    genes : iterable of str or None
        Restricts the result to those gene ids (other entries are fetched and dropped). ``None``
        (default) returns every entry.
    id_field : str
        Identifier used as key (default ``"gene_oln"``): ``"gene_oln"`` (ordered locus, the ORF
        name like ``YNR001C``; matches yeast-GEM gene ids), ``"accession"`` or ``"gene_primary"``.
    reviewed : bool
        If True (default), keep only curated Swiss-Prot entries.
    extra_query : str or None
        UniProt query string that is ANDed into the query (default ``None``).
    min_length : int
        Drops sequences shorter than it (default ``1``; DeepLoc 2.1 needs >= 10 aa).
    timeout : float
        Timeout of the HTTP request in seconds (default ``120.0``).
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
    space and many tools identify results by that token, so ids must not contain whitespace; a
    ``ValueError`` is raised otherwise.

    Parameters
    ----------
    sequences : mapping of str to str
        Maps each FASTA header to its sequence. An empty sequence gives the header line only.
    path : str or Path
        Output file. With chunking, the chunk number is added to the stem (see
        ``max_records_per_file``).
    max_records_per_file : int or None
        With a value set (for example ``500`` for the DeepLoc 2.1 web server) and more records than
        that, records are split across ``<stem>_001<suffix>``, ``<stem>_002<suffix>``, ... and
        every path is returned. Otherwise (default ``None``) a single file is written at ``path``.
    wrap : int
        Residues per line (default ``60``; ``<= 0`` gives one line per sequence).
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
    """Write a FASTA of protein sequences for ``genes`` as input for DeepLoc 2.1.

    Sequences are fetched from UniProtKB (see :func:`fetch_protein_sequences`). Returns a
    :class:`PreparedFasta` with the file path(s) and the genes that had no sequence (with the
    default ``reviewed=True``, no reviewed sequence). The CSV that DeepLoc 2.1 writes for these
    files is read with :func:`load_deeploc`; its ``Protein_ID`` column holds these gene ids.

    Parameters
    ----------
    genes : iterable of str or cobra.Model
        Gene ids, or a cobra model (its ``.genes`` ids are used). For yeast-GEM those are the ORF
        names that match ``id_field="gene_oln"``.
    organism : int or str
        UniProt organism or taxon id, as in :func:`fetch_protein_sequences`.
    path : str or Path
        Output FASTA file. When the output is split, the chunk number is added to the stem.
    max_records_per_file : int or None
        The FASTA is split into files of this many records (default ``500``, the DeepLoc 2.1
        web-server limit). ``None`` writes one file, for the standalone tool, which has no limit.
    min_length : int
        Drops sequences below this length (default ``10``, the DeepLoc minimum of 10 aa).
    **fetch_kwargs : dict
        Forwarded to :func:`fetch_protein_sequences`, for example ``reviewed``, ``id_field``,
        ``extra_query`` and ``timeout``.
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
