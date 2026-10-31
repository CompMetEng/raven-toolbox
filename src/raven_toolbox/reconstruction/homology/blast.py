"""Run BLAST+ or DIAMOND, or load precomputed hits, into a homology hits table.

Each producer returns the bidirectional hits DataFrame (``HIT_COLUMNS``) consumed by
:func:`~raven_toolbox.reconstruction.homology.get_model_from_homology`. Binaries are
located via :func:`raven_toolbox.binaries.resolve_binary` (argument, environment variable,
PATH, bundled binary, in that order).
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from raven_toolbox.binaries import resolve_binary
from raven_toolbox.reconstruction.homology.hits import HIT_COLUMNS, validate_hits

# Tabular output columns requested from BLAST+/DIAMOND, in order.
_OUTFMT_FIELDS = ["qseqid", "sseqid", "evalue", "pident", "length", "bitscore", "ppos"]
_FIELD_TO_HIT = {
    "qseqid": "from_gene", "sseqid": "to_gene", "evalue": "evalue",
    "pident": "identity", "length": "align_len", "bitscore": "bitscore", "ppos": "ppos",
}


def _parse_tabular(text: str, from_id: str, to_id: str, sep: str) -> pd.DataFrame:
    """Parse one BLAST or DIAMOND tabular output into hit rows for one direction."""
    if not text.strip():
        return pd.DataFrame(columns=HIT_COLUMNS)
    # round_trip parses each number exactly, as RAVEN does: the default parser can read
    # an e-value printed as 1e-30 as 9.999999999999999e-31, which passes max_evalue=1e-30.
    df = pd.read_csv(io.StringIO(text), sep=sep, names=_OUTFMT_FIELDS, dtype={0: str, 1: str},
                     float_precision="round_trip")
    df = df.rename(columns=_FIELD_TO_HIT)
    df["from_id"] = from_id
    df["to_id"] = to_id
    return df[HIT_COLUMNS]


def _as_list(x):
    return [x] if isinstance(x, (str, Path)) else list(x)


def _available_cpus() -> int:
    """Number of CPUs this process may run on, the count MATLAB's ``feature('numcores')``
    reports as assigned by the OS and RAVEN passes to the aligners."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def run_blast(
    organism_id: str,
    fasta: str | Path,
    model_ids: Sequence[str],
    ref_fastas: Sequence[str | Path],
    *,
    evalue: float = 1e-4,
    threads: int | None = None,
    blastp: str | Path | None = None,
    makeblastdb: str | Path | None = None,
) -> pd.DataFrame:
    """Bidirectional BLAST+ between an organism and template organisms.

    ``model_ids`` and ``ref_fastas`` are parallel: ``ref_fastas[i]`` is the
    reference proteome for template ``model_ids[i]``.

    Returns the hits DataFrame, filtered at ``evalue``. Requires BLAST+
    (``blastp``, ``makeblastdb``). ``threads`` defaults to every CPU the process
    may use, as in RAVEN's ``getBlast``.

    The default ``evalue`` of 1e-4 equals the ``-evalue 10e-5`` that MATLAB
    RAVEN's ``getBlast`` hardcodes. Here ``evalue`` is an argument. The cutoff is
    a pre-filter; the ``max_evalue``, ``min_align_len`` and ``min_identity``
    arguments of
    :func:`~raven_toolbox.reconstruction.homology.homology.get_model_from_homology`
    apply stricter limits downstream.
    """
    model_ids = list(model_ids)
    ref_fastas = _as_list(ref_fastas)
    if len(model_ids) != len(ref_fastas):
        raise ValueError("model_ids and ref_fastas must have the same length.")
    threads = _available_cpus() if threads is None else threads
    blastp = resolve_binary("blastp", binary=blastp)
    makeblastdb = resolve_binary("makeblastdb", binary=makeblastdb)
    outfmt = "10 " + " ".join(_OUTFMT_FIELDS)  # 10 = CSV

    frames = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        def blastp_dir(query, subject_fasta, from_id, to_id):
            db = tmp / f"db_{from_id}_{to_id}"
            _run([makeblastdb, "-in", str(subject_fasta), "-dbtype", "prot", "-out", str(db)])
            out = _run([
                blastp, "-query", str(query), "-db", str(db), "-evalue", str(evalue),
                "-outfmt", outfmt, "-num_threads", str(threads),
            ])
            return _parse_tabular(out, from_id, to_id, sep=",")

        for model_id, ref in zip(model_ids, ref_fastas, strict=True):
            # template -> organism, and organism -> template
            frames.append(blastp_dir(ref, fasta, model_id, organism_id))
            frames.append(blastp_dir(fasta, ref, organism_id, model_id))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HIT_COLUMNS)


def run_diamond(
    organism_id: str,
    fasta: str | Path,
    model_ids: Sequence[str],
    ref_fastas: Sequence[str | Path],
    *,
    evalue: float = 1e-3,
    threads: int | None = None,
    sensitivity: str = "--more-sensitive",
    diamond: str | Path | None = None,
) -> pd.DataFrame:
    """Bidirectional DIAMOND between an organism and template organisms.

    ``model_ids`` and ``ref_fastas`` are parallel: ``ref_fastas[i]`` is the
    reference proteome for template ``model_ids[i]``. ``sensitivity`` is
    passed unchanged as a DIAMOND command-line flag (for example ``"--more-sensitive"``
    or ``"--sensitive"``); ``""`` selects DIAMOND's default fast mode.

    Returns the hits DataFrame. Requires DIAMOND. ``threads`` defaults to
    every CPU the process may use, as in RAVEN's ``getDiamond``.

    The default ``evalue`` of 1e-3 equals DIAMOND's own default, which MATLAB
    RAVEN's ``getDiamond`` uses because it passes no ``--evalue``. ``run_blast``
    uses 1e-4, following ``getBlast``. Neither cutoff has been calibrated
    against data, unlike the KO-assignment cutoffs of
    :func:`~raven_toolbox.reconstruction.kegg.assign_kos`, where measurement justified
    diverging from RAVEN.
    """
    model_ids = list(model_ids)
    ref_fastas = _as_list(ref_fastas)
    if len(model_ids) != len(ref_fastas):
        raise ValueError("model_ids and ref_fastas must have the same length.")
    threads = _available_cpus() if threads is None else threads
    diamond = resolve_binary("diamond", binary=diamond)

    frames = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        def diamond_dir(query, subject_fasta, from_id, to_id):
            db = tmp / f"db_{from_id}_{to_id}"
            _run([diamond, "makedb", "--in", str(subject_fasta), "--db", str(db)])
            cmd = [diamond, "blastp", "--query", str(query), "--db", str(db),
                   "--evalue", str(evalue), "--outfmt", "6", *_OUTFMT_FIELDS,
                   "--threads", str(threads)]
            if sensitivity:
                cmd.append(sensitivity)
            return _parse_tabular(_run(cmd), from_id, to_id, sep="\t")

        for model_id, ref in zip(model_ids, ref_fastas, strict=True):
            frames.append(diamond_dir(ref, fasta, model_id, organism_id))
            frames.append(diamond_dir(fasta, ref, organism_id, model_id))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HIT_COLUMNS)


def blast_from_table(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a precomputed homology hits table from a CSV path or a DataFrame (not Excel).

    Must contain the ``HIT_COLUMNS`` columns.
    """
    # Force gene-id columns to str: an all-numeric gene-id column (e.g. Entrez ids)
    # would otherwise be read as int64 and never match the string gene ids in a model.
    df = (source if isinstance(source, pd.DataFrame)
          else pd.read_csv(source, dtype={"from_gene": str, "to_gene": str},
                           float_precision="round_trip"))
    validate_hits(df)
    return df[HIT_COLUMNS].copy()
