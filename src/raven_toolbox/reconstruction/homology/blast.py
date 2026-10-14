"""Run BLAST+ / DIAMOND (or load precomputed hits) into a homology hits table.

Each producer returns the bidirectional hits DataFrame (``HIT_COLUMNS``) consumed by
:func:`~raven_toolbox.reconstruction.homology.get_model_from_homology`. Binaries are
located via :func:`raven_toolbox.binaries.resolve_binary` (arg → env → PATH → bundled).
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
    """Parse one BLAST/DIAMOND tabular output into hit rows for one direction."""
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
    threads: int = max(1, (os.cpu_count() or 2) - 1),
    blastp: str | Path | None = None,
    makeblastdb: str | Path | None = None,
) -> pd.DataFrame:
    """Bidirectional BLAST+ between an organism and template organisms.

    ``model_ids`` and ``ref_fastas`` are parallel: ``ref_fastas[i]`` is the
    reference proteome for template ``model_ids[i]``.

    Returns the hits DataFrame (filtered at
    ``evalue``). Requires BLAST+ (`blastp`, `makeblastdb`).

    The default matches MATLAB RAVEN's ``getBlast``, which hardcodes
    ``-evalue 10e-5`` -- that is 1e-4, not 1e-5. Unlike RAVEN's, this one is an
    argument, so a stricter search stays available; it's also a loose
    pre-filter only, dominated downstream by
    :func:`~raven_toolbox.reconstruction.homology.homology.get_model_from_homology`'s
    much stricter ``max_evalue``/``min_align_len``/``min_identity``.
    """
    model_ids = list(model_ids)
    ref_fastas = _as_list(ref_fastas)
    if len(model_ids) != len(ref_fastas):
        raise ValueError("model_ids and ref_fastas must have the same length.")
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
    threads: int = max(1, (os.cpu_count() or 2) - 1),
    sensitivity: str = "--more-sensitive",
    diamond: str | Path | None = None,
) -> pd.DataFrame:
    """Bidirectional DIAMOND between an organism and template organisms.

    ``model_ids`` and ``ref_fastas`` are parallel: ``ref_fastas[i]`` is the
    reference proteome for template ``model_ids[i]``. ``sensitivity`` is
    passed straight through as a DIAMOND CLI flag (e.g. ``"--more-sensitive"``,
    ``"--sensitive"``, or ``""`` to use DIAMOND's default fast mode).

    Returns the hits DataFrame. Requires DIAMOND.

    The default matches MATLAB RAVEN's ``getDiamond``, which passes no
    ``--evalue`` at all and so inherits DIAMOND's own default of 1e-3. Stated
    explicitly here rather than left implicit, since the two aligners disagree:
    ``run_blast`` uses 1e-4, following ``getBlast``. Neither cutoff has been
    calibrated against data on either side -- unlike the KO-assignment cutoffs,
    where measurement did justify diverging from RAVEN.
    """
    model_ids = list(model_ids)
    ref_fastas = _as_list(ref_fastas)
    if len(model_ids) != len(ref_fastas):
        raise ValueError("model_ids and ref_fastas must have the same length.")
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
    """Load a precomputed homology hits table (CSV path or DataFrame, not Excel).

    Must contain the ``HIT_COLUMNS`` columns.
    """
    # Force gene-id columns to str: an all-numeric gene-id column (e.g. Entrez ids)
    # would otherwise be read as int64 and never match the string gene ids in a model.
    df = (source if isinstance(source, pd.DataFrame)
          else pd.read_csv(source, dtype={"from_gene": str, "to_gene": str},
                           float_precision="round_trip"))
    validate_hits(df)
    return df[HIT_COLUMNS].copy()
