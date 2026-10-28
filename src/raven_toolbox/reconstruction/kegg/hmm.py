"""Build per-KO HMM libraries from KEGG sequences (step 3b.3, maintainer-side).

Ports RAVEN ``constructMultiFasta`` plus the clustering/alignment/training stages
of ``getKEGGModelForOrganism``. Run once per KEGG release to produce the per-domain
(``kegg<version>_prokaryotes`` / ``kegg<version>_eukaryotes``) HMM libraries that the
de-novo query path (3b.5) searches.

Per KO, within one domain (prokaryote / eukaryote):

1. **Multi-FASTA** — gather the member genes' sequences from ``genes.pep``
   (:func:`build_ko_fastas`).
2. **CD-HIT** — dereplicate near-identical sequences (default 90 % identity).
3. **MAFFT** — multiple-sequence alignment (``--auto --anysymbol``).
4. **hmmbuild** — train the profile HMM.

Finally the per-KO HMMs are concatenated into a single library: end users search it
with a single ``hmmsearch`` over the multi-profile query (the fast direction), replacing
RAVEN's per-KO ``hmmsearch`` sweep — one invocation, no ``hmmpress``.

The pure parts (FASTA indexing/grouping, command construction, CD-HIT ``-n``
choice) are unit-tested; running the binaries needs HMMER/MAFFT/CD-HIT, located
via :func:`raven_toolbox.binaries.resolve_binary`.
"""
from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from raven_toolbox.binaries import resolve_binary
from raven_toolbox.reconstruction.homology.blast import _available_cpus
from raven_toolbox.reconstruction.kegg.taxonomy import organisms_in_domain

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Step 1 — per-KO multi-FASTA (constructMultiFasta)
# --------------------------------------------------------------------------- #
def _full_id(organism: str, gene: str) -> str:
    """The genes.pep header key for a gene, i.e. ``organism:gene``."""
    return f"{organism}:{gene}"


def _index_fasta(path: str | Path, wanted: set[str]) -> dict[str, tuple[int, int]]:
    """Map each wanted record id to its ``(start, end)`` byte span in ``path``.

    The record id is the first whitespace-delimited token of the ``>`` header.
    One streaming pass; only wanted ids are kept (memory stays small).
    """
    index: dict[str, tuple[int, int]] = {}
    cur_id: str | None = None
    cur_start = 0
    pos = 0
    with open(path, "rb") as handle:
        for line in handle:
            if line.startswith(b">"):
                if cur_id is not None and cur_id in wanted:
                    index[cur_id] = (cur_start, pos)
                cur_id = line[1:].split(None, 1)[0].decode()
                cur_start = pos
            pos += len(line)
    if cur_id is not None and cur_id in wanted:
        index[cur_id] = (cur_start, pos)
    return index


def build_ko_fastas(
    organism_gene_ko: pd.DataFrame,
    genes_pep: str | Path,
    out_dir: str | Path,
    *,
    organisms: set[str] | None = None,
    progress: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """Write one ``<KO>.fa`` per KO with its member genes' sequences.

    Uses a stdlib offset index instead of a Java-hashtable byte scan.
    ``organisms`` restricts to a domain's organism codes (for the prok/euk
    split). Empty KOs are skipped (no file).
    ``progress`` shows a tqdm bar over the per-KO writing pass. Returns
    ``{ko: path}`` for the files written.

    **Resumable:** a ``<KO>.fa`` that already exists is kept (not rewritten), and
    once a full pass completes a ``.ko_fastas_complete`` marker is written so a
    later call returns the existing files without re-scanning ``genes_pep`` (the
    costly part). ``force`` ignores both and rebuilds. The resume assumes the same
    inputs as the original run (as the per-KO HMM resume in :func:`build_hmm_library`
    does); use ``force`` after changing ``organism_gene_ko``/``organisms``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".ko_fastas_complete"

    rows = organism_gene_ko
    if organisms is not None:
        rows = rows[rows["organism"].isin(organisms)]

    ko_to_ids: dict[str, list[str]] = {}
    for organism, gene, ko in zip(rows["organism"], rows["gene"], rows["ko"], strict=True):
        ko_to_ids.setdefault(ko, []).append(_full_id(organism, gene))

    # Fast path: a prior run finished, so trust the existing files — no genes.pep scan.
    if marker.exists() and not force:
        return {ko: out_dir / f"{ko}.fa" for ko in ko_to_ids if (out_dir / f"{ko}.fa").exists()}

    # Keep already-written fastas; only the remaining KOs need a genes.pep scan, so a
    # partial resume re-indexes just the leftovers (a crash leaves no marker).
    written: dict[str, Path] = {}
    todo: dict[str, list[str]] = {}
    for ko, ids in ko_to_ids.items():
        path = out_dir / f"{ko}.fa"
        if path.exists() and not force:
            written[ko] = path
        else:
            todo[ko] = ids

    if todo:
        wanted = {fid for ids in todo.values() for fid in ids}
        index = _index_fasta(genes_pep, wanted)
        with open(genes_pep, "rb") as src:
            for ko, ids in tqdm(
                todo.items(), desc="multi-FASTA", unit="KO", disable=not progress
            ):
                present = sorted({i for i in ids if i in index})
                if not present:
                    continue
                path = out_dir / f"{ko}.fa"
                with open(path, "wb") as out:
                    for fid in present:
                        start, end = index[fid]
                        src.seek(start)
                        out.write(src.read(end - start))
                written[ko] = path

    marker.write_text("")  # full pass over all KOs done → safe to fast-path next time
    return written


# --------------------------------------------------------------------------- #
# Steps 2-4 — cluster, align, train (one KO)
# --------------------------------------------------------------------------- #
def _cdhit_word_size(seq_identity: float) -> str:
    """CD-HIT ``-n`` word size for a given identity threshold (per CD-HIT guide)."""
    if not 0.4 < seq_identity <= 1.0:
        raise ValueError("seq_identity must be in (0.4, 1.0] (or -1 to skip CD-HIT).")
    if seq_identity > 0.7:
        return "5"
    if seq_identity > 0.6:
        return "4"
    if seq_identity > 0.5:
        return "3"
    return "2"


def _count_sequences(fasta: Path) -> int:
    with open(fasta, "rb") as fh:
        return sum(1 for line in fh if line.startswith(b">"))


def _fasta_stats(fasta: Path) -> tuple[int, int]:
    """Return ``(sequence_count, total_residues)`` in one pass."""
    n = residues = 0
    with open(fasta, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                n += 1
            else:
                residues += len(line.strip())
    return n, residues


def _cdhit_cmd(cdhit: str, inp: Path, out: Path, seq_identity: float, threads: int) -> list[str]:
    return [
        cdhit, "-i", str(inp), "-o", str(out),
        "-c", str(seq_identity), "-n", _cdhit_word_size(seq_identity),
        "-M", "2000", "-T", str(threads),
    ]


# MAFFT uses fast progressive FFT-NS-2 until an alignment is large enough to
# threaten memory, then switches to memory-light PartTree (which keeps all
# sequences; only the guide tree is approximated).
#
# Peak FFT-NS-2 RSS is driven by the progressive-alignment DP work, ~ n_seqs ×
# (mean length)^2  (equivalently residues^2 / n_seqs) — NOT residue count alone:
# a few hundred long proteins cost far more than the same residues spread over
# many short ones. Empirical fit (real KEGG sequences, 12 threads):
#     RSS_GB ≈ _MAFFT_GB_PER_COST × (n_seqs × mean_len^2)
# Measured (residues, n_seqs, RSS): 250k/266/0.67, 500k/534/1.25, 1.0M/1066/3.16,
# 1.5M/1624/5.73, and K12047 941k/452 (mean len 2082) which OOM'd >7 GB — its
# cost 1.96e9 is the largest of all, hence the length-aware metric.
_MAFFT_GB_PER_COST = 4.2e-9  # GB per unit of (n_seqs × mean_len^2); conservative upper bound
_MAFFT_MEMORY_OVERHEAD_GB = 2.5  # RAM not for MAFFT (OS + WSL2 + Python); WSL total overcounts
_MEMORY_SAFETY = 0.65  # leave headroom; never budget MAFFT to the brink
_DEFAULT_COST_BUDGET = 5e8  # fallback DP-cost budget when total memory can't be detected
_LOW_MEMORY_BYTES = 16 * 1024**3  # below this, warn that the budget is conservative


def _total_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None


def _alignment_cost(n_seqs: int, residues: int) -> float:
    """FFT-NS-2 memory proxy: ``n_seqs × mean_len^2`` = ``residues^2 / n_seqs``."""
    return residues * residues / n_seqs if n_seqs else 0.0


@functools.lru_cache(maxsize=1)
def _auto_cost_budget() -> float:
    """Max FFT-NS-2 DP-cost (``n_seqs × mean_len^2``) before switching to PartTree.

    Derived from available RAM via the measured memory model; above it, an
    alignment is predicted to exceed a safe fraction of the RAM left for MAFFT.
    Computed and logged once; warns on low-memory hosts (more KOs then use the
    approximate PartTree).
    """
    total = _total_memory_bytes()
    if total is None:
        logger.warning(
            "Could not detect system memory; using default MAFFT cost budget %.2e. "
            "Pass parttree_residues to override.", _DEFAULT_COST_BUDGET,
        )
        return _DEFAULT_COST_BUDGET
    total_gb = total / 1024**3
    mafft_gb = max(total_gb - _MAFFT_MEMORY_OVERHEAD_GB, 0.5)
    budget = _MEMORY_SAFETY * mafft_gb / _MAFFT_GB_PER_COST
    logger.info(
        "MAFFT DP-cost budget %.2e auto-set from %.1f GB RAM (~%.1f GB for MAFFT)",
        budget, total_gb, mafft_gb,
    )
    if total < _LOW_MEMORY_BYTES:
        logger.warning(
            "Limited memory (%.1f GB total): MAFFT cost budget set conservatively to "
            "%.2e, so more (especially long-protein) KOs use the approximate PartTree "
            "alignment. With more RAM, fewer would.", total_gb, budget,
        )
    return budget


def _mafft_cmd(
    mafft: str, inp: Path, threads: int, *, fast: bool = True, parttree: bool = False
) -> list[str]:
    """Build the MAFFT command.

    ``fast`` selects FFT-NS-2 (``--retree 2 --maxiterate 0``) — fast progressive
    alignment, the right trade-off for building profile HMMs — instead of
    ``--auto`` (which picks slow iterative refinement on medium/large inputs).
    ``parttree`` adds MAFFT's PartTree approximation for very large inputs.
    """
    cmd = [mafft]
    if parttree:
        cmd += ["--retree", "2", "--parttree"]
    elif fast:
        cmd += ["--retree", "2", "--maxiterate", "0"]
    else:
        cmd += ["--auto"]
    cmd += ["--anysymbol", "--thread", str(threads), str(inp)]
    return cmd


def _hmmbuild_cmd(
    hmmbuild: str, out_hmm: Path, aligned: Path, threads: int, name: str | None = None
) -> list[str]:
    cmd = [hmmbuild, "--cpu", str(threads)]
    if name:  # name the profile after its KO so it is the query name in hmmsearch output
        cmd += ["-n", name]
    cmd += [str(out_hmm), str(aligned)]
    return cmd


def _run(cmd: list[str], *, stdout_path: Path | None = None) -> str:
    """Run a command; optionally redirect stdout to a file. Raises on failure."""
    if stdout_path is not None:
        with open(stdout_path, "w") as out:
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
        stderr = proc.stderr or ""
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stderr = proc.stderr or ""
    if proc.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed:\n{stderr.strip()}")
    return stderr


def _staged_run(
    cmd: list[str], *, label: str, stage: str, verbose: bool,
    stdout_path: Path | None = None, log: bool = True,
) -> float:
    """Run a stage's command; log one completion line per stage (when verbose).

    At INFO (when ``log``): a single ``[KO] stage: done in X.Xs`` line — the
    ``stage`` descriptor already names the tool/mode and any seq/res/cost context,
    so the timing is just appended rather than repeated on a second line. The
    tool's own stderr (MAFFT/CD-HIT/hmmbuild progress) is logged at DEBUG. Pass
    ``log=False`` to suppress the line so the caller can fold the timing into its
    own message. Returns the stage's wall-clock seconds.
    """
    start = time.perf_counter()
    stderr = _run(cmd, stdout_path=stdout_path)
    elapsed = time.perf_counter() - start
    if verbose:
        if log:
            logger.info("[%s] %s: done in %.1fs", label, stage, elapsed)
        if stderr.strip():
            logger.debug("[%s] %s output:\n%s", label, stage, stderr.strip())
    return elapsed


def build_ko_hmm(
    ko_fasta: str | Path,
    out_hmm: str | Path,
    *,
    seq_identity: float = 0.9,
    parttree_residues: int | None = None,
    threads: int | None = None,
    fast: bool = True,
    verbose: bool = False,
    cdhit: str | Path | None = None,
    mafft: str | Path | None = None,
    hmmbuild: str | Path | None = None,
) -> Path:
    """Cluster, align and train a profile HMM for one KO's multi-FASTA.

    Single-sequence KOs skip CD-HIT/MAFFT (a lone sequence is its own alignment).
    ``seq_identity=-1`` skips CD-HIT. All (deduplicated) sequences are kept —
    memory on large KOs is bounded by switching MAFFT to PartTree, not by
    dropping sequences. ``fast`` uses MAFFT FFT-NS-2 (fast progressive) rather
    than ``--auto``'s slow iterative refinement. MAFFT switches to memory-light
    PartTree once an alignment is predicted to be too memory-heavy: by default from
    its **DP cost** (``n_seqs × mean_len²`` — long proteins cost far more than the
    same residue count in short ones) against a RAM-derived budget
    (:func:`_auto_cost_budget`). Passing ``parttree_residues`` overrides this with a
    simple residue-count cutoff.
    ``verbose`` logs (via the ``logging`` module, INFO/DEBUG) which tool is running
    for this KO, sequence counts at each stage, timings, and the tools' own
    output. ``threads`` (CD-HIT, MAFFT and hmmbuild) defaults to every CPU the process may
    use. Returns ``out_hmm``.
    """
    threads = _available_cpus() if threads is None else threads
    ko_fasta = Path(ko_fasta)
    out_hmm = Path(out_hmm)
    label = out_hmm.stem
    out_hmm.parent.mkdir(parents=True, exist_ok=True)
    n = _count_sequences(ko_fasta)
    if n == 0:
        raise ValueError(f"{ko_fasta} contains no sequences.")
    if verbose:
        logger.info("[%s] start: %d sequences", label, n)

    hmmbuild = resolve_binary("hmmbuild", binary=hmmbuild)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        if n == 1:
            if verbose:
                logger.info("[%s] single sequence: skipping CD-HIT/MAFFT", label)
            aligned = ko_fasta  # trivially aligned
        else:
            clustered = ko_fasta
            cdhit_elapsed: float | None = None
            if seq_identity != -1:
                clustered = tmp / "clustered.fa"
                cdhit_elapsed = _staged_run(
                    _cdhit_cmd(
                        resolve_binary("cd-hit", binary=cdhit), ko_fasta, clustered,
                        seq_identity, threads,
                    ),
                    label=label, stage=f"CD-HIT ({seq_identity})", verbose=verbose, log=False,
                )
            n_clustered, residues = _fasta_stats(clustered)
            if verbose and cdhit_elapsed is not None:
                logger.info(
                    "[%s] CD-HIT (%s): %d -> %d sequences in %.1fs",
                    label, seq_identity, n, n_clustered, cdhit_elapsed,
                )
            aligned = tmp / "aligned.fa"
            if n_clustered == 1:
                if verbose:
                    logger.info("[%s] one sequence after CD-HIT: skipping MAFFT", label)
                shutil.copyfile(clustered, aligned)  # MAFFT can't align a single seq
            else:
                # PartTree once the alignment is too memory-heavy. Default: its DP
                # cost (n_seqs × mean_len^2) vs a RAM-derived budget — length-aware,
                # so long-protein KOs (few seqs, huge residues) route correctly.
                # parttree_residues, if given, overrides with a residue-count cutoff.
                cost = _alignment_cost(n_clustered, residues)
                if parttree_residues is None:
                    parttree = cost > _auto_cost_budget()
                else:
                    parttree = residues > parttree_residues
                _staged_run(
                    _mafft_cmd(
                        resolve_binary("mafft", binary=mafft), clustered, threads,
                        fast=fast, parttree=parttree,
                    ),
                    label=label,
                    stage=f"MAFFT {'PartTree' if parttree else 'FFT-NS-2' if fast else 'auto'} "
                    f"({n_clustered} seqs, {residues} res, cost {cost:.2e})",
                    verbose=verbose,
                    stdout_path=aligned,
                )
        _staged_run(
            _hmmbuild_cmd(hmmbuild, out_hmm, aligned, threads, name=label),
            label=label, stage="hmmbuild", verbose=verbose,
        )
    if verbose:
        logger.info("[%s] complete -> %s", label, out_hmm)
    return out_hmm


# --------------------------------------------------------------------------- #
# Orchestration — a full domain library
# --------------------------------------------------------------------------- #
def build_hmm_library(
    organism_gene_ko: pd.DataFrame,
    genes_pep: str | Path,
    taxonomy: str | Path,
    out_dir: str | Path,
    *,
    domain: str,
    seq_identity: float = 0.9,
    parttree_residues: int | None = None,
    threads: int | None = None,
    fast: bool = True,
    verbose: bool = False,
    progress: bool = False,
    force: bool = False,
    concatenate: bool = True,
    cdhit: str | Path | None = None,
    mafft: str | Path | None = None,
    hmmbuild: str | Path | None = None,
) -> dict[str, Path | list[Path] | None]:
    """Build a domain (``"prokaryotes"``/``"eukaryotes"``) HMM library.

    Restricts genes to the domain's organisms (from ``taxonomy``), builds a
    multi-FASTA and a profile HMM per KO under ``out_dir``, and (if ``concatenate``)
    concatenates them into ``out_dir/library.hmm`` — searched directly with
    ``hmmsearch`` (no ``hmmpress`` needed). Returns ``{"hmms": [...], "library": path | None}``.

    Heavy and binary-dependent — intended for the maintainer, run once per KEGG
    release. **Resumable:** skips KOs that already have an ``.hmm`` and reuses an
    already-built multi-FASTA (see :func:`build_ko_fastas`), so re-running after a
    failure continues rather than restarting. ``force`` rebuilds everything.

    ``progress`` shows tqdm bars for the multi-FASTA and per-KO HMM passes (the
    headline ``N of M`` counter for the long build); it is independent of
    ``verbose`` (per-KO INFO logging). With both on, per-KO log lines are routed
    through ``tqdm.write`` so they don't corrupt the bar. ``threads`` goes to
    :func:`build_ko_hmm` and defaults to every CPU the process may use.
    """
    out_dir = Path(out_dir)
    fasta_dir = out_dir / "fasta"
    hmm_dir = out_dir / "hmms"
    hmm_dir.mkdir(parents=True, exist_ok=True)

    organisms = organisms_in_domain(taxonomy, domain)
    if not organisms:
        raise ValueError(f"No organisms found for domain {domain!r} in {taxonomy}.")

    if progress:
        tqdm.write(f"Indexing genes and writing per-KO FASTAs for {domain}...")
    ko_fastas = build_ko_fastas(
        organism_gene_ko, genes_pep, fasta_dir, organisms=organisms,
        progress=progress, force=force,
    )

    hmms: list[Path] = []
    redirect = logging_redirect_tqdm() if progress else nullcontext()
    with redirect, tqdm(
        ko_fastas.items(), total=len(ko_fastas), desc=f"HMM {domain}",
        unit="KO", disable=not progress,
    ) as bar:
        for ko, fasta in bar:
            bar.set_postfix_str(ko, refresh=False)
            out_hmm = hmm_dir / f"{ko}.hmm"
            if force or not out_hmm.exists():
                out_hmm.unlink(missing_ok=True)  # ensure a clean rebuild under force
                build_ko_hmm(
                    fasta, out_hmm, seq_identity=seq_identity,
                    parttree_residues=parttree_residues, threads=threads, fast=fast,
                    verbose=verbose, cdhit=cdhit, mafft=mafft, hmmbuild=hmmbuild,
                )
            hmms.append(out_hmm)

    library: Path | None = None
    if concatenate and hmms:
        library = out_dir / "library.hmm"
        with open(library, "wb") as out:
            for hmm in sorted(hmms):
                with open(hmm, "rb") as fh:
                    shutil.copyfileobj(fh, out)

    return {"hmms": hmms, "library": library}
