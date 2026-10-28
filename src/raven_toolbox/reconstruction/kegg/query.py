"""De-novo KEGG draft from a proteome FASTA via HMM search (step 3b.5).

Ports the FASTA/HMM branch of RAVEN ``getKEGGModelForOrganism``: search a query
proteome against the KO profile-HMM library (3b.3), assign genes to KOs using the
score cut-off and the two score-ratio filters, then build the draft model with the
shared assembler. For organisms not in KEGG.

Improvement over RAVEN: one ``hmmsearch`` of the whole concatenated KO library (K7)
replaces RAVEN's per-KO ``hmmsearch`` loop — the same fast search direction, but one
invocation instead of thousands, and no ``hmmpress``/``hmmscan`` needed.
Phylogenetic-distance subsampling is **not** used — our prebuilt per-domain
libraries (``kegg<version>_prokaryotes`` / ``kegg<version>_eukaryotes``) already fix
the sequence set, so picking the right domain library (not per-organism distance
weighting) is the relevant choice.

The scoring/assignment logic (:func:`assign_kos`, :func:`parse_hmmsearch_tblout`) is
pure and unit-tested; running the search needs HMMER (``hmmsearch``).
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import cobra
import pandas as pd

from raven_toolbox.binaries import resolve_binary
from raven_toolbox.io.yaml import read_yaml_model
from raven_toolbox.reconstruction.homology.blast import _available_cpus
from raven_toolbox.reconstruction.kegg.assemble import assemble_model_from_ko_genes
from raven_toolbox.reconstruction.kegg.parse import _resolve_artefact, read_kegg_table

_NOTE = "Included by KEGG HMM reconstruction"
_MIN_EVALUE = 1e-250  # floor for a reported E-value of 0, to keep logs finite


def _count_profiles(library: str | Path) -> int:
    """Number of profiles in an HMMER flatfile (one ``HMMER3/`` record header each)."""
    n = 0
    with open(library, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("HMMER3/"):
                n += 1
    return n


def run_hmmsearch(
    fasta: str | Path,
    library: str | Path,
    *,
    threads: int | None = None,
    hmmsearch: str | Path | None = None,
) -> str:
    """Search the profile ``library`` against proteome ``fasta``; return tblout text.

    One ``hmmsearch`` of the whole concatenated multi-profile ``library`` (query)
    against ``fasta`` (target) — the fast search direction, parallelised with
    ``--cpu``, and no ``hmmpress`` needed. ``-Z`` is fixed to the profile count so the
    per-hit E-values match the convention :func:`assign_kos` is calibrated against
    (identical to a ``hmmscan`` against the same library). ``threads`` defaults to every
    CPU the process may use, as in RAVEN's ``getKEGGModelForOrganism``.
    """
    threads = _available_cpus() if threads is None else threads
    exe = resolve_binary("hmmsearch", binary=hmmsearch)
    nprofiles = _count_profiles(library)
    z_opt = ["-Z", str(nprofiles)] if nprofiles > 0 else []
    with tempfile.TemporaryDirectory() as tmp:
        tbl = Path(tmp) / "hits.tbl"
        cmd = [exe, "--cpu", str(threads), *z_opt, "--tblout", str(tbl), str(library), str(fasta)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"hmmsearch failed:\n{(proc.stderr or '').strip()}")
        return tbl.read_text()


def parse_hmmsearch_tblout(text: str) -> pd.DataFrame:
    """Parse ``hmmsearch --tblout`` text into a ``[ko, gene, evalue]`` table.

    With the profile library as the *query*, column 1 (target name) is the proteome
    gene, column 3 (query name) is the KO, and column 5 is the full-sequence E-value.
    """
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            continue
        rows.append((fields[2], fields[0], float(fields[4])))
    return pd.DataFrame(rows, columns=["ko", "gene", "evalue"])


def assign_kos(
    hits: pd.DataFrame,
    *,
    cutoff: float = 1e-30,
    min_score_ratio_ko: float = 0.3,
    min_score_ratio_g: float = 0.9,
) -> dict[str, list[str]]:
    """Assign genes to KOs from HMM hits, applying the cut-off and ratio filters.

    Ports RAVEN's three steps on the KO×gene E-value matrix:

    1. keep hits with ``evalue <= cutoff``;
    2. **min_score_ratio_ko** — within a KO, drop genes whose
       ``log(evalue)/log(best_evalue_in_KO) < min_score_ratio_ko`` (prune weak
       members of a KO);
    3. **min_score_ratio_g** — within a gene, drop KOs whose
       ``log(evalue)/log(best_evalue_for_gene) < min_score_ratio_g`` (stop a gene
       that clearly belongs to one KO leaking into weaker ones).

    Smaller E-value = better; since all kept values are ``< 1`` their logs are
    negative, so the best (smallest) hit gives ratio 1 and weaker hits give a
    smaller positive ratio.

    Default calibration, cross-validated against the true
    KEGG gene→KO annotation of four organisms spanning the prok/euk libraries and
    the well-/lesser-studied axis (*S. cerevisiae*, *Cyanidioschyzon merolae*,
    *E. coli*, *Mycoplasma genitalium*): real annotations score
    overwhelmingly (median E ≈ 1e-100…1e-155) while spurious hits pile up at
    ≈1e-8, so the two are separated by ~20 orders of magnitude. RAVEN's
    ``1e-50`` sits inside the *true* tail and silently drops real but divergent
    hits — costing 16% gene→KO recall on the divergent minimal genome
    (*M. genitalium*) for no noise-rejection benefit (noise is far weaker). The
    default is therefore loosened to **1e-30** (recovers that tail; still ~22
    orders above the noise floor), with the precision work moved to
    **min_score_ratio_g = 0.9** — the *effective* precision lever (it resolves
    multi-KO genes). ``min_score_ratio_ko`` proved empirically inert across all
    four organisms (identical output at 0.0/0.3/0.5) and is kept only for RAVEN
    parity.
    """
    # The ratio filters compare log(evalue)/log(best_evalue); when best == 1.0
    # the denominator is 0 → ZeroDivisionError. The default cutoff (1e-30) keeps
    # us safely away, but a caller-passed cutoff ≥ 1 is ambiguous and would
    # crash later. Reject it up front with a clear message.
    if cutoff >= 1:
        raise ValueError(
            f"cutoff must be < 1 (smaller E-value = better hit); got {cutoff!r}."
        )

    # Best (smallest) E-value per (ko, gene), filtered at the cut-off.
    mat: dict[str, dict[str, float]] = {}
    for ko, gene, evalue in zip(hits["ko"], hits["gene"], hits["evalue"], strict=True):
        if evalue > cutoff:
            continue
        e = evalue if evalue > 0 else _MIN_EVALUE
        per_ko = mat.setdefault(ko, {})
        if gene not in per_ko or e < per_ko[gene]:
            per_ko[gene] = e

    # Step 2: prune weak genes within each KO.
    for ko, genes in mat.items():
        log_best = math.log(min(genes.values()))
        mat[ko] = {
            g: e for g, e in genes.items() if math.log(e) / log_best >= min_score_ratio_ko
        }

    # Step 3: prune weak KOs within each gene (over the survivors of step 2).
    gene_kos: dict[str, dict[str, float]] = {}
    for ko, genes in mat.items():
        for g, e in genes.items():
            gene_kos.setdefault(g, {})[ko] = e
    dropped: set[tuple[str, str]] = set()
    for g, kos in gene_kos.items():
        log_best = math.log(min(kos.values()))
        for ko, e in kos.items():
            if math.log(e) / log_best < min_score_ratio_g:
                dropped.add((ko, g))

    result: dict[str, list[str]] = {}
    for ko, genes in mat.items():
        kept = sorted(g for g in genes if (ko, g) not in dropped)
        if kept:
            result[ko] = kept
    return result


def get_kegg_model_from_sequences(
    fasta: str | Path,
    reference_model: cobra.Model,
    ko_reaction: pd.DataFrame,
    library: str | Path,
    *,
    rxn_flags: pd.DataFrame | None = None,
    model_id: str | None = None,
    cutoff: float = 1e-30,
    min_score_ratio_ko: float = 0.3,
    min_score_ratio_g: float = 0.9,
    keep_spontaneous: bool = True,
    keep_undefined_stoich: bool = True,
    keep_incomplete: bool = True,
    keep_general: bool = False,
    threads: int | None = None,
    hmmsearch: str | Path | None = None,
) -> cobra.Model:
    """Reconstruct a draft model for a proteome by HMM-searching the KO library.

    Searches the ``library`` against ``fasta`` (3b.3), assigns KOs
    (:func:`assign_kos`), and assembles the model against ``reference_model`` /
    ``ko_reaction``. Genes are the query proteome's identifiers. ``threads`` is passed to
    :func:`run_hmmsearch`.
    """
    if model_id is None:
        # RAVEN always sets model.id (to organismID); default it here so the
        # draft never inherits the reference model's id. Pass model_id explicitly
        # for byte-identical parity with a specific MATLAB organismID.
        model_id = Path(fasta).stem
    hits = parse_hmmsearch_tblout(run_hmmsearch(fasta, library, threads=threads, hmmsearch=hmmsearch))
    ko_to_genes = assign_kos(
        hits,
        cutoff=cutoff,
        min_score_ratio_ko=min_score_ratio_ko,
        min_score_ratio_g=min_score_ratio_g,
    )
    model, _ = assemble_model_from_ko_genes(
        reference_model,
        ko_reaction,
        ko_to_genes,
        rxn_flags=rxn_flags,
        keep_spontaneous=keep_spontaneous,
        keep_undefined_stoich=keep_undefined_stoich,
        keep_incomplete=keep_incomplete,
        keep_general=keep_general,
        model_id=model_id,
        note=_NOTE,
        prune_orthology=True,
    )
    return model


def get_kegg_model_from_sequences_with_artefacts(
    fasta: str | Path,
    artefact_dir: str | Path | None = None,
    library: str | Path | None = None,
    *,
    domain: str = "prokaryotes",
    version: str | None = None,
    **kwargs,
) -> cobra.Model:
    """Load reference model + tables from ``artefact_dir`` and run the HMM query.

    If ``artefact_dir`` / ``library`` are ``None`` they are fetched/cached via
    :func:`raven_toolbox.data.ensure_kegg_data` / :func:`raven_toolbox.data.ensure_kegg_hmm_library`
    (``domain`` selects the prok/euk library; ``version`` the release).
    """
    if artefact_dir is None or library is None:
        from raven_toolbox.data import ensure_kegg_data, ensure_kegg_hmm_library

        if artefact_dir is None:
            artefact_dir = ensure_kegg_data(version=version)
        if library is None:
            library = ensure_kegg_hmm_library(domain, version=version)
    artefact_dir = Path(artefact_dir)
    reference_model = read_yaml_model(_resolve_artefact(artefact_dir, "reference_model.yml.gz"))
    ko_reaction = read_kegg_table(_resolve_artefact(artefact_dir, "ko_reaction.tsv.gz"))
    rxn_flags = read_kegg_table(_resolve_artefact(artefact_dir, "rxn_flags.tsv.gz"))
    return get_kegg_model_from_sequences(
        fasta, reference_model, ko_reaction, library, rxn_flags=rxn_flags, **kwargs
    )
