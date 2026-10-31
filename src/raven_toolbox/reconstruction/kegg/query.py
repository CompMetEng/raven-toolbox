"""De-novo KEGG draft from a proteome FASTA via HMM search.

Ports the FASTA and HMM branch of RAVEN ``getKEGGModelForOrganism``: search a query
proteome against the KO profile-HMM library (:mod:`.hmm`), assign genes to KOs using the
score cut-off and the two score-ratio filters, then build the draft model with the
shared assembler. For organisms not in KEGG.

Unlike RAVEN, one ``hmmsearch`` of the whole concatenated KO library replaces RAVEN's
per-KO ``hmmsearch`` loop. The search direction is the same, and neither ``hmmpress``
nor ``hmmscan`` is needed. Phylogenetic-distance subsampling is not used: the prebuilt
per-domain libraries (``kegg<version>_prokaryotes`` and ``kegg<version>_eukaryotes``)
fix the sequence set, and the choice is the domain library.

The scoring and assignment functions (:func:`assign_kos`, :func:`parse_hmmsearch_tblout`)
are pure functions; running the search needs HMMER (``hmmsearch``).
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
    against ``fasta`` (target), which is the fast search direction. It runs with
    ``--cpu`` and needs no ``hmmpress``. ``-Z`` is set to the profile count (when it
    is positive) so that the per-hit E-values match the convention :func:`assign_kos` is
    calibrated against (identical to a ``hmmscan`` against the same library). ``threads``
    defaults to every CPU the process may use, as in RAVEN's ``getKEGGModelForOrganism``.
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

    With the profile library as the query, column 1 (target name) is the proteome
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

    Ports RAVEN's three steps on the KO by gene E-value matrix. An E-value of 0 is
    replaced by 1e-250. The steps are:

    1. keep hits with ``evalue <= cutoff``;
    2. ``min_score_ratio_ko``: within a KO, drop genes whose
       ``log(evalue)/log(best_evalue_in_KO) < min_score_ratio_ko`` (removes weak
       members of a KO);
    3. ``min_score_ratio_g``: within a gene, drop KOs whose
       ``log(evalue)/log(best_evalue_for_gene) < min_score_ratio_g`` (removes the weaker
       KOs of a gene that has a better KO).

    A smaller E-value is a better hit. All kept E-values are below 1, so their logs are
    negative, the best (smallest) hit has ratio 1, and weaker hits have a smaller
    positive ratio.

    The defaults were calibrated by cross-validation against the KEGG gene and KO
    annotation of four organisms that span the prokaryote and eukaryote libraries and
    well- and less-studied species (*S. cerevisiae*, *Cyanidioschyzon merolae*,
    *E. coli*, *Mycoplasma genitalium*). Annotated genes have a median E-value of
    1e-100 to 1e-155, and spurious hits cluster at about 1e-8, a separation of ~20
    orders of magnitude. RAVEN's cut-off of ``1e-50`` lies within the E-value range of
    the annotated genes and drops real but divergent hits. For the divergent minimal genome
    (*M. genitalium*, ``mge``) the gene-to-KO recall is 0.95 at 1e-30 and 0.84 at 1e-50,
    with no gain in rejecting noise (noise E-values are far weaker). The default cut-off
    is therefore 1e-30, which recovers those hits and is ~22 orders of magnitude above
    the noise level. ``min_score_ratio_g`` (default 0.9) sets the precision, because it
    resolves genes with several KOs. Against RAVEN's settings (1e-50 and 0.8), the
    recall for ``mge`` rises from 0.84 to 0.94. ``min_score_ratio_ko`` values of 0.0, 0.3
    and 0.5 changed precision and recall by at most 0.02 across the four organisms; 0.3 is
    kept as in RAVEN.
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

    Searches the ``library`` against ``fasta``, assigns KOs
    (:func:`assign_kos`), and assembles the model against ``reference_model`` and
    ``ko_reaction``. Genes are the identifiers of the query proteome. ``model_id``
    defaults to the stem of the ``fasta`` file name. ``threads`` is passed to
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
    """Load the reference model and tables from ``artefact_dir`` and run the HMM query.

    If ``artefact_dir`` or ``library`` is ``None``, it is fetched and cached via
    :func:`raven_toolbox.data.ensure_kegg_data` or
    :func:`raven_toolbox.data.ensure_kegg_hmm_library`. ``domain`` selects the
    prokaryote or eukaryote library and ``version`` the release. Remaining keyword
    arguments go to :func:`get_kegg_model_from_sequences`.
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
