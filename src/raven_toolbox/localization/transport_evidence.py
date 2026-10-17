"""Evidence-aware transport scoring — turn transporter evidence into per-metabolite transport costs.

The localisation assignment (:func:`predict_localization`, :func:`assign_compartments`) charges a flat
cost per inter-compartment transport it must add. Applied blindly that is *indiscriminate*: it drops
real, functionally essential transporters as readily as spurious ones, because the cost ignores whether
a transporter actually exists. This module makes the
cost **evidence-aware**: a transport is cheap when a transporter gene supports it (right substrate,
right membrane) and pays the full prior otherwise.

    ``transport_cost(metabolite) = base_cost * (1 - evidence(metabolite))``

:func:`evidence_aware_transport_cost` returns exactly the ``{metabolite_base: cost}`` mapping both
assignment functions already accept as their ``transport_cost`` argument — so no MILP change is needed.

Evidence is carrier-general (any transporter family/membrane) and organism-agnostic (sequence-derived;
the only per-organism input is the proteome). :func:`annotate_proteome` runs the bundled ``hmmsearch``
(Pfam transporter families) + ``diamond`` (TCDB) back-ends against a proteome FASTA;
:func:`annotate_transporters` takes a pre-computed table instead. Substrate matching is *coarse-first*
(family → coarse class), with a finer per-substrate ChEBI layer available via ``ontology``. See
the `evidence-aware transport scoring design
<https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/localization/transport_evidence_scoring.md>`_ on raven-gecko-parity for the full design.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import cobra
import pandas as pd

from raven_toolbox.binaries import resolve_binary
from raven_toolbox.data import ensure_data_file
from raven_toolbox.localization.substrate_ontology import SubstrateOntology, load_tc_substrates
from raven_toolbox.localization.transporter_tables import PFAM_TRANSPORTERS, TC_FAMILY_CLASS

__all__ = [
    "TransporterAnnotation",
    "annotate_proteome",
    "annotate_transporters",
    "default_metabolite_chebi",
    "default_substrate_of",
    "evidence_aware_transport_cost",
]


@dataclass(frozen=True)
class TransporterAnnotation:
    """Per-gene transporter evidence, from any source (Pfam/hmmer, TCDB/diamond, orthology, or a
    hand-curated table). ``confidence`` is a 0..1 strength; ``substrate_classes`` are coarse shared
    classes (e.g. ``"amino_acid"``, ``"sugar"``, ``"organic_acid"``) used to match a transporter to a
    metabolite it can plausibly carry; ``substrate_chebi`` are the *specific* curated substrate ChEBIs
    (from TCDB), for the finer ChEBI-ontology match."""

    gene: str
    confidence: float = 0.0
    families: tuple[str, ...] = ()
    substrate_classes: frozenset[str] = field(default_factory=frozenset)
    mechanism: str | None = None  # "uniport" | "symport" | "antiport" | None
    substrate_chebi: frozenset[str] = field(default_factory=frozenset)


def _default_base(m: cobra.Metabolite) -> str:
    """Compartment-agnostic metabolite key: strip a trailing ``_<compartment>`` from the id."""
    if m.compartment and m.id.endswith(f"_{m.compartment}"):
        return m.id[: -(len(m.compartment) + 1)]
    return m.id


def default_metabolite_chebi(metabolite: cobra.Metabolite) -> frozenset[str]:
    """ChEBI identifiers from a metabolite's annotation, normalised to ``CHEBI:NNNN``.

    The model-side input to the ChEBI substrate match; pass as ``metabolite_chebi`` to
    :func:`evidence_aware_transport_cost`. Returns the empty set for an un-annotated metabolite (which
    then falls back to the coarse class match).
    """
    ann = metabolite.annotation.get("chebi")
    if not ann:
        return frozenset()
    values = ann if isinstance(ann, (list, tuple, set, frozenset)) else [ann]
    out: set[str] = set()
    for raw in values:
        v = str(raw).strip()
        if not v:
            continue
        # canonicalise to an uppercase ``CHEBI:<id>`` so lookups match the uppercase ChEBI/TCDB tables
        # (a lowercase/mixed-case ``chebi:`` prefix would otherwise silently miss every match)
        tail = v.split(":", 1)[1] if v.upper().startswith("CHEBI:") else v
        out.add(f"CHEBI:{tail}")
    return frozenset(out)


def annotate_transporters(
    table: pd.DataFrame,
    *,
    gene_col: str = "gene",
    confidence_col: str = "confidence",
    families_col: str | None = "families",
    substrate_col: str | None = "substrate_classes",
    mechanism_col: str | None = "mechanism",
    sep: str = ";",
) -> dict[str, TransporterAnnotation]:
    """Parse a per-gene transporter-annotation table into :class:`TransporterAnnotation` objects.

    This is the **bring-your-own** path: the table can come from any tool (eggNOG-mapper, InterProScan,
    a web service); use :func:`annotate_proteome` instead for the bundled ``hmmsearch``/``diamond``
    back-ends. List
    columns (``families``, ``substrate_classes``) may be real lists or ``sep``-joined strings. A gene
    appearing on several rows keeps its highest-confidence annotation and the union of families /
    substrate classes.
    """

    def _as_set(v) -> frozenset[str]:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return frozenset()
        if isinstance(v, str):
            return frozenset(x.strip() for x in v.split(sep) if x.strip())
        return frozenset(str(x).strip() for x in v if str(x).strip())

    out: dict[str, TransporterAnnotation] = {}
    for row in table.to_dict("records"):
        gene = str(row[gene_col])
        conf = float(row.get(confidence_col, 0.0) or 0.0)
        fams = _as_set(row.get(families_col)) if families_col else frozenset()
        subs = _as_set(row.get(substrate_col)) if substrate_col else frozenset()
        mech = row.get(mechanism_col) if mechanism_col else None
        mech = None if mech is None or (isinstance(mech, float) and pd.isna(mech)) else str(mech)
        prev = out.get(gene)
        if prev is None or conf >= prev.confidence:
            fam_tuple = tuple(sorted(fams | set(prev.families))) if prev else tuple(sorted(fams))
            subs = subs | prev.substrate_classes if prev else subs
            out[gene] = TransporterAnnotation(gene, max(conf, prev.confidence if prev else 0.0),
                                              fam_tuple, subs, mech or (prev.mechanism if prev else None))
        else:  # keep prev confidence but accumulate families/substrates
            out[gene] = TransporterAnnotation(
                gene, prev.confidence, tuple(sorted(set(prev.families) | fams)),
                prev.substrate_classes | subs, prev.mechanism)
    return out


def evidence_aware_transport_cost(
    model: cobra.Model,
    annotation: Mapping[str, TransporterAnnotation],
    gene_compartments: Mapping[str, Iterable[str]],
    *,
    substrate_of: Callable[[cobra.Metabolite], Iterable[str]] | None = None,
    ontology: SubstrateOntology | None = None,
    metabolite_chebi: Callable[[cobra.Metabolite], Iterable[str]] | None = None,
    sibling_weight: float = 0.0,
    base_cost: float = 0.5,
    base_metabolite: Callable[[cobra.Metabolite], str] | None = None,
) -> dict[str, float]:
    """Per-metabolite transport cost from transporter evidence, ready to pass as ``transport_cost``.

    For each (compartment-agnostic) metabolite *m*::

        evidence(m) = max over transporter genes g of  g.confidence * substrate_match(g, m),
                      restricted to genes localised to a compartment where m occurs (membrane gate).
        cost(m)     = base_cost * (1 - evidence(m))

    ``substrate_match`` is 1.0 for a coarse substrate-class overlap and, when an ``ontology`` is given,
    a graded 0..1 ChEBI roll-up (metabolite ChEBI → the gene's curated substrate ChEBI) that also
    evidences metabolites the coarse classifier misses. With neither ``substrate_of`` nor ``ontology``
    the match is compartment-only (every carrier at the right membrane counts).

    Parameters
    ----------
    model:
        the model to cost; every compartment-agnostic metabolite base in it gets an entry in the
        result.
    annotation:
        gene -> transporter evidence, as returned by :func:`annotate_transporters` or
        :func:`annotate_proteome`.
    gene_compartments:
        gene -> predicted compartment ids (the reliable DeepLoc *compartment* calls). A carrier at
        compartment *X* is taken to support transports across *X*'s boundary.
    substrate_of:
        metabolite -> its coarse substrate class(es) (e.g. :func:`default_substrate_of`); a class
        overlap with the gene is full-credit evidence.
    ontology:
        a :class:`~.substrate_ontology.SubstrateOntology` enabling the finer, graded ChEBI-substrate
        match on top of (and filling the gaps of) the coarse classes.
    metabolite_chebi:
        metabolite -> its ChEBI id(s) for that match (default: :func:`default_metabolite_chebi`, which
        reads the ``chebi`` annotation).
    sibling_weight:
        if > 0, also credit a metabolite that is a *chemical relative* of a curated substrate (shares a
        near ChEBI ancestor, not under it) at this fraction of a direct hit — trades specificity for
        recall (0 = off, the default).
    base_cost:
        the flat cost for an unsupported transport (recovers today's constant behaviour when no gene
        supports the metabolite).
    base_metabolite:
        metabolite -> the compartment-agnostic key to group it under (default: strip a trailing
        ``_<compartment>`` from the id).

    Returns
    -------
    dict[str, float]
        every metabolite base -> cost (unsupported metabolites map to ``base_cost``), so the result
        is a self-contained ``transport_cost`` mapping.
    """
    base_of = base_metabolite or _default_base
    chebi_of = metabolite_chebi or default_metabolite_chebi
    comps_of: dict[str, set[str]] = defaultdict(set)
    rep: dict[str, cobra.Metabolite] = {}
    for m in model.metabolites:
        b = base_of(m)
        comps_of.setdefault(b, set())  # every base gets a cost, even a compartment-less metabolite
        if m.compartment:
            comps_of[b].add(m.compartment)
        rep.setdefault(b, m)

    carriers: list[tuple[set[str], frozenset[str], frozenset[str], float]] = []
    for gene, ann in annotation.items():
        if ann.confidence <= 0:
            continue
        carriers.append((set(gene_compartments.get(gene, ())), ann.substrate_classes,
                         ann.substrate_chebi, ann.confidence))

    compartment_only = substrate_of is None and ontology is None
    costs: dict[str, float] = {}
    for b, bcomps in comps_of.items():
        m = rep[b]
        classes = frozenset(substrate_of(m)) if substrate_of else frozenset()
        mchebi = frozenset(chebi_of(m)) if ontology is not None else frozenset()
        best = 0.0
        for gcomps, subs, gsub_chebi, conf in carriers:
            if gcomps and not (gcomps & bcomps):
                continue  # gene's membrane does not border a compartment this metabolite is in
            if compartment_only:
                match = 1.0
            else:
                # coarse class overlap = full credit (preserves the coarse layer); a graded ChEBI
                # roll-up adds credit where the coarse class is absent or too broad. Strongest wins.
                match = 1.0 if (classes & subs) else 0.0
                if ontology is not None and match < 1.0:
                    match = max(match, ontology.match(mchebi, gsub_chebi, sibling_weight=sibling_weight))
            if match <= 0.0:
                continue  # substrate mismatch
            evidence = conf * match
            if evidence > best:
                best = evidence
        costs[b] = base_cost * (1.0 - best)
    return costs


# --------------------------------------------------------------------------- annotation back-end
_TC_FAMILY = re.compile(r"(\d+\.[A-Z]\.\d+)\.\d+\.\d+")  # group 1 = family (3 levels), group 0 = full TC-ID


def _hmmsearch_families(proteome: Path, hmm_db: Path, threads: int) -> dict[str, tuple[set[str], float]]:
    """hmmsearch (transporter Pfam HMMs vs proteome) -> gene -> (Pfam accessions hit, best bitscore)."""
    with tempfile.TemporaryDirectory() as tmp:
        tbl = Path(tmp) / "hits.tbl"
        subprocess.run([resolve_binary("hmmsearch"), "--cut_ga", "--noali", "--cpu", str(threads),
                        "--tblout", str(tbl), str(hmm_db), str(proteome)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out: dict[str, tuple[set[str], float]] = {}
        for line in tbl.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            gene, pfam_acc, score = fields[0], fields[3].split(".")[0], float(fields[5])
            fams, best = out.get(gene, (set(), 0.0))
            fams.add(pfam_acc)
            out[gene] = (fams, max(best, score))
        return out


def _diamond_tc(proteome: Path, tcdb_db: Path,
                threads: int) -> dict[str, tuple[set[str], set[str], float]]:
    """diamond blastp (proteome vs TCDB) -> gene -> (TC family prefixes, full TC-IDs, best frac id)."""
    with tempfile.TemporaryDirectory() as tmp:
        m8 = Path(tmp) / "hits.m8"
        subprocess.run([resolve_binary("diamond"), "blastp", "-q", str(proteome), "-d", str(tcdb_db),
                        "-o", str(m8), "--outfmt", "6", "qseqid", "sseqid", "pident", "-e", "1e-5",
                        "--max-target-seqs", "3", "-p", str(threads), "--quiet"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out: dict[str, tuple[set[str], set[str], float]] = {}
        for line in m8.read_text().splitlines():
            if not line.strip():
                continue
            gene, sseqid, pident = line.split("\t")[:3]
            m = _TC_FAMILY.search(sseqid)
            if m is None:
                continue
            fams, tc_ids, best = out.get(gene, (set(), set(), 0.0))
            fams.add(m.group(1))
            tc_ids.add(m.group(0))
            out[gene] = (fams, tc_ids, max(best, float(pident) / 100.0))
        return out


def annotate_proteome(
    proteome: str | Path,
    *,
    hmm_db: str | Path | None = None,
    tcdb_db: str | Path | None = None,
    substrates: str | Path | bool | None = None,
    threads: int = 1,
) -> dict[str, TransporterAnnotation]:
    """Annotate a proteome's transporters from sequence, via the bundled ``hmmsearch`` + ``diamond``.

    Scans the proteome FASTA (headers = gene ids, as written by :func:`prepare_deeploc_input`) against
    the transporter Pfam HMM database (``hmmsearch --cut_ga``) and the TCDB DIAMOND database
    (``diamond blastp``); both databases auto-download from raven-data if not supplied. Pfam families
    and TCDB TC-families map to coarse substrate classes via
    :mod:`raven_toolbox.localization.transporter_tables`, and each gene's TCDB TC-IDs map to their
    curated **substrate ChEBIs** (``substrates``: a ``tcdb_substrates.tsv`` path, or ``False`` to skip
    that lookup). Returns gene → :class:`TransporterAnnotation` (``mechanism`` stays empty). Genes with
    no transporter hit are absent, so this doubles as a ``substrate_of``-independent membrane screen.
    """
    proteome = Path(proteome)
    hmm_db = Path(hmm_db) if hmm_db else ensure_data_file("transporters", "transporter_pfam.hmm")
    tcdb_db = Path(tcdb_db) if tcdb_db else ensure_data_file("transporters", "tcdb.dmnd")
    pfam = _hmmsearch_families(proteome, hmm_db, threads)
    tcdb = _diamond_tc(proteome, tcdb_db, threads)
    tc_substrates = ({} if substrates is False
                     else load_tc_substrates(substrates if isinstance(substrates, str | Path) else None))

    out: dict[str, TransporterAnnotation] = {}
    for gene in set(pfam) | set(tcdb):
        fams, pf_score = pfam.get(gene, (set(), 0.0))
        tc_fams, tc_ids, tc_id = tcdb.get(gene, (set(), set(), 0.0))
        classes: set[str] = set()
        for acc in fams:
            if acc in PFAM_TRANSPORTERS:
                classes |= PFAM_TRANSPORTERS[acc][1]
        for tc in tc_fams:
            classes |= TC_FAMILY_CLASS.get(tc, frozenset())
        substrate_chebi: set[str] = set()
        for tc in tc_ids:
            substrate_chebi |= tc_substrates.get(tc, frozenset())
        # Coarse confidence in [0,1]: a --cut_ga Pfam hit is curated-significant (floor 0.5, scaled by
        # bitscore, saturating ~200); a TCDB hit scales with sequence identity. Take the stronger. Tunable.
        conf = max(min(1.0, 0.5 + pf_score / 400.0) if fams else 0.0, tc_id)
        out[gene] = TransporterAnnotation(
            gene=gene, confidence=round(conf, 4), families=tuple(sorted(fams | tc_fams)),
            substrate_classes=frozenset(classes), substrate_chebi=frozenset(substrate_chebi))
    return out


# ------------------------------------------------- metabolite -> coarse substrate class (model side)
# Name-keyword heuristic over the shared coarse vocabulary; does not derive classes from a
# metabolite's ChEBI/KEGG annotation.
_SUBSTRATE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "amino_acid": ("glycine", "alanine", "valine", "leucine", "isoleucine", "proline", "phenylalanine",
        "tryptophan", "methionine", "serine", "threonine", "cysteine", "cystine", "tyrosine",
        "asparagine", "glutamine", "aspartate", "glutamate", "lysine", "arginine", "histidine",
        "ornithine", "citrulline", "homoserine", "homocysteine", "aminobutyrate"),
    "sugar": ("glucose", "fructose", "mannose", "galactose", "sucrose", "maltose", "trehalose",
        "lactose", "sorbitol", "mannitol", "arabitol", "ribitol", "ribose", "xylose", "arabinose",
        "raffinose", "glycerol", "ribulose", "sorbose", "fucose", "rhamnose", "xylulose", "inositol",
        "sedoheptulose", "galactitol", "gluconate"),
    "carboxylate": ("malate", "citrate", "isocitrate", "succinate", "fumarate", "oxaloacetate",
        "oxoglutarate", "pyruvate", "lactate", "acetate", "formate", "oxalate", "malonate", "glyoxylate",
        "glutarate", "aconitate", "tartrate", "propionate", "butyrate", "oxoadipate", "adipate",
        "citramalate", "mevalonate", "acetoacetate", "hydroxybutyrate"),
    "nucleotide": ("atp", "adp", "amp", "gtp", "gdp", "gmp", "ctp", "cdp", "cmp", "utp", "udp", "ump",
        "itp", "idp", "imp", "dttp", "nadph", "nadh", "nadp", "nad(", "diphosphate", "triphosphate"),
    "nucleoside_base": ("adenine", "guanine", "cytosine", "uracil", "thymine", "hypoxanthine",
        "xanthine", "adenosine", "guanosine", "cytidine", "uridine", "thymidine", "inosine"),
    "cofactor_vitamin": ("coa", "coenzyme a", "nadph", "nadp", "nad(", "flavin", "fad", "fmn", "folate",
        "tetrahydrofolate", "biotin", "thiamine", "pantothenate", "pantoate", "pyridoxal", "pyridoxine",
        "riboflavin", "cobalamin", "lipoate", "menaquinone", "ubiquinone", "molybdopterin"),
    "lipid_fatty_acid": ("palmitate", "oleate", "stearate", "myristate", "laurate", "linoleate",
        "decanoate", "octanoate", "hexanoate", "dodecanoate", "tetradecanoate", "hexadecanoate",
        "octadecanoate", "fatty acid", "acyl", "sterol", "ergosterol", "lanosterol", "phosphatidyl",
        "diacylglycerol", "triacylglycerol", "ceramide", "sphinganine"),
    "phosphate_sulfate": ("phosphate", "sulfate", "sulphate", "thiosulfate"),
    "inorganic_ion": ("sodium", "potassium", "calcium", "magnesium", "manganese", "ferrous", "ferric",
        "zinc", "copper", "cobalt", "nickel", "chloride", "ammonium", "ammonia", "nitrate", "nitrite",
        "carbonate", "bicarbonate", "hydrogen sulfide"),
    "amine_polyamine": ("putrescine", "spermidine", "spermine", "cadaverine", "choline", "ethanolamine",
        "carnitine", "betaine", "agmatine"),
    "peptide": ("glutathione", "peptide", "dipeptide", "tripeptide"),
}


def default_substrate_of(metabolite: cobra.Metabolite) -> frozenset[str]:
    """Coarse substrate class(es) for a model metabolite, from a name-keyword heuristic.

    A dependency-free first pass over :data:`transporter_tables.COARSE_CLASSES`: it matches the
    metabolite's name against curated keyword lists, so ``(S)-malate`` → ``carboxylate``,
    ``D-glucose`` → ``sugar``, ``NADPH`` → ``nucleotide`` + ``cofactor_vitamin``. Pass it as
    ``substrate_of`` to :func:`evidence_aware_transport_cost`. It does not derive classes from a
    metabolite's ChEBI/KEGG annotation; an unmatched metabolite returns the empty set — no substrate
    support, so the parsimony prior stands (a safe default).
    """
    name = (metabolite.name or metabolite.id or "").lower()
    return frozenset(cls for cls, kws in _SUBSTRATE_KEYWORDS.items() if any(k in name for k in kws))
