"""Loaders for gene → compartment localisation predictors (DeepLoc 2, MULocDeep, COMPARTMENTS).

The localisation algorithms in :mod:`raven_toolbox.localization.predict` and
:mod:`raven_toolbox.localization.certify` consume a gene × compartment score table
(:class:`LocalizationScores`) where a higher value is stronger evidence. Each predictor or database
produces the table differently; the loaders here normalise them. A :class:`LocalizationScores` can
also be built from any other source by constructing the :class:`pandas.DataFrame` directly.

Predictors label compartments with their own names (``Mitochondrion``, ``Cytoplasm``, …).
``compartment_map`` (for example :data:`DEFAULT_COMPARTMENT_MAP`) renames them to the compartment
ids of the model. Consequences of the mapping:

* A label absent from the map is dropped with its whole column, which excludes compartments that the
  model does not have. Without a map the predictor's own labels are kept
  (:meth:`LocalizationScores.with_compartments` renames them later).
* Several labels may share one compartment id, which merges their columns by taking the
  maximum per gene, not the sum: a gene scoring 0.7 cytoplasm and 0.4 cytosol scores 0.7
  for ``c``. This applies to synonyms and to compartments that the model does not separate.
* One label cannot be split across two compartment ids, because a mapping holds one value per key.
  The split is made on the resulting frame, for example ``scores.df["v"] = scores.df["g"]``
  lets a gene predicted Golgi compete for both. The columns are then independent, and
  :func:`raven_toolbox.localization.predict_localization` trades the scores against the transport
  cost.

By default, each loader normalises every gene's row so that the best compartment is 1.0 (RAVEN's
``parseScores`` convention), so the scores of several predictors share one scale
(``load_deeploc(normalise=False)`` keeps the raw probabilities).

Sequence-based predictors are external tools; the loaders read their output files.
The module has loaders for the multi-label predictors (DeepLoc 2, MULocDeep) and the COMPARTMENTS
evidence database, and none for older single-label predictors.
"""
from __future__ import annotations

import io
import re
import urllib.parse
import urllib.request
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "LocalizationScores",
    "DEFAULT_COMPARTMENT_MAP",
    "load_deeploc",
    "load_mulocdeep",
    "load_compartments",
    "load_uniprot",
    "fetch_uniprot_localization",
    "combine_scores",
]


@dataclass
class LocalizationScores:
    """Per-gene compartment scores. ``df`` is indexed by ``gene_id`` with one column per
    compartment id; values are floats (higher = stronger evidence for that compartment).

    :func:`raven_toolbox.localization.predict_localization` counts a NaN entry as score 0, leaves a
    gene absent from ``df`` out of its gene coupling, and lists a reaction that has genes, none of
    them in ``df``, in ``unplaced_reactions``. A reaction without a GPR is placed freely and is not
    listed.

    ``raw_confidence`` (optional) is a per-gene Series of the pre-normalisation top probability of
    the predictor. The loaders normalise every gene's best compartment to 1.0, which discards the
    confidence of the call. :func:`load_deeploc` populates ``raw_confidence`` with
    ``keep_raw_confidence=True``, so downstream consumers (notably
    :func:`raven_toolbox.localization.triage_localization`) can tell a 0.97 call from a 0.40 one.
    """

    df: pd.DataFrame
    raw_confidence: pd.Series | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.df.index, pd.Index) or self.df.index.name not in (None, "gene_id"):
            # accept but normalise
            self.df = self.df.copy()
            self.df.index.name = "gene_id"

    @property
    def genes(self) -> list[str]:
        return list(self.df.index)

    @property
    def compartments(self) -> list[str]:
        return list(self.df.columns)

    def with_compartments(self, mapping: Mapping[str, str]) -> LocalizationScores:
        """Rename compartment columns via ``{old: new}`` (for example predictor labels →
        model compartments). Unmapped columns are kept; multiple sources can be merged
        with ``df.combine_first`` afterwards."""
        return LocalizationScores(self.df.rename(columns=dict(mapping)))


# --------------------------------------------------------------------- compartment mapping

#: Default predictor/database compartment label → model compartment id. Keys are matched
#: case-insensitively. Tuned for yeast/fungal models (e.g. yeast-GEM codes); override per model.
#: Labels not present here (e.g. ``Plastid`` for fungi) are dropped when this map is used.
DEFAULT_COMPARTMENT_MAP: dict[str, str] = {
    "cytoplasm": "c",
    "cytosol": "c",
    "nucleus": "n",
    "nucleoplasm": "n",
    "mitochondrion": "m",
    "mitochondria": "m",
    "mitochondrial": "m",
    "peroxisome": "p",
    "endoplasmic reticulum": "er",
    "golgi apparatus": "g",
    "golgi": "g",
    "vacuole": "v",
    "lysosome/vacuole": "v",
    "lysosome": "v",
    "extracellular": "e",
    "extracellular space": "e",
    "extracellular region": "e",
    "secreted": "e",
    "cell membrane": "ce",
    "plasma membrane": "ce",
    "cell envelope": "ce",
    "lipid particle": "lp",
    "lipid droplet": "lp",
}


def _apply_map(df: pd.DataFrame, compartment_map: Mapping[str, str] | None) -> pd.DataFrame:
    """Rename compartment columns to model ids and collapse synonyms (max). Columns whose label
    is not in ``compartment_map`` are dropped. ``None`` leaves the columns untouched."""
    if not compartment_map:
        return df
    lower = {str(k).strip().lower(): v for k, v in compartment_map.items()}
    renamed = {col: lower[str(col).strip().lower()]
               for col in df.columns if str(col).strip().lower() in lower}
    sub = df[list(renamed)].rename(columns=renamed)
    if sub.shape[1] == 0:
        return sub
    # collapse duplicate target ids (e.g. Lysosome + Vacuole → v) taking the strongest evidence
    return sub.T.groupby(level=0).max().T


# --------------------------------------------------------------------- DeepLoc 2 / MULocDeep

def _wide_mapped(df: pd.DataFrame, *, id_column, compartment_map, source: str) -> pd.DataFrame:
    """Mapped, un-normalised gene x model-compartment table (non-numeric metadata columns
    dropped)."""
    id_col = id_column if id_column is not None else df.columns[0]
    wide = df.set_index(id_col).apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if wide.shape[1] == 0:
        raise ValueError(f"{source}: no numeric compartment columns found in {list(df.columns)}")
    wide = _apply_map(wide.fillna(0.0), compartment_map)
    wide.index.name = "gene_id"
    return wide


def _finalise(wide: pd.DataFrame, *, min_confidence: float,
              normalise: bool = True) -> LocalizationScores:
    """Drop genes whose top (pre-normalisation) score is below ``min_confidence``, then
    optionally normalise each gene's row so its best compartment is 1.0."""
    if min_confidence > 0:
        wide = wide.loc[wide.max(axis=1) >= min_confidence]
    return _normalise_rows(LocalizationScores(wide)) if normalise else LocalizationScores(wide)


def load_deeploc(path: str | Path, *,
                 compartment_map: Mapping[str, str] | None = None,
                 min_confidence: float = 0.0,
                 membrane_split: Mapping[str, str] | None = None,
                 membrane_threshold: float = 0.5,
                 keep_raw_confidence: bool = False,
                 normalise: bool = True) -> LocalizationScores:
    """Parse DeepLoc 2 CSV output into a normalised :class:`LocalizationScores`.

    DeepLoc 2's per-protein CSV has ``Protein_ID, Localizations, Signals`` then one probability
    column per compartment. The first column is the gene id; non-numeric metadata columns
    (``Localizations``, ``Signals``) are dropped. The measurements behind ``membrane_split`` and
    ``normalise`` are in the `DeepLoc yeast-GEM benchmark
    <https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/localization/deeploc_yeast_benchmark.md>`_
    and the `DeepLoc normalisation benchmark
    <https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/localization/deeploc_normalisation_benchmark.md>`_
    on raven-gecko-parity.

    Parameters
    ----------
    path : str or Path
        DeepLoc 2 CSV file.
    compartment_map : mapping of str to str or None
        Renames the predictor labels to model compartment ids (default ``None``, which keeps the
        labels). A label absent from the map is dropped with its column (see the module docstring).
    min_confidence : float
        Drops genes whose top compartment probability (before per-gene normalisation) is below it
        (default ``0.0``, which keeps every gene). In the yeast-GEM benchmark of the confidence
        study, the share of calls that other evidence corroborates rises with the raw probability.
    membrane_split : mapping of str to str or None
        For example ``{"m": "mm"}`` (default ``None``, no split). It moves the probability of an
        organelle to its membrane sub-compartment when the protein is membrane-associated
        (``1 - P(Soluble) >= membrane_threshold``, using DeepLoc's ``Soluble`` column); otherwise
        it stays in the lumen. Keys and values are the compartment ids of the model (after
        ``compartment_map``). A key that is not a compartment of the table is skipped. Raises
        :class:`ValueError` when the file has no ``Soluble`` column. The evidence supports only the
        mitochondrial split (``m``/``mm``; matrix versus membrane AUC ~0.93). DeepLoc does not
        separate the ER lumen from the membrane, so a split of ``er`` has no support.
    membrane_threshold : float
        Value of ``1 - P(Soluble)`` from which a protein counts as membrane-associated (default
        ``0.5``). Used only with ``membrane_split``.
    keep_raw_confidence : bool
        If True, attach the per-gene pre-normalisation top probability to
        :attr:`LocalizationScores.raw_confidence` (default False; normalisation otherwise sets
        every top to 1.0). :func:`triage_localization` uses it for its low-confidence signal.
    normalise : bool
        If True (default), rescale the best compartment of every gene to 1.0. This follows the
        RAVEN ``parseScores`` convention and keeps the scales of several sources comparable. False
        keeps DeepLoc's raw probabilities. The raw probabilities are calibrated (a 0.97 call is
        more reliable than a 0.40 one), so a downstream assignment can weight confident genes more
        heavily.
    """
    raw = pd.read_csv(path)
    wide = _wide_mapped(raw, id_column=None, compartment_map=compartment_map, source=str(path))
    if membrane_split:
        if "Soluble" not in raw.columns:
            raise ValueError("membrane_split needs DeepLoc's 'Soluble' column (membrane-type output)")
        sol = pd.to_numeric(raw.set_index(raw.columns[0])["Soluble"], errors="coerce").fillna(1.0)
        is_membrane = ((1.0 - sol).reindex(wide.index).fillna(0.0) >= membrane_threshold)
        for lumen_id, membrane_id in membrane_split.items():
            if lumen_id not in wide.columns:
                continue
            wide[membrane_id] = wide[lumen_id].where(is_membrane, 0.0)
            wide[lumen_id] = wide[lumen_id].where(~is_membrane, 0.0)
    raw_conf = wide.max(axis=1) if keep_raw_confidence else None
    scores = _finalise(wide, min_confidence=min_confidence, normalise=normalise)
    if raw_conf is not None:
        scores.raw_confidence = raw_conf.reindex(scores.df.index)
    return scores


def load_mulocdeep(path: str | Path, *,
                   compartment_map: Mapping[str, str] | None = None,
                   id_column: str | None = None,
                   sep: str | None = None,
                   min_confidence: float = 0.0) -> LocalizationScores:
    """Parse MULocDeep output into a normalised :class:`LocalizationScores`.

    Expects a wide table: a protein or gene id column plus one numeric probability column per major
    subcellular compartment (MULocDeep predicts 10).

    Parameters
    ----------
    path : str or Path
        MULocDeep output table.
    compartment_map : mapping of str to str or None
        Renames the predictor labels to model compartment ids (default ``None``, which keeps the
        labels). A label absent from the map is dropped with its column (see the module docstring).
    id_column : str or None
        Name of the id column (default ``None``, the first column).
    sep : str or None
        Field delimiter (default ``None``, auto-detected).
    min_confidence : float
        Works as in :func:`load_deeploc` (default ``0.0``).
    """
    df = (pd.read_csv(path, sep=None, engine="python") if sep is None
          else pd.read_csv(path, sep=sep))
    wide = _wide_mapped(df, id_column=id_column, compartment_map=compartment_map, source=str(path))
    return _finalise(wide, min_confidence=min_confidence)


def combine_scores(sources: list[LocalizationScores], *,
                   weights: list[float] | None = None) -> LocalizationScores:
    """Merge several :class:`LocalizationScores` into a consensus by weighted sum.

    Genes and compartments are the union across sources; absent entries count as 0. Scores are
    summed per ``(gene, compartment)`` (optionally weighted) and the result is per-gene normalised
    (best -> 1.0), so a compartment supported by several sources scores higher than a
    compartment supported by one source. The consensus combines complementary evidence (for example
    DeepLoc, curated UniProt and COMPARTMENTS); two curated sources agree on ~90% of the genes in
    yeast-GEM. Raises :class:`ValueError` when ``sources`` is empty or ``weights`` and ``sources``
    differ in length.

    Parameters
    ----------
    sources : list of LocalizationScores
        The score tables to merge.
    weights : list of float or None
        One weight per source (default ``None``, a weight of 1.0 for every source).
    """
    sources = list(sources)
    if not sources:
        raise ValueError("combine_scores needs at least one LocalizationScores")
    weights = [1.0] * len(sources) if weights is None else list(weights)
    if len(weights) != len(sources):
        raise ValueError(f"weights ({len(weights)}) must match the number of sources ({len(sources)})")
    total: pd.DataFrame | None = None
    for s, w in zip(sources, weights, strict=True):
        df = s.df.mul(float(w))
        total = df.copy() if total is None else total.add(df, fill_value=0.0)
    assert total is not None
    total = total.fillna(0.0)
    total.index.name = "gene_id"
    return _normalise_rows(LocalizationScores(total))


# --------------------------------------------------------------------- COMPARTMENTS database

def load_compartments(path: str | Path, *,
                      compartment_map: Mapping[str, str] | None = None,
                      min_confidence: float = 0.0) -> LocalizationScores:
    """Parse a COMPARTMENTS (jensenlab.org) channel file into a normalised
    :class:`LocalizationScores`.

    Reads the tidy ``*_<channel>_full.tsv`` layout (no header): column 1 = protein or gene id,
    column 4 = GO compartment term name, last column = confidence score. Rows are aggregated per
    (gene, compartment) by the maximum confidence; ``min_confidence`` drops weak annotations
    (the integrated and text-mining channels score 0 to 5).

    The per-organism files (for example ``yeast_compartment_integrated_full.tsv``) are available
    from https://download.jensenlab.org/, also when the COMPARTMENTS Downloads web page is
    unavailable. Raises :class:`ValueError` when the file has fewer than 5 columns.

    Parameters
    ----------
    path : str or Path
        COMPARTMENTS channel file.
    compartment_map : mapping of str to str or None
        Renames the GO compartment term names to model compartment ids (default ``None``, which
        keeps the term names). A term absent from the map is dropped (see the module docstring).
    min_confidence : float
        Drops annotations whose confidence score is below it (default ``0.0``).
    """
    raw = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if raw.shape[1] < 5:
        raise ValueError(f"{path}: expected a COMPARTMENTS full TSV (>=5 columns), "
                         f"got {raw.shape[1]}")
    long = pd.DataFrame({
        "gene_id": raw[0].astype(str),
        "compartment": raw[3].astype(str),
        "score": pd.to_numeric(raw[raw.columns[-1]], errors="coerce"),
    }).dropna(subset=["score"])
    long = long[long["score"] >= min_confidence]
    wide = (long.pivot_table(index="gene_id", columns="compartment", values="score",
                             aggfunc="max").fillna(0.0))
    wide = _apply_map(wide, compartment_map)
    wide.index.name = "gene_id"
    return _normalise_rows(LocalizationScores(wide))


# --------------------------------------------------------------------- UniProt

def load_uniprot(path: str | Path, *,
                 compartment_map: Mapping[str, str] = DEFAULT_COMPARTMENT_MAP,
                 id_column: str | None = None,
                 location_column: str | None = None,
                 sep: str = "\t") -> LocalizationScores:
    """Parse a UniProtKB export into a normalised :class:`LocalizationScores`.

    UniProt's curated ``Subcellular location [CC]`` annotation is qualitative (a set of
    compartments, not probabilities), so each annotated compartment gets score ``1.0`` (a
    multi-location protein gets 1.0 in several compartments). The annotation text is scanned for the
    labels in ``compartment_map`` (which is therefore also the vocabulary), and the labels are
    mapped to the compartment ids of the model. Labels match as whole words (``cytoplasm`` does not
    match ``Cytoplasmic vesicle``). Evidence ``{ECO:…}`` braces are removed, the text is split into
    one block per ``[Isoform …]:`` and the ``Note=…`` free text is cut within each block, so
    incidental mentions there are ignored and the locations of a later isoform are kept.

    The columns are the distinct codes of ``compartment_map`` in the order they first occur when
    the map keys are sorted, whether or not a gene has a hit in them, as in RAVEN. Genes without a
    hit are absent. When a gene occurs in several rows, the first row is kept, the genes are
    sorted and a ``UserWarning`` is issued.

    Export from https://rest.uniprot.org with ``fields=accession,gene_oln,cc_subcellular_location``
    (TSV). Raises :class:`ValueError` when ``compartment_map`` is empty and when no location column
    is found.

    Parameters
    ----------
    path : str or Path
        UniProtKB export.
    compartment_map : mapping of str to str
        Maps the location labels to model compartment ids, and is the vocabulary that the
        annotation text is scanned for (default :data:`DEFAULT_COMPARTMENT_MAP`).
    id_column : str or None
        Column with the gene ids (default ``None``, the first column, the accession in this
        export). For yeast-GEM, the ordered-locus column (``Gene Names (ordered locus)``, the ORF
        id like ``YNR001C``) matches the gene ids.
    location_column : str or None
        Column with the subcellular location annotation (default ``None``, the first column whose
        name contains ``subcellular``, ignoring case).
    sep : str
        Field delimiter of the file (default ``"\t"``).
    """
    df = pd.read_csv(path, sep=sep, dtype=str).fillna("")
    return _uniprot_scores(df, id_column=id_column, location_column=location_column,
                           compartment_map=compartment_map)


#: UniProt field name → its column header in a TSV export.
_UNIPROT_ID_COLUMN = {
    "accession": "Entry",
    "gene_primary": "Gene Names (primary)",
    "gene_oln": "Gene Names (ordered locus)",
}


def fetch_uniprot_localization(organism: int | str, *,
                               compartment_map: Mapping[str, str] = DEFAULT_COMPARTMENT_MAP,
                               id_field: str = "gene_oln",
                               reviewed: bool = True,
                               extra_query: str | None = None,
                               timeout: float = 120.0) -> LocalizationScores:
    """Query the UniProtKB REST API for the subcellular locations of an organism and return
    normalised :class:`LocalizationScores`.

    Parsing is as in :func:`load_uniprot`. Raises :class:`ValueError` when ``id_field`` is not
    one of the accepted values.

    Parameters
    ----------
    organism : int or str
        UniProt organism or taxon id (for example ``559292`` for *S. cerevisiae* S288C).
    compartment_map : mapping of str to str
        Maps the location labels to model compartment ids, as in :func:`load_uniprot` (default
        :data:`DEFAULT_COMPARTMENT_MAP`).
    id_field : str
        Identifier used as gene id (default ``"gene_oln"``): ``"gene_oln"`` (ordered locus, the
        ORF name like ``YNR001C``; matches yeast-GEM), ``"accession"`` or ``"gene_primary"``.
    reviewed : bool
        If True (default), restrict the query to curated Swiss-Prot entries.
    extra_query : str or None
        UniProt query string that is ANDed into the query (default ``None``).
    timeout : float
        Timeout of the HTTP request in seconds (default ``120.0``).
    """
    if id_field not in _UNIPROT_ID_COLUMN:
        raise ValueError(f"id_field must be one of {list(_UNIPROT_ID_COLUMN)}")
    query = f"organism_id:{organism}"
    if reviewed:
        query += " AND reviewed:true"
    if extra_query:
        query += f" AND ({extra_query})"
    params = {"query": query, "format": "tsv",
              "fields": "accession,gene_primary,gene_oln,cc_subcellular_location"}
    url = "https://rest.uniprot.org/uniprotkb/stream?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "raven-toolbox"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed https host
        text = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str).fillna("")
    return _uniprot_scores(df, id_column=_UNIPROT_ID_COLUMN[id_field],
                           location_column="Subcellular location [CC]",
                           compartment_map=compartment_map)


def _uniprot_scores(df: pd.DataFrame, *, id_column, location_column,
                    compartment_map) -> LocalizationScores:
    """Shared UniProt parser: scan the location annotation for known compartment terms."""
    id_col = id_column if id_column is not None else df.columns[0]
    if location_column is None:
        cands = [c for c in df.columns if "subcellular" in str(c).lower()]
        if not cands:
            raise ValueError(f"no 'Subcellular location' column in {list(df.columns)}")
        location_column = cands[0]
    # Labels in the sorted order of the map keys; the output columns are the distinct codes
    # in the order they first occur there (RAVEN's parseScores for the uniprot predictor).
    terms = [(str(k).strip().lower(), v) for k, v in sorted(
        (compartment_map or {}).items(), key=lambda kv: str(kv[0]))]
    if not terms:
        raise ValueError("UniProt parsing needs a non-empty compartment_map (the term vocabulary)")
    codes = list(dict.fromkeys(code for _label, code in terms))
    # Whole-word patterns (RAVEN's \<...\>) so that a short label such as 'cytoplasm' does not
    # match inside a longer term such as 'Cytoplasmic vesicle'.
    patterns = [
        (re.compile(r"(?<!\w)(?=\w)" + re.escape(label) + r"(?<=\w)(?!\w)"), code)
        for label, code in terms
    ]

    ids: list[str] = []
    hit_rows: list[dict[str, float]] = []
    for gid, text in zip(df[id_col].astype(str), df[location_column].astype(str), strict=True):
        gid = gid.strip()
        if not gid:
            continue
        clean = re.sub(r"\{[^}]*\}", " ", text.lower())      # drop evidence braces
        # One block per isoform; the Note= free text is cut within each block, so a later
        # isoform's locations survive and Note prose is never read as a location.
        hits: set[str] = set()
        for block in re.split(r"\[isoform[^\]]*\]:", clean):
            seg = re.sub(r"note=.*", " ", block, flags=re.DOTALL)
            hits.update(code for pattern, code in patterns if pattern.search(seg))
        if hits:
            ids.append(gid)
            hit_rows.append({c: 1.0 for c in hits})
    out = pd.DataFrame(hit_rows, index=ids, columns=codes).fillna(0.0)
    if out.index.has_duplicates:
        warnings.warn("There are duplicate genes in the input file", stacklevel=3)
        out = out[~out.index.duplicated(keep="first")].sort_index()
    out.index.name = "gene_id"
    return _normalise_rows(LocalizationScores(out))


# ----------------------------------------------------------------------- helpers

def _normalise_rows(s: LocalizationScores) -> LocalizationScores:
    """Per-gene row normalisation: best compartment to 1.0 (RAVEN's parseScores convention).

    Rows whose max is ≤0 are left unscaled (no positive evidence to normalise against).
    """
    df = s.df.copy()
    row_max = df.max(axis=1)
    safe = row_max > 0
    df.loc[safe] = df.loc[safe].div(row_max[safe], axis=0)
    return LocalizationScores(df)
