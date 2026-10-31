"""Parse a local KEGG flat-file dump into a reference model and relational tables.

Maintainer-side build-time tooling. Produces the published raven_toolbox KEGG artefacts:

* a **gene-free reference GEM** (reactions and metabolites only) as a ``cobra.Model``;
* minimal **relational tables** (``pandas.DataFrame``) written as gzipped TSV:
  ``ko_reaction``, ``ko_names``, ``organism_gene_ko`` (the largest), and
  ``rxn_flags`` (flags spontaneous, undefined_stoich, incomplete and general).

Genes live only in ``organism_gene_ko``; per-organism GPRs are built at runtime
(:mod:`.organism` and :mod:`.query`), so the reference model stays small.

Differences from RAVEN:

* equations are read from the ``EQUATION`` field of each reaction entry and not
  from ``reaction.lst``, whose line order would have to match ``reaction``;
* undefined-stoichiometry terms (``n C00001``, ``(n+1) C00002``) keep their
  compound id with coefficient 1 and the reaction is flagged; RAVEN creates
  ``"n C00001"`` pseudo-metabolites renamed ``undefined_N``;
* quality labels are a boolean ``rxn_flags`` table; RAVEN appends free text to
  ``rxnNotes``.

The KEGG flat-file format: each entry is a block of lines terminated by ``///``;
a field label occupies columns 1-12, continuation lines are indented 12 spaces.
"""
from __future__ import annotations

import gzip
import heapq
import re
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cobra
import pandas as pd
from tqdm import tqdm

from raven_toolbox.io.yaml import write_yaml_model

# A KEGG entry id is the first token after the 12-char ENTRY label (6 chars:
# R00010, C00001, K01194, ...).
_ID_LEN = 6
_LABEL_WIDTH = 12

# Compound token inside an equation, optionally a glycan (G) or drug (D); we also
# tolerate trailing polymer suffixes like "C00404(n)" by matching the stem.
_MET_TOKEN = re.compile(r"^([CGD]\d{5})")
_NUMERIC = re.compile(r"^\d+(\.\d+)?$")


# --------------------------------------------------------------------------- #
# Generic flat-file reader
# --------------------------------------------------------------------------- #
def _iter_entries(
    path: str | Path, *, progress_desc: str | None = None
) -> Iterator[dict[str, list[str]]]:
    """Yield one ``{field_label: [value_lines]}`` dict per ``///``-delimited entry.

    Field labels (columns 1-12) map to a list of their value lines in file order;
    continuation lines (12 leading spaces) append to the current field.

    When ``progress_desc`` is set, a tqdm byte-progress bar (total = file size)
    advances as the file is read; it is used for the ``ko`` pass.
    """
    entry: dict[str, list[str]] = {}
    current: str | None = None
    bar = (
        tqdm(total=Path(path).stat().st_size, desc=progress_desc, unit="B", unit_scale=True)
        if progress_desc is not None
        else None
    )
    try:
        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                if bar is not None:
                    bar.update(len(raw))  # chars≈bytes (ASCII dump); bar is a guide, not exact
                line = raw.rstrip("\n")
                if line.startswith("///"):
                    if entry:
                        yield entry
                    entry, current = {}, None
                    continue
                if not line.strip():
                    continue
                label = line[:_LABEL_WIDTH].strip()
                value = line[_LABEL_WIDTH:].rstrip()
                if label:
                    current = label
                    entry.setdefault(current, []).append(value)
                elif current is not None:
                    entry[current].append(value)
    finally:
        if bar is not None:
            bar.close()
    if entry:  # tolerate a missing final '///'
        yield entry


# --------------------------------------------------------------------------- #
# Reactions
# --------------------------------------------------------------------------- #
@dataclass
class KeggReaction:
    """A reaction parsed from the KEGG ``reaction`` flat file."""

    id: str
    name: str = ""
    equation: str = ""
    reversible: bool = True
    eccodes: list[str] = field(default_factory=list)
    kos: list[str] = field(default_factory=list)
    pathways: list[str] = field(default_factory=list)
    spontaneous: bool = False
    incomplete: bool = False
    general: bool = False
    undefined_stoich: bool = False
    # Cached stoichiometry from ``_parse_equation(equation)``: populated by
    # :func:`parse_kegg_reactions` so :func:`build_reference_model` reuses the
    # parse instead of repeating it (KEGG has ~12k reactions; a full redundant
    # parse cost a noticeable chunk of the build).
    stoichiometry: dict[str, float] = field(default_factory=dict)


def _first_id(lines: list[str]) -> str:
    return lines[0][:_ID_LEN].strip() if lines else ""


def _comment_flags(rxn: KeggReaction, comment: str) -> None:
    text = comment.upper()
    rxn.spontaneous = "SPONTANEOUS" in text
    rxn.incomplete = any(w in text for w in ("INCOMPLETE", "ERRONEOUS", "UNCLEAR"))
    rxn.general = "GENERAL REACTION" in text


def _parse_equation(equation: str) -> tuple[dict[str, float], bool, bool]:
    """Parse a KEGG equation into ``({met_id: coef}, reversible, undefined_stoich)``.

    Reactants get negative coefficients, products positive. Non-numeric
    coefficients (``n``, ``(n+1)``, ``2n``) are treated as 1.0 and flag the
    reaction as having undefined stoichiometry.
    """
    reversible = "<=>" in equation
    parts = re.split(r"\s(?:<=>|=>|<=)\s", equation, maxsplit=1)
    lhs, rhs = (parts + ["", ""])[:2]

    stoich: dict[str, float] = {}
    undefined = False
    for side, sign in ((lhs, -1.0), (rhs, 1.0)):
        for term in filter(None, (t.strip() for t in side.split(" + "))):
            tokens = term.split()
            met_token = tokens[-1]
            coef_tokens = tokens[:-1]
            if coef_tokens and _NUMERIC.match(coef_tokens[0]):
                coef = float(coef_tokens[0])
            else:
                coef = 1.0
                if coef_tokens:  # a symbolic coefficient like 'n' or '(n+1)'
                    undefined = True
            match = _MET_TOKEN.match(met_token)
            if not match:  # unparseable term -> flag, keep raw token
                undefined = True
                met_id = met_token
            else:
                met_id = match.group(1)
            stoich[met_id] = stoich.get(met_id, 0.0) + sign * coef
    # Drop metabolites that cancel out (A <=> A + B leaves A at 0).
    stoich = {m: c for m, c in stoich.items() if c != 0.0}
    return stoich, reversible, undefined


def parse_kegg_reactions(kegg_dir: str | Path) -> list[KeggReaction]:
    """Parse ``<kegg_dir>/reaction`` into :class:`KeggReaction` records.

    Reversibility is taken from the equation arrow and, when
    ``reaction_mapformula.lst`` is present, set to irreversible for reactions that are
    irreversible across all KEGG maps (see :func:`_irreversible_from_mapformula`).
    """
    kegg_dir = Path(kegg_dir)
    reactions: list[KeggReaction] = []
    for entry in _iter_entries(kegg_dir / "reaction"):
        rxn = KeggReaction(id=_first_id(entry.get("ENTRY", [])))
        if not rxn.id:
            continue
        if entry.get("NAME"):
            rxn.name = entry["NAME"][0].rstrip(";").strip()
        if entry.get("COMMENT"):
            _comment_flags(rxn, " ".join(entry["COMMENT"]))
        if entry.get("ENZYME"):
            rxn.eccodes = [ec for line in entry["ENZYME"] for ec in line.split()]
        rxn.kos = [line[:_ID_LEN].strip() for line in entry.get("ORTHOLOGY", [])]
        for line in entry.get("PATHWAY", []):
            pid = line[:7].strip()
            if pid and not pid.startswith(("rn011", "rn012")):  # skip global/overview
                rxn.pathways.append(pid)
        if entry.get("EQUATION"):
            rxn.equation = " ".join(s.strip() for s in entry["EQUATION"])
            stoich, rxn.reversible, rxn.undefined_stoich = _parse_equation(rxn.equation)
            rxn.stoichiometry = stoich  # cached for build_reference_model
        reactions.append(rxn)

    irrev = _irreversible_from_mapformula(kegg_dir / "reaction_mapformula.lst")
    for rxn in reactions:
        if rxn.id in irrev:
            rxn.reversible = False
    return reactions


def _irreversible_from_mapformula(path: str | Path) -> set[str]:
    """Reaction ids that are irreversible in every KEGG map they appear in.

    ``reaction_mapformula.lst`` lines look like ``R00005: 00330: C01010 => C00011``.
    A reaction is considered irreversible only if no map lists it as ``<=>`` and
    every map draws it in the same direction. The direction (substrate and product
    order) is not propagated back into the model stoichiometry. This is a simplification:
    RAVEN also flips the stoichiometric column to the map direction. The difference affects
    only the reactions that have a map direction.
    """
    path = Path(path)
    if not path.is_file():
        return set()
    seen_reversible: set[str] = set()
    products: dict[str, str] = {}
    conflicting: set[str] = set()
    for entry in _iter_mapformula_lines(path):
        rid, reversible, product = entry
        if reversible:
            seen_reversible.add(rid)
        elif rid in products and products[rid] != product:
            conflicting.add(rid)  # drawn both directions across maps -> reversible
        else:
            products.setdefault(rid, product)
    return {rid for rid in products if rid not in seen_reversible and rid not in conflicting}


def _iter_mapformula_lines(path: Path) -> Iterator[tuple[str, bool, str]]:
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            rid = line[:_ID_LEN]
            reversible = "<=>" in line
            product = line.split()[-1]
            yield rid, reversible, product


# --------------------------------------------------------------------------- #
# Compounds
# --------------------------------------------------------------------------- #
@dataclass
class KeggCompound:
    """A metabolite parsed from the KEGG ``compound`` flat file."""

    id: str
    name: str = ""
    formula: str = ""
    inchi: str = ""
    chebi: list[str] = field(default_factory=list)
    pubchem: list[str] = field(default_factory=list)


def parse_kegg_compounds(kegg_dir: str | Path) -> list[KeggCompound]:
    """Parse ``<kegg_dir>/compound`` and the optional ``compound.inchi`` into records."""
    kegg_dir = Path(kegg_dir)
    compounds: list[KeggCompound] = []
    for entry in _iter_entries(kegg_dir / "compound"):
        cid = _first_id(entry.get("ENTRY", []))
        if not cid:
            continue
        cmp = KeggCompound(id=cid)
        if entry.get("NAME"):
            # Only the first synonym; KEGG separates them with ';'.
            cmp.name = entry["NAME"][0].split(";")[0].strip()
        if entry.get("FORMULA"):
            cmp.formula = entry["FORMULA"][0].strip()
        for line in entry.get("DBLINKS", []):
            if line.startswith("ChEBI:"):
                cmp.chebi += [f"CHEBI:{x}" for x in line.split(":", 1)[1].split()]
            elif line.startswith("PubChem:"):
                cmp.pubchem += line.split(":", 1)[1].split()
        compounds.append(cmp)

    inchis = _parse_inchis(kegg_dir / "compound.inchi")
    for cmp in compounds:
        if cmp.id in inchis:
            cmp.inchi = inchis[cmp.id]
            cmp.formula = ""  # prefer the InChI; matches RAVEN
    return compounds


def _parse_inchis(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            cid, _, inchi = raw.rstrip("\n").partition("\t")
            if cid and inchi:
                out[cid.strip()] = inchi.strip()
    return out


# --------------------------------------------------------------------------- #
# KOs and organism genes
# --------------------------------------------------------------------------- #
@dataclass
class KeggKO:
    """A KEGG Orthology entry: its name and the organism genes assigned to it."""

    id: str
    name: str = ""
    genes: list[tuple[str, str]] = field(default_factory=list)  # (organism, gene)


def parse_kegg_kos(kegg_dir: str | Path, *, keep: set[str] | None = None) -> list[KeggKO]:
    """Parse ``<kegg_dir>/ko`` into :class:`KeggKO` records (name and organism genes).

    ``keep`` limits parsing to those KO ids (for example only KOs linked to reactions),
    as RAVEN's ``koList`` argument does; the gene lists are large.
    """
    ko_records: list[KeggKO] = []
    for entry in _iter_entries(Path(kegg_dir) / "ko"):
        ko_id = _first_id(entry.get("ENTRY", []))
        if not ko_id or (keep is not None and ko_id not in keep):
            continue
        ko = KeggKO(id=ko_id)
        if entry.get("DEFINITION"):
            ko.name = entry["DEFINITION"][0].strip()
        ko.genes = list(_parse_gene_lines(entry.get("GENES", [])))
        ko_records.append(ko)
    return ko_records


def _parse_gene_lines(lines: list[str]) -> Iterator[tuple[str, str]]:
    """Yield ``(organism, gene)`` pairs from a KO entry's GENES block.

    Lines look like ``BSU: BSU31050(gbsB) BSU31060``: an upper-case organism
    code, a colon, then space-separated gene ids with an optional ``(name)``
    suffix, which is removed. Organism codes are lower-cased to match KEGG's protein
    sequence files, as in RAVEN.
    """
    for line in lines:
        org, sep, rest = line.partition(":")
        if not sep:
            continue
        organism = org.strip().lower()
        for token in rest.split():
            gene = token.split("(", 1)[0]
            if gene:
                yield organism, gene


# --------------------------------------------------------------------------- #
# Reference model + tables
# --------------------------------------------------------------------------- #
_COMPARTMENT = "s"  # single 'system' compartment, as in getModelFromKEGG


def build_reference_model(
    reactions: list[KeggReaction], compounds: list[KeggCompound]
) -> cobra.Model:
    """Assemble the gene-free KEGG reference model from parsed records.

    Only metabolites used by a reaction are added. Reactions carry KEGG
    annotations (reaction id, KO ids, EC codes, pathways) and no genes or GPRs.
    Reactions with empty stoichiometry are skipped. Bounds are ``(-1000, 1000)``
    for reversible reactions and ``(0, 1000)`` otherwise.
    """
    model = cobra.Model("KEGG")
    model.name = "Automatically generated from KEGG database"

    by_id = {c.id: c for c in compounds}
    # Reuse the cached parse from parse_kegg_reactions; only re-parse for
    # callers that constructed KeggReaction records without the cache.
    parsed = {
        r.id: (r.stoichiometry if r.stoichiometry else _parse_equation(r.equation)[0])
        for r in reactions
    }
    used = {m for stoich in parsed.values() for m in stoich}

    metabolites = []
    for cid in sorted(used):
        cmp = by_id.get(cid)
        met = cobra.Metabolite(cid, compartment=_COMPARTMENT)
        if cmp:
            met.name = cmp.name or cid
            met.formula = cmp.formula or None
            if cmp.chebi:
                met.annotation["chebi"] = cmp.chebi
            if cmp.pubchem:
                met.annotation["pubchem.substance"] = cmp.pubchem
            if cmp.inchi:
                met.annotation["inchi"] = cmp.inchi
        else:
            met.name = cid
        metabolites.append(met)
    model.add_metabolites(metabolites)
    met_index = {m.id: m for m in metabolites}

    cobra_reactions = []
    for rxn in reactions:
        stoich = parsed[rxn.id]
        if not stoich:  # empty (e.g. A <=> A) -> skip, as RAVEN drops bad rxns
            continue
        reaction = cobra.Reaction(rxn.id, name=rxn.name)
        reaction.bounds = (-1000.0, 1000.0) if rxn.reversible else (0.0, 1000.0)
        reaction.add_metabolites({met_index[m]: c for m, c in stoich.items()})
        reaction.annotation["kegg.reaction"] = rxn.id
        if rxn.kos:
            reaction.annotation["kegg.orthology"] = rxn.kos
        if rxn.eccodes:
            reaction.annotation["ec-code"] = rxn.eccodes
        if rxn.pathways:
            reaction.annotation["kegg.pathway"] = rxn.pathways
        cobra_reactions.append(reaction)
    model.add_reactions(cobra_reactions)
    return model


def build_kegg_tables(
    reactions: list[KeggReaction], kos: list[KeggKO]
) -> dict[str, pd.DataFrame]:
    """Build the minimal relational tables from parsed records.

    Returns a dict of ``DataFrame`` objects keyed by table name: ``ko_reaction``,
    ``ko_names``, ``organism_gene_ko``, ``rxn_flags``.
    """
    ko_reaction = pd.DataFrame(
        [(ko, r.id) for r in reactions for ko in r.kos],
        columns=["ko", "reaction"],
    ).drop_duplicates(ignore_index=True)

    ko_names = pd.DataFrame(
        [(ko.id, ko.name) for ko in kos], columns=["ko", "name"]
    )

    organism_gene_ko = pd.DataFrame(
        [(org, gene, ko.id) for ko in kos for org, gene in ko.genes],
        columns=["organism", "gene", "ko"],
    ).drop_duplicates(ignore_index=True)

    rxn_flags = pd.DataFrame(
        [
            (r.id, r.spontaneous, r.undefined_stoich, r.incomplete, r.general)
            for r in reactions
        ],
        columns=["reaction", "spontaneous", "undefined_stoich", "incomplete", "general"],
    )

    return {
        "ko_reaction": ko_reaction,
        "ko_names": ko_names,
        "organism_gene_ko": organism_gene_ko,
        "rxn_flags": rxn_flags,
    }


def write_kegg_tables(
    tables: dict[str, pd.DataFrame], out_dir: str | Path, *, prefix: str = ""
) -> list[Path]:
    """Write each table as a gzipped TSV (``<prefix><name>.tsv.gz``) into ``out_dir``.

    Gzipped TSV is the format shared with MATLAB RAVEN; MATLAB's built-in ``gunzip``
    reads it without an external tool. ``prefix`` version-tags the filenames (for
    example ``kegg116_``). Returns the written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in tables.items():
        path = out_dir / f"{prefix}{name}.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, sep="\t", index=False)
        written.append(path)
    return written


def read_kegg_table(path: str | Path) -> pd.DataFrame:
    """Read a KEGG table from :func:`write_kegg_tables` or :func:`stream_organism_gene_ko`.

    Compression is inferred from the file suffix. The published tables are gzipped TSV
    (``.tsv.gz``), including those with a version prefix (``kegg116_<name>.tsv.gz``).
    All columns are read as strings, and empty fields stay empty strings.
    """
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _resolve_artefact(directory: str | Path, base: str) -> Path:
    """Locate an artefact file by its base name, tolerating a version prefix.

    Returns ``directory/base`` if it exists, else the single version-prefixed
    match ``directory/<version>_<base>`` (the published asset name, for example
    ``kegg116_ko_reaction.tsv.gz``). This lets one reader consume both a user's own
    unprefixed build directory and the version-pinned download cache.
    """
    directory = Path(directory)
    exact = directory / base
    if exact.exists():
        return exact
    matches = sorted(directory.glob(f"*_{base}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"{base!r} (or a <version>_ prefixed form) not found in {directory}"
        )
    raise ValueError(f"Ambiguous {base!r} in {directory}: {sorted(p.name for p in matches)}")


def _flush_sorted_run(rows: list[str], tmp_dir: Path, run_no: int) -> Path:
    """Sort a buffer of ``organism\\tgene\\tko\\n`` lines and write one gzipped run."""
    rows.sort(key=_ogk_sort_key)
    run_path = tmp_dir / f"run_{run_no:04d}.gz"
    with gzip.open(run_path, "wt", encoding="utf-8", newline="") as run:
        run.writelines(rows)
    return run_path


def _ogk_sort_key(line: str) -> tuple[str, str]:
    """Sort key ``(organism, gene)`` for an ``organism\\tgene\\tko`` line."""
    organism, gene, _ = line.split("\t", 2)
    return organism, gene


def stream_organism_gene_ko(
    kegg_dir: str | Path,
    keep: set[str],
    ogk_path: str | Path,
    *,
    chunk_rows: int = 1_000_000,
    progress: bool = False,
) -> pd.DataFrame:
    """Stream the ``ko`` file to a sorted, gzipped ``organism_gene_ko.tsv.gz``.

    ``keep`` restricts output to that set of KO ids (for example only KOs linked to
    reactions), mirroring ``keep`` on :func:`parse_kegg_kos`.

    KEGG has ~9M gene and KO associations, which are too many to hold in memory as a
    DataFrame. Rows are sorted by ``(organism, gene)`` before writing: gene IDs
    from one organism share long common prefixes (locus tags, numeric runs), so
    sorting makes them adjacent and improves compression; the order also matches
    the by-organism query pattern in :func:`get_kegg_model_for_organism`. The table is
    compressed with gzip, which MATLAB's built-in ``gunzip`` reads without an external
    tool; xz would give a smaller file.

    The sort is an **external merge sort** that holds at most ``chunk_rows`` rows in
    memory at a time (sorted runs are written to gzipped temporary files and merged with
    :func:`heapq.merge`), so peak memory does not grow with the size of KEGG. Only
    the ``ko_names`` table (one row per KO) is held in full; it is returned with the
    columns ``ko`` and ``name``.

    ``progress`` shows a tqdm byte-progress bar over the (large) ``ko`` read pass.
    """
    ogk_path = Path(ogk_path)
    names: list[tuple[str, str]] = []
    buffer: list[str] = []
    runs: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="ogk_sort_", dir=ogk_path.parent) as tmp:
        tmp_dir = Path(tmp)
        for entry in _iter_entries(
            Path(kegg_dir) / "ko",
            progress_desc="organism_gene_ko (ko)" if progress else None,
        ):
            ko_id = _first_id(entry.get("ENTRY", []))
            if not ko_id or ko_id not in keep:
                continue
            names.append((ko_id, entry["DEFINITION"][0].strip() if entry.get("DEFINITION") else ""))
            for organism, gene in _parse_gene_lines(entry.get("GENES", [])):
                buffer.append(f"{organism}\t{gene}\t{ko_id}\n")
            if len(buffer) >= chunk_rows:
                runs.append(_flush_sorted_run(buffer, tmp_dir, len(runs)))
                buffer = []
        if buffer:
            runs.append(_flush_sorted_run(buffer, tmp_dir, len(runs)))

        handles = [gzip.open(r, "rt", encoding="utf-8") for r in runs]
        try:
            with gzip.open(ogk_path, "wt", encoding="utf-8", newline="") as out:
                out.write("organism\tgene\tko\n")
                out.writelines(heapq.merge(*handles, key=_ogk_sort_key))
        finally:
            for h in handles:
                h.close()
    return pd.DataFrame(names, columns=["ko", "name"])


def parse_kegg_dump(
    kegg_dir: str | Path,
    out_dir: str | Path,
    *,
    version: str | None = None,
    progress: bool = False,
) -> dict[str, Path]:
    """Parse a full KEGG dump into the reference model + tables and write them out.

    Writes ``reference_model.yml.gz`` (gzipped RAVEN or cobra YAML) plus the
    gzipped TSV tables into ``out_dir`` and returns ``{name: path}`` for
    everything written. The ``organism_gene_ko`` table is streamed to disk (see
    :func:`stream_organism_gene_ko`) and not built in memory, so the full KEGG database
    can be processed; the other tables are built in memory.

    When ``version`` is given (for example ``"kegg116"``), the output filenames are
    version-prefixed (``<version>_<name>``), as in the published release
    assets; the keys of the returned dict are the logical table names.

    ``progress`` reports each stage and shows a byte-progress bar over the
    ``organism_gene_ko`` streaming pass, the longest stage.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{version}_" if version else ""

    if progress:
        tqdm.write("Parsing reactions and compounds...")
    reactions = parse_kegg_reactions(kegg_dir)
    compounds = parse_kegg_compounds(kegg_dir)
    linked_kos = {ko for r in reactions for ko in r.kos}

    if progress:
        tqdm.write(f"Building reference model ({len(reactions)} reactions)...")
    model = build_reference_model(reactions, compounds)

    small = {
        "ko_reaction": pd.DataFrame(
            [(ko, r.id) for r in reactions for ko in r.kos], columns=["ko", "reaction"]
        ).drop_duplicates(ignore_index=True),
        "rxn_flags": pd.DataFrame(
            [(r.id, r.spontaneous, r.undefined_stoich, r.incomplete, r.general) for r in reactions],
            columns=["reaction", "spontaneous", "undefined_stoich", "incomplete", "general"],
        ),
    }
    paths = {
        name: p
        for name, p in zip(small, write_kegg_tables(small, out_dir, prefix=prefix), strict=True)
    }

    if progress:
        tqdm.write("Streaming organism_gene_ko (the large pass)...")
    ogk_path = out_dir / f"{prefix}organism_gene_ko.tsv.gz"
    ko_names = stream_organism_gene_ko(kegg_dir, linked_kos, ogk_path, progress=progress)
    paths["organism_gene_ko"] = ogk_path
    paths.update(
        zip(
            ["ko_names"],
            write_kegg_tables({"ko_names": ko_names}, out_dir, prefix=prefix),
            strict=True,
        )
    )

    if progress:
        tqdm.write("Writing reference model (YAML)...")
    ref_path = out_dir / f"{prefix}reference_model.yml.gz"
    write_yaml_model(model, ref_path)
    paths["reference_model"] = ref_path
    return paths
