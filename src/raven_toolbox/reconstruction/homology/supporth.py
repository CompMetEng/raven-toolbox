"""Read SuppOrth consensus orthologs into a homology hits table.

`SuppOrth <https://github.com/fma91/SuppOrth>`_ runs several ortholog predictors
(OrthoFinder, SonicParanoid2, Broccoli, and optionally imports an OMA run) on a set of
proteomes and writes ``suppOrth.tsv``: one row per cross-species protein pair, with the
number of predictors that support it. :func:`hits_from_supporth` turns that table into
the ``HIT_COLUMNS`` table that
:func:`~raven_toolbox.reconstruction.homology.get_model_from_homology` takes, and
:func:`run_supporth` runs the ``suppOrth`` command and reads its output.

SuppOrth is an external tool and is not a Python dependency of raven-toolbox: nothing
here imports it. Install it with
``pip install 'git+https://github.com/fma91/SuppOrth.git@stand_alone'`` followed by
``suppOrth install``. Its predictors run on Linux and macOS.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from raven_toolbox.reconstruction.homology.blast import _available_cpus
from raven_toolbox.reconstruction.homology.hits import HIT_COLUMNS

SOURCE = "SuppOrth"

_REQUIRED = ["Species1", "Species2", "Protein1", "Protein2", "Lv_support", "SupportedBy"]
_ALIGNMENT = ["Identity", "e_val", "Bitscore"]
_DEFAULT_TOOLS = ("orthofinder", "sonicparanoid", "broccoli")
_EXTRA_COLUMNS = ["support", "supported_by", "identity_raw", "evalue_raw", "bitscore_raw"]
_LABEL_ORDER = ("F", "S", "B", "O")
_PREFIX_SEP = "__"
_ENV_VAR = "RAVEN_SUPPORTH"
_INSTALL_HINT = (
    "Install it with `pip install 'git+https://github.com/fma91/SuppOrth.git@stand_alone'` "
    "followed by `suppOrth install`, then put `suppOrth` on PATH, set "
    f"{_ENV_VAR}, or pass supporth="
)


def _read_table(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        # Protein ids such as "NA" stay strings; only empty alignment fields become NaN.
        df = pd.read_csv(
            source, sep="\t", keep_default_na=False,
            na_values={c: [""] for c in _ALIGNMENT}, float_precision="round_trip",
            dtype={c: str for c in ("Species1", "Species2", "Protein1", "Protein2", "SupportedBy")},
        )
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Not a suppOrth.tsv table (SuppOrth 0.4): missing columns {missing}. "
            "suppOrth.genes.tsv is not supported; pass protein_to_gene instead."
        )
    for col in ("Species1", "Species2", "Protein1", "Protein2", "SupportedBy"):
        df[col] = df[col].astype(str)
    for col in _ALIGNMENT:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else float("nan")
    df["Lv_support"] = pd.to_numeric(df["Lv_support"]).astype(int)
    return df


def _tool_columns(df: pd.DataFrame) -> list[str]:
    """The per-predictor 0/1 columns: those between ``SupportedBy`` and ``Identity``."""
    cols = list(df.columns)
    start = cols.index("SupportedBy") + 1
    stop = cols.index("Identity") if "Identity" in cols else len(cols)
    return [c for c in cols[start:stop] if c not in _ALIGNMENT]


def _load_gene_map(protein_to_gene) -> dict[str, str]:
    if isinstance(protein_to_gene, Mapping):
        return {str(k): str(v) for k, v in protein_to_gene.items()}
    path = Path(protein_to_gene)
    if path.suffix.lower() in (".pkl", ".pickle"):
        raise ValueError(
            f"{path}: pickle maps are not read; pass a dict, a JSON file or a two-column "
            "tab-separated file."
        )
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as fh:
            return {str(k): str(v) for k, v in json.load(fh).items()}
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"{path}: expected two tab-separated columns, got {line!r}")
            mapping[fields[0].strip()] = fields[1].strip()
    return mapping


def _join_labels(labels: pd.Series) -> str:
    """Union of comma-separated predictor labels, in SuppOrth's F, S, B, O order."""
    found = {lb.strip() for text in labels for lb in text.split(",") if lb.strip()}
    rank = {lb: i for i, lb in enumerate(_LABEL_ORDER)}
    return ",".join(sorted(found, key=lambda lb: (rank.get(lb, len(rank)), lb)))


def hits_from_supporth(
    source: str | Path | pd.DataFrame,
    organism_id: str,
    species_to_model: Mapping[str, str | None] | None = None,
    *,
    min_support: int = 1,
    require_tools: Sequence[str] = (),
    min_identity: float | None = None,
    max_evalue: float | None = None,
    protein_to_gene: Mapping[str, str] | str | Path | None = None,
) -> pd.DataFrame:
    """Build a homology hits table from a SuppOrth ``suppOrth.tsv``.

    Only the pairs between ``organism_id`` and another species are kept. Each pair is
    written in both directions, so every ``strictness`` of
    :func:`~raven_toolbox.reconstruction.homology.get_model_from_homology` accepts it,
    and ``strictness`` 1 and 2 give the same result.

    SuppOrth decides orthology by agreement between predictors, so the filters that
    apply to it are in this function, and the hit metrics are placeholders that pass
    the alignment filters of ``get_model_from_homology`` (``evalue`` 0, ``identity``
    100, ``align_len`` 1000, ``ppos`` 100). ``bitscore`` is SuppOrth's own value (0
    when it has none), which ``strictness=3`` ranks by. Because the identity is a
    placeholder, ``review_identity`` reports no candidates for these hits.

    Parameters
    ----------
    source
        Path to ``suppOrth.tsv``, or the table as a DataFrame. A DataFrame read with
        pandas defaults has protein ids such as ``NA`` turned into NaN; read it with
        ``keep_default_na=False`` or pass the path.
    organism_id
        SuppOrth species label of the organism to build a model for: the FASTA file
        name without extension, or the ``--labels`` value. It becomes ``from_id`` /
        ``to_id`` unless ``species_to_model`` maps it to another id.
    species_to_model
        SuppOrth species label to model id. A label mapped to ``None`` is left out.
        When given, every label in the table must be a key. When ``None``, labels are
        used as model ids.
    min_support
        Minimum number of predictors that found the pair (``Lv_support``). SuppOrth has
        already filtered the table with its own ``--supportLv`` and
        ``--identity_filter``; this filter is applied in addition.
    require_tools
        Predictors that must all have found the pair, by column name
        (``"orthofinder"``, ``"sonicparanoid"``, ``"broccoli"``, ``"oma"``).
    min_identity, max_evalue
        Filters on SuppOrth's alignment values. A pair without an alignment hit fails
        both.
    protein_to_gene
        Protein id to model gene id, as a dict, a JSON file, or a two-column
        tab-separated file (lines starting with ``#`` are skipped). Proteins missing
        from the map keep their id, and their number is reported in a warning. A gene
        pair takes the highest support over its protein pairs, the union of their
        predictors, the highest identity and bitscore, and the lowest e-value; these
        values can come from different protein pairs.

    Returns
    -------
    pandas.DataFrame
        ``HIT_COLUMNS`` followed by ``support``, ``supported_by``, ``identity_raw``,
        ``evalue_raw`` and ``bitscore_raw`` (SuppOrth's values). ``attrs["source"]`` is
        ``"SuppOrth"``.
    """
    df = _read_table(source)
    organism_id = str(organism_id)

    labels = sorted(set(df.Species1) | set(df.Species2))
    if organism_id not in labels:
        raise ValueError(f"organism_id {organism_id!r} is not a species in the table: {labels}")
    if species_to_model is not None:
        unmapped = [lb for lb in labels if lb not in species_to_model]
        if unmapped:
            raise ValueError(
                f"species_to_model has no entry for {unmapped}; map each label to a "
                "model id, or to None to leave it out."
            )
        if species_to_model.get(organism_id, organism_id) is None:
            raise ValueError(f"species_to_model maps organism_id {organism_id!r} to None.")
        id_of = {lb: species_to_model[lb] for lb in labels}
    else:
        id_of = {lb: lb for lb in labels}

    tool_cols = _tool_columns(df)
    unknown = [t for t in require_tools if t not in tool_cols]
    if unknown:
        raise ValueError(f"require_tools {unknown} not in the table; its predictors are {tool_cols}")

    keep = df.Lv_support >= min_support
    for tool in require_tools:
        keep &= pd.to_numeric(df[tool], errors="coerce").fillna(0).astype(int) == 1
    if min_identity is not None:
        keep &= df.Identity.notna() & (df.Identity >= min_identity)
    if max_evalue is not None:
        keep &= df.e_val.notna() & (df.e_val <= max_evalue)
    df = df[keep]

    # Orient every pair as (organism protein, template species, template protein).
    first = df[(df.Species1 == organism_id) & (df.Species2 != organism_id)]
    second = df[(df.Species2 == organism_id) & (df.Species1 != organism_id)]
    columns = ["Lv_support", "SupportedBy", "Identity", "e_val", "Bitscore"]
    pairs = pd.concat([
        first[columns].assign(new=first.Protein1, label=first.Species2, template=first.Protein2),
        second[columns].assign(new=second.Protein2, label=second.Species1, template=second.Protein1),
    ], ignore_index=True)
    pairs["model_id"] = pairs.label.map(id_of)
    pairs = pairs[pairs.model_id.notna()]

    if protein_to_gene is not None:
        gene_of = _load_gene_map(protein_to_gene)
        proteins = set(pairs.new) | set(pairs.template)
        n_missing = len(proteins - gene_of.keys())
        if n_missing:
            warnings.warn(
                f"{n_missing} of {len(proteins)} proteins are not in protein_to_gene and "
                "keep their protein id.",
                stacklevel=2,
            )
        pairs["new"] = pairs.new.map(lambda p: gene_of.get(p, p))
        pairs["template"] = pairs.template.map(lambda p: gene_of.get(p, p))

    if pairs.empty:
        raise ValueError(
            f"No SuppOrth pairs between {organism_id!r} and a template species remain "
            "after filtering."
        )

    genes = pairs.groupby(["model_id", "new", "template"], sort=True).agg(
        support=("Lv_support", "max"),
        supported_by=("SupportedBy", _join_labels),
        identity_raw=("Identity", "max"),
        evalue_raw=("e_val", "min"),
        bitscore_raw=("Bitscore", "max"),
    ).reset_index()

    new_id = str(id_of[organism_id])
    bitscore = genes.bitscore_raw.fillna(0.0)
    extras = genes[_EXTRA_COLUMNS]
    fwd = pd.DataFrame({
        "from_id": new_id, "to_id": genes.model_id, "from_gene": genes.new,
        "to_gene": genes.template,
    })
    rev = pd.DataFrame({
        "from_id": genes.model_id, "to_id": new_id, "from_gene": genes.template,
        "to_gene": genes.new,
    })
    frames = []
    for view in (fwd, rev):
        view = view.assign(evalue=0.0, identity=100.0, align_len=1000, bitscore=bitscore,
                           ppos=100.0)
        frames.append(pd.concat([view[HIT_COLUMNS], extras], axis=1))
    hits = pd.concat(frames, ignore_index=True)
    hits.attrs["source"] = SOURCE
    return hits


def _resolve_supporth(supporth: str | os.PathLike | None) -> str:
    if supporth is not None:
        return os.fspath(supporth)
    if os.environ.get(_ENV_VAR):
        return os.environ[_ENV_VAR]
    for name in ("suppOrth", "supporth"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(f"Could not find the SuppOrth command `suppOrth`. {_INSTALL_HINT}.")


def _check_labels(organism_id: str, model_ids: list[str]) -> None:
    ids = [organism_id, *model_ids]
    for label in ids:
        if (not label or label in (".", "..") or label != label.strip()
                or any(ch in label for ch in ",/\\")):
            raise ValueError(
                f"{label!r} cannot be a SuppOrth species label: it must be non-empty, not "
                "'.' or '..', have no leading or trailing space, and contain no ',', '/' "
                "or '\\'."
            )
    if len(set(ids)) != len(ids):
        raise ValueError(f"organism_id and model_ids must be distinct, got {ids}")


def _open_fasta(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _fasta_ids(path: Path) -> list[str]:
    """Sequence ids as SuppOrth reads them: the first word of each header line."""
    with _open_fasta(path) as fh:
        return [line[1:].split(maxsplit=1)[0] for line in fh
                if line.startswith(">") and line[1:].strip()]


def _write_prefixed(src: Path, dest: Path, prefix: str) -> None:
    with _open_fasta(src) as fin, open(dest, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(">" + prefix + line[1:] if line.startswith(">") else line)


def _strip_prefixes(table: pd.DataFrame) -> pd.DataFrame:
    """Remove the ``<label>__`` prefix that :func:`run_supporth` added to protein ids."""
    table = table.copy()
    for species, protein in (("Species1", "Protein1"), ("Species2", "Protein2")):
        table[protein] = [
            p[len(s) + len(_PREFIX_SEP):] if p.startswith(s + _PREFIX_SEP) else p
            for s, p in zip(table[species], table[protein], strict=True)
        ]
    return table


def run_supporth(
    organism_id: str,
    fasta: str | Path,
    model_ids: Sequence[str],
    ref_fastas: Sequence[str | Path],
    outdir: str | Path,
    *,
    tools: str | Sequence[str] = "all",
    support_lv: int | None = None,
    identity_filter: float = 80.0,
    alignment: str = "diamond",
    threads: int | None = None,
    extra_args: Sequence[str] = (),
    supporth: str | os.PathLike | None = None,
    **reader_kwargs,
) -> pd.DataFrame:
    """Run SuppOrth on an organism and template proteomes and read its consensus.

    ``model_ids`` and ``ref_fastas`` are parallel: ``ref_fastas[i]`` is the proteome
    of template ``model_ids[i]``, with FASTA ids equal to that model's gene ids (or
    mapped to them with ``protein_to_gene``). The ids are passed to SuppOrth as
    ``--labels``, so the table uses them as species labels.

    ``support_lv`` and ``identity_filter`` are SuppOrth's ``--supportLv`` and
    ``--identity_filter``: a pair is kept if at least ``support_lv`` predictors found
    it, or if its identity is above ``identity_filter``. ``support_lv`` defaults to
    3, or to the number of ``tools`` when fewer are selected. ``extra_args`` are
    appended to the command, for example ``["--resume"]`` or
    ``["--oma", "path/to/Output"]``. Output is written to ``outdir``. ``threads`` defaults to
    every CPU the process may use, as in RAVEN's ``runSuppOrth``.

    SuppOrth requires protein ids to be unique over all proteomes. When two FASTA
    files share an id (for example two templates of the same species), every id is
    given a ``<label>__`` prefix in copies written to ``outdir/prefixed_inputs``, and
    the prefix is removed from the returned hits. ``outdir/suppOrth.tsv`` then
    contains the prefixed ids. SuppOrth's console output is written to
    ``outdir/run_supporth.log``.

    Remaining keyword arguments go to :func:`hits_from_supporth`.

    Requires the ``suppOrth`` command, found from ``supporth``, then the
    ``RAVEN_SUPPORTH`` environment variable, then PATH. SuppOrth's predictors do not
    run on Windows; use WSL or a Linux host, and read the result with
    :func:`hits_from_supporth`.
    """
    if sys.platform == "win32":
        raise NotImplementedError(
            "SuppOrth's predictors run on Linux and macOS only. Run SuppOrth under WSL or on "
            "a Linux host, and read suppOrth.tsv with hits_from_supporth()."
        )
    threads = _available_cpus() if threads is None else threads
    organism_id = str(organism_id)
    model_ids = [model_ids] if isinstance(model_ids, str) else [str(m) for m in model_ids]
    ref_fastas = [ref_fastas] if isinstance(ref_fastas, (str, Path)) else list(ref_fastas)
    if len(model_ids) != len(ref_fastas):
        raise ValueError("model_ids and ref_fastas must have the same length.")
    if not model_ids:
        raise ValueError("At least one template is needed.")
    _check_labels(organism_id, model_ids)

    if isinstance(tools, str):
        tool_list = [t.strip() for t in tools.split(",") if t.strip()]
    else:
        tool_list = [str(t) for t in tools]
    if tool_list == ["all"]:
        tool_list = list(_DEFAULT_TOOLS)
    if support_lv is None:
        support_lv = min(3, len(tool_list))
    if not 1 <= support_lv <= 4:
        raise ValueError(f"support_lv must be between 1 and 4, got {support_lv}")
    if support_lv > len(tool_list) and "--oma" not in extra_args:
        warnings.warn(
            f"support_lv={support_lv} is more than the {len(tool_list)} predictor(s) "
            f"selected; SuppOrth then keeps only pairs above identity_filter "
            f"({identity_filter} %).",
            stacklevel=2,
        )

    exe = _resolve_supporth(supporth)
    outdir = Path(outdir)
    labels = [organism_id, *model_ids]
    fastas = [Path(fasta), *(Path(f) for f in ref_fastas)]

    # SuppOrth requires protein ids to be unique over all proteomes. Templates of the
    # same species often share gene ids, so in that case every id gets a
    # "<label>__" prefix in a copy of its FASTA, and the prefix is removed again below.
    seen: dict[str, str] = {}
    shared = set()
    for label, path in zip(labels, fastas, strict=True):
        for pid in _fasta_ids(path):
            if seen.setdefault(pid, label) != label:
                shared.add(pid)
    if shared:
        inputs = outdir / "prefixed_inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        prefixed = []
        for label, path in zip(labels, fastas, strict=True):
            dest = inputs / f"{label}.faa"
            _write_prefixed(path, dest, label + _PREFIX_SEP)
            prefixed.append(dest)
        fastas = prefixed

    cmd = [
        exe, "run",
        *(str(f) for f in fastas),
        "--labels", ",".join(labels),
        "-o", str(outdir),
        "--tools", ",".join(tool_list),
        "--supportLv", str(support_lv),
        "--identity_filter", str(identity_filter),
        "--alignment", alignment,
        "-t", str(threads),
        *extra_args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = "\n".join(s.strip() for s in (proc.stdout, proc.stderr) if s and s.strip())
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "run_supporth.log"
    log.write_text(" ".join(cmd) + "\n\n" + output + "\n", encoding="utf-8")
    if proc.returncode != 0:
        lines = output.splitlines()
        # SuppOrth reports a failed stage before the consensus summary, which can push
        # it out of the last lines.
        failed = [ln for ln in lines[:-20] if "FAILED" in ln or ln.startswith("error")]
        tail = "\n".join(failed + lines[-20:])
        raise RuntimeError(
            f"suppOrth exited with code {proc.returncode}; its output in {outdir} is "
            f"incomplete and was not read. Full output in {log}.\n{tail}"
        )
    table = _read_table(outdir / "suppOrth.tsv")
    if shared:
        table = _strip_prefixes(table)
    return hits_from_supporth(table, organism_id, **reader_kwargs)
