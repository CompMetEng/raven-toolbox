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

import json
import os
import shutil
import subprocess
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from raven_toolbox.reconstruction.homology.hits import HIT_COLUMNS

SOURCE = "SuppOrth"

_REQUIRED = ["Species1", "Species2", "Protein1", "Protein2", "Lv_support", "SupportedBy"]
_ALIGNMENT = ["Identity", "e_val", "Bitscore"]
_DEFAULT_TOOLS = ("orthofinder", "sonicparanoid", "broccoli")
_EXTRA_COLUMNS = ["support", "supported_by", "identity_raw", "evalue_raw", "bitscore_raw"]
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
            na_values={c: [""] for c in _ALIGNMENT},
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
    seen: list[str] = []
    for text in labels:
        for label in text.split(","):
            label = label.strip()
            if label and label not in seen:
                seen.append(label)
    return ",".join(seen)


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
        Path to ``suppOrth.tsv``, or the table as a DataFrame.
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
        pair takes the highest support over its protein pairs.

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
        if not label or label in (".", "..") or any(ch in label for ch in ",/\\"):
            raise ValueError(
                f"{label!r} cannot be a SuppOrth species label: it must be non-empty, not "
                "'.' or '..', and contain no ',', '/' or '\\'."
            )
    if len(set(ids)) != len(ids):
        raise ValueError(f"organism_id and model_ids must be distinct, got {ids}")


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
    threads: int = max(1, (os.cpu_count() or 2) - 1),
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
    ``["--oma", "path/to/Output"]``. Output is written to ``outdir``.

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
    organism_id = str(organism_id)
    model_ids = [str(m) for m in model_ids]
    ref_fastas = [ref_fastas] if isinstance(ref_fastas, (str, Path)) else list(ref_fastas)
    if len(model_ids) != len(ref_fastas):
        raise ValueError("model_ids and ref_fastas must have the same length.")
    if not model_ids:
        raise ValueError("At least one template is needed.")
    _check_labels(organism_id, model_ids)

    tool_list = list(_DEFAULT_TOOLS) if tools == "all" else (
        [tools] if isinstance(tools, str) else list(tools))
    if support_lv is None:
        support_lv = min(3, len(tool_list))
    if not 1 <= support_lv <= 4:
        raise ValueError(f"support_lv must be between 1 and 4, got {support_lv}")

    exe = _resolve_supporth(supporth)
    outdir = Path(outdir)
    cmd = [
        exe, "run",
        str(fasta), *(str(f) for f in ref_fastas),
        "--labels", ",".join([organism_id, *model_ids]),
        "-o", str(outdir),
        "--tools", "all" if tools == "all" else ",".join(tool_list),
        "--supportLv", str(support_lv),
        "--identity_filter", str(identity_filter),
        "--alignment", alignment,
        "-t", str(threads),
        *extra_args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-20:])
        raise RuntimeError(
            f"suppOrth exited with code {proc.returncode}; its output in {outdir} is "
            f"incomplete and was not read.\n{tail}"
        )
    return hits_from_supporth(outdir / "suppOrth.tsv", organism_id, **reader_kwargs)
