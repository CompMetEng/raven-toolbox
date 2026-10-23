"""Export a model into a Standard-GEM versioned-repository layout.

Writes the model in several formats into the Standard-GEM folder structure (a
``model/`` directory with one subfolder per format), ready to commit to a
Git-maintained model repository (Metabolic Atlas / Human-GEM / yeast-GEM style),
plus a ``dependencies.txt`` recording tool versions.

Thin orchestration over the writers raven_toolbox already exposes: ``write_yaml_model``,
cobra's ``write_sbml_model``, ``write_matlab_model``, ``export_to_excel``, plus a
single-file reaction table (txt).
"""
from __future__ import annotations

import datetime
import importlib.metadata as _md
import math
import platform
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

import cobra
import libsbml
from cobra.core import Group

from raven_toolbox.io.excel import _equation, export_to_excel
from raven_toolbox.io.mat import write_matlab_model
from raven_toolbox.io.metadata import get_model_metadata, set_model_metadata
from raven_toolbox.io.yaml import write_yaml_model
from raven_toolbox.utils.sort import sort_identifiers

_ALL_FORMATS = ("yml", "xml", "mat", "xlsx", "txt")


def _version(package: str) -> str:
    try:
        return _md.version(package)
    except _md.PackageNotFoundError:
        return "unknown"


# RAVEN's importModel reads these reaction fields out of the SBML <notes>
# block, keyed by its own labels (see parseNote in importModel.m). cobra
# writes a note as "<p>key: value</p>" straight from the notes dict, so the
# keys are renamed to RAVEN's labels for the SBML export only.
_SBML_NOTE_LABELS = {
    "note": "NOTES",
    "references": "AUTHORS",
    "confidence_score": "Confidence Level",
}


def _for_sbml(model: cobra.Model) -> cobra.Model:
    """A copy of ``model`` shaped so RAVEN's importModel finds everything.

    Two things are lost otherwise, both silently:

    * **subSystems.** RAVEN reads them from the SBML groups package, which
      cobra writes from ``model.groups``. A model read with read_yaml_model
      carries its subsystems on the reaction instead, so without groups the
      SBML has no subsystem information at all.
    * **rxnNotes / rxnReferences / rxnConfidenceScores.** Present in the
      file, but under cobra's key names rather than the labels RAVEN parses.

    The model passed in is not changed.
    """
    out = model.copy()

    for rxn in out.reactions:
        notes = rxn.notes or {}
        renamed = {_SBML_NOTE_LABELS.get(key, key): value for key, value in notes.items()}
        if renamed != notes:
            rxn.notes = renamed

    # RAVEN recovers model.annotation.taxonomy from the identifiers.org URL
    # cobra emits for a model-level annotation entry.
    meta = get_model_metadata(model)
    taxonomy = meta.get("taxonomy")
    if taxonomy and "taxonomy" not in (out.annotation or {}):
        annotation = dict(out.annotation or {})
        annotation["taxonomy"] = str(taxonomy).replace("taxonomy/", "")
        out.annotation = annotation

    # The model's own note goes in as its own notes entry, so cobra writes it
    # as "<p>note: ...</p>". RAVEN's exportModel instead puts a bare note
    # inside a <body> that cobra never emits, which is why importModel has to
    # look for both shapes.
    # The note, version and date go in as their own notes entries, so cobra
    # writes each as "<p>key: value</p>". RAVEN's exportModel instead puts a
    # bare note inside a <body> that cobra never emits, which is why
    # importModel has to look for both shapes; the version and date reached
    # the file only inside the metaData repr dropped below.
    stamped = dict(out.notes or {})
    for key in ("note", "version", "date"):
        if meta.get(key) and key not in stamped:
            stamped[key] = str(meta[key])
    out.notes = stamped

    # cobra writes a notes entry as "<p>key: value</p>" with str(value), so a
    # non-string value is published as its Python repr -- read_yaml_model parks
    # the parsed metaData block in notes["metaData"], which came out as
    # "metaData: {&apos;id&apos;: &apos;HumanGEM&apos;, ...}" in the file. Its
    # content already goes in as the model id, name, annotation and note, so
    # only string-valued entries are written.
    out.notes = {
        key: value
        for key, value in (out.notes or {}).items()
        if isinstance(value, str)
    }

    if not out.groups:
        members: dict[str, list] = {}
        for rxn in out.reactions:
            subsystem = rxn.subsystem
            if not subsystem:
                continue
            names = subsystem if isinstance(subsystem, (list, tuple)) else [subsystem]
            for name in names:
                name = str(name).strip()
                if name:
                    members.setdefault(name, []).append(rxn)
        groups = []
        for index, (name, rxns) in enumerate(members.items(), start=1):
            group = Group(id=f"group{index}", name=name, kind="partonomy")
            group.add_members(rxns)
            groups.append(group)
        if groups:
            out.add_groups(groups)

    return out


def _write_sbml(model: cobra.Model, path: Path) -> None:
    """Write ``model`` as SBML with each gene product's fbc:label set to the gene name.

    cobra's writer sets fbc:label to the gene product id (``G_<id>``), while RAVEN's
    exportModel writes the gene short name there, and RAVEN's importModel reads
    geneShortNames from fbc:label before fbc:name. cobra writes the gene name as
    fbc:name, or the id when the gene has no name, so copying fbc:name into fbc:label
    keeps the id as label for unnamed genes. cobra reads gene ids from fbc:id and
    names from fbc:name, so the label does not change what cobra reads back.
    """
    cobra.io.write_sbml_model(_for_sbml(model), str(path))
    doc = libsbml.readSBMLFromFile(str(path))
    for gene_product in doc.getModel().getPlugin("fbc").getListOfGeneProducts():
        gene_product.setLabel(gene_product.getName())
    libsbml.writeSBMLToFile(doc, str(path))


def _shortest(value: float) -> str:
    """The shortest text that reads back as exactly ``value`` (1/3 is 0.3333333333333333).

    The same rule as RAVEN's exportForGit, so a bound is never rounded in the file.
    """
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if math.isnan(value):
        return "NaN"
    if value == math.floor(value) and abs(value) < 1e15:
        return str(int(value))
    for digits in range(1, 18):
        text = f"{value:.{digits}g}"
        if float(text) == value:
            return text
    return repr(value)


def _write_txt(model: cobra.Model, path: Path) -> None:
    """Single-file, human-readable reaction table (RAVEN exportForGit txt).

    Bounds and objective coefficients are written with the digits needed to read them back
    unchanged, and no field is padded.
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Rxn id\tFormula\tGene-reaction association\tLB\tUB\tObjective\n")
        for r in model.reactions:
            fh.write(
                f"{r.id}\t{_equation(r).strip()}\t{r.gene_reaction_rule}\t"
                f"{_shortest(r.lower_bound)}\t{_shortest(r.upper_bound)}\t"
                f"{_shortest(r.objective_coefficient)}\n"
            )


def _write_dependencies(path: Path, model: cobra.Model) -> None:
    """Versions of everything that writes the files, then the model's own version and date.

    The order is fixed and nothing depends on the time of the export, so the same model
    exported twice gives the same file.
    """
    lines = [
        ("python", platform.python_version()),
        ("cobra", _version("cobra")),
        ("libSBML", libsbml.getLibSBMLDottedVersion()),
        ("ruamel.yaml", _version("ruamel.yaml")),
        ("scipy", _version("scipy")),
        ("openpyxl", _version("openpyxl")),
        ("raven_toolbox", _version("raven_toolbox")),
    ]
    meta = get_model_metadata(model)
    for key in ("version", "date"):
        if meta.get(key) is not None:
            lines.append((f"model_{key}", str(meta[key])))
    with open(path, "w", encoding="utf-8") as fh:
        for name, value in lines:
            fh.write(f"{name}\t{value}\n")


def _check_main_branch() -> None:
    """Raise when raven_toolbox is a git checkout that is not on ``main`` or ``master``.

    The checkout is the repository whose top level holds this package as
    ``src/raven_toolbox`` next to a ``pyproject.toml`` named ``raven-toolbox``. A package
    outside such a checkout, including one installed in the ignored virtual environment of
    an unrelated repository, passes. GECKO is not checked.
    """
    package = Path(__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(package), *args], capture_output=True, text=True, check=True,
        ).stdout.strip()

    try:
        top = Path(git("rev-parse", "--show-toplevel")).resolve()
        if (top / "src" / "raven_toolbox").resolve() != package:
            return
        pyproject = (top / "pyproject.toml").read_text(encoding="utf-8")
        if not re.search(r'^name\s*=\s*"raven-toolbox"', pyproject, re.MULTILINE):
            return
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return
    if branch not in ("main", "master"):
        raise RuntimeError(
            f"raven_toolbox is on branch '{branch}', not main or master. Check out the main "
            "branch before submitting a model to Git."
        )


def export_for_git(
    model: cobra.Model,
    path: str | Path = ".",
    *,
    prefix: str = "model",
    formats: Iterable[str] = _ALL_FORMATS,
    main_branch_flag: bool = False,
    sub_dirs: bool = True,
    varname: str | None = None,
    version: str | None = None,
    date: str | datetime.date | None = None,
) -> Path:
    """Write ``model`` into a Standard-GEM repository layout.

    Parameters
    ----------
    path
        Directory to populate.
    prefix
        Base filename for every format (default ``"model"``).
    formats
        Which formats to write; any of ``"yml"``, ``"xml"``, ``"mat"``,
        ``"xlsx"``, ``"txt"`` (default: all five, as RAVEN's ``exportForGit``).
    main_branch_flag
        A ``bool``. If True, raise ``RuntimeError`` when raven_toolbox runs from a git checkout that is not
        on ``main`` or ``master`` (default False), as RAVEN's ``mainBranchFlag``. The check
        passes when the package is not in a raven_toolbox checkout (for example a pip
        installation) and when git is missing or refuses the checkout (``OSError`` or a
        failing ``git`` call such as ``safe.directory``). GECKO is not checked.
    sub_dirs
        A ``bool``. If True (default), write ``model/<fmt>/<prefix>.<fmt>`` (standard-GEM
        layout); otherwise all files go directly in ``path``.
    varname
        Variable name for the MATLAB (``.mat``) struct. ``None`` (default) lets
        cobra use its own default (the model id); set it when a repository pins a
        specific name (e.g. Human-GEM's ``humanGEM``).
    version, date
        Stamped into the exported copy's metadata (see
        :func:`raven_toolbox.io.set_model_metadata`); the model passed in is not changed.
        Left out, the model's own version and date are written.

    Returns
    -------
    pathlib.Path
        The root directory written to.

    Raises
    ------
    TypeError
        If ``main_branch_flag`` or ``sub_dirs`` is not a ``bool``.
    """
    for name, flag in (("main_branch_flag", main_branch_flag), ("sub_dirs", sub_dirs)):
        if not isinstance(flag, bool):
            raise TypeError(f"{name} must be a bool, got {type(flag).__name__}")
    if main_branch_flag:
        _check_main_branch()
    formats = list(formats)
    unknown = set(formats) - set(_ALL_FORMATS)
    if unknown:
        raise ValueError(f"Unknown format(s): {sorted(unknown)}; allowed: {_ALL_FORMATS}")

    # Sort a copy so the caller's model is untouched.
    model = sort_identifiers(model.copy())
    if version is not None or date is not None:
        set_model_metadata(model, version=version, date=date)

    root = Path(path) / "model" if sub_dirs else Path(path)
    root.mkdir(parents=True, exist_ok=True)

    def target(fmt: str) -> Path:
        folder = root / fmt if sub_dirs else root
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{prefix}.{fmt}"

    if "yml" in formats:
        write_yaml_model(model, target("yml"))
    if "xml" in formats:
        _write_sbml(model, target("xml"))
    if "mat" in formats:
        write_matlab_model(model, target("mat"), varname=varname)
    if "xlsx" in formats:
        export_to_excel(model, target("xlsx"))
    if "txt" in formats:
        _write_txt(model, target("txt"))

    _write_dependencies(root / "dependencies.txt", model)

    return root
