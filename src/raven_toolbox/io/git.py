"""Export a model into a Standard-GEM versioned-repository layout.

Writes the model in several formats into the Standard-GEM folder structure (a
``model/`` directory with one subfolder per format), ready to commit to a
Git-maintained model repository (Metabolic Atlas, Human-GEM and yeast-GEM style),
and a ``dependencies.txt`` that records tool versions.

The module calls the existing writers ``write_yaml_model``, cobra's ``write_sbml_model``,
``write_matlab_model`` and ``export_to_excel``, and adds a single-file reaction table (txt).
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
import numpy as np
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

# RAVEN's exportModel writes each character of the model id outside [0-9_a-zA-Z] as
# __<character code>__ and importModel decodes it again. cobra passes the model id to libSBML
# unchanged, and libSBML writes no id attribute for an id that is not a valid SId.
_SBML_ID_UNSAFE = re.compile(r"[^0-9_a-zA-Z]")

# The SBO term RAVEN's exportModel gives every gene product that has no SBO term of its own.
_SBO_GENE_PRODUCT = "SBO:0000243"


def _sbml_id(identifier: str) -> str:
    """``identifier`` with each character outside [0-9_a-zA-Z] written as ``__<code>__``."""
    return _SBML_ID_UNSAFE.sub(lambda match: f"__{ord(match.group())}__", identifier)


def _for_sbml(model: cobra.Model) -> cobra.Model:
    """A copy of ``model`` shaped so RAVEN's importModel finds everything.

    Two kinds of information would be missing from the SBML file otherwise:

    * **subSystems.** RAVEN reads them from the SBML groups package, which
      cobra writes from ``model.groups``. A model read with read_yaml_model
      carries its subsystems on the reaction, so without groups the
      SBML has no subsystem information.
    * **rxnNotes, rxnReferences and rxnConfidenceScores.** The file contains them under
      cobra's key names, while RAVEN parses other labels.

    The model id is escaped as in RAVEN's exportModel, each gene gets the SBO term
    ``SBO:0000243`` unless it has one of its own, and the protein of a gene leaves the gene's
    notes because :func:`_write_sbml` writes it as ``fbc:name``.

    The model passed in is not changed.
    """
    out = model.copy()

    if out.id:
        out.id = _sbml_id(out.id)

    for gene in out.genes:
        # model.copy() shares each gene's notes and annotation dicts with the original, so the
        # copy gets new dicts instead of changes in place.
        gene.annotation = {"sbo": _SBO_GENE_PRODUCT, **gene.annotation}
        gene.notes = {key: value for key, value in (gene.notes or {}).items() if key != "protein"}

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


def _sort_sbml_compartments(sbml_model: libsbml.Model) -> None:
    """Order the compartments of ``sbml_model`` by id.

    cobra writes them in the order in which the metabolites first mention them.
    """
    compartments = sorted(
        (compartment.clone() for compartment in sbml_model.getListOfCompartments()),
        key=lambda compartment: compartment.getId(),
    )
    while sbml_model.getNumCompartments():
        sbml_model.removeCompartment(0)
    for compartment in compartments:
        sbml_model.addCompartment(compartment)


def _write_sbml(model: cobra.Model, path: Path) -> None:
    """Write ``model`` as SBML in the shape of RAVEN's exportModel.

    Each gene product gets the protein as fbc:name, as RAVEN writes it, and no fbc:name
    when the gene has no protein; RAVEN's importModel reads proteins from fbc:name. The
    gene short name, or the gene id when there is none, goes into fbc:label, from which
    importModel reads geneShortNames; cobra reads gene names from fbc:name, so the label
    does not change what cobra reads back. The compartments are ordered by id, as in
    RAVEN's sortIdentifiers.
    """
    cobra.io.write_sbml_model(_for_sbml(model), str(path))
    doc = libsbml.readSBMLFromFile(str(path))
    sbml_model = doc.getModel()
    products = sbml_model.getPlugin("fbc").getListOfGeneProducts()
    if len(products) != len(model.genes):
        raise RuntimeError(
            f"the SBML has {len(products)} gene products for {len(model.genes)} genes"
        )
    # cobra writes the gene products in the order of model.genes.
    for gene, product in zip(model.genes, products, strict=True):
        protein = (gene.notes or {}).get("protein")
        product.setLabel(gene.name or gene.id)
        if protein:
            product.setName(str(protein))
        else:
            product.unsetName()
    _sort_sbml_compartments(sbml_model)
    libsbml.writeSBMLToFile(doc, str(path))


def _shortest(value: float) -> str:
    """The shortest text that reads back as ``value`` (1/3 is 0.3333333333333333).

    The rule is the same as in RAVEN's exportForGit. A bound is not rounded in the file.
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


def _write_dependencies(path: Path, model: cobra.Model, *, model_version: bool = False) -> None:
    """Versions of everything that writes the files, then optionally the model's version and date.

    The order is fixed and nothing depends on the time of the export, so the same model
    exported twice gives the same file. With ``model_version`` false the file lists the
    toolchain only, as RAVEN's exportForGit does.
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
    if model_version:
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
    model_version_in_dependencies: bool = False,
) -> Path:
    """Write ``model`` into a Standard-GEM repository layout.

    Parameters
    ----------
    model : cobra.Model
        Model to export. A sorted copy is written; the model passed in is not changed.
    path : str or Path
        Directory to populate.
    prefix : str
        Base filename for every format (default ``"model"``).
    formats : Iterable of str
        Which formats to write; any of ``"yml"``, ``"xml"``, ``"mat"``,
        ``"xlsx"``, ``"txt"`` (default: all five, as RAVEN's ``exportForGit``).
    main_branch_flag : bool
        A ``bool``. If True, raise ``RuntimeError`` when raven_toolbox runs from a git
        checkout that is not on ``main`` or ``master`` (default False), as RAVEN's
        ``mainBranchFlag``. The check
        passes when the package is not in a raven_toolbox checkout (for example a pip
        installation) and when git is missing or refuses the checkout (``OSError`` or a
        failing ``git`` call such as ``safe.directory``). GECKO is not checked.
    sub_dirs : bool
        A ``bool``. If True (default), write ``model/<fmt>/<prefix>.<fmt>`` (standard-GEM
        layout); otherwise all files go directly in ``path``.
    varname : str or None
        Variable name for the MATLAB (``.mat``) struct. ``None`` (default) lets
        cobra use its own default (the model id); set it when a repository pins a
        specific name (for example Human-GEM's ``humanGEM``).
    version : str or None
        Version stamped into the exported copy's metadata (see
        :func:`raven_toolbox.io.set_model_metadata`); the model passed in is not changed.
        When not given, the model's own version is written.
    date : str or datetime.date or None
        Date stamped into the exported copy's metadata in the same way. When not given, the
        model's own date is written.
    model_version_in_dependencies : bool
        A ``bool``. If True, ``dependencies.txt`` also lists ``model_version`` and
        ``model_date`` from the model metadata. The default (False) lists the toolchain
        only, as RAVEN's ``exportForGit`` does.

    Returns
    -------
    pathlib.Path
        The root directory written to.

    Raises
    ------
    TypeError
        If ``main_branch_flag``, ``sub_dirs`` or ``model_version_in_dependencies`` is not a
        ``bool`` or ``numpy.bool_``.
    """
    for name, flag in (
        ("main_branch_flag", main_branch_flag),
        ("sub_dirs", sub_dirs),
        ("model_version_in_dependencies", model_version_in_dependencies),
    ):
        if not isinstance(flag, (bool, np.bool_)):
            raise TypeError(
                f"{name} must be a bool or numpy.bool_, got {type(flag).__module__}.{type(flag).__qualname__}"
            )
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

    _write_dependencies(root / "dependencies.txt", model, model_version=model_version_in_dependencies)

    return root
