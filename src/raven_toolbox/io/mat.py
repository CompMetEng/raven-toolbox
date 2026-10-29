"""RAVEN-format MATLAB (``.mat``) model I/O.

cobra's ``save_matlab_model``/``load_matlab_model`` target the COBRA Toolbox
structure, which differs from RAVEN's in three ways that matter to a RAVEN
user:

* **Field names.** RAVEN keeps the model identifier in ``id`` and ``name``;
  COBRA uses ``description`` and ``modelName``. RAVEN's ``eccodes`` and
  ``geneShortNames`` are COBRA's ``rxnECNumbers`` and ``geneNames``.
* **Fields COBRA has no slot for.** ``version``, ``date``, the ``annotation``
  struct, ``metFrom``/``rxnFrom``, ``rxnNotes``/``rxnReferences``/
  ``rxnConfidenceScores``, ``inchis``, ``*DeltaG`` and the ``*Miriams``
  structs (COBRA spreads cross-references over one flat field per
  namespace instead).
* **Numeric classes.** Every numeric field in a RAVEN model is a double.
  cobra writes ``rev`` as a platform-width integer, which makes RAVEN's
  ``mergeLinear`` fail on ``S(...).' > 0 | rev`` — MATLAB rejects the
  sparse-logical/integer mix with "Sparse integer array arithmetic
  operations are not supported".

The layout written here follows what RAVEN's own MATLAB exporter produces,
taken from the released yeast-GEM and Human-GEM ``.mat`` files rather than
from a field-table transcribed out of the (GPL) RAVEN repository.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import cobra
import numpy as np
import scipy.io
import scipy.sparse

from raven_toolbox.io.metadata import get_model_metadata
from raven_toolbox.io.yaml import _default_bounds
from raven_toolbox.utils.gpr import set_gene_rule

# Field order RAVEN's exporter emits. Writing in this order means a model
# round-tripped through here keeps the same field order as one written by
# MATLAB, so `isequal` on the field names holds and diffs stay readable.
_FIELD_ORDER = (
    "id", "name", "description", "version", "date", "annotation",
    "rxns", "rxnNames", "mets", "metNames", "S", "lb", "ub", "rev", "c", "b",
    "genes", "grRules", "rxnGeneMat", "subSystems", "eccodes", "rxnMiriams",
    "rxnDeltaG", "rxnNotes", "rxnReferences", "rxnConfidenceScores",
    "metComps", "metSmiles", "inchis", "metFormulas", "metMiriams",
    "metDeltaG", "metCharges", "comps", "compNames", "geneMiriams",
    "geneShortNames", "metFrom", "rxnFrom",
)

# Model-level metaData keys that live inside the `annotation` struct, in the
# order RAVEN writes them. `id`, `name`, `version` and `date` sit at the top
# level of the struct instead and are handled separately.
_ANNOTATION_KEYS = (
    "givenName", "familyName", "authors", "email", "organization",
    "taxonomy", "note", "sourceUrl", "defaultLB", "defaultUB",
)

# RAVEN-only per-entry data that read_yaml_model parks in `notes`, mapped to
# the RAVEN field carrying it. Kept in step with _MET_FIELDS/_RXN_FIELDS in
# raven_toolbox.io.yaml, which define the same mapping for the YAML format.
_MET_NOTE_FIELDS = (("inchis", "inchis"), ("metFrom", "metFrom"))
_RXN_NOTE_FIELDS = (
    ("rxnNotes", "note"),
    ("rxnReferences", "references"),
    ("rxnFrom", "rxnFrom"),
)

# Annotation keys with a dedicated RAVEN field, so they are written there
# rather than into the generic *Miriams struct.
_EC_KEY = "ec-code"
_SMILES_KEY = "smiles"


def _cell(values) -> np.ndarray:
    """An n-by-1 MATLAB cell array of char from a sequence of strings."""
    values = list(values)
    out = np.empty((len(values), 1), dtype=object)
    for i, value in enumerate(values):
        out[i, 0] = "" if value is None else str(value)
    return out


def _cell_of_cells(values) -> np.ndarray:
    """An n-by-1 cell array whose every element is itself a cell of char.

    The shape RAVEN uses for ``subSystems``: one reaction's subsystems are a
    cell array, even when there is only one of them.
    """
    values = list(values)
    out = np.empty((len(values), 1), dtype=object)
    for i, entry in enumerate(values):
        out[i, 0] = _cell(entry)
    return out


def _col(values, *, default: float = 0.0) -> np.ndarray:
    """An n-by-1 double column vector.

    Every numeric field in a RAVEN model is a double — see the module
    docstring for what an integer column costs.
    """
    return np.array(
        [[default if value is None else float(value)] for value in values],
        dtype=np.float64,
    )


def _index_col(values) -> np.ndarray:
    """An n-by-1 integer column for a 1-based index field such as ``metComps``.

    RAVEN's field definition calls this a double, and MATLAB's own ``save``
    does store it as one — but narrowed on disk to the smallest integer type
    that holds it losslessly. ``scipy.io.savemat`` has no way to write that
    combination (a double class with integer storage), and a reader that
    subtracts 1 and indexes with the result needs an integer, so the class
    itself is an integer here. The type is signed: an unsigned column would
    wrap 0 (a metabolite with no compartment) round to its maximum on that
    subtraction instead of going negative.
    """
    values = [int(value) for value in values]
    widest = max((abs(value) for value in values), default=0)
    dtype = np.int8 if widest < 128 else (np.int16 if widest < 32768 else np.int32)
    return np.array([[value] for value in values], dtype=dtype)


def _empty() -> np.ndarray:
    """MATLAB's ``[]``, which RAVEN uses for an absent per-entry value."""
    return np.zeros((0, 0))


def _miriams(annotations) -> np.ndarray:
    """An n-by-1 cell of RAVEN MIRIAM structs from cobra annotation dicts.

    Each non-empty entry is a 1-by-1 struct with ``name`` and ``value`` cell
    columns holding one row per cross-reference. A namespace with several
    values contributes one row each, repeating the name — the shape RAVEN
    itself writes, and what makes the pairs order-preserving. An entry with
    no cross-references is ``[]``.
    """
    annotations = list(annotations)
    out = np.empty((len(annotations), 1), dtype=object)
    for i, annotation in enumerate(annotations):
        names: list[str] = []
        values: list[str] = []
        for key, value in (annotation or {}).items():
            if key in (_EC_KEY, _SMILES_KEY):
                continue
            for item in value if isinstance(value, (list, tuple)) else [value]:
                names.append(str(key))
                values.append(str(item))
        out[i, 0] = (
            {"name": _cell(names), "value": _cell(values)} if names else _empty()
        )
    return out


def _annotation_values(annotation, key) -> list[str]:
    """The values stored under ``key``, as a list (cobra allows either shape)."""
    value = (annotation or {}).get(key)
    if value is None:
        return []
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]


def _notes_column(objects, note_key) -> list[str]:
    """One note value per object, '' where the object carries none."""
    return [str((obj.notes or {}).get(note_key, "") or "") for obj in objects]


def _any_set(values) -> bool:
    """True when at least one entry carries content.

    RAVEN omits an optional field entirely rather than writing a column of
    empties, so the writer only emits one when there is something in it.
    """
    return any(value not in ("", None) for value in values)


def _deltaG(objects) -> np.ndarray | None:
    """A ``*DeltaG`` column, or None when no object carries one.

    Unknown values are NaN, as RAVEN's field definition requires — distinct
    from a genuine 0 kJ/mol.
    """
    raw = [(obj.notes or {}).get("deltaG") for obj in objects]
    if not any(value not in (None, "") for value in raw):
        return None
    return _col(
        [np.nan if value in (None, "") else value for value in raw], default=np.nan
    )


def _subsystems(reactions) -> list[list[str]]:
    """Each reaction's subsystems as a list of strings.

    read_yaml_model leaves a subsystem as the list the YAML carried, while a
    model built in cobra or read from SBML has a single string; both shapes
    reach here.
    """
    out: list[list[str]] = []
    for rxn in reactions:
        subsystem = rxn.subsystem
        if not subsystem:
            out.append([])
        elif isinstance(subsystem, (list, tuple)):
            out.append([str(s) for s in subsystem])
        else:
            out.append([str(subsystem)])
    return out


def _model_struct(model: cobra.Model) -> dict:
    """The RAVEN model structure for ``model``, as a savemat-ready dict."""
    rxns = list(model.reactions)
    mets = list(model.metabolites)
    genes = list(model.genes)
    comps = list(model.compartments)
    meta = get_model_metadata(model)

    struct: dict = {}
    struct["id"] = model.id or "blankID"
    struct["name"] = model.name or "blankName"
    struct["description"] = str(meta.get("description", "") or "")
    version = meta.get("version")
    if version is not None:
        struct["version"] = str(version)
    struct["date"] = str(meta.get("date") or datetime.date.today().isoformat())

    annotation = {}
    for key in _ANNOTATION_KEYS:
        value = meta.get(key)
        if value in (None, ""):
            continue
        annotation[key] = (
            float(value) if key in ("defaultLB", "defaultUB") else str(value)
        )
    # Recomputed from the model's current bounds rather than echoed from a
    # stored value, the way write_yaml_model derives them, so a model whose
    # bounds changed after loading still gets correct defaults.
    bounds = _default_bounds(model)
    if bounds is not None:
        annotation["defaultLB"], annotation["defaultUB"] = (
            float(bounds[0]),
            float(bounds[1]),
        )
    if annotation:
        struct["annotation"] = annotation

    struct["rxns"] = _cell(rxn.id for rxn in rxns)
    struct["rxnNames"] = _cell(rxn.name for rxn in rxns)
    struct["mets"] = _cell(met.id for met in mets)
    struct["metNames"] = _cell(met.name for met in mets)
    met_index = {met.id: i for i, met in enumerate(mets)}
    stoich = scipy.sparse.lil_matrix((len(mets), len(rxns)), dtype=np.float64)
    for j, rxn in enumerate(rxns):
        for met, coefficient in rxn.metabolites.items():
            stoich[met_index[met.id], j] = float(coefficient)
    struct["S"] = stoich.tocsc()
    struct["lb"] = _col(rxn.lower_bound for rxn in rxns)
    struct["ub"] = _col(rxn.upper_bound for rxn in rxns)
    struct["rev"] = _col(1 if rxn.reversibility else 0 for rxn in rxns)
    struct["c"] = _col(rxn.objective_coefficient for rxn in rxns)
    struct["b"] = _col(0.0 for _ in mets)

    struct["genes"] = _cell(gene.id for gene in genes)
    struct["grRules"] = _cell(rxn.gene_reaction_rule for rxn in rxns)
    gene_index = {gene.id: i for i, gene in enumerate(genes)}
    rxn_gene = scipy.sparse.lil_matrix((len(rxns), len(genes)), dtype=np.float64)
    for i, rxn in enumerate(rxns):
        for gene in rxn.genes:
            rxn_gene[i, gene_index[gene.id]] = 1.0
    struct["rxnGeneMat"] = rxn_gene.tocsc()

    struct["subSystems"] = _cell_of_cells(_subsystems(rxns))

    eccodes = [";".join(_annotation_values(rxn.annotation, _EC_KEY)) for rxn in rxns]
    if _any_set(eccodes):
        struct["eccodes"] = _cell(eccodes)
    rxn_miriams = _miriams(rxn.annotation for rxn in rxns)
    if any(np.size(entry) for entry in rxn_miriams[:, 0]):
        struct["rxnMiriams"] = rxn_miriams

    rxn_delta_g = _deltaG(rxns)
    if rxn_delta_g is not None:
        struct["rxnDeltaG"] = rxn_delta_g
    for field, note_key in _RXN_NOTE_FIELDS:
        values = _notes_column(rxns, note_key)
        if _any_set(values):
            struct[field] = _cell(values)
    scores = [(rxn.notes or {}).get("confidence_score") for rxn in rxns]
    if any(score not in (None, "") for score in scores):
        struct["rxnConfidenceScores"] = _col(scores)

    comp_index = {comp: i + 1 for i, comp in enumerate(comps)}
    struct["metComps"] = _index_col(
        comp_index.get(met.compartment, 0) for met in mets
    )
    smiles = [
        ";".join(_annotation_values(met.annotation, _SMILES_KEY)) for met in mets
    ]
    if _any_set(smiles):
        struct["metSmiles"] = _cell(smiles)
    for field, note_key in _MET_NOTE_FIELDS:
        values = _notes_column(mets, note_key)
        if _any_set(values):
            struct[field] = _cell(values)
    struct["metFormulas"] = _cell(met.formula or "" for met in mets)
    met_miriams = _miriams(met.annotation for met in mets)
    if any(np.size(entry) for entry in met_miriams[:, 0]):
        struct["metMiriams"] = met_miriams
    met_delta_g = _deltaG(mets)
    if met_delta_g is not None:
        struct["metDeltaG"] = met_delta_g
    struct["metCharges"] = _col(met.charge for met in mets)

    struct["comps"] = _cell(comps)
    struct["compNames"] = _cell(model.compartments[comp] for comp in comps)

    gene_miriams = _miriams(gene.annotation for gene in genes)
    if any(np.size(entry) for entry in gene_miriams[:, 0]):
        struct["geneMiriams"] = gene_miriams
    short_names = [gene.name or "" for gene in genes]
    if _any_set(short_names):
        struct["geneShortNames"] = _cell(short_names)

    return {field: struct[field] for field in _FIELD_ORDER if field in struct}


def write_matlab_model(
    model: cobra.Model,
    path: str | Path,
    *,
    varname: str | None = None,
) -> Path:
    """Write ``model`` as a RAVEN model structure in a MATLAB ``.mat`` file.

    Parameters
    ----------
    model
        The model to write.
    path
        Destination file.
    varname
        Name of the struct inside the file. Defaults to the model's id; set
        it when a repository pins a specific name (Human-GEM's ``humanGEM``).

    Returns
    -------
    pathlib.Path
        The file written.
    """
    path = Path(path)
    name = varname or model.id or "model"
    scipy.io.savemat(
        str(path),
        {name: _model_struct(model)},
        format="5",
        do_compression=True,
        oned_as="column",
    )
    return path


def _as_str(value) -> str:
    """A plain string from MATLAB's nested char wrappers.

    scipy hands a cell's char back as an array inside an array; unwrap until
    something that is not an array is left. An empty array is an empty
    string, which is how RAVEN spells "no value" for a cell field.
    """
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.flat[0]
    return "" if value is None else str(value)


def _as_list(value) -> list[str]:
    """A column cell array as a list of strings, in file order."""
    if value is None:
        return []
    array = np.asarray(value)
    if array.size == 0:
        return []
    return [_as_str(item) for item in array.ravel(order="F")]


def _as_floats(value) -> list[float]:
    """A numeric column as a list of floats."""
    if value is None:
        return []
    array = np.asarray(value, dtype=np.float64)
    return [float(x) for x in array.ravel(order="F")]


def _field(struct, name):
    """A model-struct field, or None when the file does not carry it."""
    if struct.dtype.names is None or name not in struct.dtype.names:
        return None
    return struct[name][0, 0]


def _column(struct, name, length) -> list[str]:
    """A cell field padded to ``length``, '' where the file has nothing."""
    values = _as_list(_field(struct, name))
    return values + [""] * (length - len(values))


def _annotation_from_miriams(entry) -> dict:
    """A cobra annotation dict from one RAVEN MIRIAM struct.

    Every namespace maps to a list, including a namespace carrying a single
    value, so a model read from a ``.mat`` presents its cross-references the
    same way :func:`read_yaml_model` does. A namespace appearing more than
    once (several KEGG pathways, say) keeps its values in file order.
    """
    if entry is None or np.size(entry) == 0:
        return {}
    names = _as_list(entry["name"][0, 0] if entry.dtype.names else None)
    values = _as_list(entry["value"][0, 0] if entry.dtype.names else None)
    annotation: dict = {}
    for name, value in zip(names, values, strict=False):
        annotation.setdefault(name, []).append(value)
    return annotation


def _miriam_column(struct, name, length) -> list[dict]:
    """Per-entry annotation dicts from a ``*Miriams`` field."""
    field = _field(struct, name)
    if field is None:
        return [{} for _ in range(length)]
    array = np.asarray(field).reshape(-1, order="F")
    return [
        _annotation_from_miriams(array[i]) if i < array.size else {}
        for i in range(length)
    ]


def _set_note(obj, key, value) -> None:
    """Store ``value`` in ``obj.notes[key]`` when there is something to store."""
    if value in ("", None):
        return
    notes = dict(obj.notes or {})
    notes[key] = value
    obj.notes = notes



# Fields that only a COBRA Toolbox structure carries. 'rules' is decisive --
# cobra writes it and RAVEN never does -- and the rest catch a file written
# without it. A RAVEN structure always has 'id', which cobra's writer never
# emits (it puts the identifier in 'description').
_COBRA_MARKERS = frozenset(
    {"rules", "description", "modelName", "rxnECNumbers", "geneNames", "osenseStr"}
)


def _is_cobra_struct(names) -> bool:
    """Whether the struct is a COBRA Toolbox model rather than a RAVEN one."""
    names = set(names or ())
    if "rules" in names:
        return True
    return "id" not in names and bool(names & _COBRA_MARKERS)


def _listify_annotations(model: cobra.Model) -> None:
    """Make every annotation value a list, in place.

    cobra's MATLAB reader leaves a single cross-reference as a bare string;
    :func:`read_yaml_model` and the RAVEN branch of this reader both give a
    list, so a model read from any of them presents annotations the same way.
    """
    for entity in (*model.reactions, *model.metabolites, *model.genes):
        annotation = entity.annotation
        if not annotation:
            continue
        entity.annotation = {
            key: value if isinstance(value, list) else [value]
            for key, value in annotation.items()
        }


def _read_cobra_struct(path: Path) -> cobra.Model:
    """Read a COBRA Toolbox .mat through cobra, then normalise it.

    The COBRA structure keeps the same data under its own field names
    (``description``/``modelName`` for the identifier, ``rxnECNumbers`` for EC
    codes, ``geneNames`` for gene names, one flat field per cross-reference
    namespace), so cobra's own reader already knows every mapping; this wraps
    it rather than repeating the table. Human-GEM 2.0.1 and 2.1.0 shipped such
    a file, so reading one is the way back from those releases.
    """
    model = cobra.io.load_matlab_model(str(path))
    # cobra hands these back as numpy strings, which surprise anything that
    # serialises the model later.
    model.id = str(model.id)
    model.name = str(model.name or "")
    _listify_annotations(model)
    return model

def read_matlab_model(path: str | Path, *, varname: str | None = None) -> cobra.Model:
    """Read a model structure from a MATLAB ``.mat`` file.

    Reads a RAVEN structure natively. A COBRA Toolbox structure is recognised
    and handed to cobra's own reader instead, so a file written by
    ``save_matlab_model`` — Human-GEM 2.0.1 and 2.1.0 shipped one — is read
    rather than silently coming back without its identifier, EC codes, gene
    names and cross-references. A struct that is neither raises.

    The inverse of :func:`write_matlab_model`: RAVEN-only fields land where
    :func:`read_yaml_model` puts the same data, so a model read from either
    format looks the same to the rest of raven-toolbox — cross-references in
    ``annotation``, ``metFrom``/``rxnFrom``/notes/references/deltaG and
    ``inchis`` in ``notes``, and the model's own metadata in
    ``notes["metaData"]`` and ``notes["version"]``.

    Parameters
    ----------
    path
        File to read.
    varname
        Name of the struct to read. Defaults to the file's only struct
        variable; required when the file holds more than one.

    Returns
    -------
    cobra.Model
    """
    path = Path(path)
    raw = scipy.io.loadmat(str(path), mat_dtype=True)
    candidates = [key for key in raw if not key.startswith("__")]
    if varname is not None:
        if varname not in candidates:
            raise KeyError(f"{path} has no variable {varname!r}; found {candidates}")
        name = varname
    elif len(candidates) == 1:
        name = candidates[0]
    else:
        raise KeyError(f"{path} holds {candidates}; pass varname to choose one")
    struct = raw[name]
    names = struct.dtype.names
    if _is_cobra_struct(names):
        return _read_cobra_struct(path)
    if "rxns" not in (names or ()) or "mets" not in (names or ()):
        raise ValueError(
            f"{path} holds a struct with neither RAVEN nor COBRA model fields "
            f"(found {sorted(names or ())[:10]}); it is not a model file this "
            f"reader understands"
        )

    model = cobra.Model(_as_str(_field(struct, "id")) or name)
    model.name = _as_str(_field(struct, "name"))

    comps = _as_list(_field(struct, "comps"))
    comp_names = _as_list(_field(struct, "compNames"))
    if comps:
        model.compartments = {
            comp: (comp_names[i] if i < len(comp_names) else comp)
            for i, comp in enumerate(comps)
        }

    met_ids = _as_list(_field(struct, "mets"))
    n_mets = len(met_ids)
    met_names = _column(struct, "metNames", n_mets)
    met_formulas = _column(struct, "metFormulas", n_mets)
    met_charges = _as_floats(_field(struct, "metCharges"))
    met_comps = _as_floats(_field(struct, "metComps"))
    met_miriams = _miriam_column(struct, "metMiriams", n_mets)
    met_smiles = _column(struct, "metSmiles", n_mets)

    metabolites = []
    for i, met_id in enumerate(met_ids):
        met = cobra.Metabolite(met_id, name=met_names[i])
        met.formula = met_formulas[i] or None
        if i < len(met_charges) and not np.isnan(met_charges[i]):
            met.charge = met_charges[i]
        if i < len(met_comps) and comps:
            index = int(met_comps[i]) - 1
            if 0 <= index < len(comps):
                met.compartment = comps[index]
        annotation = dict(met_miriams[i])
        if met_smiles[i]:
            annotation[_SMILES_KEY] = met_smiles[i].split(";")
        met.annotation = annotation
        metabolites.append(met)
    model.add_metabolites(metabolites)

    for field, note_key in _MET_NOTE_FIELDS:
        for met, value in zip(metabolites, _column(struct, field, n_mets), strict=False):
            _set_note(met, note_key, value)
    for met, number in zip(
        metabolites, _as_floats(_field(struct, "metDeltaG")), strict=False
    ):
        if not np.isnan(number):
            _set_note(met, "deltaG", number)

    rxn_ids = _as_list(_field(struct, "rxns"))
    n_rxns = len(rxn_ids)
    rxn_names = _column(struct, "rxnNames", n_rxns)
    lbs = _as_floats(_field(struct, "lb"))
    ubs = _as_floats(_field(struct, "ub"))
    objective = _as_floats(_field(struct, "c"))
    gr_rules = _column(struct, "grRules", n_rxns)
    subsystems = _field(struct, "subSystems")
    eccodes = _column(struct, "eccodes", n_rxns)
    rxn_miriams = _miriam_column(struct, "rxnMiriams", n_rxns)
    stoich = scipy.sparse.csc_matrix(_field(struct, "S"))

    reactions = []
    for j, rxn_id in enumerate(rxn_ids):
        rxn = cobra.Reaction(rxn_id, name=rxn_names[j])
        rxn.lower_bound = lbs[j] if j < len(lbs) else 0.0
        rxn.upper_bound = ubs[j] if j < len(ubs) else 1000.0
        reactions.append(rxn)
    model.add_reactions(reactions)

    for j, rxn in enumerate(reactions):
        column = stoich.getcol(j).tocoo()
        rxn.add_metabolites(
            {metabolites[i]: float(v) for i, v in zip(column.row, column.data, strict=False)}
        )
        if gr_rules[j]:
            set_gene_rule(rxn, gr_rules[j])
        if subsystems is not None:
            entry = np.asarray(subsystems).reshape(-1, order="F")
            if j < entry.size:
                found = _as_list(entry[j])
                if found:
                    rxn.subsystem = found
        annotation = dict(rxn_miriams[j])
        if eccodes[j]:
            annotation[_EC_KEY] = eccodes[j].split(";")
        rxn.annotation = annotation

    if objective:
        model.objective = {
            rxn: coefficient
            for rxn, coefficient in zip(reactions, objective, strict=False)
            if coefficient
        } or model.objective

    for field, note_key in _RXN_NOTE_FIELDS:
        for rxn, value in zip(reactions, _column(struct, field, n_rxns), strict=False):
            _set_note(rxn, note_key, value)
    for rxn, number in zip(
        reactions, _as_floats(_field(struct, "rxnConfidenceScores")), strict=False
    ):
        _set_note(rxn, "confidence_score", number)
    for rxn, number in zip(
        reactions, _as_floats(_field(struct, "rxnDeltaG")), strict=False
    ):
        if not np.isnan(number):
            _set_note(rxn, "deltaG", number)

    gene_ids = _as_list(_field(struct, "genes"))
    short_names = _column(struct, "geneShortNames", len(gene_ids))
    gene_miriams = _miriam_column(struct, "geneMiriams", len(gene_ids))
    for i, gene_id in enumerate(gene_ids):
        gene = model.genes.get_by_id(gene_id) if gene_id in model.genes else None
        if gene is None:
            continue
        if short_names[i]:
            gene.name = short_names[i]
        if gene_miriams[i]:
            gene.annotation = dict(gene_miriams[i])

    metadata: dict[str, str | float] = {}
    for key in ("id", "name"):
        value = _as_str(_field(struct, key))
        if value:
            metadata[key] = value
    for key in ("date", "description"):
        value = _as_str(_field(struct, key))
        if value:
            metadata[key] = value
    annotation = _field(struct, "annotation")
    if annotation is not None and annotation.dtype.names:
        for key in annotation.dtype.names:
            value = annotation[key][0, 0]
            if key in ("defaultLB", "defaultUB"):
                numbers = _as_floats(value)
                if numbers:
                    metadata[key] = numbers[0]
            else:
                text = _as_str(value)
                if text:
                    metadata[key] = text
    notes = dict(model.notes or {})
    version = _as_str(_field(struct, "version"))
    if version:
        notes["version"] = metadata["version"] = version
    if metadata:
        notes["metaData"] = metadata
    model.notes = notes

    return model
