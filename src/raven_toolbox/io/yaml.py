"""Read and write RAVEN/cobrapy YAML models.

Aligned to RAVEN ``writeYAMLmodel.m`` / ``readYAMLmodel.m``, whose writer emits
**cobra's native ``!!omap`` YAML**. Because the format *is* cobra's, the standard model
content — id, name, compartments, and per-entry id/name/compartment/formula/
charge/bounds/gene_reaction_rule/objective_coefficient/subsystem/metabolites and
the whole ``annotation`` block (which carries ``smiles`` for metabolites,
``ec-code`` for reactions, and all MIRIAM cross-references) — is read and written
by ``cobra.io`` directly.

This module only handles what cobra drops or mishandles:

* **RAVEN-only top-level per-entry keys** that cobra ignores: ``inchis``,
  ``deltaG``, ``metFrom`` and the free-text ``notes`` (metNotes) on metabolites;
  ``confidence_score``, ``references``, ``rxnFrom``, ``deltaG`` and ``notes``
  (rxnNotes) on reactions; ``protein`` on genes. These are stashed in the cobra
  object's ``.notes`` dict on read and lifted back to top-level keys on write.
* **Model-level extras** cobra ignores: ``version`` and the ``metaData``
  provenance block, preserved on ``model.notes`` for round-tripping.
* **The GECKO ec sections** (``ec-rxns``/``ec-enzymes``/``gecko_light``): on
  read, parsed into a typed :class:`~raven_toolbox.io.ec_data.EcData` and
  attached as ``model.ec``; on write, serialised back to top-level sections
  whenever ``model.ec`` is present. This mirrors RAVEN ``readYAMLmodel.m`` /
  ``writeYAMLmodel.m``, which populate the ``model.ec`` struct when the YAML
  defines it. Any other unknown top-level keys are still preserved opaquely
  via ``model.notes['_yaml_sections']``.

Legacy quirks the reader also accepts (silent normalisation):

* older RAVEN files with ``id`` / ``name`` nested in ``metaData``;
* per-metabolite top-level ``smiles`` (lifted into ``annotation['smiles']``);
* per-reaction top-level ``eccodes`` (lifted into ``annotation['ec-code']`` —
  the cobra-standard place where geckopy reads EC numbers);
* very old RAVEN files written as a bare ``-`` sequence of single-key mappings
  rather than one big mapping;
* MATLAB GECKO ecModels whose ``usage_prot_*`` and ``prot_pool_exchange``
  reactions use the older reverse-direction convention (negative lower bound,
  swapped stoichiometry signs); these are flipped on load so consumers always
  see the forward convention.
"""
from __future__ import annotations

import gzip
import re
import warnings
from collections import OrderedDict
from datetime import date
from pathlib import Path

import cobra
from cobra.io.dict import model_from_dict, model_to_dict
from cobra.io.yaml import yaml as _cobra_yaml  # ruamel round-trip YAML (handles !!omap)
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from raven_toolbox.io.ec_data import (
    EcData,
    ec_data_from_yaml_sections,
    ec_data_to_yaml_sections,
)

# A dedicated instance for *writing* only (reading still goes through
# _cobra_yaml, whose parsing is unaffected by any of this). Mutating
# _cobra_yaml's own width would also change cobra.io.save_yaml_model for
# the rest of the process, so this module gets its own ruamel round-trip
# instance instead. `width` is set far past any real line length so
# ruamel never folds a long scalar onto a continuation line, matching
# writeYAMLmodel.m. `indent` is pinned explicitly (to the values ruamel
# already used implicitly, verified byte-identical) rather than left for
# ruamel to pick, so a future ruamel version can't silently change the
# on-disk layout out from under either writer. Every list in this format
# is itself a block sequence of single-key `!!omap` entries (reactions,
# a reaction's metabolites, ...), so `sequence=2, offset=0` still nests
# each entry two spaces past its parent key, matching writeYAMLmodel.m's
# own hand-indented style — it would only read as flush-with-the-key for
# a plain `key: [list]` mapping, which this format never emits at any
# level. Quoting still only happens when YAML requires it; the quote
# *character* is double, matching Prettier's YAML default and
# writeYAMLmodel.m (see _needs_quote below for why this needs its own
# check rather than ruamel's default single-quote-when-needed behaviour).
_write_yaml = YAML(typ="rt")
_write_yaml.width = 1_000_000
_write_yaml.indent(mapping=2, sequence=2, offset=0)


def _open_text(path: str | Path, mode: str):
    """Open ``path`` as a text handle, transparently gzipping when it ends ``.gz``."""
    if str(path).endswith(".gz"):
        return gzip.open(path, f"{mode}t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")

# RAVEN-only top-level per-entry keys -> the key used inside the cobra object's
# .notes dict. ('notes' is RAVEN's free-text metNotes/rxnNotes; stored under
# 'note' to avoid colliding with the notes container itself.)
_MET_FIELDS = (("inchis", "inchis"), ("deltaG", "deltaG"), ("metFrom", "metFrom"), ("notes", "note"))
_RXN_FIELDS = (
    ("references", "references"),
    ("rxnFrom", "rxnFrom"),
    ("deltaG", "deltaG"),
    ("confidence_score", "confidence_score"),
    ("notes", "note"),
)
# Legacy YAML keys accepted on READ for reaction notes. Old RAVEN MATLAB writers
# used "rxnNotes"; the canonical key (matching cobrapy and the current MATLAB
# writer) is "notes". When both appear, "notes" wins.
_LEGACY_RXN_KEY_ALIASES = (("rxnNotes", "notes"),)
_GENE_FIELDS = (("protein", "protein"),)

_COBRA_TOP_KEYS = frozenset({"metabolites", "reactions", "genes", "compartments", "id", "name"})

# Top-level keys consumed by the typed EcData layer; not re-emitted as opaque
# `_yaml_sections` on round-trip (the source of truth is ``model.ec``).
_EC_TOP_KEYS = frozenset({"ec-rxns", "ec-enzymes", "gecko_light"})


def _to_plain(obj):
    """Coerce ruamel/numpy scalars to plain Python primitives.

    Dicts are returned as ``OrderedDict`` so that the round-trip dumper
    emits them with the ``!!omap`` tag (ruamel's CommentedMap subclass
    is replaced by a plain ``OrderedDict`` to avoid carrying ruamel's
    own type tags through). Returning a plain ``dict`` instead would
    drop the ``!!omap`` tag and produce files that RAVEN's MATLAB
    reader (a line-based parser keyed on ``!!omap``) cannot load.
    """
    if isinstance(obj, dict):
        return OrderedDict((str(k), _to_plain(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    return obj if isinstance(obj, str) else str(obj)


def _capture_entry_fields(entries, fields):
    """Pop RAVEN-only top-level keys off each entry into a parallel notes dict.

    Returns a list of ``{notes_key: value}`` dicts aligned with ``entries`` (so
    cobra never sees these keys), to be attached to the built objects afterwards.
    """
    captured = []
    for entry in entries:
        notes = {}
        for yaml_key, notes_key in fields:
            if yaml_key in entry:
                notes[notes_key] = entry.pop(yaml_key)
        captured.append(notes)
    return captured


def read_yaml_model(path: str | Path) -> cobra.Model:
    """Read a RAVEN/cobrapy YAML model into a ``cobra.Model``.

    Convenience wrapper around :func:`model_from_yaml_data` that opens the
    file (transparently un-gzipping ``.gz``) and parses the YAML. Callers
    that need to pre-process the document (e.g. lift further non-standard
    fields that cobra doesn't recognise) can read+normalise themselves and
    call :func:`model_from_yaml_data` with the resulting dict.

    Accepts both the cobra `!!omap` shape and a very old RAVEN shape where
    the document root is a bare ``-`` sequence of single-key mappings; the
    latter is merged into one mapping before parsing.
    """
    with _open_text(path, "r") as handle:
        raw = _to_plain(_cobra_yaml.load(handle))

    if isinstance(raw, list):
        # Very old RAVEN files: a sequence of one-key mappings instead of
        # one big !!omap. Merge into a single dict before parsing.
        merged: dict = {}
        for item in raw:
            if isinstance(item, dict):
                merged.update(item)
        raw = merged

    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: top-level YAML must be a mapping or a sequence of "
            f"single-key mappings, got {type(raw).__name__}."
        )
    return model_from_yaml_data(raw)


def model_from_yaml_data(raw: dict) -> cobra.Model:
    """Build a ``cobra.Model`` from an already-parsed RAVEN/cobrapy YAML dict.

    Performs three jobs in order:

    1. **cobra-shaped portion:** strips and restores RAVEN-only per-entry
       side-fields onto each entry's ``.notes``; lifts ``id`` / ``name``
       out of legacy ``metaData``; preserves ``version`` and ``metaData``
       on ``model.notes`` for round-trip.
    2. **legacy quirks:** lifts per-metabolite top-level ``smiles`` into
       ``annotation['smiles']`` and per-reaction top-level ``eccodes`` into
       ``annotation['ec-code']`` (older RAVEN/MATLAB GECKO files emitted
       these at the top level); flips the older reverse-direction
       ``usage_prot_*`` / ``prot_pool_exchange`` convention to the
       forward convention.
    3. **GECKO ec sections:** when ``ec-rxns`` / ``ec-enzymes`` are
       present, parses them into a typed :class:`EcData` and attaches
       it as ``model.ec``. Other unknown top-level keys land opaquely on
       ``model.notes['_yaml_sections']`` for round-trip.

    ``raw`` is mutated in place — copy it first if the caller needs the
    original.
    """
    metadata = raw.pop("metaData", None) or {}
    version = raw.pop("version", None)
    foreign = {k: raw.pop(k) for k in list(raw) if k not in _COBRA_TOP_KEYS}

    # RAVEN MATLAB omits a section entirely when it is empty -- a model with no
    # genes has no `genes:` block at all, which YAML treats the same as an
    # absent key. But a *present*, bare `genes:` key (nothing after the colon)
    # parses to `None`, not an empty list, and every `raw.get(section)` call
    # below would then iterate over `None` and crash. Normalise both shapes to
    # `[]` up front, before anything reads these keys.
    for section in ("metabolites", "reactions", "genes"):
        if raw.get(section) is None:
            raw[section] = []

    # Legacy quirk: per-metabolite top-level `smiles` -> annotation.smiles.
    # Done before model_from_dict so cobra sees the annotation in its
    # canonical place. No-op on current files.
    _lift_smiles_to_annotation(raw.get("metabolites"))

    # Legacy quirk: per-reaction top-level `eccodes` -> annotation['ec-code'].
    # EC numbers are standard cobra annotation; older RAVEN/MATLAB files put
    # them at the reaction top level, which hid them from cobra/geckopy (which
    # read annotation['ec-code']). Lift before model_from_dict. No-op on
    # current cobra-shaped files.
    _lift_eccodes_to_annotation(raw.get("reactions"))

    # A bare scalar annotation value (writeYAMLmodel.m/write_yaml_model both
    # collapse a single-value MIRIAM entry to one) is wrapped into a
    # single-item list here, before model_from_dict, so cobra/geckopy always
    # see annotation[key] as list[str] regardless of whether the source file
    # used a scalar or a list — cobra's own model_from_dict does not do this
    # normalisation itself. ``sbo`` stays a bare string.
    for section in ("metabolites", "reactions", "genes"):
        _normalize_annotation_values(raw.get(section))

    # A metabolite with no explicit compartment defaults to the first one,
    # matching readYAMLmodel.m's own convention — cobra's model_from_dict
    # would otherwise leave met.compartment as None.
    _default_missing_compartment(raw.get("metabolites"), raw.get("compartments"))

    # writeYAMLmodel.m (RAVEN 3) and write_yaml_model write every number as a
    # float, so a charge arrives as -1.0. cobra models charge as an int, and its
    # SBML writer passes it unchanged to libsbml's setCharge, which does not keep
    # a float: without this the charges are lost on the way to SBML.
    _integer_charges(raw.get("metabolites"))

    # Normalise legacy reaction-side YAML keys (e.g. RAVEN MATLAB's
    # ``rxnNotes`` -> the canonical ``notes``) before any field capture so
    # the capture step sees a single key per concept.
    for entry in raw.get("reactions", []):
        if not isinstance(entry, dict):
            continue
        for legacy_key, canonical_key in _LEGACY_RXN_KEY_ALIASES:
            if legacy_key in entry and canonical_key not in entry:
                entry[canonical_key] = entry.pop(legacy_key)
            elif legacy_key in entry:
                # Canonical wins; just drop the legacy duplicate.
                entry.pop(legacy_key)

    met_notes = _capture_entry_fields(raw.get("metabolites", []), _MET_FIELDS)
    rxn_notes = _capture_entry_fields(raw.get("reactions", []), _RXN_FIELDS)
    gene_notes = _capture_entry_fields(raw.get("genes", []), _GENE_FIELDS)

    model = model_from_dict(raw)

    for met, notes in zip(model.metabolites, met_notes, strict=False):
        met.notes = notes
    for rxn, notes in zip(model.reactions, rxn_notes, strict=False):
        rxn.notes = notes
    for gene, notes in zip(model.genes, gene_notes, strict=False):
        gene.notes = notes

    # Legacy quirk: flip reverse-direction usage_prot_* / prot_pool_exchange.
    # No-op when none match.
    _flip_legacy_prot_direction(model)

    # RAVEN convention keeps id/name/version inside metaData; lift them
    # onto the model so cobra-shaped accessors find them too. Older
    # cobra-style files (id/name/version at root) are handled by the
    # cobra reader; the metaData lookups below are pure fallbacks.
    if metadata.get("id") and not model.id:
        model.id = metadata["id"]
    if metadata.get("name") and not model.name:
        model.name = metadata["name"]
    if version is None and metadata.get("version") is not None:
        version = metadata["version"]
    if metadata:
        model.notes["metaData"] = metadata
    if version is not None:
        model.notes["version"] = version

    # Pop the ec sections out of `foreign` and into a typed EcData.
    # The remaining unknown keys round-trip opaquely. Pre-shim RAVEN
    # MATLAB writes wrote `geckoLight: "true"` inside metaData (rather
    # than the current top-level `gecko_light`); honour the legacy
    # placement too — keep the metaData entry untouched (round-trip)
    # and surface it at the top level so EcData picks it up.
    legacy_gecko = metadata.get("geckoLight")
    if legacy_gecko is not None and "gecko_light" not in foreign:
        if isinstance(legacy_gecko, str):
            foreign["gecko_light"] = legacy_gecko.lower() == "true"
        else:
            foreign["gecko_light"] = bool(legacy_gecko)
    ec_sections = {k: foreign.pop(k) for k in list(foreign) if k in _EC_TOP_KEYS}
    ec_data = ec_data_from_yaml_sections(ec_sections)
    if ec_data is not None:
        model.ec = ec_data
    if foreign:
        model.notes["_yaml_sections"] = foreign

    return model


# --------------------------------------------------------------------------- #
# Legacy quirk normalisers
# --------------------------------------------------------------------------- #

def _normalize_annotation_values(entries) -> None:
    """Wrap a bare scalar annotation value into a single-item list.

    Both writers collapse a singleton MIRIAM value to a scalar (see
    :func:`_collapse_singleton_annotations`), so a file can carry either
    form for the same key. Normalising here — rather than leaving cobra to
    store whatever shape the source used — means ``annotation[key]`` is
    always ``list[str]`` however the file spelled it, matching what cobra
    and geckopy expect and keeping a scalar-collapsed value from coming
    back as a bare string after a write/read round trip. The exception is
    ``sbo``, which is single-valued: it is kept as a bare string, the type
    cobra's SBML reader and :func:`~raven_toolbox.annotation.add_sbo_terms`
    give it. Normalises in place; a no-op when ``entries`` is falsy.
    """
    if not entries:
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        annotation = entry.get("annotation")
        if not isinstance(annotation, dict):
            continue
        for key, value in annotation.items():
            if key == "sbo":
                if isinstance(value, list) and len(value) == 1:
                    annotation[key] = value[0]
            elif not isinstance(value, list):
                annotation[key] = [value]


def _integer_charges(metabolites) -> None:
    """Turn whole-number charges (``-1.0``) into ints, in place.

    A charge that is not a whole number is left as it is, so no information is
    lost; ``None`` (an unset charge) is left alone too.
    """
    for met in metabolites or ():
        if not isinstance(met, dict):
            continue
        charge = met.get("charge")
        if isinstance(charge, float) and charge.is_integer():
            met["charge"] = int(charge)


def _default_missing_compartment(metabolites, compartments) -> None:
    """Default a metabolite's missing compartment to the first compartment.

    Matches readYAMLmodel.m's own convention (a metabolite with no
    explicit compartment is assigned index 1, i.e. the first entry of
    ``compartments:``) — cobra's ``model_from_dict`` has no equivalent
    default and would otherwise leave ``met.compartment`` as ``None``.
    "First" is the first key of the ``compartments`` mapping, in file
    order (preserved by the round-trip YAML loader), matching
    ``model.comps{1}`` on the MATLAB side. Normalises in place; a no-op
    when there is no compartments section to default to.
    """
    if not compartments or not metabolites:
        return
    first = next(iter(compartments))
    for met in metabolites:
        if isinstance(met, dict) and not met.get("compartment"):
            met["compartment"] = first


def _lift_smiles_to_annotation(metabolites) -> None:
    """Move per-metabolite top-level ``smiles`` into ``annotation['smiles']``.

    Older MATLAB GECKO ecModel writers placed SMILES at the metabolite top
    level; the cobra/raven convention is to nest them inside ``annotation``.
    Normalises in place; no-op when no metabolite carries a top-level
    ``smiles`` key.
    """
    if not isinstance(metabolites, list):
        return
    for met in metabolites:
        if not (isinstance(met, dict) and "smiles" in met):
            continue
        smiles = met.pop("smiles")
        annotation = met.get("annotation")
        if not isinstance(annotation, dict):
            annotation = {}
            met["annotation"] = annotation
        if "smiles" not in annotation and smiles:
            annotation["smiles"] = (
                smiles if isinstance(smiles, list) else [smiles]
            )


def _lift_eccodes_to_annotation(reactions) -> None:
    """Move a reaction's legacy top-level ``eccodes`` into ``annotation['ec-code']``.

    EC numbers are a standard MIRIAM cross-reference, so the cobra/raven
    convention is to carry them inside ``annotation`` under the ``ec-code``
    key — where cobra and geckopy read them — not as a RAVEN-only top-level
    field. Older RAVEN/MATLAB writers emitted a top-level ``eccodes`` (a
    ``;``-joined string or a list of codes); lift it into the canonical
    place. Normalises in place; no-op when no reaction carries a top-level
    ``eccodes`` key. A native ``annotation['ec-code']`` (if already present)
    wins and is left untouched.
    """
    if not isinstance(reactions, list):
        return
    for rxn in reactions:
        if not (isinstance(rxn, dict) and "eccodes" in rxn):
            continue
        codes = _eccodes_to_list(rxn.pop("eccodes"))
        if not codes:
            continue
        annotation = rxn.get("annotation")
        if not isinstance(annotation, dict):
            annotation = {}
            rxn["annotation"] = annotation
        annotation.setdefault("ec-code", codes)


def _eccodes_to_list(value) -> list:
    """Normalise a RAVEN ``eccodes`` value to a list of trimmed code strings.

    Accepts a ``;``-joined string (RAVEN MATLAB's ``getECstring`` form) or an
    already-split list; drops empty tokens.
    """
    if value is None:
        return []
    items = value if isinstance(value, list) else str(value).split(";")
    return [str(s).strip() for s in items if str(s).strip()]


def _flip_legacy_prot_direction(model: cobra.Model) -> None:
    """Flip pre-forward-direction protein reactions in place.

    Older MATLAB GECKO ecModels defined ``usage_prot_*`` and
    ``prot_pool_exchange`` as "reverse" reactions: their flux was
    negative, and the stoichiometry signs were correspondingly swapped.
    The current convention (in both geckopy and recent MATLAB GECKO)
    treats them as ordinary forward reactions with positive flux. When a
    loaded model still uses the older convention we flip the affected
    reactions in place so consumers never have to handle two shapes.

    The signature we look for is any ``usage_prot_*`` or
    ``prot_pool_exchange`` reaction whose lower bound is negative.
    """
    flipped: list[str] = []
    for rxn in model.reactions:
        if not (
            rxn.id.startswith("usage_prot_")
            or rxn.id == "prot_pool_exchange"
        ):
            continue
        if rxn.lower_bound >= -1e-9:
            continue
        rxn.add_metabolites(
            {met: -2.0 * coef for met, coef in rxn.metabolites.items()},
            combine=True,
        )
        lb, ub = rxn.lower_bound, rxn.upper_bound
        rxn.lower_bound = -ub
        rxn.upper_bound = -lb
        flipped.append(rxn.id)
    if flipped:
        warnings.warn(
            f"ecModel uses the older reverse-direction convention for "
            f"{len(flipped)} protein usage/pool reaction(s); flipping to "
            "the current forward convention.",
            stacklevel=3,
        )


def _emit_entry_fields(entries, fields):
    """Lift RAVEN-only keys out of each entry's ``notes`` dict to top level."""
    for entry in entries:
        notes = entry.pop("notes", None)
        if not isinstance(notes, dict):
            continue
        notes = dict(notes)
        for yaml_key, notes_key in fields:
            if notes_key in notes:
                entry[yaml_key] = notes.pop(notes_key)
        # Preserve any remaining (non-RAVEN) notes. The RAVEN free-text note is lifted
        # to the YAML key "notes"; if leftovers also exist, merge them with it under
        # that key (rather than silently dropping the leftovers).
        if notes:
            if "notes" in entry:
                notes["note"] = entry["notes"]
            entry["notes"] = notes


def _coerce_floats(obj):
    """Recursively coerce every plain ``int`` in ``obj`` to ``float``.

    Every number is written as an explicit float (``2.0``, not ``2``),
    matching writeYAMLmodel.m. ``bool`` is checked first since it is an
    ``int`` subclass in Python and must not be turned into ``0.0``/``1.0``.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return float(obj)
    if isinstance(obj, dict):
        return type(obj)((k, _coerce_floats(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return [_coerce_floats(v) for v in obj]
    return obj


_QUOTE_RESOLVER_PATTERN = re.compile(
    r"^(true|True|TRUE|false|False|FALSE)$"
    r"|^(?i:y|n|yes|no|on|off)$"
    r"|^[-+]?[0-9][0-9_]*\.[0-9_]*([eE][-+]?[0-9]+)?$"
    r"|^[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+$"
    r"|^[-+]?\.[0-9_]+([eE][-+][0-9]+)?$"
    r"|^[-+]?\.(inf|Inf|INF)$"
    r"|^\.(nan|NaN|NAN)$"
    r"|^[-+]?0b[01_]+$"
    r"|^[-+]?0o?[0-7_]+$"
    r"|^[-+]?[0-9_]+$"
    r"|^[-+]?0x[0-9a-fA-F_]+$"
    r"|^(~|[Nn]ull|NULL)$"
    r"|^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
)


def _is_ws(ch: str) -> bool:
    """The four characters writeYAMLmodel.m's needsQuote treats as whitespace.

    Deliberately narrower than str.isspace(), which also matches \\v, \\f and
    other Unicode whitespace the MATLAB port never considered.
    """
    return ch in (" ", "\t", "\n", "\r")


def _needs_quote(s: str) -> bool:
    """Whether this scalar needs quoting when written as YAML.

    Ports writeYAMLmodel.m's needsQuote, for the same two reasons: a plain
    (bare) reading would resolve to a non-string YAML type (bool/int/float/
    null/timestamp, as neither writer emits explicit type tags), or the text
    itself is not valid as a plain block scalar (leading indicator
    character, a mid-string ": "/" #", leading/trailing whitespace, or an
    embedded line break). Besides the YAML 1.2 booleans it quotes the YAML
    1.1 ones (y, n, yes, no, on, off, in any case), which a YAML 1.1 loader
    such as PyYAML would otherwise read as booleans: ``name: NO`` (nitric
    oxide) is written ``name: "NO"``.
    """
    if not s:
        return True
    if _QUOTE_RESOLVER_PATTERN.match(s):
        return True
    first = s[0]
    if first in "#,[]{}&*!|>'\"%@`":
        return True
    followed_by_ws = len(s) == 1 or _is_ws(s[1])
    if first in "?:" and followed_by_ws:
        return True
    if first == "-" and followed_by_ws:
        return True
    if _is_ws(first) or _is_ws(s[-1]):
        return True
    if "\n" in s or "\r" in s:
        return True
    for i in range(1, len(s)):
        ch = s[i]
        if ch == ":" and (i == len(s) - 1 or _is_ws(s[i + 1])):
            return True
        if ch == "#" and _is_ws(s[i - 1]):
            return True
    return False


def _prefer_double_quotes(obj):
    """Recursively mark every scalar string that needs quoting as double-quoted.

    ruamel's round-trip dumper single-quotes a plain ``str`` when quoting
    is syntactically required; there is no global "quote with this
    character when needed" switch, only a per-value style override. Wrap
    exactly the values :func:`_needs_quote` says need quoting in
    ``DoubleQuotedScalarString`` — never unconditionally, or every scalar
    would end up quoted regardless of whether YAML requires it.
    """
    if isinstance(obj, str):
        return DoubleQuotedScalarString(obj) if _needs_quote(obj) else obj
    if isinstance(obj, dict):
        return type(obj)((k, _prefer_double_quotes(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return [_prefer_double_quotes(v) for v in obj]
    return obj


_LIST_ONLY_ANNOTATION_KEYS = frozenset({"ec-code", "smiles"})


def _collapse_singleton_annotations(entries) -> None:
    """Collapse a one-item annotation list to a bare scalar.

    Matches writeYAMLmodel.m's MIRIAM handling: every annotation entry
    collapses to a scalar when it carries a single value, except
    ``ec-code``/``smiles`` — RAVEN's own writer marks exactly those two
    ``forceList``, which are always a list, even for one value, since cobra and
    geckopy read them as ``list[str]``. cobra stores a single identifier as a
    bare string, so a string under one of those keys is wrapped in a list.
    Normalises in place.
    """
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        annotation = entry.get("annotation")
        if not isinstance(annotation, dict):
            continue
        for key, value in annotation.items():
            if key in _LIST_ONLY_ANNOTATION_KEYS:
                if isinstance(value, str):
                    annotation[key] = [value]
            elif isinstance(value, list) and len(value) == 1:
                annotation[key] = value[0]


_META_ANNOTATION_FIELDS = (
    "givenName",
    "familyName",
    "authors",
    "email",
    "organization",
    "taxonomy",
    "note",
    "sourceUrl",
)

# Every key _build_metadata places itself, from either the model or
# stored_meta — anything else in stored_meta is a caller-specific field with
# no RAVEN-defined slot and gets passed through instead of dropped (see
# _build_metadata).
_HANDLED_METADATA_KEYS = frozenset(
    {"id", "name", "version", "date", "defaultLB", "defaultUB"}
) | frozenset(_META_ANNOTATION_FIELDS)


def _default_bounds(model: cobra.Model) -> tuple[float, float] | None:
    """The model's ``(min lower_bound, max upper_bound)`` across all reactions.

    Mirrors readYAMLmodel.m's ``defaultLB``/``defaultUB``
    (``min(model.lb)``/``max(model.ub)``), recomputed fresh from the
    model's current bounds — like the MATLAB reader recomputes them from
    the just-parsed bounds — rather than echoed from a stored value, so a
    model whose bounds changed after loading still gets a correct
    default. ``None`` for a model with no reactions (nothing to derive
    from).
    """
    if not model.reactions:
        return None
    lbs = [rxn.lower_bound for rxn in model.reactions]
    ubs = [rxn.upper_bound for rxn in model.reactions]
    return min(lbs), max(ubs)


def _build_metadata(model: cobra.Model, stored_meta: dict, version) -> OrderedDict:
    """Assemble the ``metaData`` block in writeYAMLmodel.m's fixed field order.

    ``id``/``name`` are always present (``"blankID"``/``"blankName"`` when
    unset, matching writeMetadata's own ``valueOrDefault``), as is
    ``date`` (today's date when the model carries none). ``defaultLB``/
    ``defaultUB`` are written only when the model's stored metaData has them,
    as RAVEN's writer does, and are then recomputed from the model's current
    bounds (see :func:`_default_bounds`) rather than echoed. The remaining fields follow in the same order
    writeYAMLmodel.m's ``annoFields`` emits them, each only when present
    and non-empty. Any other key left in ``stored_meta`` — a
    caller-specific provenance field with no RAVEN-defined slot, e.g.
    geckopy's ``geckopy_version`` — is appended last, sorted
    alphabetically for deterministic output, rather than dropped;
    ``model_from_yaml_data`` already restores the full parsed metaData
    dict verbatim on read, so this is the write-side half of a lossless
    round trip (mirrors writeYAMLmodel.m's matching fallback).
    """
    metadata: OrderedDict = OrderedDict()
    metadata["id"] = model.id or "blankID"
    metadata["name"] = model.name or "blankName"
    if version is not None:
        metadata["version"] = version
    metadata["date"] = stored_meta.get("date") or date.today().isoformat()
    bounds = _default_bounds(model) if "defaultLB" in stored_meta or "defaultUB" in stored_meta else None
    if bounds is not None:
        if "defaultLB" in stored_meta:
            metadata["defaultLB"] = bounds[0]
        if "defaultUB" in stored_meta:
            metadata["defaultUB"] = bounds[1]
    for key in _META_ANNOTATION_FIELDS:
        value = stored_meta.get(key)
        if value:
            metadata[key] = value
    for key in sorted(stored_meta):
        if key in _HANDLED_METADATA_KEYS:
            continue
        value = stored_meta[key]
        if value:
            metadata[key] = value
    return metadata


def _normalize_subsystems(reactions) -> None:
    """Drop a reaction's ``subsystem`` key when it carries no subsystem.

    A reaction that does carry one or more subsystems is normalised to a
    plain list of non-empty strings, even for a single subsystem, matching
    writeYAMLmodel.m. Normalises in place.
    """
    for rxn in reactions:
        if not isinstance(rxn, dict) or "subsystem" not in rxn:
            continue
        value = rxn["subsystem"]
        items = value if isinstance(value, list) else [value]
        cleaned = [str(s) for s in items if s]
        if cleaned:
            rxn["subsystem"] = cleaned
        else:
            del rxn["subsystem"]


def _normalize_bounds(reactions) -> None:
    """Convert a stringified lower_bound/upper_bound back to float.

    cobra's own model_to_dict stringifies an infinite or NaN bound (JSON has
    no literal for either) --- its own reader tolerates this, unconditionally
    casting both fields through float(v) on the way back in, so the round
    trip survives. But left as the string "inf" rather than a real float,
    ruamel has no reason to emit the YAML 1.1 infinity token (.inf) for it,
    and writes the bare word instead --- which is not that token, and parses
    back as the string "inf" on any reader that does not share cobra's
    specific convention. writeYAMLmodel.m emits .inf here regardless.
    Normalises in place.
    """
    for rxn in reactions:
        if not isinstance(rxn, dict):
            continue
        for key in ("lower_bound", "upper_bound"):
            value = rxn.get(key)
            if isinstance(value, str):
                rxn[key] = float(value)


def write_yaml_model(
    model: cobra.Model, path: str | Path, *, sort_ids: bool = False
) -> None:
    """Write a ``cobra.Model`` to RAVEN/cobrapy (``!!omap``) YAML.

    When ``model.ec`` is a populated :class:`EcData`, the ``gecko_light``
    flag and the ``ec-rxns`` / ``ec-enzymes`` top-level sections are
    emitted from it (numpy/ruamel scalars are coerced to plain Python
    primitives en route, so the dumper never sees them).

    With ``sort_ids=True`` metabolites/reactions/genes/compartments are
    written in alphabetical order (diff-friendly), without modifying
    ``model``.
    """
    model_notes = dict(model.notes or {})
    stored_meta = model_notes.pop("metaData", None) or {}
    version = model_notes.pop("version", None)
    if version is None:  # a version kept only in metaData is still the model's version
        version = stored_meta.get("version")
    foreign = model_notes.pop("_yaml_sections", None) or {}

    unnamed = [met.id for met in model.metabolites if not met.name]
    if unnamed:
        # RAVEN's writeYAMLmodel stops on these (checkModelStruct: "metabolites have
        # empty names"), so a file written here with `name: ""` may not load there.
        shown = ", ".join(unnamed[:5]) + (", ..." if len(unnamed) > 5 else "")
        warnings.warn(
            f"{len(unnamed)} metabolite(s) have an empty name ({shown}); the YAML is written "
            "anyway, but RAVEN's writeYAMLmodel refuses such a model.",
            stacklevel=2,
        )

    seen: dict[tuple[str, str], str] = {}
    duplicated: list[str] = []
    for met in model.metabolites:
        if not met.name:
            continue
        key = (met.name, met.compartment or "")
        if key in seen:
            duplicated.append(f"{met.id} and {seen[key]}")
        else:
            seen[key] = met.id
    if duplicated:
        shown = ", ".join(duplicated[:5]) + (", ..." if len(duplicated) > 5 else "")
        warnings.warn(
            f"{len(duplicated)} metabolite(s) share a name and compartment with another ({shown}); "
            "the YAML is written anyway, but RAVEN's writeYAMLmodel refuses such a model.",
            stacklevel=2,
        )

    doc = OrderedDict(_to_plain(model_to_dict(model)))

    # cobra's model_to_dict serialises model.notes verbatim into doc["notes"],
    # so the three management keys we just lifted out would otherwise also
    # appear nested inside the notes section. Strip them; preserve any other
    # genuine notes the caller stored on the model.
    doc_notes = doc.get("notes")
    if isinstance(doc_notes, dict):
        for note_key in ("metaData", "version", "_yaml_sections"):
            doc_notes.pop(note_key, None)
        if not doc_notes:
            doc.pop("notes", None)

    if sort_ids:
        for section in ("metabolites", "reactions", "genes"):
            if section in doc:
                doc[section] = sorted(doc[section], key=lambda e: e.get("id", ""))
        if isinstance(doc.get("compartments"), dict):
            # OrderedDict, not dict --- see _to_plain's docstring: a plain
            # dict here would drop the !!omap tag ruamel gives OrderedDict,
            # producing a file readYAMLmodel.m cannot load.
            doc["compartments"] = OrderedDict(sorted(doc["compartments"].items()))

    _emit_entry_fields(doc.get("metabolites", []), _MET_FIELDS)
    _emit_entry_fields(doc.get("reactions", []), _RXN_FIELDS)
    _emit_entry_fields(doc.get("genes", []), _GENE_FIELDS)
    _normalize_subsystems(doc.get("reactions", []) or ())
    _normalize_bounds(doc.get("reactions", []) or ())
    # An unset charge is left omitted (cobra's own default), not defaulted to
    # 0 — readYAMLmodel.m's own fill-missing-fields step is a positional
    # backfill shared by several columns, not a "charge defaults to 0" rule:
    # a gap before some later metabolite that does carry a charge parses as
    # NaN (omitted on write, like deltaG); only a gap at the very end of the
    # metabolite list — after the last one with any charge — hits the
    # different, unrelated tail-padding branch and becomes a real 0. That is
    # a position-dependent side effect of shared fill code, not a
    # convention worth reproducing here.
    for section in ("metabolites", "reactions", "genes"):
        _collapse_singleton_annotations(doc.get(section, []) or ())

    # cobra's _gene_to_dict always emits `name: ''` because name is a
    # required attribute; RAVEN MATLAB skips empty names. Drop the
    # empty-string entry so the two writers produce identical genes.
    for gene in doc.get("genes", []) or ():
        if isinstance(gene, dict) and gene.get("name") == "":
            gene.pop("name", None)

    # Same deal for gene_reaction_rule: cobra's _reaction_to_dict always
    # emits it (also a required attribute), but a GPR-less reaction reads
    # back the same way with the key absent — cobra defaults it to '',
    # and nothing here ever indexes the key directly — so RAVEN MATLAB
    # drops it too. Match that rather than cobra's own convention.
    for rxn in doc.get("reactions", []) or ():
        if isinstance(rxn, dict) and rxn.get("gene_reaction_rule") == "":
            rxn.pop("gene_reaction_rule", None)

    # An unnamed reaction has no `name` key in RAVEN's writer either.
    for rxn in doc.get("reactions", []) or ():
        if isinstance(rxn, dict) and rxn.get("name") == "":
            rxn.pop("name", None)

    # ec sections come from the typed model.ec (when present), not from the
    # opaque foreign-keys stash. Drop any stale ec-* entries in `foreign` so
    # they can't conflict with the EcData-derived ones.
    for ec_key in _EC_TOP_KEYS:
        foreign.pop(ec_key, None)
    ec = getattr(model, "ec", None)
    ec_sections = (
        _to_plain(ec_data_to_yaml_sections(ec))
        if isinstance(ec, EcData) and (ec.n_rxns or ec.n_enzymes)
        else None
    )

    # Final document order (matches RAVEN MATLAB writeYAMLmodel):
    #   metaData, metabolites, reactions, genes, compartments,
    #   gecko_light, ec-rxns, ec-enzymes, <opaque foreign>
    # id/name/version live inside metaData (the RAVEN convention) — they
    # are NOT emitted at root level. Cobra reading such a file recovers
    # the lists and compartments and leaves model.id empty (consistent
    # with how RAVEN MATLAB has always laid these files out).
    metadata = _build_metadata(model, stored_meta, version)

    # cobra's model_to_dict put id / name at root level; drop them so they
    # don't duplicate the metaData copy.
    doc.pop("id", None)
    doc.pop("name", None)

    ordered = OrderedDict()
    ordered["metaData"] = metadata
    for section in ("metabolites", "reactions", "genes", "compartments", "notes"):
        if section in doc:
            ordered[section] = doc.pop(section)
    if ec_sections is not None:
        ordered["gecko_light"] = ec_sections["gecko_light"]
        ordered["ec-rxns"] = ec_sections["ec-rxns"]
        ordered["ec-enzymes"] = ec_sections["ec-enzymes"]
    # Carry over any cobra-shaped fields we didn't classify above
    # (defensive: keeps forward compatibility with future cobra additions).
    for key, value in doc.items():
        ordered.setdefault(key, value)
    for key, value in foreign.items():
        ordered.setdefault(key, value)

    with _open_text(path, "w") as handle:
        _write_yaml.dump(_prefer_double_quotes(_coerce_floats(ordered)), handle)
