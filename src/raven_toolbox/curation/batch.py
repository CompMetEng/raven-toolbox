"""Generic batch-curation engine driven by tabular data."""
from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import cobra
import pandas as pd

from raven_toolbox.utils.ids import next_ids
from raven_toolbox.utils.parse import subsystem_to_str

#: Core columns recognised in ``mets_df``. Anything else is treated as a
#: MIRIAM annotation column (the header becomes the namespace key).
DEFAULT_CORE_MET_COLUMNS: tuple[str, ...] = (
    "metNames", "comps", "formula", "charge", "inchi", "metNotes",
)
#: Core columns recognised in ``genes_df``.
DEFAULT_CORE_GENE_COLUMNS: tuple[str, ...] = ("genes", "geneShortNames")
#: Core columns recognised in ``rxns_df``.
DEFAULT_CORE_RXN_COLUMNS: tuple[str, ...] = (
    "rxnNames", "grRules", "lb", "ub", "rev",
    "subSystems", "eccodes", "rxnNotes", "rxnReferences", "rxnConfidenceScores",
)
#: Required columns in ``rxns_coeffs_df``. (Linkage column ``rxnNames``
#: + one row per ``(reaction, metabolite)`` pair.) An optional ``rxnIdx`` column
#: (also read as ``index``) numbers the reactions; when both tables have it, a
#: coefficient row belongs to the reaction with the same index and name.
DEFAULT_CORE_RXN_COEFFS_COLUMNS: tuple[str, ...] = (
    "rxnNames", "metNames", "comps", "coefficient",
)

#: Header spellings read for each column, matched case-insensitively and mapped
#: to the canonical name (the same spellings ``curateModelFromTables`` accepts).
_MET_ALIASES = {
    "metNames": ("metNames", "metName"),
    "comps": ("comps", "comp"),
    "formula": ("formula", "formulas", "metFormula", "metFormulas"),
    "charge": ("charge", "charges", "metCharge", "metCharges"),
    "inchi": ("inchi", "inchis"),
    "metNotes": ("metNotes", "metNote"),
}
_GENE_ALIASES = {
    "genes": ("genes", "gene"),
    "geneShortNames": ("geneShortNames", "geneShortName"),
}
_RXN_ALIASES = {
    "rxnIdx": ("rxnIdx", "index"),
    "rxnNames": ("rxnNames", "rxnName"),
}
_COEFF_ALIASES = {
    "rxnIdx": ("rxnIdx", "index"),
    "rxnNames": ("rxnNames", "rxnName"),
    "metNames": ("metNames", "metName"),
    "comps": ("comps", "comp"),
    "coefficient": ("coefficient", "coefficients", "coeffs", "stoichCoeffs"),
}


@dataclass
class CurationResult:
    """Record of what :func:`batch_curate` added / updated.

    Each list holds the cobra ids of the affected entities, in the
    order they were processed.
    """

    added_metabolites: list[str] = field(default_factory=list)
    updated_metabolites: list[str] = field(default_factory=list)
    added_genes: list[str] = field(default_factory=list)
    updated_genes: list[str] = field(default_factory=list)
    added_reactions: list[str] = field(default_factory=list)
    updated_reactions: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(
            self.added_metabolites or self.updated_metabolites
            or self.added_genes or self.updated_genes
            or self.added_reactions or self.updated_reactions
        )


# --- public entry points ----------------------------------------------

def batch_curate(
    model: cobra.Model,
    *,
    mets_df: pd.DataFrame | None = None,
    genes_df: pd.DataFrame | None = None,
    rxns_df: pd.DataFrame | None = None,
    rxns_coeffs_df: pd.DataFrame | None = None,
    met_id_prefix: str = "M_",
    rxn_id_prefix: str = "R_",
    overwrite_with_empty: bool = True,
) -> CurationResult:
    """Add or update metabolites / reactions / genes from data tables.

    Each table is optional. ``rxns_df`` and ``rxns_coeffs_df`` must be
    provided together (one describes the per-reaction attributes, the
    other carries the stoichiometric coefficients).

    Parameters
    ----------
    model
        Model to curate in place.
    mets_df, genes_df, rxns_df, rxns_coeffs_df
        Tables matching the schema described in
        :mod:`raven_toolbox.curation`.
    met_id_prefix, rxn_id_prefix
        Prefixes for freshly-generated metabolite / reaction ids
        (e.g. ``s_`` and ``r_`` for yeast-GEM, ``M_`` and ``R_`` for
        the cobrapy / BiGG default). New entity ids are formed by
        finding the largest existing zero-padded suffix matching the
        prefix and incrementing from there.
    overwrite_with_empty
        How an empty cell is applied to an entity that already exists in the model, as
        ``curateModelFromTables``'s ``overwriteWithEmpty``. With True (default) a column
        present in the table overwrites the stored value, so an empty cell clears it: formula,
        gene short name, reaction name, subsystem, EC code, notes and InchI become empty. For
        charge and confidence score an empty cell becomes 0 when at least one cell of the
        column has a value, and a column without any value is not written. With False an
        empty cell leaves the stored value as it is. Independent of this option, an empty
        cell never changes a gene rule, a lower or upper bound (no bound given) or a MIRIAM
        annotation, and a new entity takes only the cells that have a value.

    Returns
    -------
    A :class:`CurationResult` summarising the changes.
    """
    result = CurationResult()

    if mets_df is not None:
        mets_df = _canonical_columns(mets_df, _MET_ALIASES)
    if genes_df is not None:
        genes_df = _canonical_columns(genes_df, _GENE_ALIASES)
    if rxns_df is not None:
        rxns_df = _canonical_columns(rxns_df, _RXN_ALIASES, lower_rest=True)
    if rxns_coeffs_df is not None:
        rxns_coeffs_df = _canonical_columns(rxns_coeffs_df, _COEFF_ALIASES)

    if mets_df is not None:
        _apply_mets(model, mets_df, met_id_prefix, result, overwrite_with_empty)
    if genes_df is not None:
        _apply_genes(model, genes_df, result, overwrite_with_empty)

    if (rxns_df is None) != (rxns_coeffs_df is None):
        raise ValueError(
            "rxns_df and rxns_coeffs_df must be provided together; got "
            f"rxns_df={'set' if rxns_df is not None else 'None'}, "
            f"rxns_coeffs_df="
            f"{'set' if rxns_coeffs_df is not None else 'None'}."
        )
    if rxns_df is not None:
        _apply_reactions(model, rxns_df, rxns_coeffs_df, rxn_id_prefix, result, overwrite_with_empty)
    return result


def batch_curate_from_tsv(
    model: cobra.Model,
    *,
    mets_tsv: str | Path | None = None,
    genes_tsv: str | Path | None = None,
    rxns_tsv: str | Path | None = None,
    rxns_coeffs_tsv: str | Path | None = None,
    met_id_prefix: str = "M_",
    rxn_id_prefix: str = "R_",
    overwrite_with_empty: bool = True,
) -> CurationResult:
    """File-path convenience wrapper for :func:`batch_curate`.

    Each path is optional. TSVs are read with pandas' default
    ``read_csv(sep='\\t')``; empty cells become ``NaN`` (not the
    empty string), which the engine treats as "skip this field".
    """
    def _read(path):
        return pd.read_csv(path, sep="\t") if path is not None else None

    return batch_curate(
        model,
        mets_df=_read(mets_tsv),
        genes_df=_read(genes_tsv),
        rxns_df=_read(rxns_tsv),
        rxns_coeffs_df=_read(rxns_coeffs_tsv),
        met_id_prefix=met_id_prefix,
        rxn_id_prefix=rxn_id_prefix,
        overwrite_with_empty=overwrite_with_empty,
    )


# --- metabolites ------------------------------------------------------

def _apply_mets(
    model: cobra.Model,
    df: pd.DataFrame,
    id_prefix: str,
    result: CurationResult,
    overwrite_with_empty: bool,
) -> None:
    miriam_cols = [c for c in df.columns if c not in DEFAULT_CORE_MET_COLUMNS]
    any_charge = "charge" in df.columns and bool(df["charge"].map(_has_value).any())
    name_index = _name_compartment_index(model)

    new_rows: list[pd.Series] = []
    for _, row in df.iterrows():
        name = row["metNames"]
        comp = row["comps"]
        existing = name_index.get((name, comp))
        if existing is not None:
            _update_metabolite(existing, row, miriam_cols, overwrite_with_empty, any_charge)
            result.updated_metabolites.append(existing.id)
        else:
            new_rows.append(row)

    if new_rows:
        new_mets = _build_new_metabolites(model, new_rows, miriam_cols, id_prefix)
        model.add_metabolites(new_mets)
        result.added_metabolites.extend(m.id for m in new_mets)

    if result.updated_metabolites:
        warnings.warn(
            f"{len(result.updated_metabolites)} metabolite(s) already "
            "existed and were overwritten: "
            f"{result.updated_metabolites[:5]}"
            f"{'...' if len(result.updated_metabolites) > 5 else ''}",
            stacklevel=4,
        )


def _canonical_columns(df: pd.DataFrame, aliases: Mapping[str, tuple[str, ...]], lower_rest: bool = False) -> pd.DataFrame:
    """Rename the columns of ``df`` that match an alias (case-insensitive) to the canonical name.

    The first column matching a canonical name wins and the others stay as they are.
    With ``lower_rest`` a column that matches a core reaction column in another case
    is renamed to that spelling.
    """
    lookup = {a.lower(): canonical for canonical, names in aliases.items() for a in names}
    if lower_rest:
        lookup.update({c.lower(): c for c in DEFAULT_CORE_RXN_COLUMNS})
    rename: dict[str, str] = {}
    taken: set[str] = set()
    for col in df.columns:
        canonical = lookup.get(str(col).strip().lower())
        if canonical is not None and canonical not in taken:
            rename[col] = canonical
            taken.add(canonical)
    return df.rename(columns=rename)


def _name_compartment_index(model: cobra.Model) -> dict[tuple[str, str], cobra.Metabolite]:
    return {(m.name, m.compartment): m for m in model.metabolites}


def _update_metabolite(
    met: cobra.Metabolite,
    row: pd.Series,
    miriam_cols: list[str],
    overwrite_with_empty: bool,
    any_charge: bool,
) -> None:
    if _is_written(row, "formula", overwrite_with_empty):
        met.formula = _text(row["formula"])
    if _is_written(row, "charge", overwrite_with_empty) and any_charge:
        met.charge = _coerce_int(row["charge"]) if _has_value(row["charge"]) else 0
    _apply_met_notes(met, row, overwrite_with_empty)
    _apply_miriam(met, row, miriam_cols)


def _build_new_metabolites(
    model: cobra.Model,
    rows: list[pd.Series],
    miriam_cols: list[str],
    id_prefix: str,
) -> list[cobra.Metabolite]:
    ids = _generate_ids(model.metabolites, id_prefix, len(rows))
    new_mets: list[cobra.Metabolite] = []
    for new_id, row in zip(ids, rows, strict=True):
        met = cobra.Metabolite(
            id=new_id,
            name=str(row["metNames"]),
            compartment=str(row["comps"]),
        )
        if _has_value(row.get("formula")):
            met.formula = str(row["formula"])
        if _has_value(row.get("charge")):
            met.charge = _coerce_int(row["charge"])
        _apply_met_notes(met, row)
        _apply_miriam(met, row, miriam_cols)
        new_mets.append(met)
    return new_mets


# --- genes ------------------------------------------------------------

def _apply_genes(
    model: cobra.Model, df: pd.DataFrame, result: CurationResult, overwrite_with_empty: bool
) -> None:
    miriam_cols = [c for c in df.columns if c not in DEFAULT_CORE_GENE_COLUMNS]
    existing_genes = {g.id: g for g in model.genes}

    new_rows: list[pd.Series] = []
    for _, row in df.iterrows():
        gid = str(row["genes"])
        existing = existing_genes.get(gid)
        if existing is not None:
            _update_gene(existing, row, miriam_cols, overwrite_with_empty)
            result.updated_genes.append(gid)
        else:
            new_rows.append(row)

    if new_rows:
        # cobrapy doesn't have a direct "add a free-standing gene" API;
        # genes are typically added via reactions. Use cobra.Gene + an
        # explicit registration through the DictList.
        from cobra.core.gene import Gene

        for row in new_rows:
            g = Gene(str(row["genes"]))
            if _has_value(row.get("geneShortNames")):
                g.name = str(row["geneShortNames"])
            _apply_miriam(g, row, miriam_cols)
            model.genes.append(g)
            result.added_genes.append(g.id)

    if result.updated_genes:
        warnings.warn(
            f"{len(result.updated_genes)} gene(s) already existed and "
            f"were overwritten: {result.updated_genes[:5]}"
            f"{'...' if len(result.updated_genes) > 5 else ''}",
            stacklevel=4,
        )


def _update_gene(gene, row: pd.Series, miriam_cols: list[str], overwrite_with_empty: bool) -> None:
    if _is_written(row, "geneShortNames", overwrite_with_empty):
        gene.name = _text(row["geneShortNames"])
    _apply_miriam(gene, row, miriam_cols)


# --- reactions --------------------------------------------------------

def _apply_reactions(
    model: cobra.Model,
    rxns_df: pd.DataFrame,
    coeffs_df: pd.DataFrame,
    id_prefix: str,
    result: CurationResult,
    overwrite_with_empty: bool,
) -> None:
    miriam_cols = [c for c in rxns_df.columns if c not in DEFAULT_CORE_RXN_COLUMNS and c != "rxnIdx"]
    any_score = (
        "rxnConfidenceScores" in rxns_df.columns
        and bool(rxns_df["rxnConfidenceScores"].map(_has_value).any())
    )
    _validate_rxns_coeffs(coeffs_df)
    by_index = "rxnIdx" in rxns_df.columns and "rxnIdx" in coeffs_df.columns
    coeffs_by_key = _group_coeffs(model, coeffs_df, by_index)

    # Build {stoichiometry_signature: existing_reaction} for the lookup
    # of "is this reaction already in the model?".
    by_signature = {
        _stoich_signature(r): r for r in model.reactions
    }

    new_rows: list[tuple[pd.Series, dict[cobra.Metabolite, float]]] = []
    for _, row in rxns_df.iterrows():
        rxn_name = row["rxnNames"]
        key = _reaction_key(row, by_index)
        if key not in coeffs_by_key:
            raise ValueError(
                f"Reaction {rxn_name!r} in rxns_df has no matching "
                "row(s) in rxns_coeffs_df."
            )
        stoich = coeffs_by_key[key]
        sig = _stoich_signature_from_dict(stoich)
        existing = by_signature.get(sig)
        if existing is not None:
            _update_reaction(existing, row, miriam_cols, overwrite_with_empty, any_score)
            result.updated_reactions.append(existing.id)
        else:
            new_rows.append((row, stoich))

    if new_rows:
        new_rxns = _build_new_reactions(model, new_rows, miriam_cols, id_prefix)
        model.add_reactions(new_rxns)
        result.added_reactions.extend(r.id for r in new_rxns)

    if result.updated_reactions:
        warnings.warn(
            f"{len(result.updated_reactions)} reaction(s) had matching "
            "stoichiometry in the model and were overwritten: "
            f"{result.updated_reactions[:5]}"
            f"{'...' if len(result.updated_reactions) > 5 else ''}",
            stacklevel=4,
        )


def _validate_rxns_coeffs(df: pd.DataFrame) -> None:
    missing = set(DEFAULT_CORE_RXN_COEFFS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"rxns_coeffs_df is missing required columns: {sorted(missing)}"
        )


def _reaction_key(row: pd.Series, by_index: bool):
    """Key linking a reaction row to its coefficient rows: ``(rxnIdx, name)`` or the name."""
    if by_index:
        return (str(row["rxnIdx"]).strip(), str(row["rxnNames"]))
    return str(row["rxnNames"])


def _group_coeffs(
    model: cobra.Model,
    coeffs_df: pd.DataFrame,
    by_index: bool,
) -> dict:
    """Group coefficient rows by reaction and resolve the metabolites.

    The key is ``(rxnIdx, name)`` when ``by_index`` and the reaction name otherwise,
    so reactions that share a name stay apart when the tables number them.

    Each value is a ``{metabolite_object: coefficient}`` dict ready for
    ``cobra.Reaction.add_metabolites``. Metabolites are looked up by
    ``(name, comp)``; if a coefficient references an unknown
    metabolite, that's an error (the caller should add the metabolite
    via ``mets_df`` first).
    """
    name_index = _name_compartment_index(model)
    grouped: dict = {}
    for _, row in coeffs_df.iterrows():
        rxn_name = str(row["rxnNames"])
        met_name = str(row["metNames"])
        comp = str(row["comps"])
        coef = float(row["coefficient"])
        met = name_index.get((met_name, comp))
        if met is None:
            raise ValueError(
                f"Reaction {rxn_name!r} references metabolite "
                f"{met_name!r}[{comp}] which is not in the model. "
                "Add it via mets_df first, or include it in the same "
                "batch_curate call."
            )
        grouped.setdefault(_reaction_key(row, by_index), {})[met] = coef
    return grouped


def _stoich_signature(rxn: cobra.Reaction) -> frozenset:
    return frozenset((m.id, c) for m, c in rxn.metabolites.items())


def _stoich_signature_from_dict(stoich: dict[cobra.Metabolite, float]) -> frozenset:
    return frozenset((m.id, c) for m, c in stoich.items())


def _update_reaction(
    rxn: cobra.Reaction,
    row: pd.Series,
    miriam_cols: list[str],
    overwrite_with_empty: bool,
    any_score: bool,
) -> None:
    if _is_written(row, "rxnNames", overwrite_with_empty):
        rxn.name = _text(row["rxnNames"])
    if _has_value(row.get("grRules")):
        rxn.gene_reaction_rule = str(row["grRules"])
    if _has_value(row.get("lb")):
        rxn.lower_bound = float(row["lb"])
    if _has_value(row.get("ub")):
        rxn.upper_bound = float(row["ub"])
    if _is_written(row, "subSystems", overwrite_with_empty):
        rxn.subsystem = subsystem_to_str(row["subSystems"]) if _has_value(row["subSystems"]) else ""
    if _is_written(row, "eccodes", overwrite_with_empty):
        if _has_value(row["eccodes"]):
            rxn.annotation["ec-code"] = str(row["eccodes"])
        else:
            rxn.annotation.pop("ec-code", None)
    _apply_rxn_notes(rxn, row, overwrite_with_empty, any_score)
    _apply_miriam(rxn, row, miriam_cols)


def _stored_default_ub(model: cobra.Model) -> float:
    """The ``defaultUB`` stored in the model's ``notes["metaData"]``, else in
    ``model.annotation``, else +inf.

    A reaction without ``ub`` takes this bound, as RAVEN's ``addRxns`` takes
    ``model.annotation.defaultUB``. A reaction without ``lb`` keeps ``-inf``: ``addRxns``
    sets the lower bound of reactions given by ``mets`` and ``stoichCoeffs`` to ``-inf``
    before it reads ``defaultLB``, and ``curateModelFromTables`` adds reactions that way.
    """
    meta = model.notes.get("metaData")
    if isinstance(meta, dict) and meta.get("defaultUB") not in (None, ""):
        return float(meta["defaultUB"])
    annotation = getattr(model, "annotation", None)
    if isinstance(annotation, dict) and annotation.get("defaultUB") not in (None, ""):
        return float(annotation["defaultUB"])
    return math.inf


def _build_new_reactions(
    model: cobra.Model,
    rows: list[tuple[pd.Series, dict[cobra.Metabolite, float]]],
    miriam_cols: list[str],
    id_prefix: str,
) -> list[cobra.Reaction]:
    ids = _generate_ids(model.reactions, id_prefix, len(rows))
    default_ub = _stored_default_ub(model)
    new_rxns: list[cobra.Reaction] = []
    for new_id, (row, stoich) in zip(ids, rows, strict=True):
        rxn = cobra.Reaction(
            id=new_id,
            name=str(row["rxnNames"]),
            lower_bound=float(row["lb"]) if _has_value(row.get("lb")) else -math.inf,
            upper_bound=float(row["ub"]) if _has_value(row.get("ub")) else default_ub,
        )
        rxn.add_metabolites(stoich)
        if _has_value(row.get("grRules")):
            rxn.gene_reaction_rule = str(row["grRules"])
        if _has_value(row.get("subSystems")):
            rxn.subsystem = subsystem_to_str(row["subSystems"])
        if _has_value(row.get("eccodes")):
            rxn.annotation["ec-code"] = str(row["eccodes"])
        _apply_rxn_notes(rxn, row)
        _apply_miriam(rxn, row, miriam_cols)
        new_rxns.append(rxn)
    return new_rxns


# --- shared helpers ---------------------------------------------------

# Table columns stored in ``.notes``, mapped to the keys read_yaml_model and
# read_mat_model use for the same RAVEN fields (see _MET_FIELDS/_RXN_FIELDS in
# raven_toolbox.io.yaml), so the writers emit them as RAVEN fields.
_MET_NOTE_COLUMNS = (("inchi", "inchis"), ("metNotes", "note"))
_RXN_NOTE_COLUMNS = (("rxnNotes", "note"), ("rxnReferences", "references"))


def _apply_met_notes(met: cobra.Metabolite, row: pd.Series, overwrite_with_empty: bool = False) -> None:
    for col, key in _MET_NOTE_COLUMNS:
        if _is_written(row, col, overwrite_with_empty):
            _set_note(met.notes, key, row[col])
    # An InChI already kept as a MIRIAM annotation is updated as well, so the
    # metabolite does not carry two different structures.
    if _is_written(row, "inchi", overwrite_with_empty) and "inchi" in met.annotation:
        if _has_value(row["inchi"]):
            met.annotation["inchi"] = str(row["inchi"])
        else:
            del met.annotation["inchi"]


def _apply_rxn_notes(
    rxn: cobra.Reaction,
    row: pd.Series,
    overwrite_with_empty: bool = False,
    any_score: bool = True,
) -> None:
    for col, key in _RXN_NOTE_COLUMNS:
        if _is_written(row, col, overwrite_with_empty):
            _set_note(rxn.notes, key, row[col])
    if _is_written(row, "rxnConfidenceScores", overwrite_with_empty) and any_score:
        score = float(row["rxnConfidenceScores"]) if _has_value(row["rxnConfidenceScores"]) else 0.0
        rxn.notes["confidence_score"] = int(score) if score.is_integer() else score


def _apply_miriam(entity, row: pd.Series, miriam_cols: list[str]) -> None:
    for col in miriam_cols:
        value = row.get(col)
        if _has_value(value):
            entity.annotation[col] = str(value).strip()


def _is_written(row: pd.Series, col: str, overwrite_with_empty: bool) -> bool:
    """Whether the cell ``col`` of ``row`` is applied to an existing entity: it has a value,
    or the column is present and empty cells overwrite."""
    if col not in row.index:
        return False
    return overwrite_with_empty or _has_value(row[col])


def _text(v) -> str:
    return str(v) if _has_value(v) else ""


def _set_note(notes: dict, key: str, value) -> None:
    if _has_value(value):
        notes[key] = str(value)
    else:
        notes.pop(key, None)


def _has_value(v) -> bool:
    """Return True if ``v`` is a non-empty, non-NaN cell value."""
    if v is None:
        return False
    if isinstance(v, float) and v != v:  # NaN
        return False
    if isinstance(v, str) and v.strip() == "":
        return False
    return True


def _coerce_int(v) -> int:
    """Pandas reads integer-like strings as floats. Recover the int."""
    return int(round(float(v)))


def _generate_ids(existing, prefix: str, count: int) -> list[str]:
    """Generate ``count`` fresh ids after the largest existing number behind ``prefix``.

    Same rule as :func:`raven_toolbox.utils.ids.generate_new_ids`: with prefix ``s_`` and
    existing ``s_4100``, the next ids are ``s_4101``, ``s_4102``, … The suffix has the width of
    the largest existing one, or 4 digits when no id has the prefix.
    """
    return next_ids([e.id for e in existing], prefix, count)
