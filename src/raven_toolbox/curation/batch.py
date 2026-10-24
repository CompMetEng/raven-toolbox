"""Generic batch-curation engine driven by tabular data."""
from __future__ import annotations

import csv
import logging
import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
from cobra.core.gene import GPR

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
        gene short name, reaction name, subsystem, EC code, notes and InChI become empty. For
        charge and confidence score an empty cell becomes 0 when at least one cell of the
        column has a value, and a column without any value is not written. For a MIRIAM
        column an empty cell removes the namespace from the existing entity when at least
        one existing entity of the table has a value in that column. With False an empty cell
        leaves the stored value as it is. Independent of this option, an empty cell never
        changes a gene rule or a lower or upper bound (no bound given). A new entity takes
        the cells that have a value; an empty charge or confidence score is 0 when the column
        has a value in some row of the table.
        Must be a ``bool`` or ``numpy.bool_``, otherwise :class:`TypeError` is raised.

    Numeric cells (charge, confidence score, lower and upper bound, reversibility, ``rxnIdx``
    and coefficient) are read as RAVEN reads them: an empty cell or ``NaN`` (any case, with an
    optional sign) is not given, and other text must be a plain decimal number or ``Inf``. A
    comma, a hexadecimal number or any expression raises :class:`ValueError`, and nothing is
    evaluated. ``true`` and ``false`` in any case are 1 and 0 in the lower bound, upper bound and
    ``rev`` cells and are not numbers in the other columns. A cell is trimmed as MATLAB's
    ``strtrim`` trims it, after dropping the backspaces at its start as ``textscan`` does; the
    non-breaking space, NUL and the next line character stay in the text.

    A charge must be a finite whole number (``2``, ``-3``, ``2.0`` and ``2e0`` are accepted,
    ``0.5``, ``2.5`` and ``Inf`` raise :class:`ValueError`; nothing is rounded). A ``rev`` cell
    must be 0, 1, ``true`` or ``false`` (any other number raises, although the value is not
    otherwise used). A ``rxnIdx`` cell must be a finite whole number of at least 1; the two
    reaction tables are matched on the index and the name, and the index is compared by value
    (``1``, ``1.0`` and ``1e0`` are the same). A coefficient must be a finite number other than
    zero; two rows of one reaction for the same metabolite add their coefficients. A row of a
    reaction table without any content is ignored (in the coefficient table the content is the
    index, reaction name, metabolite name and compartment); a row with content needs a
    ``rxnIdx`` when the table has that column, and a coefficient row needs a coefficient. A
    metabolite row needs a compartment. A reaction of one table without a row in the other
    raises :class:`ValueError`. An empty MIRIAM column removes no annotation.

    A lower bound above the upper bound (also against the stored bound of an existing
    reaction, since both bounds are set together) raises :class:`ValueError`; RAVEN writes such
    bounds without a message, but a cobra reaction cannot hold them.

    Every check above, the required columns, the metabolites of the coefficient rows and the bounds
    run before the model is touched, so a table that fails leaves the model unchanged. What
    can still fail while the tables are applied is an error raised by cobra itself. A malformed
    ``grRules`` cell raises :class:`ValueError` before anything is applied (cobra would log it
    and store an empty rule); RAVEN stops for some malformed rules of new reactions and stores
    the others as text.

    Returns
    -------
    A :class:`CurationResult` summarising the changes.
    """
    if not isinstance(overwrite_with_empty, (bool, np.bool_)):
        raise TypeError(
            "overwrite_with_empty must be a bool or numpy.bool_, "
            f"got {type(overwrite_with_empty).__name__}"
        )
    overwrite_with_empty = bool(overwrite_with_empty)
    result = CurationResult()

    if (rxns_df is None) != (rxns_coeffs_df is None):
        raise ValueError(
            "rxns_df and rxns_coeffs_df must be provided together; got "
            f"rxns_df={'set' if rxns_df is not None else 'None'}, "
            f"rxns_coeffs_df="
            f"{'set' if rxns_coeffs_df is not None else 'None'}."
        )

    # Validation phase: reads the tables and the model and changes nothing.
    if mets_df is not None:
        mets_df = _prepare_mets(mets_df)
    if genes_df is not None:
        genes_df = _prepare_genes(genes_df)
    plan = None
    if rxns_df is not None:
        rxns_df, rxns_coeffs_df = _prepare_reaction_tables(rxns_df, rxns_coeffs_df)
        _check_reaction_tables(model, mets_df, rxns_df, rxns_coeffs_df)
        plan = _plan_reactions(model, rxns_df, rxns_coeffs_df)

    # Apply phase.
    if mets_df is not None:
        _apply_mets(model, mets_df, met_id_prefix, result, overwrite_with_empty)
    if genes_df is not None:
        _apply_genes(model, genes_df, result, overwrite_with_empty)
    if plan is not None:
        _apply_reactions(model, plan, rxns_df, rxn_id_prefix, result, overwrite_with_empty)
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

    Each path is optional. A table file is read as RAVEN's ``curateModelFromTables`` reads it:
    tab-delimited text with a header row, every cell kept as text (``NA``, ``null`` or ``None``
    stay text and ``22`` stays ``22``), a double quote around a cell, spaces before a cell
    dropped, a trailing tab on a row ignored, and a row with more non-empty cells than the
    header has columns refused with :class:`ValueError`. A line without any text is skipped.
    A repeated header is stored under the name with ``.1`` appended (RAVEN replaces the
    annotation in the order of the columns).
    """
    def _read(path):
        return None if path is None else _read_table(path)

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


# --- reading table files ------------------------------------------------

def _read_table(path) -> pd.DataFrame:
    """A tab-delimited table file as a data frame of text, read like ``readCurationTable``.

    MATLAB's ``textscan`` drops the spaces before a cell (also before its quote) and reads a
    quoted cell up to the closing quote; the ``csv`` module does the same, accepts a cell
    longer than 131072 characters (the limit is lifted while the file is read) and keeps a
    NUL character.
    """
    path = Path(path)
    limit = csv.field_size_limit(2**31 - 1)
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle, delimiter="\t", quotechar='"', skipinitialspace=True)
            lines = [(reader.line_num, row) for row in reader]
    finally:
        csv.field_size_limit(limit)
    lines = [(n, row) for n, row in lines if not (len(row) == 0 or (len(row) == 1 and not _trim(row[0])))]
    if not lines:
        raise ValueError(f"The file {str(path)!r} is empty")
    names: list[str] = []
    for cell in lines[0][1]:
        name = _trim(cell)
        seen = sum(1 for earlier in names if earlier == name or earlier.startswith(f"{name}."))
        names.append(name if name not in names else f"{name}.{seen}")
    width = len(names)
    data = []
    for n, row in lines[1:]:
        if len(row) > width and any(_trim(cell) for cell in row[width:]):
            raise ValueError(
                f"Line {n} of {path.name!r} has {len(row)} cells but the header has {width} columns")
        data.append(list(row[:width]) + [""] * (width - len(row)))
    return pd.DataFrame(data, columns=names, dtype=object)


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
    updates: list[tuple[cobra.Metabolite, pd.Series]] = []
    for _, row in df.iterrows():
        name = row["metNames"]
        comp = row["comps"]
        existing = name_index.get((name, comp))
        if existing is not None:
            updates.append((existing, row))
        else:
            new_rows.append(row)

    active = _miriam_with_values(miriam_cols, [r for _, r in updates])
    for existing, row in updates:
        _update_metabolite(existing, row, miriam_cols, active, overwrite_with_empty, any_charge)
        result.updated_metabolites.append(existing.id)

    if new_rows:
        new_mets = _build_new_metabolites(model, new_rows, miriam_cols, id_prefix, any_charge)
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


def _require_columns(df: pd.DataFrame, columns, label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _prepare_mets(df: pd.DataFrame) -> pd.DataFrame:
    """Metabolite table with canonical column names and a numeric charge column; every cell is
    checked before anything is applied."""
    df = _canonical_columns(df, _MET_ALIASES)
    _require_columns(df, ("metNames", "comps"), "mets_df")
    for position, comp in enumerate(df["comps"], start=1):
        if not _has_value(comp):
            raise ValueError(f"Row {position} of the metsInfo table has no compartment")
    return _parse_numeric_columns(
        df, ("charge",), "metsInfo", checks={"charge": _check_whole_charge})


def _prepare_genes(df: pd.DataFrame) -> pd.DataFrame:
    df = _canonical_columns(df, _GENE_ALIASES)
    _require_columns(df, ("genes",), "genes_df")
    return df


def _row_has_content(row, columns=None) -> bool:
    values = row if columns is None else [row[c] for c in columns]
    return any(_has_value(v) for v in values)


def _check_gene_rules(rxns_df: pd.DataFrame) -> None:
    """Raise :class:`ValueError` for a ``grRules`` cell that is not a valid gene rule.

    cobra logs a malformed rule and stores an empty one, which would lose the rule silently.
    RAVEN stops for some malformed rules of new reactions (``G1 and and (``) and stores others as
    text (``(G1 or G2``); Python refuses every malformed rule.
    """
    if "grRules" not in rxns_df.columns:
        return
    gene_logger = logging.getLogger("cobra.core.gene")
    for _, row in rxns_df.iterrows():
        rule = row["grRules"]
        if not _has_value(rule):
            continue
        disabled = gene_logger.disabled
        gene_logger.disabled = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = GPR.from_string(str(rule))
        finally:
            gene_logger.disabled = disabled
        if parsed.body is None:
            raise ValueError(
                f'The grRules value "{rule}" of reaction {str(row["rxnNames"])!r} is not a valid gene rule')


def _prepare_reaction_tables(rxns_df: pd.DataFrame, coeffs_df: pd.DataFrame):
    """Reaction and coefficient tables with canonical column names and numeric columns.

    A row without any content is ignored (for the coefficient table only the reaction index,
    reaction name, metabolite name and compartment count as content). A row with content needs
    a reaction index when the table has that column, and a coefficient row needs a coefficient.
    """
    rxns_df = _canonical_columns(rxns_df, _RXN_ALIASES, lower_rest=True)
    _require_columns(rxns_df, ("rxnNames",), "rxns_df")
    coeffs_df = _canonical_columns(coeffs_df, _COEFF_ALIASES)
    _validate_rxns_coeffs(coeffs_df)

    rxns_df = rxns_df.reset_index(drop=True)
    rxns_df = rxns_df[[_row_has_content(r) for r in rxns_df.itertuples(index=False)]]
    coeff_content = [c for c in ("rxnIdx", "rxnNames", "metNames", "comps") if c in coeffs_df.columns]
    coeffs_df = coeffs_df.reset_index(drop=True)
    coeffs_df = coeffs_df[[_row_has_content(r, coeff_content) for _, r in coeffs_df.iterrows()]]

    rxns_df = _parse_numeric_columns(
        rxns_df, ("rxnIdx", "lb", "ub", "rev", "rxnConfidenceScores"), "rxnsInfo",
        checks={"rxnIdx": _check_rxn_index, "rev": _check_boolean_rev},
        boolean_columns=("lb", "ub", "rev"))
    coeffs_df = _parse_numeric_columns(
        coeffs_df, ("rxnIdx", "coefficient"), "rxnsCoeffs",
        checks={"rxnIdx": _check_rxn_index, "coefficient": _check_coefficient})
    for df, label in ((rxns_df, "rxnsInfo"), (coeffs_df, "rxnsCoeffs")):
        if "rxnIdx" in df.columns and df["rxnIdx"].isna().any():
            row = int(df.index[df["rxnIdx"].isna()][0]) + 1
            raise ValueError(
                f"Data row {row} of the {label} table has values but its rxnIdx is empty or NaN")
    _check_gene_rules(rxns_df)
    if coeffs_df["coefficient"].isna().any():
        row = int(coeffs_df.index[coeffs_df["coefficient"].isna()][0]) + 1
        raise ValueError(
            f"The coefficient in row {row} of the rxnsCoeffs table is empty or NaN; "
            "every row with a reaction, metabolite or compartment needs a coefficient")
    return rxns_df, coeffs_df


def _name_compartment_index(model: cobra.Model) -> dict[tuple[str, str], cobra.Metabolite]:
    return {(m.name, m.compartment): m for m in model.metabolites}


def _update_metabolite(
    met: cobra.Metabolite,
    row: pd.Series,
    miriam_cols: list[str],
    active_miriam: set[str],
    overwrite_with_empty: bool,
    any_charge: bool,
) -> None:
    if _is_written(row, "formula", overwrite_with_empty):
        met.formula = _text(row["formula"])
    if _is_written(row, "charge", overwrite_with_empty) and any_charge:
        met.charge = _coerce_int(row["charge"]) if _has_value(row["charge"]) else 0
    _apply_met_notes(met, row, overwrite_with_empty)
    _apply_miriam(met, row, miriam_cols, active_miriam, overwrite_with_empty)


def _build_new_metabolites(
    model: cobra.Model,
    rows: list[pd.Series],
    miriam_cols: list[str],
    id_prefix: str,
    any_charge: bool,
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
        elif any_charge:
            met.charge = 0
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
    updates: list[tuple[cobra.Gene, pd.Series]] = []
    for _, row in df.iterrows():
        gid = str(row["genes"])
        existing = existing_genes.get(gid)
        if existing is not None:
            updates.append((existing, row))
        else:
            new_rows.append(row)

    active = _miriam_with_values(miriam_cols, [r for _, r in updates])
    for existing, row in updates:
        _update_gene(existing, row, miriam_cols, active, overwrite_with_empty)
        result.updated_genes.append(existing.id)

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


def _update_gene(
    gene, row: pd.Series, miriam_cols: list[str], active_miriam: set[str], overwrite_with_empty: bool
) -> None:
    if _is_written(row, "geneShortNames", overwrite_with_empty):
        gene.name = _text(row["geneShortNames"])
    _apply_miriam(gene, row, miriam_cols, active_miriam, overwrite_with_empty)


# --- reactions --------------------------------------------------------

@dataclass
class _ReactionPlan:
    """What the reaction tables do to the model: existing reactions to update and new ones to add.

    ``new`` pairs a table row with its stoichiometry as ``{(metabolite name, compartment): coefficient}``;
    the metabolites are looked up when the reaction is built, after the metabolite table is applied.
    """

    updates: list[tuple[cobra.Reaction, pd.Series]]
    new: list[tuple[pd.Series, dict[tuple[str, str], float]]]


def _plan_reactions(model: cobra.Model, rxns_df: pd.DataFrame, coeffs_df: pd.DataFrame) -> _ReactionPlan:
    """Decide which reactions of ``rxns_df`` exist in the model, and check their bounds.

    A reaction whose stoichiometry matches a model reaction (compared by metabolite id and
    coefficient) is an update; every other one is new, also when it uses a metabolite that
    ``mets_df`` adds. Raises :class:`ValueError` for a lower bound above the upper bound.
    """
    by_index = "rxnIdx" in rxns_df.columns and "rxnIdx" in coeffs_df.columns
    coeffs_by_key = _group_coeffs(coeffs_df, by_index)
    name_index = _name_compartment_index(model)
    by_signature = {_stoich_signature(r): r for r in model.reactions}
    default_ub = _stored_default_ub(model)

    updates: list[tuple[cobra.Reaction, pd.Series]] = []
    new: list[tuple[pd.Series, dict[tuple[str, str], float]]] = []
    for _, row in rxns_df.iterrows():
        stoich = coeffs_by_key[_reaction_key(row, by_index)]
        existing = None
        if all(key in name_index for key in stoich):
            existing = by_signature.get(
                frozenset((name_index[key].id, coef) for key, coef in stoich.items()))
        if existing is not None:
            lb = float(row["lb"]) if _has_value(row.get("lb")) else existing.lower_bound
            ub = float(row["ub"]) if _has_value(row.get("ub")) else existing.upper_bound
            updates.append((existing, row))
        else:
            lb = float(row["lb"]) if _has_value(row.get("lb")) else -math.inf
            ub = float(row["ub"]) if _has_value(row.get("ub")) else default_ub
            new.append((row, stoich))
        if lb > ub:
            raise ValueError(
                f"The lower bound {lb:g} of reaction {str(row['rxnNames'])!r} is larger than its "
                f"upper bound {ub:g}")
    return _ReactionPlan(updates, new)


def _apply_reactions(
    model: cobra.Model,
    plan: _ReactionPlan,
    rxns_df: pd.DataFrame,
    id_prefix: str,
    result: CurationResult,
    overwrite_with_empty: bool,
) -> None:
    miriam_cols = [c for c in rxns_df.columns if c not in DEFAULT_CORE_RXN_COLUMNS and c != "rxnIdx"]
    any_score = (
        "rxnConfidenceScores" in rxns_df.columns
        and bool(rxns_df["rxnConfidenceScores"].map(_has_value).any())
    )

    active = _miriam_with_values(miriam_cols, [r for _, r in plan.updates])
    for existing, row in plan.updates:
        _update_reaction(existing, row, miriam_cols, active, overwrite_with_empty, any_score)
        result.updated_reactions.append(existing.id)

    if plan.new:
        name_index = _name_compartment_index(model)
        rows = [(row, {name_index[key]: coef for key, coef in stoich.items()})
                for row, stoich in plan.new]
        new_rxns = _build_new_reactions(model, rows, miriam_cols, id_prefix, any_score)
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


def _check_reaction_tables(model, mets_df, rxns_df, coeffs_df) -> None:
    """Raise :class:`ValueError` before anything is applied when a coefficient row names a
    metabolite that neither the model nor ``mets_df`` has, or when a reaction of one table has
    no row in the other."""
    by_index = "rxnIdx" in rxns_df.columns and "rxnIdx" in coeffs_df.columns
    known = set(_name_compartment_index(model))
    if mets_df is not None:
        known |= {(str(n), str(c)) for n, c in zip(mets_df["metNames"], mets_df["comps"],
                                                  strict=False)}
    coeff_keys = set()
    for _, row in coeffs_df.iterrows():
        if (str(row["metNames"]), str(row["comps"])) not in known:
            raise ValueError(
                f"Reaction {str(row['rxnNames'])!r} references metabolite "
                f"{str(row['metNames'])!r}[{str(row['comps'])}] which is not in the model. "
                "Add it via mets_df first, or include it in the same "
                "batch_curate call."
            )
        coeff_keys.add(_reaction_key(row, by_index))
    rxn_keys = {_reaction_key(row, by_index) for _, row in rxns_df.iterrows()}
    for _, row in rxns_df.iterrows():
        if _reaction_key(row, by_index) not in coeff_keys:
            raise ValueError(
                f"Reaction {_reaction_label(row, by_index)} in rxns_df has no matching "
                "row(s) in rxns_coeffs_df."
            )
    for _, row in coeffs_df.iterrows():
        if _reaction_key(row, by_index) not in rxn_keys:
            raise ValueError(
                f"Reaction {_reaction_label(row, by_index)} in rxns_coeffs_df has no matching "
                "row in rxns_df."
            )


def _reaction_label(row: pd.Series, by_index: bool) -> str:
    """The identifiers of a reaction row for an error message: its name and, when the tables are
    numbered, its index."""
    if by_index:
        return f"{str(row['rxnNames'])!r} (rxnIdx {_reaction_key(row, by_index)[0]})"
    return repr(str(row["rxnNames"]))


def _validate_rxns_coeffs(df: pd.DataFrame) -> None:
    missing = set(DEFAULT_CORE_RXN_COEFFS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"rxns_coeffs_df is missing required columns: {sorted(missing)}"
        )


def _reaction_key(row: pd.Series, by_index: bool):
    """Key linking a reaction row to its coefficient rows: ``(rxnIdx, name)`` or the name.

    The index is compared by value, so ``1``, ``1.0`` and ``1e0`` are the same index.
    """
    if by_index:
        return (int(row["rxnIdx"]), str(row["rxnNames"]))
    return str(row["rxnNames"])


def _group_coeffs(coeffs_df: pd.DataFrame, by_index: bool) -> dict:
    """Group coefficient rows by reaction.

    The key is ``(rxnIdx, name)`` when ``by_index`` and the reaction name otherwise, so
    reactions that share a name stay apart when the tables number them. Each value maps
    ``(metabolite name, compartment)`` to the coefficient; two rows of one reaction for the
    same metabolite add their coefficients, as RAVEN's ``addRxns`` does.
    """
    grouped: dict = {}
    for _, row in coeffs_df.iterrows():
        stoich = grouped.setdefault(_reaction_key(row, by_index), {})
        key = (str(row["metNames"]), str(row["comps"]))
        stoich[key] = stoich.get(key, 0.0) + float(row["coefficient"])
    return grouped


def _stoich_signature(rxn: cobra.Reaction) -> frozenset:
    return frozenset((m.id, c) for m, c in rxn.metabolites.items())


def _update_reaction(
    rxn: cobra.Reaction,
    row: pd.Series,
    miriam_cols: list[str],
    active_miriam: set[str],
    overwrite_with_empty: bool,
    any_score: bool,
) -> None:
    if _is_written(row, "rxnNames", overwrite_with_empty):
        rxn.name = _text(row["rxnNames"])
    if _has_value(row.get("grRules")):
        rxn.gene_reaction_rule = str(row["grRules"])
    if _has_value(row.get("lb")) or _has_value(row.get("ub")):
        # Both bounds are set together, so that a pair that is valid as a whole is accepted
        # whatever the stored bounds are.
        rxn.bounds = (
            float(row["lb"]) if _has_value(row.get("lb")) else rxn.lower_bound,
            float(row["ub"]) if _has_value(row.get("ub")) else rxn.upper_bound,
        )
    if _is_written(row, "subSystems", overwrite_with_empty):
        rxn.subsystem = subsystem_to_str(row["subSystems"]) if _has_value(row["subSystems"]) else ""
    if _is_written(row, "eccodes", overwrite_with_empty):
        if _has_value(row["eccodes"]):
            rxn.annotation["ec-code"] = str(row["eccodes"])
        else:
            rxn.annotation.pop("ec-code", None)
    _apply_rxn_notes(rxn, row, overwrite_with_empty, any_score)
    _apply_miriam(rxn, row, miriam_cols, active_miriam, overwrite_with_empty)


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
    any_score: bool,
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
        if any_score and not _has_value(row.get("rxnConfidenceScores")):
            rxn.notes["confidence_score"] = 0
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


def _miriam_with_values(miriam_cols: list[str], rows: list[pd.Series]) -> set[str]:
    """The MIRIAM columns with a value in at least one of ``rows`` (the existing entities of
    the table); an empty cell of such a column removes the namespace from its entity."""
    return {c for c in miriam_cols if any(_has_value(r.get(c)) for r in rows)}


def _apply_miriam(
    entity,
    row: pd.Series,
    miriam_cols: list[str],
    active_miriam: frozenset[str] | set[str] = frozenset(),
    overwrite_with_empty: bool = False,
) -> None:
    for col in miriam_cols:
        value = row.get(col)
        if _has_value(value):
            entity.annotation[col] = _trim(value)
        elif overwrite_with_empty and col in active_miriam:
            entity.annotation.pop(col, None)


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


_DECIMAL = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", re.ASCII)
_NAN = re.compile(r"[+-]?nan")
_INF = re.compile(r"[+-]?inf")

#: The characters MATLAB's ``strtrim`` removes from both ends of a cell: the code points 9 to 13
#: and 28 to 32 (tab, line feed, vertical tab, form feed, carriage return, the four separators
#: and the space), 5760, 8192 to 8198, 8200 to 8202, 8232, 8233, 8287 and 12288. NUL (0), the
#: non-breaking space (160), the next line character (133), the figure space (8199) and the
#: narrow no-break space (8239) are not removed.
_BLANKS = "".join(chr(c) for c in (
    *range(9, 14), *range(28, 33), 5760, *range(8192, 8199), *range(8200, 8203), 8232, 8233,
    8287, 12288))


def _trim(text) -> str:
    """A cell as RAVEN reads it: ``strtrim`` of the text, after MATLAB's ``textscan`` has dropped
    the backspace characters at its start."""
    text = str(text).strip(_BLANKS)
    if text.startswith("\b"):
        text = text.lstrip("\b").strip(_BLANKS)
    return text


def _number_cell(value, field: str, label: str, allow_boolean: bool = False) -> float:
    """Number in a table cell, as RAVEN reads it: NaN for an empty cell or a cell saying NaN.

    ``true`` and ``false`` in any case are 1 and 0 where ``allow_boolean`` is set (the bound and
    reversibility columns) and are not numbers elsewhere. Other text must be a plain decimal
    number (digits with an optional sign, decimal point and exponent), ``Inf`` or ``NaN`` with
    an optional sign; a comma, a hexadecimal number and any expression raise
    :class:`ValueError` and nothing is evaluated.
    """
    if not _has_value(value):
        return math.nan
    if isinstance(value, (bool, np.bool_)):
        if allow_boolean:
            return float(value)
        raise ValueError(f'The {field} value "{value}" in the {label} table is not a number')
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = _trim(value)
    low = text.lower()
    if low in ("true", "false") and allow_boolean:
        return 1.0 if low == "true" else 0.0
    if _NAN.fullmatch(low):
        return math.nan
    if _INF.fullmatch(low):
        return float(low)
    if _DECIMAL.fullmatch(text):
        return float(text)
    raise ValueError(f'The {field} value "{text}" in the {label} table is not a number')


def _check_whole_charge(number: float, text: str, field: str, label: str) -> None:
    """A charge is a finite whole number: ``0.5``, ``2.5`` and ``Inf`` raise, nothing is rounded."""
    if not math.isnan(number) and (not math.isfinite(number) or number != math.floor(number)):
        raise ValueError(
            f'The {field} value "{text}" in the {label} table is not a finite whole number')


def _check_boolean_rev(number: float, text: str, field: str, label: str) -> None:
    """A reversibility cell is 0, 1, ``true`` or ``false``; other numbers raise."""
    if not math.isnan(number) and number not in (0.0, 1.0):
        raise ValueError(
            f'The {field} value "{text}" in the {label} table must be 0, 1, true or false')


def _check_rxn_index(number: float, text: str, field: str, label: str) -> None:
    """A reaction index is a finite whole number of at least 1; an empty or NaN cell is not given."""
    if not math.isnan(number) and (not math.isfinite(number) or number != math.floor(number)
                                   or number < 1):
        raise ValueError(
            f'The {field} value "{text}" in the {label} table is not a whole number of at least 1')


def _check_coefficient(number: float, text: str, field: str, label: str) -> None:
    """A coefficient is a finite number other than zero; an empty or NaN cell is refused later."""
    if not math.isnan(number) and (not math.isfinite(number) or number == 0):
        raise ValueError(
            f'The {field} value "{text}" in the {label} table must be a finite number other than 0')


def _parse_numeric_columns(
    df: pd.DataFrame, columns, label: str, checks=None, boolean_columns=()
) -> pd.DataFrame:
    """Copy of ``df`` whose numeric columns hold floats, NaN where the cell is empty.

    ``checks`` maps a column to a function ``(number, text, column, label)`` that raises
    :class:`ValueError` for a number the column does not accept. Only the columns in
    ``boolean_columns`` read ``true`` and ``false``.
    """
    present = [c for c in columns if c in df.columns]
    if not present:
        return df
    checks = checks or {}
    df = df.copy()
    for column in present:
        numbers = []
        for v in df[column]:
            number = _number_cell(v, column, label, column in boolean_columns)
            if column in checks:
                checks[column](number, _trim(v) if _has_value(v) else "", column, label)
            numbers.append(number)
        df[column] = pd.Series(numbers, index=df.index, dtype=float)
    return df


def _has_value(v) -> bool:
    """Return True if ``v`` is a non-empty, non-NaN cell value."""
    if v is None or v is pd.NA or v is pd.NaT:
        return False
    if isinstance(v, (float, np.floating)) and v != v:  # NaN
        return False
    if isinstance(v, str) and _trim(v) == "":
        return False
    return True


def _coerce_int(v) -> int:
    """Pandas reads integer-like strings as floats. Recover the int.

    The charge column has been checked to hold finite whole numbers; any other value raises
    :class:`ValueError` here too, and nothing is rounded.
    """
    value = float(v)
    if not math.isfinite(value) or value != math.floor(value):
        raise ValueError(f"The charge {v!r} is not a finite whole number")
    return int(value)


def _generate_ids(existing, prefix: str, count: int) -> list[str]:
    """Generate ``count`` fresh ids after the largest existing number behind ``prefix``.

    Same rule as :func:`raven_toolbox.utils.ids.generate_new_ids`: with prefix ``s_`` and
    existing ``s_4100``, the next ids are ``s_4101``, ``s_4102``, … The suffix has the width of
    the largest existing one, or 4 digits when no id has the prefix.
    """
    return next_ids([e.id for e in existing], prefix, count)
