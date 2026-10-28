"""Generic batch-curation engine driven by tabular data."""
from __future__ import annotations

import csv
import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import cobra
import numpy as np
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
    otherwise used). A ``rxnIdx`` cell must be a finite whole number of at least 1; each index
    may be used by one row of the reaction table (the coefficient table repeats it, one row per
    metabolite), otherwise :class:`ValueError` names the index and the data rows. The two
    reaction tables are matched on the index and the name, and the index is compared by value
    (``1``, ``1.0`` and ``1e0`` are the same). A coefficient must be a finite number other than
    zero; two rows of one reaction for the same metabolite add their coefficients. A row of a
    reaction table without any content is ignored (in the coefficient table the content is the
    index, reaction name, metabolite name and compartment); a row with content needs a
    ``rxnIdx`` when the table has that column, and a coefficient row needs a coefficient. A
    metabolite row needs a compartment. A reaction of one table without a row in the other
    raises :class:`ValueError`. An empty MIRIAM column removes no annotation.

    A lower bound above the upper bound (also against the stored bound of an existing
    reaction, since both bounds are set together) raises :class:`ValueError`. RAVEN's
    ``curateModelFromTables`` writes such bounds as given and ``solveLP`` rejects the model when
    it is run ("Invalid set of bounds for reaction(s): ..."); a cobra reaction cannot hold them.

    Every check above, the required columns, the metabolites of the coefficient rows, the bounds
    (several rows that update one reaction are checked one after the other), a gene listed twice
    in ``genes_df`` and the gene rules run before the model is touched, so a table that fails
    leaves the model unchanged. What is not checked beforehand is an error that cobra itself
    raises while the tables are applied.

    A ``grRules`` cell must be a well-formed gene rule, in the grammar of RAVEN's
    ``parseGrRule``: genes joined by ``and``/``or`` in any case or by ``&``, ``&&``, ``|`` and
    ``||``, with balanced parentheses. A rule with an operator that lacks an operand, empty
    parentheses or two genes without an operator between them raises :class:`ValueError`, as it
    raises ``RAVEN:badInput`` there. The rule of a new reaction that names a gene which neither
    the model nor ``genes_df`` has also raises :class:`ValueError`; RAVEN's ``addRxns`` is
    called without permission to add genes and stops with an error of its own. The rule of an
    existing reaction may name a gene the model does not have. A gene id that ``genes_df``
    lists more than once raises :class:`ValueError`, whether the model has the gene or not.

    A reaction table with a header and no row is empty: when both reaction tables are empty
    nothing happens, and a coefficient row without a reaction row (or the reverse) raises
    :class:`ValueError`. RAVEN matches the stoichiometry by ``rxnIdx`` alone, Python by the
    index and the name: for the rows ``1 A`` and ``1 B`` with coefficient rows of ``A`` and ``B``
    RAVEN adds two reactions that both take all coefficients of index 1.

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
        _check_genes(model, genes_df)
    plan = None
    if rxns_df is not None:
        rxns_df, rxns_coeffs_df = _prepare_reaction_tables(rxns_df, rxns_coeffs_df)
        _check_reaction_tables(model, mets_df, rxns_df, rxns_coeffs_df)
        plan = _plan_reactions(model, rxns_df, rxns_coeffs_df, genes_df)

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
    the first line is the header (also when it is blank, which leaves no usable column) and
    its names are trimmed and lose one pair of surrounding double quotes; a column with an empty
    header is ignored; when two columns match the same core name the first one is used and the
    others are ignored, and a repeated name of another column gets ``.1``, ``.2`` ... appended
    (RAVEN uses both columns). The rest is tab-delimited text with every cell kept as text
    (``NA``, ``null`` or ``None`` stay text and ``22`` stays ``22``), a double quote around a
    cell, spaces before a cell dropped, a trailing tab on a row ignored, and a row with more
    non-empty cells than the header has columns refused with :class:`ValueError` (RAVEN wraps
    the extra cell into a new row). A line without any text is skipped; RAVEN makes an empty row
    from a line that holds only ``""`` or from a row that ends in two or more tabs.
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

    The first line of the file is the header, also when it is blank: it is split at the tabs,
    each name is trimmed and one pair of double quotes around it is removed, and a column with
    an empty name is ignored. MATLAB's ``textscan`` reads the rest: it drops the spaces before
    a cell (also before its quote) and reads a quoted cell up to the closing quote; the ``csv``
    module does the same, accepts a cell longer than 131072 characters (the limit is lifted
    while the file is read) and keeps a NUL character. Repeated header names stay repeated in
    the data frame.
    """
    path = Path(path)
    limit = csv.field_size_limit(2**31 - 1)
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            header_line = handle.readline()
            if header_line == "":
                raise ValueError(f"The file {str(path)!r} is empty")
            reader = csv.reader(handle, delimiter="\t", quotechar='"', skipinitialspace=True)
            lines = [(reader.line_num + 1, row) for row in reader]
    finally:
        csv.field_size_limit(limit)
    names = [re.sub(r'^"(.*)"$', r"\1", cell.strip(_BLANKS))
             for cell in header_line.replace("\ufeff", "").rstrip("\n").split("\t")]
    width = len(names)
    data = []
    for n, row in lines:
        if len(row) == 0 or (len(row) == 1 and not _trim(row[0])):
            continue
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
        existing = name_index.get((_text(row["metNames"]), _text(row["comps"])))
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

    As in RAVEN's ``matchColumns``, the first column that matches a canonical name is the core
    column and every later column that matches it is ignored; a column with an empty header is
    ignored. With ``lower_rest`` a column that matches a core reaction column in another case
    is renamed to that spelling. The remaining columns keep their names, a repeated name gets
    ``.1``, ``.2`` ... appended.
    """
    lookup = {a.lower(): canonical for canonical, names in aliases.items() for a in names}
    if lower_rest:
        lookup.update({c.lower(): c for c in DEFAULT_CORE_RXN_COLUMNS})
    keep: list[int] = []
    names: list = []
    taken: set[str] = set()
    for position, col in enumerate(df.columns):
        if isinstance(col, str) and _trim(col) == "":
            continue
        canonical = lookup.get(str(col).strip().lower())
        if canonical is not None:
            if canonical in taken:
                continue
            taken.add(canonical)
            names.append(canonical)
        else:
            names.append(col)
        keep.append(position)
    out = df.iloc[:, keep].copy()
    used = set(names)
    seen: set = set()
    for i, name in enumerate(names):
        if name in seen:
            k = 1
            while f"{name}.{k}" in used:
                k += 1
            names[i] = f"{name}.{k}"
            used.add(names[i])
        seen.add(name)
    out.columns = names
    return out


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
        df, ("charge",), "metsInfo", checks={"charge": _check_whole_charge}).astype(object)


def _prepare_genes(df: pd.DataFrame) -> pd.DataFrame:
    df = _canonical_columns(df, _GENE_ALIASES)
    _require_columns(df, ("genes",), "genes_df")
    return df.astype(object)


def _check_genes(model: cobra.Model, genes_df: pd.DataFrame) -> None:
    """Raise :class:`ValueError` for a gene id that the table lists more than once.

    This holds for a gene the model has as well as for a new one: two rows for one gene would
    say two things about it (RAVEN would add a second gene with the same id).
    """
    seen: set[str] = set()
    repeated: list[str] = []
    for value in genes_df["genes"]:
        gid = _text(value)
        if gid in seen and gid not in repeated:
            repeated.append(gid)
        seen.add(gid)
    if repeated:
        raise ValueError(f"These genes are listed more than once in genes_df: {repeated}")


def _row_has_content(row, columns=None) -> bool:
    values = row if columns is None else [row[c] for c in columns]
    return any(_has_value(v) for v in values)


_RULE_TOKEN = re.compile(r"\(|\)|[^\s()]+")
_AND_WORDS = frozenset({"and", "&", "&&"})
_OR_WORDS = frozenset({"or", "|", "||"})
#: Spellings of the operators that cobra reads as they are.
_COBRA_OPERATORS = frozenset({"and", "AND", "&", "or", "OR", "|"})


def _rule_tokens(rule: str) -> list[tuple[str, str]]:
    """Split a gene rule into ``(kind, text)`` pairs; kind is ``(``, ``)``, ``and``, ``or`` or ``gene``.

    The operators are ``and``/``or`` in any case and the symbols ``&``, ``&&``, ``|`` and ``||``,
    as RAVEN's ``parseGrRule`` reads them; every other word is a gene id.
    """
    tokens = []
    for match in _RULE_TOKEN.finditer(rule):
        word = match.group()
        low = word.lower()
        if word in ("(", ")"):
            tokens.append((word, word))
        elif low in _AND_WORDS:
            tokens.append(("and", word))
        elif low in _OR_WORDS:
            tokens.append(("or", word))
        else:
            tokens.append(("gene", word))
    return tokens


def _rule_genes(rule: str) -> list[str]:
    """Gene ids of a gene rule; :class:`ValueError` when the rule is not well formed.

    The grammar is that of RAVEN's ``parseGrRule``: ``expr := term (or term)*``,
    ``term := factor (and factor)*``, ``factor := gene | ( expr )``.
    """
    tokens = _rule_tokens(rule)
    genes: list[str] = []

    def factor(pos: int) -> int:
        if pos >= len(tokens):
            raise ValueError("the rule ends after an operator")
        kind, text = tokens[pos]
        if kind == "(":
            pos = expr(pos + 1)
            if pos >= len(tokens) or tokens[pos][0] != ")":
                raise ValueError("unbalanced parentheses")
            return pos + 1
        if kind == "gene":
            genes.append(text)
            return pos + 1
        raise ValueError(f'expected a gene but found "{text}"')

    def term(pos: int) -> int:
        pos = factor(pos)
        while pos < len(tokens) and tokens[pos][0] == "and":
            pos = factor(pos + 1)
        return pos

    def expr(pos: int) -> int:
        pos = term(pos)
        while pos < len(tokens) and tokens[pos][0] == "or":
            pos = term(pos + 1)
        return pos

    if not tokens:
        return genes
    end = expr(0)
    if end < len(tokens):
        raise ValueError(f'unexpected "{tokens[end][1]}"')
    return genes


def _row_rule_genes(row: pd.Series) -> list[str]:
    """Gene ids of the ``grRules`` cell of a reaction row; :class:`ValueError` when it is malformed."""
    rule = row.get("grRules")
    if not _has_value(rule):
        return []
    try:
        return _rule_genes(str(rule))
    except ValueError as exc:
        raise ValueError(
            f'The grRules value "{rule}" of reaction {_text(row["rxnNames"])!r} is not a valid gene rule: {exc}'
        ) from None


def _stored_gene_rule(rule) -> str:
    """The rule as cobra reads it: operator spellings cobra does not know (``And``, ``&&``,
    ``||``) are written as ``and``/``or``; everything else is kept as it is."""
    text = str(rule)
    pieces: list[str] = []
    last = 0
    for match in _RULE_TOKEN.finditer(text):
        word = match.group()
        low = word.lower()
        if word in _COBRA_OPERATORS or word in ("(", ")"):
            continue
        if low in _AND_WORDS:
            replacement = "and"
        elif low in _OR_WORDS:
            replacement = "or"
        else:
            continue
        pieces.append(text[last:match.start()])
        pieces.append(replacement)
        last = match.end()
    pieces.append(text[last:])
    return "".join(pieces)


def _check_gene_rules(rxns_df: pd.DataFrame) -> None:
    """Raise :class:`ValueError` for a ``grRules`` cell that is not a well-formed gene rule.

    The grammar is RAVEN's (``parseGrRule``), so both tools refuse the same rules: an operator
    without two operands, unbalanced or empty parentheses, and two genes without an operator
    between them. cobra would log a malformed rule and store an empty one.
    """
    if "grRules" not in rxns_df.columns:
        return
    for _, row in rxns_df.iterrows():
        _row_rule_genes(row)


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

    rxns_df = rxns_df.reset_index(drop=True).astype(object)
    rxns_df = rxns_df.loc[pd.Series(
        [_row_has_content(r) for r in rxns_df.itertuples(index=False)], index=rxns_df.index,
        dtype=bool)]
    coeff_content = [c for c in ("rxnIdx", "rxnNames", "metNames", "comps") if c in coeffs_df.columns]
    coeffs_df = coeffs_df.reset_index(drop=True).astype(object)
    coeffs_df = coeffs_df.loc[pd.Series(
        [_row_has_content(r, coeff_content) for _, r in coeffs_df.iterrows()],
        index=coeffs_df.index, dtype=bool)]

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
    if "rxnIdx" in rxns_df.columns:
        repeated = rxns_df["rxnIdx"].duplicated(keep=False)
        if repeated.any():
            index = rxns_df["rxnIdx"][repeated].iloc[0]
            rows = ", ".join(str(int(i) + 1) for i in rxns_df.index[rxns_df["rxnIdx"] == index])
            raise ValueError(
                f"The rxnIdx {index:g} is used by data rows {rows} of the rxnsInfo table; "
                "each reaction needs its own index")
    rxns_df = rxns_df.astype(object)
    coeffs_df = coeffs_df.astype(object)
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
            name=_text(row["metNames"]),
            compartment=_text(row["comps"]),
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
        gid = _text(row["genes"])
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
            g = Gene(_text(row["genes"]))
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


def _plan_reactions(
    model: cobra.Model, rxns_df: pd.DataFrame, coeffs_df: pd.DataFrame,
    genes_df: pd.DataFrame | None = None,
) -> _ReactionPlan:
    """Decide which reactions of ``rxns_df`` exist in the model, and check them.

    A reaction whose stoichiometry matches a model reaction (compared by metabolite id and
    coefficient) is an update; every other one is new, also when it uses a metabolite that
    ``mets_df`` adds. Raises :class:`ValueError` for a lower bound above the upper bound (several
    rows that update one reaction are checked one after the other, as they are applied) and for
    the gene rule of a new reaction that names a gene which neither the model nor ``genes_df``
    has (RAVEN's ``addRxns`` does not add genes either).
    """
    by_index = "rxnIdx" in rxns_df.columns and "rxnIdx" in coeffs_df.columns
    coeffs_by_key = _group_coeffs(coeffs_df, by_index)
    name_index = _name_compartment_index(model)
    by_signature = {_stoich_signature(r): r for r in model.reactions}
    default_ub = _stored_default_ub(model)
    known_genes = {g.id for g in model.genes}
    if genes_df is not None:
        known_genes |= {_text(g) for g in genes_df["genes"]}

    updates: list[tuple[cobra.Reaction, pd.Series]] = []
    new: list[tuple[pd.Series, dict[tuple[str, str], float]]] = []
    bounds: dict[str, tuple[float, float]] = {}
    for _, row in rxns_df.iterrows():
        stoich = coeffs_by_key[_reaction_key(row, by_index)]
        existing = None
        if all(key in name_index for key in stoich):
            existing = by_signature.get(
                frozenset((name_index[key].id, coef) for key, coef in stoich.items()))
        if existing is not None:
            stored = bounds.get(existing.id, (existing.lower_bound, existing.upper_bound))
            lb = float(row["lb"]) if _has_value(row.get("lb")) else stored[0]
            ub = float(row["ub"]) if _has_value(row.get("ub")) else stored[1]
            bounds[existing.id] = (lb, ub)
            updates.append((existing, row))
        else:
            lb = float(row["lb"]) if _has_value(row.get("lb")) else -math.inf
            ub = float(row["ub"]) if _has_value(row.get("ub")) else default_ub
            _check_new_reaction_genes(row, known_genes)
            new.append((row, stoich))
        if lb > ub:
            raise ValueError(
                f"The lower bound {lb:g} of reaction {_text(row['rxnNames'])!r} is larger than its "
                f"upper bound {ub:g}")
    return _ReactionPlan(updates, new)


def _check_new_reaction_genes(row: pd.Series, known_genes: set[str]) -> None:
    """Raise :class:`ValueError` when the gene rule of a new reaction names an unknown gene."""
    unknown = sorted(set(_row_rule_genes(row)) - known_genes)
    if unknown:
        raise ValueError(
            f"The gene rule of the new reaction {_text(row['rxnNames'])!r} names genes that are "
            f"not in the model or in genes_df: {unknown}")


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
        known |= {(_text(n), _text(c)) for n, c in zip(mets_df["metNames"], mets_df["comps"],
                                                  strict=False)}
    coeff_keys = set()
    for _, row in coeffs_df.iterrows():
        if (_text(row["metNames"]), _text(row["comps"])) not in known:
            raise ValueError(
                f"Reaction {_text(row['rxnNames'])!r} references metabolite "
                f"{_text(row['metNames'])!r}[{_text(row['comps'])}] which is not in the model. "
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
        return f"{_text(row['rxnNames'])!r} (rxnIdx {_reaction_key(row, by_index)[0]})"
    return repr(_text(row["rxnNames"]))


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
        return (int(row["rxnIdx"]), _text(row["rxnNames"]))
    return _text(row["rxnNames"])


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
        key = (_text(row["metNames"]), _text(row["comps"]))
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
        rxn.gene_reaction_rule = _stored_gene_rule(row["grRules"])
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
            name=_text(row["rxnNames"]),
            lower_bound=float(row["lb"]) if _has_value(row.get("lb")) else -math.inf,
            upper_bound=float(row["ub"]) if _has_value(row.get("ub")) else default_ub,
        )
        rxn.add_metabolites(stoich)
        if _has_value(row.get("grRules")):
            rxn.gene_reaction_rule = _stored_gene_rule(row["grRules"])
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
