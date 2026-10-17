"""Generic batch-curation engine driven by tabular data."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cobra
import pandas as pd

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
#: + one row per ``(reaction, metabolite)`` pair.) A leading ``index``
#: column from the legacy yeast-GEM schema is silently ignored.
DEFAULT_CORE_RXN_COEFFS_COLUMNS: tuple[str, ...] = (
    "rxnNames", "metNames", "comps", "coefficient",
)


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

    Returns
    -------
    A :class:`CurationResult` summarising the changes.
    """
    result = CurationResult()

    if mets_df is not None:
        _apply_mets(model, mets_df, met_id_prefix, result)
    if genes_df is not None:
        _apply_genes(model, genes_df, result)

    if (rxns_df is None) != (rxns_coeffs_df is None):
        raise ValueError(
            "rxns_df and rxns_coeffs_df must be provided together; got "
            f"rxns_df={'set' if rxns_df is not None else 'None'}, "
            f"rxns_coeffs_df="
            f"{'set' if rxns_coeffs_df is not None else 'None'}."
        )
    if rxns_df is not None:
        _apply_reactions(model, rxns_df, rxns_coeffs_df, rxn_id_prefix, result)
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
    )


# --- metabolites ------------------------------------------------------

def _apply_mets(
    model: cobra.Model,
    df: pd.DataFrame,
    id_prefix: str,
    result: CurationResult,
) -> None:
    miriam_cols = [c for c in df.columns if c not in DEFAULT_CORE_MET_COLUMNS]
    name_index = _name_compartment_index(model)

    new_rows: list[pd.Series] = []
    for _, row in df.iterrows():
        name = row["metNames"]
        comp = row["comps"]
        existing = name_index.get((name, comp))
        if existing is not None:
            _update_metabolite(existing, row, miriam_cols)
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


def _name_compartment_index(model: cobra.Model) -> dict[tuple[str, str], cobra.Metabolite]:
    return {(m.name, m.compartment): m for m in model.metabolites}


def _update_metabolite(met: cobra.Metabolite, row: pd.Series, miriam_cols: list[str]) -> None:
    if _has_value(row.get("formula")):
        met.formula = str(row["formula"])
    if _has_value(row.get("charge")):
        met.charge = _coerce_int(row["charge"])
    _apply_met_notes(met, row)
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

def _apply_genes(model: cobra.Model, df: pd.DataFrame, result: CurationResult) -> None:
    miriam_cols = [c for c in df.columns if c not in DEFAULT_CORE_GENE_COLUMNS]
    existing_genes = {g.id: g for g in model.genes}

    new_rows: list[pd.Series] = []
    for _, row in df.iterrows():
        gid = str(row["genes"])
        existing = existing_genes.get(gid)
        if existing is not None:
            _update_gene(existing, row, miriam_cols)
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


def _update_gene(gene, row: pd.Series, miriam_cols: list[str]) -> None:
    if _has_value(row.get("geneShortNames")):
        gene.name = str(row["geneShortNames"])
    _apply_miriam(gene, row, miriam_cols)


# --- reactions --------------------------------------------------------

def _apply_reactions(
    model: cobra.Model,
    rxns_df: pd.DataFrame,
    coeffs_df: pd.DataFrame,
    id_prefix: str,
    result: CurationResult,
) -> None:
    miriam_cols = [c for c in rxns_df.columns if c not in DEFAULT_CORE_RXN_COLUMNS]
    _validate_rxns_coeffs(coeffs_df)
    coeffs_by_name = _group_coeffs_by_rxn_name(model, coeffs_df)

    # Build {stoichiometry_signature: existing_reaction} for the lookup
    # of "is this reaction already in the model?".
    by_signature = {
        _stoich_signature(r): r for r in model.reactions
    }

    new_rows: list[tuple[pd.Series, dict[cobra.Metabolite, float]]] = []
    for _, row in rxns_df.iterrows():
        rxn_name = row["rxnNames"]
        if rxn_name not in coeffs_by_name:
            raise ValueError(
                f"Reaction {rxn_name!r} in rxns_df has no matching "
                "row(s) in rxns_coeffs_df."
            )
        stoich = coeffs_by_name[rxn_name]
        sig = _stoich_signature_from_dict(stoich)
        existing = by_signature.get(sig)
        if existing is not None:
            _update_reaction(existing, row, miriam_cols)
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


def _group_coeffs_by_rxn_name(
    model: cobra.Model,
    coeffs_df: pd.DataFrame,
) -> dict[str, dict[cobra.Metabolite, float]]:
    """Group coefficient rows by rxn name and resolve the metabolites.

    Each value is a ``{metabolite_object: coefficient}`` dict ready for
    ``cobra.Reaction.add_metabolites``. Metabolites are looked up by
    ``(name, comp)``; if a coefficient references an unknown
    metabolite, that's an error (the caller should add the metabolite
    via ``mets_df`` first).
    """
    name_index = _name_compartment_index(model)
    grouped: dict[str, dict[cobra.Metabolite, float]] = {}
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
        grouped.setdefault(rxn_name, {})[met] = coef
    return grouped


def _stoich_signature(rxn: cobra.Reaction) -> frozenset:
    return frozenset((m.id, c) for m, c in rxn.metabolites.items())


def _stoich_signature_from_dict(stoich: dict[cobra.Metabolite, float]) -> frozenset:
    return frozenset((m.id, c) for m, c in stoich.items())


def _update_reaction(rxn: cobra.Reaction, row: pd.Series, miriam_cols: list[str]) -> None:
    if _has_value(row.get("rxnNames")):
        rxn.name = str(row["rxnNames"])
    if _has_value(row.get("grRules")):
        rxn.gene_reaction_rule = str(row["grRules"])
    if _has_value(row.get("lb")):
        rxn.lower_bound = float(row["lb"])
    if _has_value(row.get("ub")):
        rxn.upper_bound = float(row["ub"])
    if _has_value(row.get("subSystems")):
        rxn.subsystem = subsystem_to_str(row["subSystems"])
    if _has_value(row.get("eccodes")):
        rxn.annotation["ec-code"] = str(row["eccodes"])
    _apply_rxn_notes(rxn, row)
    _apply_miriam(rxn, row, miriam_cols)


def _build_new_reactions(
    model: cobra.Model,
    rows: list[tuple[pd.Series, dict[cobra.Metabolite, float]]],
    miriam_cols: list[str],
    id_prefix: str,
) -> list[cobra.Reaction]:
    ids = _generate_ids(model.reactions, id_prefix, len(rows))
    new_rxns: list[cobra.Reaction] = []
    for new_id, (row, stoich) in zip(ids, rows, strict=True):
        rxn = cobra.Reaction(
            id=new_id,
            name=str(row["rxnNames"]),
            lower_bound=float(row["lb"]) if _has_value(row.get("lb")) else -1000.0,
            upper_bound=float(row["ub"]) if _has_value(row.get("ub")) else 1000.0,
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


def _apply_met_notes(met: cobra.Metabolite, row: pd.Series) -> None:
    for col, key in _MET_NOTE_COLUMNS:
        if _has_value(row.get(col)):
            met.notes[key] = str(row[col])
    # An InChI already kept as a MIRIAM annotation is updated as well, so the
    # metabolite does not carry two different structures.
    if _has_value(row.get("inchi")) and "inchi" in met.annotation:
        met.annotation["inchi"] = str(row["inchi"])


def _apply_rxn_notes(rxn: cobra.Reaction, row: pd.Series) -> None:
    for col, key in _RXN_NOTE_COLUMNS:
        if _has_value(row.get(col)):
            rxn.notes[key] = str(row[col])
    if _has_value(row.get("rxnConfidenceScores")):
        score = float(row["rxnConfidenceScores"])
        rxn.notes["confidence_score"] = int(score) if score.is_integer() else score


def _apply_miriam(entity, row: pd.Series, miriam_cols: list[str]) -> None:
    for col in miriam_cols:
        value = row.get(col)
        if _has_value(value):
            entity.annotation[col] = str(value).strip()


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
    """Generate ``count`` fresh ids by incrementing from the largest
    existing zero-padded suffix matching ``prefix``.

    e.g. with prefix ``s_`` and existing ``s_4100``, the next ids are
    ``s_4101``, ``s_4102``, … Width is preserved.
    """
    max_n = 0
    width = 1
    for entity in existing:
        if not entity.id.startswith(prefix):
            continue
        suffix = entity.id[len(prefix):]
        if suffix.isdigit():
            max_n = max(max_n, int(suffix))
            width = max(width, len(suffix))
    return [f"{prefix}{(max_n + i):0{width}d}" for i in range(1, count + 1)]
