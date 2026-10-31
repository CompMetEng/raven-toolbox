"""Replace gene identifiers in a model; port of RAVEN's ``renameModelGenes``.

Builds an old-id to new-id mapping from a gene table (for example from
:mod:`raven_toolbox.curation.genome_data`, or supplied manually) and applies it with
:func:`cobra.manipulation.rename_genes`. cobra parses the GPR into an AST and renames
the matched gene nodes, so a gene id that is a substring of another (``"gene1"`` in
``"gene10"``) is not renamed by mistake; RAVEN guards against this case with a regex
word boundary. cobra merges the gene objects when a rename target already exists in
the model. The GPR string keeps one occurrence per reference (``"G1 or G1"`` stays
as written); only the gene object is shared. RAVEN has no such check and can leave a
model with two ``model.genes`` entries with the same id. cobra normalises GPR syntax
on assignment, so the steps of RAVEN's ``removeUnnecessaryParentheses`` and
``standardizeGrRules`` are not needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cobra
import cobra.manipulation
import pandas as pd

__all__ = ["RenameGenesResult", "rename_model_genes"]


@dataclass
class RenameGenesResult:
    """Result of a gene-renaming pass.

    Parameters
    ----------
    renamed:
        Number of model genes for which a mapping was found and applied.
    unmapped:
        Model gene ids with no entry in ``from_col`` (or an empty ``to_col``),
        left unchanged. Sorted, unique.
    """

    renamed: int
    unmapped: list[str]


def rename_model_genes(
    model: cobra.Model,
    gene_table: pd.DataFrame | str | Path,
    from_col: str,
    to_col: str,
) -> RenameGenesResult:
    """Rename ``model``'s genes in place using a mapping table.

    Parameters
    ----------
    model:
        Model to rename genes in.
    gene_table:
        A DataFrame, or the path of a tab-delimited file that holds one.
    from_col:
        Column whose values match the ids currently in ``model.genes``.
    to_col:
        Column whose values replace them.

    Returns
    -------
    RenameGenesResult

    Raises
    ------
    ValueError
        If ``from_col`` or ``to_col`` is not a column of ``gene_table``.
    """
    if isinstance(gene_table, (str, Path)):
        gene_table = pd.read_csv(gene_table, sep="\t")

    for col in (from_col, to_col):
        if col not in gene_table.columns:
            raise ValueError(
                f"Column {col!r} not found in gene_table. "
                f"Available columns: {list(gene_table.columns)}"
            )

    # First non-empty (from, to) pair wins for a duplicated from_col value;
    # rows missing either side are skipped.
    mapping: dict[str, str] = {}
    for from_val, to_val in zip(gene_table[from_col], gene_table[to_col], strict=True):
        if pd.isna(from_val) or pd.isna(to_val):
            continue
        old = str(from_val).strip()
        new = str(to_val).strip()
        if old and new and old not in mapping:
            mapping[old] = new

    model_gene_ids = [g.id for g in model.genes]
    rename_dict = {g: mapping[g] for g in model_gene_ids if g in mapping}
    unmapped = sorted(g for g in model_gene_ids if g not in mapping)

    cobra.manipulation.rename_genes(model, rename_dict)

    return RenameGenesResult(renamed=len(rename_dict), unmapped=unmapped)
