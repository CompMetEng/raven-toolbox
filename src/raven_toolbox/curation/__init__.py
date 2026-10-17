"""Batch curation of metabolites / reactions / genes from data tables.

Port of yeast-GEM's MATLAB ``curateMetsRxnsGenes`` into a generic
DataFrame-driven engine. Other GEM projects (Human-GEM, custom
reconstructions, …) can use the same machinery with their own TSV
layouts; the only required pieces are the data tables and (optionally)
project-specific id prefixes for fresh metabolites and reactions.

Public API:

* :func:`batch_curate` — entrypoint taking pandas DataFrames.
* :func:`batch_curate_from_tsv` — file-path convenience wrapper.
* :class:`CurationResult` — record of what was added / updated.
* :func:`rename_model_genes` — rename gene identifiers from a mapping table.
* :func:`get_gene_data` / :func:`download_genome_data` — build that mapping
  table from an NCBI genome assembly's GFF3 annotation.
* :func:`process_protein_fasta_file` — rename a protein FASTA's headers
  from the same mapping table.

Schema (mirrors yeast-GEM's ``data/modelCuration/template/`` layout):

- **mets_df**: ``metNames, comps, formula, charge, inchi, metNotes``
  + any number of MIRIAM-annotation columns. Match key is
  ``(name, comp)``.
- **genes_df**: ``genes, geneShortNames`` + MIRIAM columns. Match key
  is ``genes``.
- **rxns_df**: ``rxnNames, grRules, lb, ub, rev, subSystems, eccodes,
  rxnNotes, rxnReferences, rxnConfidenceScores`` + MIRIAM columns.
  Match key is the reaction's *stoichiometry* — same metabolites and
  coefficients ⇒ same reaction.
- **rxns_coeffs_df**: ``rxnNames, metNames, comps, coefficient``. One
  row per ``(reaction, metabolite)`` pair. The ``rxnNames`` column
  links each coefficient back to a row in ``rxns_df``. An optional
  ``index`` first column from the legacy yeast-GEM schema is silently
  ignored.

Everything after the core columns in any of the four tables is
interpreted as a MIRIAM annotation: the column header becomes the
namespace key (``met.annotation[<header>] = <cell>``).

``inchi``, ``metNotes``, ``rxnNotes``, ``rxnReferences`` and
``rxnConfidenceScores`` are RAVEN fields and go to ``.notes`` under the keys
:func:`~raven_toolbox.io.read_yaml_model` uses for them (``inchis``,
``note``, ``note``, ``references`` and ``confidence_score``, the last as a
number), so the YAML and MAT writers emit them as those fields. An InChI the
metabolite already carries as ``annotation["inchi"]`` is updated as well.
"""
from raven_toolbox.curation.batch import (
    DEFAULT_CORE_GENE_COLUMNS,
    DEFAULT_CORE_MET_COLUMNS,
    DEFAULT_CORE_RXN_COEFFS_COLUMNS,
    DEFAULT_CORE_RXN_COLUMNS,
    CurationResult,
    batch_curate,
    batch_curate_from_tsv,
)
from raven_toolbox.curation.genome_data import (
    download_genome_data,
    get_gene_data,
    parse_gff_gene_table,
)
from raven_toolbox.curation.process_protein_fasta import process_protein_fasta_file
from raven_toolbox.curation.rename_genes import RenameGenesResult, rename_model_genes

__all__ = [
    "DEFAULT_CORE_GENE_COLUMNS",
    "DEFAULT_CORE_MET_COLUMNS",
    "DEFAULT_CORE_RXN_COEFFS_COLUMNS",
    "DEFAULT_CORE_RXN_COLUMNS",
    "CurationResult",
    "RenameGenesResult",
    "batch_curate",
    "batch_curate_from_tsv",
    "download_genome_data",
    "get_gene_data",
    "parse_gff_gene_table",
    "process_protein_fasta_file",
    "rename_model_genes",
]
