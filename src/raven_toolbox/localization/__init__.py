"""Sub-cellular localisation: predictor-independent placement of reactions in compartments.

:func:`predict_localization` is the score-driven MILP entry point; :func:`assign_compartments`
is the functionality-constrained variant (a flux-free placement master and FBA certification,
with optional gap-fill, transport pruning and reaction-level multi-localisation validated by FVA).
:func:`load_deeploc`, :func:`load_mulocdeep` and :func:`load_compartments` parse predictor and
evidence-database outputs into the ``gene × compartment`` :class:`LocalizationScores` DataFrame
that the algorithms consume (:data:`DEFAULT_COMPARTMENT_MAP` maps labels to the compartment ids of
the model).

For sequence-based predictors (DeepLoc 2.1, MULocDeep), :func:`prepare_deeploc_input` writes a FASTA
of the gene sequences of the model (fetched from UniProtKB, headers are the gene ids) as input to
the predictor, and :func:`load_deeploc` reads the output.
"""
from raven_toolbox.localization.assign import (
    AssignmentProposal,
    GrowthCondition,
    apply_assignment,
)
from raven_toolbox.localization.certify import assign_compartments
from raven_toolbox.localization.curation import curation_priority
from raven_toolbox.localization.predict import (
    LocalizationProposal,
    LocalizationResult,
    apply_localization,
    predict_localization,
)
from raven_toolbox.localization.relocate import RelocationResult, relocate_reactions
from raven_toolbox.localization.scores import (
    DEFAULT_COMPARTMENT_MAP,
    LocalizationScores,
    combine_scores,
    fetch_uniprot_localization,
    load_compartments,
    load_deeploc,
    load_mulocdeep,
    load_uniprot,
)
from raven_toolbox.localization.sequences import (
    PreparedFasta,
    fetch_protein_sequences,
    prepare_deeploc_input,
    write_fasta,
)
from raven_toolbox.localization.substrate_ontology import SubstrateOntology
from raven_toolbox.localization.transport_evidence import (
    TransporterAnnotation,
    annotate_proteome,
    annotate_transporters,
    default_metabolite_chebi,
    default_substrate_of,
    evidence_aware_transport_cost,
)
from raven_toolbox.localization.triage import (
    DEEPLOC_COMPARTMENT_TRUST,
    ReviewReport,
    triage_localization,
)

__all__ = [
    "AssignmentProposal",
    "GrowthCondition",
    "DEFAULT_COMPARTMENT_MAP",
    "DEEPLOC_COMPARTMENT_TRUST",
    "LocalizationProposal",
    "LocalizationResult",
    "LocalizationScores",
    "PreparedFasta",
    "RelocationResult",
    "ReviewReport",
    "SubstrateOntology",
    "TransporterAnnotation",
    "annotate_proteome",
    "annotate_transporters",
    "apply_assignment",
    "apply_localization",
    "assign_compartments",
    "combine_scores",
    "curation_priority",
    "default_metabolite_chebi",
    "default_substrate_of",
    "evidence_aware_transport_cost",
    "fetch_protein_sequences",
    "fetch_uniprot_localization",
    "load_compartments",
    "load_deeploc",
    "load_mulocdeep",
    "load_uniprot",
    "prepare_deeploc_input",
    "predict_localization",
    "relocate_reactions",
    "triage_localization",
    "write_fasta",
]
