"""Generic cobra.Model structural transforms that cobrapy does not cover cleanly:
reaction building from equations, batch GPR / bound changes, irreversibility splitting,
isozyme expansion, compartment merge / copy, model merging by name, and leak-driven
reaction removal."""
from .add import add_reactions_from_equations
from .bad_reactions import remove_bad_reactions
from .boundary import close_model
from .change import change_gene_reaction_rules, change_reaction_equations
from .compartments import copy_to_compartment, merge_compartments
from .expand import expand_model, gpr_to_dnf
from .irreversible import convert_to_irreversible
from .merge import merge_models
from .parameters import set_exchange_bounds, set_variance_bounds
from .remove import remove_genes, remove_metabolites
from .replace import replace_metabolite
from .simplify import (
    constrain_reversible_reactions,
    contract_model,
    find_duplicate_reactions,
    group_linear_reactions,
    remove_dead_end_reactions,
    remove_no_flux_reactions,
    remove_zero_interval_reactions,
    simplify_model,
)
from .transfer import add_reactions_from_model
from .transport import add_transport_reactions

__all__ = [
    "add_reactions_from_equations",
    "add_reactions_from_model",
    "add_transport_reactions",
    "change_gene_reaction_rules",
    "change_reaction_equations",
    "close_model",
    "constrain_reversible_reactions",
    "contract_model",
    "convert_to_irreversible",
    "copy_to_compartment",
    "expand_model",
    "find_duplicate_reactions",
    "gpr_to_dnf",
    "group_linear_reactions",
    "merge_compartments",
    "merge_models",
    "remove_bad_reactions",
    "remove_dead_end_reactions",
    "remove_genes",
    "remove_metabolites",
    "remove_no_flux_reactions",
    "remove_zero_interval_reactions",
    "replace_metabolite",
    "set_exchange_bounds",
    "set_variance_bounds",
    "simplify_model",
]
