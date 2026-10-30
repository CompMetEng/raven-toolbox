"""Change the stoichiometry of existing reactions from equation strings.

Editing the same ``Reaction`` object changes only its stoichiometry — its id, name,
bounds, GPR, subsystem, and position are preserved automatically by cobra.

So this port simply re-parses the equation (reusing the same metabolite
matching as :func:`~raven_toolbox.manipulation.add.add_reactions_from_equations`,
including name and ``name[comp]`` modes that cobra lacks) and swaps the
metabolites in place.

Like RAVEN, **bounds are left unchanged** even if the new equation's arrow
implies a different reversibility — use a bounds setter for that.
"""
from __future__ import annotations

from collections.abc import Mapping

import cobra
from cobra import Reaction

from raven_toolbox.manipulation.add import _build_met_index, _stoichiometry
from raven_toolbox.utils.gpr import set_gene_rule

__all__ = ["change_reaction_equations", "change_gene_reaction_rules"]


def change_reaction_equations(
    model: cobra.Model,
    equations: Mapping[str, str],
    *,
    mets_by: str = "id",
    compartment: str | None = None,
    allow_new_mets: bool = True,
    new_met_prefix: str = "m",
) -> list[Reaction]:
    """Replace the stoichiometry of existing reactions.

    Parameters
    ----------
    model
        Target ``cobra.Model``, mutated in place.
    equations
        Mapping of ``reaction_id -> equation string``. Every ID must already
        exist in the model. Equation syntax is identical to
        :func:`~raven_toolbox.manipulation.add.add_reactions_from_equations`.
    mets_by, compartment, allow_new_mets, new_met_prefix
        Metabolite-matching options, as in ``add_reactions_from_equations``.

    Returns
    -------
    list of cobra.Reaction
        The reactions changed, in input order.

    Notes
    -----
    Bounds are **not** modified, matching RAVEN. Changing an equation from
    ``-->`` to ``<=>`` does not by itself make the reaction reversible; adjust
    the bounds separately.
    """
    if mets_by not in ("id", "name"):
        raise ValueError(f"mets_by must be 'id' or 'name', got {mets_by!r}")

    changed: list[Reaction] = []
    met_index = _build_met_index(model)
    for rxn_id, equation in equations.items():
        if rxn_id not in model.reactions:
            raise ValueError(f"Reaction {rxn_id!r} not found in the model.")
        rxn = model.reactions.get_by_id(rxn_id)

        coeffs, _reversible = _stoichiometry(
            model,
            equation,
            mets_by=mets_by,
            compartment=compartment,
            allow_new_mets=allow_new_mets,
            new_met_prefix=new_met_prefix,
            met_index=met_index,
        )

        rxn.subtract_metabolites(dict(rxn.metabolites), combine=True)
        rxn.add_metabolites(coeffs)
        changed.append(rxn)

    return changed


def change_gene_reaction_rules(
    model: cobra.Model,
    rules: Mapping[str, str],
    *,
    replace: bool = True,
) -> list[Reaction]:
    """Set or append gene-reaction rules on existing reactions.

    Rules are read with :func:`~raven_toolbox.utils.parse_gr_rule`, the grammar of RAVEN's
    ``parseGrRule`` (``G1&G2`` is one gene id); a rule that grammar refuses is read with
    cobra's parser. New genes are created in the model. ``replace=False`` appends the rule
    as an isozyme, as RAVEN's ``changeGrRules`` with ``replace`` false does.

    Parameters
    ----------
    model
        Target ``cobra.Model``, mutated in place.
    rules
        Mapping of ``reaction_id -> GPR string``. Every ID must already exist.
    replace
        If True (default), overwrite the existing GPR. If False, append the new
        rule as an isozyme: ``(old) or (new)`` (just ``new`` if the reaction had
        no GPR).

    Returns
    -------
    list of cobra.Reaction
        The reactions changed, in input order.
    """
    changed: list[Reaction] = []
    for rxn_id, rule in rules.items():
        if rxn_id not in model.reactions:
            raise ValueError(f"Reaction {rxn_id!r} not found in the model.")
        rxn = model.reactions.get_by_id(rxn_id)

        if replace or not rxn.gene_reaction_rule:
            new_rule = rule
        else:
            new_rule = f"({rxn.gene_reaction_rule}) or ({rule})"

        set_gene_rule(rxn, new_rule)
        changed.append(rxn)

    return changed
