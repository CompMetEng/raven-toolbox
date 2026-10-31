"""SBO term assignment.

Port of the generic core of yeast-GEM's ``code/missingFields/addSBOterms.m``.
Defaults reproduce the yeast-GEM behaviour. Arguments set the pseudo-metabolite and
pseudo-reaction name conventions, the transport detector, and a yeast-specific
bug-compatibility flag.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import cobra

DEFAULT_BIOMASS_MET_NAMES: frozenset[str] = frozenset(
    {"biomass", "DNA", "RNA", "protein", "carbohydrate",
     "lipid", "cofactor", "ion"}
)
DEFAULT_BIOMASS_RXN_NAME: str = "biomass pseudoreaction"
DEFAULT_NGAM_RXN_NAME: str = "non-growth associated maintenance reaction"


def add_sbo_terms(
    model: cobra.Model,
    *,
    biomass_met_names: Iterable[str] = DEFAULT_BIOMASS_MET_NAMES,
    biomass_met_suffixes: Iterable[str] = (" backbone", " chain"),
    biomass_rxn_name: str = DEFAULT_BIOMASS_RXN_NAME,
    ngam_rxn_name: str = DEFAULT_NGAM_RXN_NAME,
    pseudoreaction_name_substrings: Iterable[str] = ("pseudoreaction", "SLIME rxn"),
    transport_detector: Callable[[cobra.Model], set[str]] | None = None,
    only_last_reaction_for_pseudo: bool = False,
) -> cobra.Model:
    """Assign SBO terms to metabolites and reactions in-place.

    **Metabolite SBO assignment**

    - SBO:0000649 (Biomass) for metabolites whose ``name`` is in
      ``biomass_met_names`` or ends with any of ``biomass_met_suffixes``
      (default: ``" backbone"`` and ``" chain"``).
    - SBO:0000247 (simple chemical) otherwise.

    **Reaction SBO assignment**

    - SBO:0000176 (metabolic reaction) by default.
    - Single-metabolite reactions (exchange, sink or demand):
        * extracellular: SBO:0000627 (exchange)
        * coefficient < 0: SBO:0000632 (sink)
        * otherwise: SBO:0000628 (demand)
    - Transport reactions: SBO:0000655 (default detector: the same
      metabolite name appears in at least 2 compartments).
    - The reaction whose ``name`` matches ``biomass_rxn_name``: SBO:0000629.
    - The reaction whose ``name`` matches ``ngam_rxn_name``: SBO:0000630.
    - Other reactions whose ``name`` contains any of ``pseudoreaction_name_substrings``:
      SBO:0000395.

    The SBO term is written to ``annotation['sbo']`` only when that key is missing or
    empty, as in RAVEN's ``editMiriam(..., 'fill')`` mode.

    Parameters
    ----------
    model : cobra.Model
        Model to annotate. It is modified in place and returned.
    biomass_met_names : Iterable[str]
        Metabolite names that receive SBO:0000649. Default
        ``DEFAULT_BIOMASS_MET_NAMES`` (``biomass``, ``DNA``, ``RNA``, ``protein``,
        ``carbohydrate``, ``lipid``, ``cofactor`` and ``ion``). Names are matched exactly.
    biomass_met_suffixes : Iterable[str]
        Name suffixes that mark a metabolite as biomass. Default ``(" backbone", " chain")``.
    biomass_rxn_name : str
        Name of the biomass pseudoreaction, which receives SBO:0000629. The match is exact.
        Default ``"biomass pseudoreaction"``.
    ngam_rxn_name : str
        Name of the non-growth associated maintenance reaction, which receives SBO:0000630.
        The match is exact. Default ``"non-growth associated maintenance reaction"``.
    pseudoreaction_name_substrings : Iterable[str]
        Reactions that are not the biomass or NGAM reaction and whose name contains any of
        these substrings receive SBO:0000395. Default ``("pseudoreaction", "SLIME rxn")``.
    transport_detector
        Optional callable that takes the model and returns the set of
        reaction ids classified as transport. When ``None``, the default
        heuristic is used: a reaction is transport when one metabolite name
        occurs in at least 2 compartments. A custom detector replaces this
        convention (for example RAVEN's ``getTransportRxns`` via the MATLAB
        engine).
    only_last_reaction_for_pseudo
        If True, the biomass, NGAM and pseudoreaction terms are assigned to the last
        reaction only, as in yeast-GEM's ``addSBOterms.m``, whose loop header
        ``for i = numel(...)`` lacks ``1:``. Default False.
    """
    biomass_met_names = frozenset(biomass_met_names)
    biomass_met_suffixes = tuple(biomass_met_suffixes)
    pseudoreaction_name_substrings = tuple(pseudoreaction_name_substrings)
    if transport_detector is None:
        transport_detector = _default_transport_detector

    _assign_metabolite_sbo(model, biomass_met_names, biomass_met_suffixes)
    _assign_reaction_sbo(
        model,
        biomass_rxn_name=biomass_rxn_name,
        ngam_rxn_name=ngam_rxn_name,
        pseudo_substrings=pseudoreaction_name_substrings,
        transport_detector=transport_detector,
        only_last_reaction_for_pseudo=only_last_reaction_for_pseudo,
    )
    return model


# --- internals ---------------------------------------------------------

def _assign_metabolite_sbo(
    model: cobra.Model,
    biomass_met_names: frozenset[str],
    biomass_met_suffixes: tuple[str, ...],
) -> None:
    for met in model.metabolites:
        if met.name in biomass_met_names or met.name.endswith(biomass_met_suffixes):
            sbo = "SBO:0000649"
        else:
            sbo = "SBO:0000247"
        _fill_sbo(met, sbo)


def _assign_reaction_sbo(
    model: cobra.Model,
    *,
    biomass_rxn_name: str,
    ngam_rxn_name: str,
    pseudo_substrings: tuple[str, ...],
    transport_detector: Callable[[cobra.Model], set[str]],
    only_last_reaction_for_pseudo: bool,
) -> None:
    """Compute the SBO term of each reaction by successive overrides, then
    fill the terms into the model in a single pass, as the MATLAB ``rxnSBO``
    array and ``editMiriam(..., 'fill')`` do.
    """
    transport_set = transport_detector(model)
    rxns = list(model.reactions)

    sbo_for_rxn: dict[str, str] = {}
    for rxn in rxns:
        sbo = "SBO:0000176"
        if len(rxn.metabolites) == 1:
            (met,) = rxn.metabolites
            coef = rxn.metabolites[met]
            if met.compartment == "e" or model.compartments.get(
                met.compartment
            ) == "extracellular":
                sbo = "SBO:0000627"
            elif coef < 0:
                sbo = "SBO:0000632"
            else:
                sbo = "SBO:0000628"
        sbo_for_rxn[rxn.id] = sbo

    # Transport override
    for rxn_id in transport_set:
        sbo_for_rxn[rxn_id] = "SBO:0000655"

    # Pseudoreaction override. The legacy bug-compat flag scopes this
    # loop to the last reaction only; the default walks the whole list.
    pseudo_targets = [rxns[-1]] if only_last_reaction_for_pseudo else rxns
    for rxn in pseudo_targets:
        if rxn.name == biomass_rxn_name:
            sbo_for_rxn[rxn.id] = "SBO:0000629"
        elif rxn.name == ngam_rxn_name:
            sbo_for_rxn[rxn.id] = "SBO:0000630"
        elif any(s in rxn.name for s in pseudo_substrings):
            sbo_for_rxn[rxn.id] = "SBO:0000395"

    for rxn in rxns:
        _fill_sbo(rxn, sbo_for_rxn[rxn.id])


def _fill_sbo(entity, sbo: str) -> None:
    """Set ``annotation['sbo']`` only if it is missing or empty."""
    if not entity.annotation.get("sbo"):
        entity.annotation["sbo"] = sbo


def _default_transport_detector(model: cobra.Model) -> set[str]:
    """Return the ids of reactions with one metabolite name in at least 2 compartments.

    Approximates RAVEN's ``getTransportRxns``. :func:`add_sbo_terms` accepts a custom
    callable in its place.
    """
    out: set[str] = set()
    for rxn in model.reactions:
        by_name: dict[str, set[str | None]] = {}
        for met in rxn.metabolites:
            by_name.setdefault(met.name, set()).add(met.compartment)
        if any(len(comps) >= 2 for comps in by_name.values()):
            out.add(rxn.id)
    return out
