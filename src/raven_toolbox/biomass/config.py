"""Configuration types for the biomass module."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

#: Strategies for computing a component's mass contribution from the
#: stoichiometry of its pseudoreaction. See :func:`sum_biomass`.
#:
#: ``"mw"``                 multiply each substrate coefficient by the
#:                          molecular weight derived from its
#:                          ``formula`` (g/mol), divide by 1000 to get
#:                          g/gDW. Used for carbohydrate, ion and cofactor.
#: ``"mw_minus_2h"``        as ``"mw"`` but with each MW reduced by
#:                          2.016 g/mol (two protons released per
#:                          charged tRNA on protein-pseudoreaction
#:                          substrates).
#: ``"mw_minus_water"``     as ``"mw"`` but with each MW reduced by
#:                          18.015 g/mol (one water released per
#:                          polymerisation step; RNA and DNA).
#: ``"grams"``              substrate coefficients are already in
#:                          g/gDW (e.g. the lipid-backbone
#:                          pseudoreaction). No MW lookup takes place.
MassStrategy = Literal["mw", "mw_minus_2h", "mw_minus_water", "grams"]


@dataclass(frozen=True)
class BiomassComponent:
    """One component of the biomass equation (protein, carbohydrate, and so on).

    Attributes
    ----------
    name
        Short name of the component. It is also the ``model.metabolites`` name of
        the metabolite that the pseudoreaction produces, which
        :func:`rescale_pseudoreaction` uses to identify the product.
    pseudoreaction_name
        ``model.reactions[*].name`` of the pseudoreaction. The reaction is
        looked up by ``name``, not by ``id``, following the yeast-GEM naming
        convention.
    mass_strategy
        How to convert the pseudoreaction's substrates into a mass
        fraction. :data:`MassStrategy` lists the choices and their
        offsets.
    """

    name: str
    pseudoreaction_name: str
    mass_strategy: MassStrategy = "mw"


@dataclass(frozen=True)
class BiomassConfig:
    """Container for the per-organism biomass layout.

    Attributes
    ----------
    biomass_rxn
        ``model.reactions[*].id`` of the top-level biomass
        pseudoreaction (the one whose flux is the growth rate).
    proton_met
        ``model.metabolites[*].id`` of the cytosolic H+ metabolite.
        :func:`rescale_pseudoreaction` sets its coefficient so that each
        pseudoreaction is charge-balanced after rescaling.
    components
        Ordered tuple of :class:`BiomassComponent` describing every
        non-top-level pseudoreaction that contributes to mass.
    """

    biomass_rxn: str
    proton_met: str
    components: tuple[BiomassComponent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Freeze the components into a tuple even if a list was passed.
        if not isinstance(self.components, tuple):
            object.__setattr__(self, "components", tuple(self.components))

    @classmethod
    def from_components(
        cls,
        biomass_rxn: str,
        proton_met: str,
        components: Iterable[BiomassComponent],
    ) -> BiomassConfig:
        """Build a :class:`BiomassConfig` from any iterable of components.

        The result equals the one from calling the class directly."""
        return cls(biomass_rxn=biomass_rxn, proton_met=proton_met,
                   components=tuple(components))

    def get(self, component_name: str) -> BiomassComponent:
        """Return the component with the given name; raise ``KeyError`` if there is none."""
        for c in self.components:
            if c.name == component_name:
                return c
        raise KeyError(f"BiomassConfig has no component named {component_name!r}")
