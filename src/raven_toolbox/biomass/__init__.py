"""Biomass equation manipulation, component scaling and biomass-fraction reporting.

The functions take a :class:`BiomassConfig` that describes the component layout of
the model. The example builds one ``BiomassConfig`` and passes it to each operation:

.. code-block:: python

    cfg = BiomassConfig(
        biomass_rxn="r_4041",
        proton_met="s_0794",
        components=[
            BiomassComponent("protein", "protein pseudoreaction",
                             mass_strategy="mw_minus_2h"),
            BiomassComponent("carbohydrate", "carbohydrate pseudoreaction"),
            BiomassComponent("lipid_backbone", "lipid backbone pseudoreaction",
                             mass_strategy="grams"),
            BiomassComponent("RNA", "RNA pseudoreaction",
                             mass_strategy="mw_minus_water"),
            BiomassComponent("DNA", "DNA pseudoreaction",
                             mass_strategy="mw_minus_water"),
            BiomassComponent("ion", "ion pseudoreaction"),
            BiomassComponent("cofactor", "cofactor pseudoreaction"),
        ],
    )
    fractions = sum_biomass(model, cfg)            # {'protein': 0.46, ...}
    scale_biomass(model, cfg, "protein", 0.50, balance_out="carbohydrate")
    set_gam(model, 80, biomass_rxn=cfg.biomass_rxn,
            cofactor_met_names=("ATP", "ADP", "H2O", "H+", "phosphate"))
"""
from raven_toolbox.biomass.config import BiomassComponent, BiomassConfig
from raven_toolbox.biomass.fit_parameters import (
    FitParametersResult,
    ParameterPosition,
    fit_parameters,
)
from raven_toolbox.biomass.gam import set_gam
from raven_toolbox.biomass.scale import (
    rescale_pseudoreaction,
    scale_biomass,
    sum_biomass,
)

__all__ = [
    "BiomassComponent",
    "BiomassConfig",
    "FitParametersResult",
    "ParameterPosition",
    "fit_parameters",
    "rescale_pseudoreaction",
    "scale_biomass",
    "set_gam",
    "sum_biomass",
]
