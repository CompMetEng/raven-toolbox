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

GAM and NGAM are fitted to chemostat uptakes and a maintenance coefficient with
:func:`fit_maintenance` (see :mod:`raven_toolbox.biomass.maintenance`).
"""
from raven_toolbox.biomass.config import BiomassComponent, BiomassConfig
from raven_toolbox.biomass.fit_parameters import (
    FitParametersResult,
    ParameterPosition,
    fit_parameters,
)
from raven_toolbox.biomass.gam import get_gam, set_gam
from raven_toolbox.biomass.maintenance import (
    MaintenanceConfig,
    MaintenanceFit,
    bootstrap_gam,
    fit_gam,
    fit_maintenance,
    huber_loss,
    log_ratio,
    min_uptake,
    ngam_from_maintenance_coefficient,
    study_weights,
    uptake_grid,
)
from raven_toolbox.biomass.scale import (
    rescale_pseudoreaction,
    scale_biomass,
    sum_biomass,
)

__all__ = [
    "BiomassComponent",
    "BiomassConfig",
    "FitParametersResult",
    "MaintenanceConfig",
    "MaintenanceFit",
    "ParameterPosition",
    "bootstrap_gam",
    "fit_gam",
    "fit_maintenance",
    "fit_parameters",
    "get_gam",
    "huber_loss",
    "log_ratio",
    "min_uptake",
    "ngam_from_maintenance_coefficient",
    "rescale_pseudoreaction",
    "scale_biomass",
    "set_gam",
    "study_weights",
    "sum_biomass",
    "uptake_grid",
]
