"""Load the measured growth rates and fluxes of the conditions.

The flux and max-growth files, ``fluxData.tsv`` and ``maxGrowth.tsv``,
share the column layout
``Condition Ptot grRate <met> (<rxn>) ... weight source``, so both use
:func:`~raven_toolbox.experiments.flux_data.load_flux_data`; only ``zeroFlux.tsv``
(a single ``Rxns`` id column) needs a loader of its own. A ``weight`` of 0 leaves a
condition out of the distance to the measurements.

All three files are optional: a missing file yields ``None`` (or an empty
list for ``zero_flux``), never an exception.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .flux_data import FluxData, load_flux_data

_FLUX_DATA_FILENAME = "fluxData.tsv"
_MAX_GROWTH_FILENAME = "maxGrowth.tsv"
_ZERO_EXCH_FILENAME = "zeroFlux.tsv"


@dataclass
class ExperimentData:
    """Measured conditions.

    Attributes
    ----------
    flux_data
        Per-condition growth/exchange-flux measurements, from
        ``fluxData.tsv``. ``None`` if that file isn't present.
    max_growth
        Per-condition maximum-growth-rate measurements (one active
        carbon source per condition, at ``-1000`` i.e. unconstrained),
        from ``maxGrowth.tsv``. ``None`` if that file isn't
        present.
    zero_flux
        Reaction IDs assumed to carry zero flux, from
        ``zeroFlux.tsv``. Empty if that file isn't present.
    """

    flux_data: Optional[FluxData] = None
    max_growth: Optional[FluxData] = None
    zero_flux: list[str] = field(default_factory=list)


def load_experiment_data(data_dir: str | Path) -> ExperimentData:
    """Load the three experimental data files.

    Each file is optional and resolved as
    ``<data_dir>/<filename>``; a missing file yields
    ``None`` (``flux_data``/``max_growth``) or an empty list
    (``zero_flux``) rather than raising.

    Parameters
    ----------
    data_dir
        The folder that holds the files.

    Returns
    -------
    ExperimentData
    """
    data_dir = Path(data_dir)
    return ExperimentData(
        flux_data=_load_optional_flux_data(data_dir / _FLUX_DATA_FILENAME),
        max_growth=_load_optional_flux_data(data_dir / _MAX_GROWTH_FILENAME),
        zero_flux=_load_zero_exch(data_dir / _ZERO_EXCH_FILENAME),
    )


def _load_optional_flux_data(path: Path) -> Optional[FluxData]:
    if not path.is_file():
        return None
    return load_flux_data(path)


def _load_zero_exch(path: Path) -> list[str]:
    """Parse ``zeroFlux.tsv``: a single ``Rxns`` header, then
    one reaction ID per line."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rxn_ids = [line.strip() for line in lines[1:] if line.strip()]
    return rxn_ids
