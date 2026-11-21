"""Read the measured conditions of a model from a folder.

A folder with measurements can hold three files, and each of them is optional:

* ``fluxData.tsv``: flux conditions. The uptake of the carbon source of a condition is limited
  to the measured rate, and the growth rate and the measured exchange fluxes of the model are
  compared with the measurements.
* ``maxGrowth.tsv``: maximum-growth conditions. The uptake of the carbon source is unlimited,
  and only the maximal growth rate of the model is compared with the measured growth rate. The
  model objective has to be the growth for the simulation to give the maximal growth rate, and
  the simulation warns if it is not (see
  :func:`~raven_toolbox.experiments.simulate_conditions`).
* ``zeroFlux.tsv``: reactions that carry no flux in any condition, one reaction id per line
  below the header ``Rxns``. A model that carries flux in them is penalised in the flux
  conditions.

``fluxData.tsv`` and ``maxGrowth.tsv`` have the layout described in
:func:`~raven_toolbox.experiments.flux_data.load_flux_data`. A missing file gives ``None`` (or
an empty list for ``zeroFlux.tsv``), not an error.
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
    """The measured conditions of a model, as read from a folder by :func:`load_experiment_data`.

    Attributes
    ----------
    flux_data : FluxData or None
        Flux conditions, from ``fluxData.tsv``: growth rate and exchange fluxes that the model is
        compared with, at a limited uptake of the carbon source. ``None`` if the file is missing.
    max_growth : FluxData or None
        Maximum-growth conditions, from ``maxGrowth.tsv``: the growth rate that the model reaches
        when the uptake of the carbon source is unlimited. ``None`` if the file is missing.
    zero_flux : list of str
        Ids of reactions that carry no flux in any condition, from ``zeroFlux.tsv``. Empty if the
        file is missing.
    """

    flux_data: Optional[FluxData] = None
    max_growth: Optional[FluxData] = None
    zero_flux: list[str] = field(default_factory=list)


def load_experiment_data(data_dir: str | Path) -> ExperimentData:
    r"""Read ``fluxData.tsv``, ``maxGrowth.tsv`` and ``zeroFlux.tsv`` from a folder.

    Each file is optional; a missing file gives ``None`` (``flux_data`` and ``max_growth``) or an
    empty list (``zero_flux``) and no error. See the module documentation for what the files hold.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Folder that holds the files.

    Returns
    -------
    ExperimentData
        The measured conditions.

    See Also
    --------
    load_flux_data : Reads one flux file.
    simulate_conditions : Simulates the conditions that were read.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from raven_toolbox.experiments import load_experiment_data
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> _ = (folder / "zeroFlux.tsv").write_text("Rxns\nEX_byproduct\n")
    >>> experiment_data = load_experiment_data(folder)
    >>> experiment_data.flux_data is None, experiment_data.zero_flux
    (True, ['EX_byproduct'])
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
