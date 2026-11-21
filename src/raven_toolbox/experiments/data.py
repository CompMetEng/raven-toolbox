"""Read the measured conditions of a model from a folder.

The folder holds up to three files, each optional:

* ``fluxData.tsv``: flux conditions (carbon-source uptake limited to the measured rate; growth
  rate and measured exchange fluxes compared).
* ``maxGrowth.tsv``: maximum-growth conditions (uptake unlimited; only the maximal growth rate
  compared). The model objective must be growth, see
  :func:`~raven_toolbox.experiments.simulate_conditions`.
* ``zeroFlux.tsv``: the header ``Rxns`` and one reaction id per line, for reactions that carry no
  flux in any condition. Flux through them is penalised in the flux conditions.

The first two have the layout of :func:`~raven_toolbox.experiments.flux_data.load_flux_data`. A
missing file gives ``None`` (an empty list for ``zeroFlux.tsv``), not an error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .flux_data import FluxData, load_flux_data

_FLUX_DATA_FILENAME = "fluxData.tsv"
_MAX_GROWTH_FILENAME = "maxGrowth.tsv"
_ZERO_EXCH_FILENAME = "zeroFlux.tsv"


@dataclass
class ExperimentData:
    """The measured conditions of a model, as read by :func:`load_experiment_data`.

    Attributes
    ----------
    flux_data : FluxData or None
        Flux conditions from ``fluxData.tsv``; ``None`` if the file is missing.
    max_growth : FluxData or None
        Maximum-growth conditions from ``maxGrowth.tsv``; ``None`` if the file is missing.
    zero_flux : list of str
        Ids of reactions that carry no flux in any condition, from ``zeroFlux.tsv``. Empty if the file
        is missing.
    """

    flux_data: FluxData | None = None
    max_growth: FluxData | None = None
    zero_flux: list[str] = field(default_factory=list)


def load_experiment_data(data_dir: str | Path) -> ExperimentData:
    r"""Read ``fluxData.tsv``, ``maxGrowth.tsv`` and ``zeroFlux.tsv`` from a folder.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Folder that holds the files. Each is optional.

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


def _load_optional_flux_data(path: Path) -> FluxData | None:
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
