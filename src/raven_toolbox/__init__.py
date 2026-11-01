"""raven_toolbox: Python counterpart of the RAVEN Toolbox, built on cobrapy.

raven_toolbox uses cobrapy for simulation, standard analyses, SBML I/O, and model
manipulation, and adds the RAVEN-specific functionality: de novo
reconstruction (KEGG and homology), context-specific modelling (tINIT and ftINIT),
metabolic task validation, connectivity gap-filling, omics integration (HPA),
sub-cellular localisation, N-model comparison, and the RAVEN-style I/O formats.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("raven-toolbox")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
