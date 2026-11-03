# Contributing to raven-toolbox

User documentation lives on [raven-docs](https://github.com/CompMetEng/raven-docs),
which covers both this package and MATLAB RAVEN. This file is for working on the
package itself.

## Development setup

```bash
pip install -e ".[dev,excel]"
```

Three checks run in CI, and all three run locally the same way:

```bash
ruff check .                                          # lint, pinned to 0.15.15
mypy                                                  # types
pytest -v --durations=20 --timeout=300 --timeout-method=thread
```

`ruff` is pinned to an exact version so a lint failure is never a version
difference. Line length is 100 and the target is Python 3.11.

A separate nightly workflow runs the parity suite in `tests/parity/` against a
checkout of MATLAB RAVEN, comparing the two implementations on the same inputs.
It is slower than the rest and is not part of the per-push CI.

## Documentation

Prose documentation for users belongs on raven-docs, not here. Function-level
documentation belongs in the docstrings: raven-docs generates its API reference
from them at build time, so a docstring edit reaches the site without anything
being copied across.

Write new and changed public docstrings in NumPy style (Parameters, Returns and, where
useful, Examples sections). Many existing docstrings are still prose.

## Publishing the assets

The maintainers publish the KEGG assets and the binary bundles. [Update the KEGG assets](https://raven-docs.readthedocs.io/en/latest/installation/updating-kegg-assets.html) and [Update the binary bundles](https://raven-docs.readthedocs.io/en/latest/installation/updating-binaries.html) on raven-docs describe the procedures, and the scripts they use are in `scripts/`.
