"""Fixtures for the cross-language parity checks.

Everything here skips rather than fails when its reference material is absent:
a checkout without a RAVEN clone, or without regenerated oracles, still has a
green suite. What must never happen is a parity test that silently passes
because it compared nothing -- so each fixture skips with a message naming
exactly what is missing and how to supply it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ORACLE_DIR = Path(__file__).resolve().parent.parent / "data" / "parity" / "oracles"
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "parity"


@pytest.fixture(scope="session")
def raven_root() -> Path:
    """A MATLAB RAVEN checkout, from ``$RAVEN_ROOT``.

    RAVEN is GPL and this package is MIT, so its files are read in place and
    never vendored here.
    """
    root = os.environ.get("RAVEN_ROOT")
    if not root:
        pytest.skip(
            "RAVEN_ROOT is not set. Clone RAVEN and point at it to run the "
            "cross-language checks: git clone --depth 1 -b develop "
            "https://github.com/CompMetEng/RAVEN && RAVEN_ROOT=$PWD/RAVEN "
            "pytest -m parity"
        )
    path = Path(root)
    if not (path / "src" / "io").is_dir():
        pytest.skip(f"RAVEN_ROOT={path} does not look like a RAVEN checkout")
    return path


@pytest.fixture(scope="session")
def raven_models(raven_root: Path) -> dict[str, Path]:
    """RAVEN-authored YAML models, by file name.

    These were written by MATLAB RAVEN's own ``writeYAMLmodel``, which makes
    them reference data for the Python reader without needing MATLAB.
    """
    tutorial = raven_root / "tutorials" / "data"
    found = {p.name: p for p in sorted(tutorial.rglob("*.yml"))}
    if not found:
        pytest.skip(f"no YAML models under {tutorial}")
    return found


@pytest.fixture
def oracle():
    """Load a recorded MATLAB answer, or skip if it has not been generated."""

    def _load(name: str):
        path = ORACLE_DIR / f"{name}.json"
        if not path.is_file():
            pytest.skip(
                f"oracle '{name}' has not been generated. Run "
                f"scripts/parity/generate_oracles.m in MATLAB with RAVEN on the "
                f"path, then commit tests/data/parity/oracles/{name}.json"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    return _load


@pytest.fixture(scope="session")
def parity_fixture_dir() -> Path:
    """Models authored in this repository, used as input on both sides."""
    return FIXTURE_DIR
