"""Load a toolbox-assets manifest into the runtime resolver registries.

A manifest is the language-agnostic source of truth for every downloadable
artefact (KEGG tables, HMMs, …) and external-binary bundle (BLAST, DIAMOND, HMMER).
It is stored in the data repository, in a Zenodo record, or in both; raven-toolbox and MATLAB RAVEN
both read the same JSON and verify each file's SHA256 after download. See
``data/manifest.schema.json`` for the format and ``data/manifest.example.json`` for a
worked example.

The manifest is a superset of the two runtime registries:

* ``manifest["data"]``     → :data:`raven_toolbox.data._DATA_REGISTRY`
* ``manifest["binaries"]`` → :data:`raven_toolbox.binaries._REGISTRY`

Usage::

    from raven_toolbox import manifest
    manifest.load_into_registries("https://github.com/CompMetEng/toolbox-assets/releases/download/manifest-v1/manifest.json")

or set ``RAVEN_PYTHON_MANIFEST`` (a path or URL) and the resolvers load it lazily on first
use::

    export RAVEN_PYTHON_MANIFEST=/path/to/manifest.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import urlopen

#: Environment variable holding a manifest path or URL; consulted lazily by the
#: resolvers when their registry is still empty.
ENV_MANIFEST = "RAVEN_PYTHON_MANIFEST"

#: Manifest format version this module understands.
SUPPORTED_VERSION = 1


def _read(source: str | os.PathLike) -> str:
    """Read manifest text from a local path or an http(s)/ftp URL."""
    s = str(source)
    if s.startswith(("http://", "https://", "ftp://")):
        with urlopen(s) as resp:  # noqa: S310 (trusted, user-supplied manifest source)
            return resp.read().decode("utf-8")
    return Path(s).read_text(encoding="utf-8")


def load_manifest(source: str | os.PathLike | None = None) -> dict:
    """Read and validate a manifest from ``source`` (path or URL) or ``$RAVEN_PYTHON_MANIFEST``."""
    source = source or os.environ.get(ENV_MANIFEST)
    if not source:
        raise ValueError(
            f"No manifest source: pass a path/URL or set ${ENV_MANIFEST}."
        )
    manifest = json.loads(_read(source))
    version = manifest.get("manifest_version")
    if version != SUPPORTED_VERSION:
        raise ValueError(
            f"Unsupported manifest_version {version!r} (this raven-toolbox understands "
            f"{SUPPORTED_VERSION})."
        )
    return manifest


def to_data_registry(manifest: dict) -> dict:
    """Project ``manifest['data']`` onto the ``raven_toolbox.data._DATA_REGISTRY`` shape."""
    return {
        dataset: {
            "version": spec["version"],
            "files": {
                name: {"url": f["url"], "sha256": f["sha256"]}
                for name, f in spec["files"].items()
            },
        }
        for dataset, spec in manifest.get("data", {}).items()
    }


def to_binary_registry(manifest: dict) -> dict:
    """Project ``manifest['binaries']`` onto the ``raven_toolbox.binaries._REGISTRY`` shape."""
    return {
        bundle: {
            "version": spec["version"],
            "provides": list(spec["provides"]),
            "platforms": {
                key: {"url": f["url"], "sha256": f["sha256"]}
                for key, f in spec["platforms"].items()
            },
        }
        for bundle, spec in manifest.get("binaries", {}).items()
    }


def load_into_registries(
    source: str | os.PathLike | None = None, *, replace: bool = False
) -> dict:
    """Load a manifest and merge it into the live data and binary registries.

    Parameters
    ----------
    source
        Manifest path or URL; defaults to ``$RAVEN_PYTHON_MANIFEST``.
    replace
        If True, clear the existing registries first; otherwise merge; on a conflict the manifest entry replaces the registry entry.

    Returns
    -------
    dict
        The parsed manifest.
    """
    manifest = load_manifest(source)
    # Imported here (not at module top) so data/binaries can lazily call back
    # into this module without an import cycle.
    from raven_toolbox import binaries as _binaries
    from raven_toolbox import data as _data

    if replace:
        _data._DATA_REGISTRY.clear()
        _binaries._REGISTRY.clear()
    _data._DATA_REGISTRY.update(to_data_registry(manifest))
    _binaries._REGISTRY.update(to_binary_registry(manifest))
    return manifest
