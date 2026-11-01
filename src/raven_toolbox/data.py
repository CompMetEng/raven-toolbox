"""Fetch and cache published data artefacts (KEGG reference model, tables, HMMs).

The counterpart of :mod:`raven_toolbox.binaries` for data: a version-pinned registry of
downloadable artefacts, fetched on first use, SHA256-verified, and cached under
platformdirs so end users do not rebuild them from a KEGG dump (maintainers build them;
see docs/maintaining_kegg_data.md).

Resolution for any artefact file:

    explicit local dir  →  cached copy  →  download from the registry (verify,
        cache)  →  FileNotFoundError with guidance

The registry lists the ``kegg`` and ``transporters`` datasets. A dataset that is not
registered makes ``ensure_data_file`` raise ``FileNotFoundError`` with guidance. Cache
layout::

    $XDG_CACHE_HOME/raven_toolbox/data/<dataset>-<version>/<filename>
    (or ~/.cache/raven_toolbox/data/... if XDG_CACHE_HOME is unset)
"""
from __future__ import annotations

import gzip
import os
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen

from raven_toolbox.binaries import _sha256

# dataset -> {"version": str, "files": {filename: {"url": str, "sha256": str}}}
# Baked snapshot of data/manifest.json (the cross-language source of truth). Assets
# are hosted in the toolbox-assets repo's releases. Regenerate this block from the
# manifest with `python scripts/make_registry_snippet.py sync` (never hand-edit).
_DATA_REGISTRY: dict = {
    "kegg": {
        "version": "kegg118",
        "files": {
            "kegg118_core.tar.gz": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/kegg118/kegg118_core.tar.gz",
                "sha256": "520ba8bb09d6ca479ca1ab84ea70bf7c1d7f82d8f6406b717cee293b7c8609ca",
            },
            "kegg118_eukaryotes.hmm.gz": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/kegg118/kegg118_eukaryotes.hmm.gz",
                "sha256": "05999a38f3a1a5186d19d4e415f5029c33af56a08edb75190bd98ea19437765a",
            },
            "kegg118_prokaryotes.hmm.gz": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/kegg118/kegg118_prokaryotes.hmm.gz",
                "sha256": "ff9f25389e4c9c1cd58c13ee8d48d68a23152d04761d2e5875a9b5de8051b7b4",
            },
            "kegg118_taxonomy.gz": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/kegg118/kegg118_taxonomy.gz",
                "sha256": "5499c9537ac2697592dd1b484f309027509472e25a9d39d925dc581b9eef46e2",
            },
        },
    },
    "transporters": {
        "version": "transporters-2026-07-01",
        "files": {
            "tcdb.dmnd": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/transporters-2026-07-01/tcdb.dmnd",
                "sha256": "ae8db23287ef415cad8a3004134b0d94024230f82afb5eb00a45941f8b8c04e6",
            },
            "transporter_pfam.hmm": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/transporters-2026-07-01/transporter_pfam.hmm",
                "sha256": "0c44e7bfc1390a08c914915a87b9b85398528be5db4294f91c33387c46c48f19",
            },
            "tcdb_substrates.tsv": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/transporters-2026-07-01/tcdb_substrates.tsv",
                "sha256": "f85f7b866b325d1e221089ccb5fd7a0491d1c26f8282c9ebb7c2d11bf3f1e79c",
            },
            "chebi_relations.tsv.gz": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/transporters-2026-07-01/chebi_relations.tsv.gz",
                "sha256": "b6910bf9c7a405e4f1fc112030123dd79c982112b8f460cebefd9235c621d192",
            },
        },
    },
}

# The core KEGG artefacts needed to build a model (no HMM libraries). These are
# the *base* names of the files bundled into the published ``<version>_core.tar.gz``
# (each stored version-prefixed inside the archive); ``ensure_kegg_data`` fetches the
# bundle and extracts these, and the build groups exactly this set.
CORE_KEGG_FILES = (
    "reference_model.yml.gz",
    "ko_reaction.tsv.gz",
    "ko_names.tsv.gz",
    "organism_gene_ko.tsv.gz",
    "rxn_flags.tsv.gz",
)


def _data_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "raven_toolbox" / "data"


def _maybe_autoload(registry: dict) -> None:
    """Populate the default registry from ``$RAVEN_PYTHON_MANIFEST`` on first use, if set.

    Fires only when the caller uses the default ``_DATA_REGISTRY``, that registry is empty
    and the environment variable points at a manifest. The registry as shipped lists the
    ``kegg`` and ``transporters`` datasets, so the manifest is loaded only when the
    registry has been emptied. Local import avoids an import cycle with
    :mod:`raven_toolbox.manifest`.
    """
    if registry is _DATA_REGISTRY and not registry and os.environ.get("RAVEN_PYTHON_MANIFEST"):
        from raven_toolbox import manifest as _manifest

        _manifest.load_into_registries()


def _bundle(dataset: str, registry: dict) -> dict:
    _maybe_autoload(registry)
    bundle = registry.get(dataset)
    if bundle is None:
        raise FileNotFoundError(
            f"No data artefacts registered for {dataset!r}. Either pass a local "
            f"directory of artefacts, or build them per docs/maintaining_kegg_data.md."
        )
    return bundle


def ensure_data_file(
    dataset: str,
    filename: str,
    *,
    version: str | None = None,
    registry: dict | None = None,
    verify: bool = False,
) -> Path:
    """Download (if needed) and return the cached path to one artefact file.

    Looks the file up in the registry for ``dataset`` (at ``version`` or the
    registry's default), downloads it to the version-pinned cache directory,
    verifies its SHA256, and returns the path. Reuses an already-cached copy.

    A freshly downloaded file is always SHA256-checked. ``verify`` additionally
    re-checks the SHA256 of an already-cached file (a mismatch means a corrupted cache, and
    the file is discarded and downloaded again); it is off by default, so a cache hit does
    not compute the hash.
    """
    registry = _DATA_REGISTRY if registry is None else registry
    bundle = _bundle(dataset, registry)
    ver = version or bundle["version"]
    entry = bundle.get("files", {}).get(filename)
    if entry is None:
        raise FileNotFoundError(
            f"{filename!r} is not registered for {dataset!r} {ver}. "
            f"Available: {sorted(bundle.get('files', {}))}."
        )

    dest_dir = _data_cache_dir() / f"{dataset}-{ver}"
    dest = dest_dir / filename
    if dest.exists():
        if not verify or _sha256(dest) == entry["sha256"]:
            return dest
        dest.unlink()  # corrupted cache → fall through and re-download

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urlopen(entry["url"], timeout=60) as resp, open(tmp, "wb") as out:  # noqa: S310 (trusted registry URLs)
        shutil.copyfileobj(resp, out)
    digest = _sha256(tmp)
    if digest != entry["sha256"]:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"SHA256 mismatch for {dataset}/{filename} ({ver}): "
            f"expected {entry['sha256']}, got {digest}."
        )
    tmp.replace(dest)
    return dest


def ensure_kegg_data(*, version: str | None = None, registry: dict | None = None) -> Path:
    """Ensure the core KEGG artefacts are cached; return their directory.

    Fetches the single ``<version>_core.tar.gz`` bundle (the gene-free reference
    model + the KO, reaction and organism-gene tables of :data:`CORE_KEGG_FILES`),
    SHA256-verifies it, and extracts the version-prefixed members into the cache
    directory on first use. The returned directory can be passed as the ``artefact_dir`` of
    :func:`get_kegg_model_for_organism_from_artefacts`. The HMM libraries and the
    taxonomy file are separate artefacts (see :func:`ensure_kegg_hmm_library`,
    :func:`ensure_kegg_taxonomy`).
    """
    registry = _DATA_REGISTRY if registry is None else registry
    ver = version or _bundle("kegg", registry)["version"]
    dest_dir = _data_cache_dir() / f"kegg-{ver}"
    archive = ensure_data_file("kegg", f"{ver}_core.tar.gz", version=ver, registry=registry)
    # Extract once; a marker avoids re-extracting (and re-reading the archive) per call.
    marker = dest_dir / ".core-extracted"
    if not marker.exists():
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest_dir, filter="data")  # safe extraction (matches download.py)
        marker.touch()
    return dest_dir


def ensure_kegg_hmm_library(
    domain: str,
    *,
    version: str | None = None,
    registry: dict | None = None,
) -> Path:
    """Ensure a domain HMM library is cached and decompressed; return the ``.hmm`` path.

    ``domain`` is ``"prokaryotes"`` or ``"eukaryotes"``. Fetches the gzipped
    concatenated library ``<version>_<domain>.hmm.gz`` and decompresses it once
    (cached). Returns the path to the ``.hmm`` flatfile, which is the argument for
    :func:`run_hmmsearch`; it searches the flatfile directly, without ``hmmpress``.

    The gzip flatfile is ~10x smaller than a binary index, is portable across HMMER
    versions and platforms, and serves MATLAB RAVEN as well.
    """
    registry = _DATA_REGISTRY if registry is None else registry
    ver = version or _bundle("kegg", registry)["version"]
    archive = ensure_data_file("kegg", f"{ver}_{domain}.hmm.gz", version=ver, registry=registry)
    library = archive.with_suffix("")  # strip ".gz" -> <version>_<domain>.hmm
    if not library.exists():
        tmp = library.with_name(library.name + ".part")
        with gzip.open(archive, "rb") as src, open(tmp, "wb") as out:
            shutil.copyfileobj(src, out)
        tmp.replace(library)
    return library


def ensure_kegg_taxonomy(*, version: str | None = None, registry: dict | None = None) -> Path:
    """Ensure the KEGG ``taxonomy`` artefact is cached; return its (gzipped) path.

    The gzipped KEGG ``taxonomy`` file is the source for domain classification and for
    regenerating the phylogenetic distance matrix (RAVEN's ``keggPhylDist``, which GECKO
    uses to pick the closest organism for kcat assignment) via
    :func:`raven_toolbox.reconstruction.kegg.phyl_dist`, which reads ``.gz`` directly. That
    capability needs only this published artefact and no MATLAB ``.mat`` file.
    """
    registry = _DATA_REGISTRY if registry is None else registry
    ver = version or _bundle("kegg", registry)["version"]
    return ensure_data_file("kegg", f"{ver}_taxonomy.gz", version=ver, registry=registry)
