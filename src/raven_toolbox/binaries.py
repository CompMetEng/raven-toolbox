"""Locate and provision external command-line binaries (BLAST+, DIAMOND, …).

Shared across tools (not homology-specific). Resolution order for any executable::

    explicit path arg  →  env var (RAVEN_PYTHON_<TOOL>)  →  shutil.which (PATH)
      →  ensure_binary  (download the version-pinned ZIP from a raven_toolbox release,
                         verify SHA256, cache, return the path)
      →  FileNotFoundError with install guidance

So a pre-installed/conda binary always wins; the bundled ZIP is the zero-setup
fallback. See https://github.com/SysBioChalmers/raven-gecko-parity/blob/main/docs/maintaining_binaries.md
for how the release ZIPs and the registry are produced and updated.
"""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

# Registry of bundled binaries, hosted as release assets in the toolbox-assets repo.
# Keyed by *bundle*; one bundle can provide several executables (e.g. "blast" ->
# blastp + makeblastdb). Baked snapshot of data/manifest.json's `binaries` block —
# regenerate from the manifest with `python scripts/make_registry_snippet.py sync`
# (never hand-edit). See https://github.com/SysBioChalmers/raven-gecko-parity/blob/main/docs/maintaining_binaries.md.
#   bundle -> {version, provides:[exe...], platforms:{"<os>-<arch>": {url, sha256}}}
_REGISTRY: dict = {
    "blast": {
        "version": "2.17.0",
        "provides": ["blastp", "makeblastdb"],
        "platforms": {
            "linux-x86_64": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/blast-2.17.0/blast-2.17.0-linux-x86_64.zip",
                "sha256": "c887deae9f3ac85bb5a0bbbf9af254d00f3bebf9764304a9582db39e1c843085",
            },
            "macos-arm64": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/blast-2.17.0/blast-2.17.0-macos-arm64.zip",
                "sha256": "f1ff0b9e718fbb252d240e31e4f85f842f6ef8e99a0e97bf5f66390295c795da",
            },
            "windows-x86_64": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/blast-2.17.0/blast-2.17.0-windows-x86_64.zip",
                "sha256": "b80d6ee1c1f85b8c18ae009e7d2ee844738d1a17338928570f5113ecd16e8326",
            },
        },
    },
    "diamond": {
        "version": "2.1.17",
        "provides": ["diamond"],
        "platforms": {
            "linux-x86_64": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/diamond-2.1.17/diamond-2.1.17-linux-x86_64.zip",
                "sha256": "6e432f8b205fb14c355c330cd4d802acb7decaaf17cfbe90573c614b6dc976c8",
            },
            "macos-arm64": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/diamond-2.1.17/diamond-2.1.17-macos-arm64.zip",
                "sha256": "3f929c151b8d8061391d73510b2011e9c8820fa5e2b6b9ae142c3b4dfccced5d",
            },
            "windows-x86_64": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/diamond-2.1.17/diamond-2.1.17-windows-x86_64.zip",
                "sha256": "daabab3d6072f4a37e701a5e02944f0fbcf8791ae5daa125fb7b5daed4a9be6d",
            },
        },
    },
    "hmmer": {
        "version": "3.4.0",
        "provides": ["hmmsearch"],
        "platforms": {
            "windows-x86_64": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/hmmer-3.3.2/hmmer-3.3.2-windows-x86_64.zip",
                "sha256": "19ce00c5dcdead0cdcb1f89cfafac299521afa79296e9a671baf73a64b7385eb",
            },
            "linux-x86_64": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/hmmer-3.4.0/hmmer-3.4.0-linux-x86_64.zip",
                "sha256": "d60e6b4bead44cf6d0cbd5711f9bda4a11cf72d7db07545ee0a43160b60b9fac",
            },
            "macos-arm64": {
                "url": "https://github.com/CompMetEng/toolbox-assets/releases/download/hmmer-3.4.0/hmmer-3.4.0-macos-arm64.zip",
                "sha256": "89596237e4a6a325a5f8526983696a1197e6970b1352db81a9817a165cd62f82",
            },
        },
    },
}

# Environment variable overrides per executable.
_ENV_VARS = {
    "diamond": "RAVEN_PYTHON_DIAMOND",
    "blastp": "RAVEN_PYTHON_BLASTP",
    "makeblastdb": "RAVEN_PYTHON_MAKEBLASTDB",
    "hmmbuild": "RAVEN_PYTHON_HMMBUILD",
    "hmmpress": "RAVEN_PYTHON_HMMPRESS",
    "hmmsearch": "RAVEN_PYTHON_HMMSEARCH",
    "hmmscan": "RAVEN_PYTHON_HMMSCAN",
    "mafft": "RAVEN_PYTHON_MAFFT",
    "cd-hit": "RAVEN_PYTHON_CDHIT",
}

# Named binary sets for the two audiences (provisioned together by the
# ``raven-toolbox-binaries`` CLI). Which of these actually have a bundle for a
# given OS/arch is decided by the registry, not here — e.g. native Windows has no
# MAFFT/CD-HIT build, so those resolve to an actionable "use conda/WSL2" error.
BINARY_SETS: dict[str, tuple[str, ...]] = {
    # End users: homology search (BLAST/DIAMOND) + KEGG HMM query (hmmsearch).
    "runtime": ("blastp", "makeblastdb", "diamond", "hmmsearch"),
    # Maintainers/developers building the KEGG HMM libraries (step 3b.3).
    "build": ("hmmbuild", "mafft", "cd-hit"),
}

# Env var to disable lazy first-use downloads (auto-fetch). Unset/anything-else =
# enabled (the zero-setup default); these values turn it off.
_AUTOFETCH_ENV = "RAVEN_PYTHON_AUTOFETCH"
_AUTOFETCH_OFF = {"0", "false", "no", "off"}


def platform_key() -> str:
    """Return the ``<os>-<arch>`` key used in the registry (e.g. ``linux-x86_64``)."""
    system = {"linux": "linux", "darwin": "macos", "windows": "windows"}.get(
        platform.system().lower(), platform.system().lower()
    )
    machine = platform.machine().lower()
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(
        machine, machine
    )
    return f"{system}-{arch}"


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "raven_toolbox" / "binaries"


def _bundle_for(executable: str, registry: dict):
    for name, bundle in registry.items():
        if executable in bundle.get("provides", []):
            return name, bundle
    return None, None


def _maybe_autoload(registry: dict) -> None:
    """Populate the default registry from ``$RAVEN_PYTHON_MANIFEST`` on first use, if set.

    Only fires when the caller is using the default (still-empty) ``_REGISTRY`` and the
    environment variable points at a manifest; a caller that passes its own ``registry``
    is left untouched. The import is local to avoid a cycle with :mod:`raven_toolbox.manifest`.
    """
    if registry is _REGISTRY and not registry and os.environ.get("RAVEN_PYTHON_MANIFEST"):
        from raven_toolbox import manifest as _manifest

        _manifest.load_into_registries()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract_zip(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    """Extract ``zf`` into ``dest_dir``, rejecting members that escape it.

    ``ZipFile.extractall`` has no path-traversal guard (unlike tarfile's
    ``filter="data"`` used in reconstruction/kegg/download.py), so a malicious or
    corrupt archive could write outside the cache via absolute paths, ``..``, or
    symlink members. SHA256 + HTTPS already make a hostile archive unlikely; this
    is defence in depth.
    """
    dest = dest_dir.resolve()
    for info in zf.infolist():
        member = info.filename
        target = (dest_dir / member).resolve()
        if target != dest and dest not in target.parents:
            raise ValueError(f"unsafe path in archive (path traversal): {member!r}")
        # Reject symlink members: extractall recreates them as real symlinks whose
        # target the path check above never validates, so one could point outside
        # dest_dir (or a later member be written through it).
        if (info.external_attr >> 16) & 0o170000 == 0o120000:  # stat.S_IFLNK
            raise ValueError(f"unsafe symlink in archive: {member!r}")
    zf.extractall(dest_dir)


def _make_executable(path: Path) -> None:
    """Add the execute bit for everyone who can already read the file.

    ZIP archives carry Unix permissions in a field ``zipfile`` does not restore,
    so every extracted file arrives non-executable on Linux and macOS. Derived
    from the read bits rather than set to a fixed 0o755 so a restrictive umask
    is not silently widened.
    """
    mode = path.stat().st_mode
    path.chmod(mode | ((mode & 0o444) >> 2))


def ensure_binary(executable: str, *, registry: dict | None = None) -> Path:
    """Download (if needed) and return the path to a bundled ``executable``.

    Consults the registry for the current platform, downloads the pinned ZIP,
    verifies its SHA256, extracts it into the cache, and returns the executable
    path. Raises ``FileNotFoundError`` if no bundle for this platform is hosted.
    Pass ``registry`` to consult an alternate bundle registry instead of the
    built-in one (mainly for tests or a custom manifest).
    """
    registry = _REGISTRY if registry is None else registry
    _maybe_autoload(registry)
    bundle_name, bundle = _bundle_for(executable, registry)
    if bundle is None:
        raise FileNotFoundError(
            f"No bundled binary registered for {executable!r}. Install it (e.g. "
            f"`conda install -c bioconda {executable}`) or pass an explicit path."
        )
    key = platform_key()
    entry = bundle.get("platforms", {}).get(key)
    if entry is None:
        raise FileNotFoundError(
            f"No bundled {executable!r} for platform {key!r}. Install it "
            f"(e.g. `conda install -c bioconda {executable}`), set "
            f"{_ENV_VARS.get(executable, 'the binary path')}, or pass binary=."
        )

    # Windows bundles ship the executable with a .exe suffix (e.g. hmmsearch.exe,
    # blastp.exe); prefer that on Windows but tolerate a bare name too.
    candidates = [f"{executable}.exe", executable] if key.startswith("windows-") else [executable]
    dest_dir = _cache_dir() / f"{bundle_name}-{bundle['version']}-{key}"

    def _find_exe() -> Path | None:
        return next((dest_dir / name for name in candidates if (dest_dir / name).is_file()), None)

    cached = _find_exe()
    if cached is not None:
        # Re-applied on the cached path too, not only after extraction: a stale
        # cache dir may only have the one executable it was asked for marked,
        # leaving the rest unusable.
        _make_executable(cached)
        return cached

    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / "_download.zip"
    # Download into a sibling .part file and rename on success — an interrupted
    # download leaves the partial behind .part, never as a half-complete .zip
    # that a later run might mistake for a finished one. Mirrors data.py.
    part = archive.with_suffix(archive.suffix + ".part")
    try:
        with urlopen(entry["url"], timeout=60) as resp, open(part, "wb") as out:  # noqa: S310
            shutil.copyfileobj(resp, out)
        digest = _sha256(part)
        if digest != entry["sha256"]:
            raise ValueError(
                f"SHA256 mismatch for {executable!r} ({key}): "
                f"expected {entry['sha256']}, got {digest}."
            )
        os.replace(part, archive)
    finally:
        part.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as zf:
        _safe_extract_zip(zf, dest_dir)
    archive.unlink(missing_ok=True)
    exe = _find_exe()
    if exe is None:
        raise FileNotFoundError(
            f"None of {candidates} found in the extracted bundle at {dest_dir}."
        )
    # Every executable the bundle provides, not just the one asked for. The BLAST
    # bundle ships blastp *and* makeblastdb; marking only the requested one would
    # leave the other non-executable.
    for provided in bundle.get("provides", []):
        for name in ([f"{provided}.exe", provided] if key.startswith("windows-") else [provided]):
            sibling = dest_dir / name
            if sibling.is_file():
                _make_executable(sibling)
    return exe


def autofetch_enabled() -> bool:
    """Whether lazy first-use downloads are allowed.

    On by default (the zero-setup behaviour). Set ``RAVEN_PYTHON_AUTOFETCH`` to
    ``0``/``false``/``no``/``off`` (any case) to disable, so :func:`resolve_binary`
    stops at PATH and never reaches the network — for air-gapped or
    strictly conda/system-managed setups.
    """
    val = os.environ.get(_AUTOFETCH_ENV)
    return val is None or val.strip().lower() not in _AUTOFETCH_OFF


def resolve_binary(executable: str, *, binary: str | os.PathLike | None = None) -> str:
    """Resolve an executable to a path: arg → env var → PATH → bundled ZIP → error.

    The bundled-ZIP step is skipped when auto-fetch is disabled
    (:func:`autofetch_enabled`); resolution then stops at PATH with an actionable
    error instead of downloading.
    """
    if binary is not None:
        return os.fspath(binary)
    env_var = _ENV_VARS.get(executable)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    found = shutil.which(executable)
    if found:
        return found
    hint = (
        f"Install it (e.g. `conda install -c bioconda {executable}`), put it on "
        f"PATH, set {env_var or 'the binary path'}, pass binary=, or run "
        f"`raven-toolbox-binaries`"
    )
    if not autofetch_enabled():
        raise FileNotFoundError(
            f"Could not find {executable!r} and auto-fetch is disabled "
            f"({_AUTOFETCH_ENV}={os.environ.get(_AUTOFETCH_ENV)!r}). {hint}."
        )
    try:
        return os.fspath(ensure_binary(executable))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Could not find {executable!r}. {hint}. ({exc})") from exc


# --------------------------------------------------------------------------- #
# Provisioning a whole set (used by the raven-toolbox-binaries CLI)
# --------------------------------------------------------------------------- #
@dataclass
class BinaryStatus:
    """Outcome of provisioning one executable.

    ``status`` is one of ``"present"`` (already on PATH / via env var),
    ``"downloaded"`` (fetched from a bundle just now), ``"unavailable"`` (no bundle
    hosted for this OS/arch — install via conda/WSL2), or ``"error"`` (download or
    verification failed). ``detail`` is the path (present/downloaded) or message.
    """

    executable: str
    status: str
    detail: str


def executables_for_set(set_name: str) -> tuple[str, ...]:
    """Return the executables in a named set (``"all"`` = the union of every set)."""
    if set_name == "all":
        seen: list[str] = []
        for execs in BINARY_SETS.values():
            seen.extend(e for e in execs if e not in seen)
        return tuple(seen)
    try:
        return BINARY_SETS[set_name]
    except KeyError:
        raise ValueError(
            f"Unknown binary set {set_name!r}. Choose from "
            f"{sorted(BINARY_SETS) + ['all']}."
        ) from None


def provision_binaries(
    executables: tuple[str, ...] | list[str],
    *,
    registry: dict | None = None,
    prefer_existing: bool = True,
) -> list[BinaryStatus]:
    """Ensure each executable is available, reporting per-tool outcomes.

    With ``prefer_existing`` (default) a tool already on PATH or pointed at by its
    env var is left as-is (``"present"``) and not downloaded. Otherwise the bundle
    is fetched via :func:`ensure_binary` (``registry``, if given, is forwarded to
    it in place of the built-in registry). Never raises for an individual tool —
    a missing platform bundle becomes ``"unavailable"`` and a failed download
    ``"error"``, so a caller can report the whole set at once.
    """
    out: list[BinaryStatus] = []
    for exe in executables:
        if prefer_existing:
            env_var = _ENV_VARS.get(exe)
            existing = (os.environ.get(env_var) if env_var else None) or shutil.which(exe)
            if existing:
                out.append(BinaryStatus(exe, "present", existing))
                continue
        try:
            out.append(BinaryStatus(exe, "downloaded", str(ensure_binary(exe, registry=registry))))
        except FileNotFoundError as exc:
            out.append(BinaryStatus(exe, "unavailable", str(exc)))
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            out.append(BinaryStatus(exe, "error", str(exc)))
    return out
