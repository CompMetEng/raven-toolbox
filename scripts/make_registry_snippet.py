#!/usr/bin/env python
"""Emit ready-to-paste registry entries for published artefacts / binary ZIPs.

Computes the SHA256 of each file and prints the Python/JSON entry to merge into
``raven_toolbox.data._DATA_REGISTRY`` (data artefacts) or ``raven_toolbox.binaries._REGISTRY``
(binary bundles). Run once per release, after uploading the files to the release.

Examples
--------
Pass ``--tag`` (the GitHub release tag the assets were uploaded to); the script builds the
``https://github.com/CompMetEng/toolbox-assets/releases/download/<tag>`` asset prefix
itself, so you never hand it a full URL (and can't fumble the ``releases/download`` path).

Data artefacts (KEGG reference model + tables + HMM libraries) for one release::

    python scripts/make_registry_snippet.py data \\
        --dataset kegg --version kegg116 --dir artefacts --tag v0.3.0

Binary bundle (one ZIP per platform, named ``<bundle>-<version>-<os>-<arch>.zip``)::

    python scripts/make_registry_snippet.py binary \\
        --bundle blast --version 2.16.0 --provides blastp makeblastdb --dir zips \\
        --tag blast-2.16.0

Add/update an entry in the shared ``manifest.json`` (the single source of truth read by
both raven-toolbox and MATLAB RAVEN — see data/manifest.schema.json)::

    python scripts/make_registry_snippet.py manifest --manifest data/manifest.json \\
        --target data --dataset kegg --version kegg116 --dir artefacts --tag v0.3.0 \\
        --doi 10.5281/zenodo.0000000

    python scripts/make_registry_snippet.py manifest --manifest data/manifest.json \\
        --target binary --bundle diamond --version 2.1.9 --provides diamond --dir zips \\
        --tag diamond-2.1.9 --license GPL-3.0-only

The SHA256 helper is shared with the runtime resolvers (``raven_toolbox.binaries``), so
published checksums always match what ``ensure_data`` / ``ensure_binary`` verify.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from raven_toolbox.binaries import _sha256

#: Default GitHub repository whose release assets the registry/manifest point at.
DEFAULT_REPO = "CompMetEng/toolbox-assets"


def release_base_url(tag: str, repo: str = DEFAULT_REPO) -> str:
    """GitHub release-asset download prefix for ``tag`` (e.g. 'v0.3.0').

    Builds the full ``https://github.com/<repo>/releases/download/<tag>`` URL so callers
    only supply the release tag — the fixed ``releases/download`` path can't be mistyped.
    """
    return f"https://github.com/{repo}/releases/download/{tag}"


def _files_in(directory: Path) -> list[Path]:
    """Regular, non-hidden files in ``directory``, sorted by name."""
    return sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))


def data_entry(dataset: str, version: str, base_url: str, directory: Path) -> dict:
    """Build the ``_DATA_REGISTRY[dataset]`` entry for every file in ``directory``."""
    base = base_url.rstrip("/")
    files = {
        p.name: {"url": f"{base}/{p.name}", "sha256": _sha256(p)} for p in _files_in(directory)
    }
    if not files:
        raise SystemExit(f"No files found in {directory}")
    return {"version": version, "files": files}


def binary_entry(
    bundle: str, version: str, provides: list[str], base_url: str, directory: Path
) -> dict:
    """Build the ``_REGISTRY[bundle]`` entry from ``<bundle>-<version>-<os>-<arch>.zip``."""
    base = base_url.rstrip("/")
    prefix = f"{bundle}-{version}-"
    platforms = {}
    for zip_path in directory.glob(f"{prefix}*.zip"):
        platform = zip_path.name[len(prefix) : -len(".zip")]
        platforms[platform] = {"url": f"{base}/{zip_path.name}", "sha256": _sha256(zip_path)}
    if not platforms:
        raise SystemExit(f"No {prefix}*.zip files found in {directory}")
    return {"version": version, "provides": provides, "platforms": dict(sorted(platforms.items()))}


def render(key: str, entry: dict) -> str:
    """Render ``{key: entry}`` as an indented JSON block (valid Python to paste)."""
    return json.dumps({key: entry}, indent=4)


# --- sync: regenerate the baked Python registries from manifest.json ---------
# The manifest is the single source of truth; the two baked dicts are derived from
# it, so `sync` rewrites them in place after a manifest update (kills 3-copy drift).


def _render_data_registry(reg: dict) -> str:
    out = ["_DATA_REGISTRY: dict = {"]
    for ds, e in reg.items():
        out += [f'    "{ds}": {{', f'        "version": "{e["version"]}",', '        "files": {']
        for name, f in e["files"].items():
            out += [f'            "{name}": {{', f'                "url": "{f["url"]}",',
                    f'                "sha256": "{f["sha256"]}",', "            },"]
        out += ["        },", "    },"]
    out.append("}")
    return "\n".join(out)


def _render_binary_registry(reg: dict) -> str:
    out = ["_REGISTRY: dict = {"]
    for bundle, e in reg.items():
        prov = ", ".join(f'"{x}"' for x in e["provides"])
        out += [f'    "{bundle}": {{', f'        "version": "{e["version"]}",',
                f'        "provides": [{prov}],', "        \"platforms\": {"]
        for key, f in e["platforms"].items():
            out += [f'            "{key}": {{', f'                "url": "{f["url"]}",',
                    f'                "sha256": "{f["sha256"]}",', "            },"]
        out += ["        },", "    },"]
    out.append("}")
    return "\n".join(out)


def _replace_block(path: Path, varname: str, literal: str) -> None:
    """Swap the top-level ``<varname>: dict = {...}`` literal in ``path`` for ``literal``."""
    import re

    text = path.read_text(encoding="utf-8")
    pat = re.compile(rf"^{re.escape(varname)}: dict = \{{.*?^\}}", re.DOTALL | re.MULTILINE)
    if not pat.search(text):
        raise SystemExit(f"could not find `{varname}: dict = {{...}}` in {path}")
    path.write_text(pat.sub(lambda _: literal, text, count=1), encoding="utf-8")


def sync_registries(manifest_path: Path, data_py: Path, binaries_py: Path) -> None:
    """Regenerate ``_DATA_REGISTRY`` / ``_REGISTRY`` in the .py files from the manifest."""
    from raven_toolbox.manifest import load_manifest, to_binary_registry, to_data_registry

    m = load_manifest(manifest_path)
    _replace_block(data_py, "_DATA_REGISTRY", _render_data_registry(to_data_registry(m)))
    _replace_block(binaries_py, "_REGISTRY", _render_binary_registry(to_binary_registry(m)))


# --- manifest.json (shared source of truth) --------------------------------


def _file_meta(path: Path, base: str) -> dict:
    """Manifest file record: url + sha256 + byte size."""
    return {"url": f"{base}/{path.name}", "sha256": _sha256(path), "bytes": path.stat().st_size}


def manifest_data_entry(version: str, base_url: str, directory: Path, **meta: str) -> dict:
    """Build a manifest ``data`` dataset entry (registry fields + optional metadata)."""
    base = base_url.rstrip("/")
    files = {p.name: _file_meta(p, base) for p in _files_in(directory)}
    if not files:
        raise SystemExit(f"No files found in {directory}")
    entry = {"version": version}
    entry.update({k: v for k, v in meta.items() if v})  # description/license/doi/source
    entry["files"] = files
    return entry


def manifest_binary_entry(
    bundle: str, version: str, provides: list[str], base_url: str, directory: Path, **meta: str
) -> dict:
    """Build a manifest ``binaries`` bundle entry from per-platform ZIPs."""
    base = base_url.rstrip("/")
    prefix = f"{bundle}-{version}-"
    platforms = {
        zp.name[len(prefix) : -len(".zip")]: _file_meta(zp, base)
        for zp in sorted(directory.glob(f"{prefix}*.zip"))
    }
    if not platforms:
        raise SystemExit(f"No {prefix}*.zip files found in {directory}")
    entry = {"version": version, "provides": provides}
    entry.update({k: v for k, v in meta.items() if v})  # description/license
    entry["platforms"] = platforms
    return entry


def update_manifest(manifest_path: Path, section: str, key: str, entry: dict) -> None:
    """Insert ``entry`` under ``manifest[section][key]`` and write the manifest back."""
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"manifest_version": 1}
    manifest.setdefault("manifest_version", 1)
    manifest.setdefault(section, {})[key] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="kind", required=True)

    d = sub.add_parser("data", help="data-artefact registry entry (raven_toolbox.data)")
    d.add_argument("--dataset", required=True, help="dataset key, e.g. 'kegg'")
    d.add_argument("--version", required=True)
    d.add_argument("--dir", required=True, type=Path, help="directory of uploaded artefacts")
    d.add_argument("--tag", required=True, help="GitHub release tag the assets were uploaded to")

    b = sub.add_parser("binary", help="binary-bundle registry entry (raven_toolbox.binaries)")
    b.add_argument("--bundle", required=True, help="bundle key, e.g. 'blast'")
    b.add_argument("--version", required=True)
    b.add_argument("--provides", nargs="+", required=True, help="executables the bundle provides")
    b.add_argument("--dir", required=True, type=Path, help="directory of uploaded ZIPs")
    b.add_argument("--tag", required=True, help="GitHub release tag the ZIPs were uploaded to")

    m = sub.add_parser("manifest", help="add/update an entry in the shared manifest.json")
    m.add_argument("--manifest", required=True, type=Path, help="manifest.json to create/update")
    m.add_argument("--target", required=True, choices=["data", "binary"])
    m.add_argument("--version", required=True)
    m.add_argument("--dir", required=True, type=Path, help="directory of uploaded files")
    m.add_argument("--tag", required=True, help="GitHub release tag the assets were uploaded to")
    m.add_argument("--dataset", help="data: dataset key, e.g. 'kegg'")
    m.add_argument("--bundle", help="binary: bundle key, e.g. 'diamond'")
    m.add_argument("--provides", nargs="+", help="binary: executables the bundle provides")
    m.add_argument("--description")
    m.add_argument("--license")
    m.add_argument("--doi", help="data: Zenodo (or other) DOI for this version")
    m.add_argument("--source", help="data: human-facing release/record page")

    s = sub.add_parser("sync", help="regenerate the baked Python registries from manifest.json")
    s.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    s.add_argument("--data-py", type=Path, default=Path("src/raven_toolbox/data.py"))
    s.add_argument("--binaries-py", type=Path, default=Path("src/raven_toolbox/binaries.py"))

    args = parser.parse_args(argv)
    if args.kind == "sync":
        sync_registries(args.manifest, args.data_py, args.binaries_py)
        print(f"Synced _DATA_REGISTRY + _REGISTRY from {args.manifest}", file=sys.stderr)
        return
    base_url = release_base_url(args.tag)
    if args.kind == "data":
        key, entry = args.dataset, data_entry(args.dataset, args.version, base_url, args.dir)
        target = "raven_toolbox/data.py  _DATA_REGISTRY"
        print(f"# Merge into {target}:", file=sys.stderr)
        print(render(key, entry))
    elif args.kind == "binary":
        key = args.bundle
        entry = binary_entry(args.bundle, args.version, args.provides, base_url, args.dir)
        target = "raven_toolbox/binaries.py  _REGISTRY"
        print(f"# Merge into {target}:", file=sys.stderr)
        print(render(key, entry))
    else:  # manifest
        if args.target == "data":
            if not args.dataset:
                parser.error("--dataset is required for --target data")
            entry = manifest_data_entry(
                args.version, base_url, args.dir,
                description=args.description, license=args.license, doi=args.doi, source=args.source,
            )
            update_manifest(args.manifest, "data", args.dataset, entry)
            print(f"Updated {args.manifest}: data/{args.dataset} ({len(entry['files'])} files)", file=sys.stderr)
        else:
            if not (args.bundle and args.provides):
                parser.error("--bundle and --provides are required for --target binary")
            entry = manifest_binary_entry(
                args.bundle, args.version, args.provides, base_url, args.dir,
                description=args.description, license=args.license,
            )
            update_manifest(args.manifest, "binaries", args.bundle, entry)
            print(f"Updated {args.manifest}: binaries/{args.bundle} ({len(entry['platforms'])} platforms)", file=sys.stderr)


if __name__ == "__main__":
    main()
