#!/usr/bin/env python
"""Publish release assets to the toolbox-assets repo with the GitHub CLI (``gh``).

Maintainer-side. Creates per-artefact, **immutable** release tags and uploads the
assets to them — idempotently: an existing tag is reused and only *missing* assets
are uploaded, so re-running after, say, a single DIAMOND bump touches only that
tool's tag and never re-uploads the unchanged BLAST/HMMER/KEGG assets (no
duplication). Requires ``gh auth`` with write access to the target repo.

Subcommands::

    # one release tag from explicit files or a whole directory
    publish_to_toolbox_assets.py release --tag kegg118 --dir C:/Work/GitHub/kegg118
    publish_to_toolbox_assets.py release --tag manifest-v1 data/manifest.json

    # every binary ZIP in a dir, auto-grouped into <bundle>-<version> tags
    publish_to_toolbox_assets.py binaries --dir dist/binaries

Add ``--dry-run`` to print the ``gh`` calls without executing them.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

DEFAULT_REPO = "CompMetEng/toolbox-assets"
# Metadata sidecars produced by the build scripts — never uploaded as assets.
_SKIP = {"checksums.txt", "PROVENANCE.txt", "manifest_binaries.json"}
_BUNDLE_RE = re.compile(r"^([a-z0-9]+)-(\d+\.\d+(?:\.\d+)?)-")  # blast-2.17.0-linux-x86_64.zip


def _run(args: list[str], *, dry: bool, capture: bool = False) -> str:
    if dry:
        print("DRY-RUN:", " ".join(args))
        return ""
    proc = subprocess.run(args, capture_output=capture, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(args)}\n{proc.stderr or ''}")
    return proc.stdout or ""


def _release_assets(repo: str, tag: str) -> set[str] | None:
    """Asset names already on the release, or None if the release doesn't exist."""
    proc = subprocess.run(
        ["gh", "release", "view", tag, "-R", repo, "--json", "assets"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    return {a["name"] for a in json.loads(proc.stdout).get("assets", [])}


def publish(repo: str, tag: str, files: list[Path], *, title: str | None,
            notes: str | None, dry: bool) -> None:
    files = sorted(files)
    for f in files:
        if not f.is_file():
            raise SystemExit(f"missing asset: {f}")
    existing = None if dry else _release_assets(repo, tag)
    if existing is None:
        print(f"  {tag}: creating release with {len(files)} asset(s)")
        args = ["gh", "release", "create", tag, "-R", repo, "--title", title or tag,
                "--notes", notes or f"Automated asset release: {tag}", *map(str, files)]
        _run(args, dry=dry)
    else:
        new = [f for f in files if f.name not in existing]
        if not new:
            print(f"  {tag}: up to date ({len(existing)} asset(s) already present)")
            return
        print(f"  {tag}: exists; uploading {len(new)} new asset(s) "
              f"(immutable — not clobbering {len(existing)} existing)")
        _run(["gh", "release", "upload", tag, "-R", repo, *map(str, new)], dry=dry)


def _files_from(directory: Path | None, explicit: list[str]) -> list[Path]:
    if directory:
        return [p for p in sorted(directory.iterdir())
                if p.is_file() and p.name not in _SKIP]
    return [Path(f) for f in explicit]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--dry-run", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("release", help="one tag from --dir or explicit files")
    r.add_argument("--tag", required=True)
    r.add_argument("--dir", type=Path, help="upload every (non-metadata) file in this dir")
    r.add_argument("--title")
    r.add_argument("--notes-file", type=Path)
    r.add_argument("files", nargs="*")

    b = sub.add_parser("binaries", help="auto-group <bundle>-<version>-*.zip into tags")
    b.add_argument("--dir", required=True, type=Path)

    args = p.parse_args(argv)

    if args.cmd == "release":
        files = _files_from(args.dir, args.files)
        if not files:
            p.error("no files to upload (pass --dir or file paths)")
        notes = args.notes_file.read_text(encoding="utf-8") if args.notes_file else None
        publish(args.repo, args.tag, files, title=args.title, notes=notes, dry=args.dry_run)
    else:  # binaries
        groups: dict[str, list[Path]] = {}
        for z in sorted(args.dir.glob("*.zip")):
            m = _BUNDLE_RE.match(z.name)
            if not m:
                print(f"  (skip, unrecognised name) {z.name}")
                continue
            groups.setdefault(f"{m.group(1)}-{m.group(2)}", []).append(z)
        if not groups:
            p.error(f"no <bundle>-<version>-*.zip files in {args.dir}")
        for tag, files in groups.items():
            publish(args.repo, tag, files, title=tag, notes=None, dry=args.dry_run)


if __name__ == "__main__":
    main()
