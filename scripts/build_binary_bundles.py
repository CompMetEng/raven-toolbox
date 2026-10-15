#!/usr/bin/env python
"""Build per-platform binary-bundle ZIPs from RAVEN's vetted ``software/`` binaries.

Maintainer-side, run when refreshing the bundled BLAST+/DIAMOND/HMMER binaries.
Downloads the executables RAVEN ships (pinned commits), repackages each into the
flat ``<bundle>-<version>-<os>-<arch>.zip`` layout
:func:`raven_toolbox.binaries.ensure_binary` expects — executables at the root,
plus the upstream ``LICENSE`` and any required runtime DLL — and records each
ZIP's SHA256 + size. Output goes to ``--out`` (default ``dist/binaries``, which is
gitignored); upload its ZIPs as GitHub release assets and paste the emitted
``manifest_binaries.json`` into ``data/manifest.json``.

Sources (pinned for reproducibility):

* RAVEN ``develop3`` — BLAST+ 2.17.0, DIAMOND 2.1.17, HMMER 3.4.0 ``hmmsearch``
  (Linux + macOS). ``.mac`` builds: ``hmmsearch`` is arm64, BLAST/DIAMOND are
  universal — so ``macos-arm64`` serves all three (Intel macOS → conda).
* RAVEN ``v2.10.5`` — HMMER **3.3.2** native Windows ``hmmsearch.exe`` (+ the
  ``cygwin1.dll`` it needs). No native-Windows HMMER 3.4 exists; the shared
  ``HMMER3/f`` ASCII format makes 3.3.2 search 3.4-built libraries identically.

MAFFT and CD-HIT (the ``build`` set) are not bundled by RAVEN and are not produced
here — install them via conda, or add upstream sources later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

RAW = "https://raw.githubusercontent.com/SysBioChalmers/RAVEN"
# Pinned RAVEN refs (commit SHA for the mutable branch; tag is already immutable).
DEV3 = "aa5e2c670f2f9cb7d040557974bb493a8ce0ba91"  # develop3 HEAD at build time
V2105 = "v2.10.5"

PROVIDES = {"blast": ["blastp", "makeblastdb"], "diamond": ["diamond"], "hmmer": ["hmmsearch"]}

# One dict per published ZIP. members: (path under software/, arcname in ZIP, executable?)
SPECS: list[dict] = [
    # BLAST+ 2.17.0
    {"bundle": "blast", "version": "2.17.0", "plat": "linux-x86_64", "ref": DEV3, "members": [
        ("blast+/blastp", "blastp", True),
        ("blast+/makeblastdb", "makeblastdb", True),
        ("blast+/LICENSE", "LICENSE", False)]},
    {"bundle": "blast", "version": "2.17.0", "plat": "macos-arm64", "ref": DEV3, "members": [
        ("blast+/blastp.mac", "blastp", True),
        ("blast+/makeblastdb.mac", "makeblastdb", True),
        ("blast+/LICENSE", "LICENSE", False)]},
    {"bundle": "blast", "version": "2.17.0", "plat": "windows-x86_64", "ref": DEV3, "members": [
        ("blast+/blastp.exe", "blastp.exe", True),
        ("blast+/makeblastdb.exe", "makeblastdb.exe", True),
        ("blast+/nghttp2.dll", "nghttp2.dll", False),
        ("blast+/LICENSE", "LICENSE", False)]},
    # DIAMOND 2.1.17
    {"bundle": "diamond", "version": "2.1.17", "plat": "linux-x86_64", "ref": DEV3, "members": [
        ("diamond/diamond", "diamond", True), ("diamond/LICENSE", "LICENSE", False)]},
    {"bundle": "diamond", "version": "2.1.17", "plat": "macos-arm64", "ref": DEV3, "members": [
        ("diamond/diamond.mac", "diamond", True), ("diamond/LICENSE", "LICENSE", False)]},
    {"bundle": "diamond", "version": "2.1.17", "plat": "windows-x86_64", "ref": DEV3, "members": [
        ("diamond/diamond.exe", "diamond.exe", True), ("diamond/LICENSE", "LICENSE", False)]},
    # HMMER — 3.4.0 (Linux/macOS), 3.3.2 native Windows from v2.10.5
    {"bundle": "hmmer", "version": "3.4.0", "plat": "linux-x86_64", "ref": DEV3, "members": [
        ("hmmer/hmmsearch", "hmmsearch", True), ("hmmer/LICENSE", "LICENSE", False)]},
    {"bundle": "hmmer", "version": "3.4.0", "plat": "macos-arm64", "ref": DEV3, "members": [
        ("hmmer/hmmsearch.mac", "hmmsearch", True), ("hmmer/LICENSE", "LICENSE", False)]},
    {"bundle": "hmmer", "version": "3.3.2", "plat": "windows-x86_64", "ref": V2105, "members": [
        ("hmmer/hmmsearch.exe", "hmmsearch.exe", True),
        ("hmmer/cygwin1.dll", "cygwin1.dll", False),
        ("hmmer/LICENSE", "LICENSE", False)]},
]


def _download(ref: str, src: str, cache: Path) -> Path:
    """Fetch ``software/<src>`` at ``ref`` into the cache (skip if already complete)."""
    dest = cache / ref / src
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW}/{ref}/software/{src}"
    with urlopen(url, timeout=120) as resp:  # noqa: S310 (pinned RAVEN raw URLs)
        expected = int(resp.headers.get("Content-Length", 0)) or None
        if dest.exists() and expected is not None and dest.stat().st_size == expected:
            return dest
        data = resp.read()
    if expected is not None and len(data) != expected:
        raise RuntimeError(f"short read for {url}: got {len(data)} of {expected} bytes")
    if data[:1] == b"<":  # an HTML error page, not a binary
        raise RuntimeError(f"unexpected HTML response for {url} (ref/path wrong?)")
    dest.write_bytes(data)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(out: Path, base_url: str) -> None:
    cache = out / "_src"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, int]] = []
    provenance: list[str] = [f"RAVEN develop3 @ {DEV3}", f"RAVEN tag {V2105}", ""]
    manifest: dict[str, dict] = {}

    for spec in SPECS:
        bundle, version, plat, ref = spec["bundle"], spec["version"], spec["plat"], spec["ref"]
        asset = f"{bundle}-{version}-{plat}.zip"
        zip_path = out / asset
        print(f"  building {asset} ...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for src, arcname, is_exec in spec["members"]:
                blob = _download(ref, src, cache)
                info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))  # reproducible
                info.external_attr = (0o755 if is_exec else 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, blob.read_bytes())
                provenance.append(f"{asset}  <-  software/{src}  @ {ref}")
        sha, size = _sha256(zip_path), zip_path.stat().st_size
        rows.append((asset, sha, size))
        provenance.append("")
        # Assemble the manifest 'binaries' snippet (bundle version = the Linux/macOS one).
        entry = manifest.setdefault(bundle, {
            "version": "3.4.0" if bundle == "hmmer" else version,
            "provides": PROVIDES[bundle],
            "platforms": {},
        })
        entry["platforms"][plat] = {
            "url": f"{base_url.rstrip('/')}/{bundle}-{version}/{asset}",
            "sha256": sha,
            "bytes": size,
        }

    (out / "checksums.txt").write_text(
        "".join(f"{sha}  {asset}  ({size} bytes)\n" for asset, sha, size in rows), encoding="utf-8"
    )
    (out / "PROVENANCE.txt").write_text("\n".join(provenance), encoding="utf-8")
    (out / "manifest_binaries.json").write_text(
        json.dumps({"binaries": manifest}, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n{len(rows)} ZIPs written to {out}")
    for asset, sha, size in rows:
        print(f"  {size/1e6:7.1f} MB  {sha[:12]}…  {asset}")
    print(f"\nSHA256s: {out / 'checksums.txt'}\nProvenance: {out / 'PROVENANCE.txt'}")
    print(f"Manifest snippet (set the real release base-url): {out / 'manifest_binaries.json'}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("dist/binaries"), help="output folder")
    p.add_argument(
        "--base-url",
        default="https://github.com/CompMetEng/toolbox-assets/releases/download",
        help="release-download base; the snippet URLs become <base>/<bundle>-<version>/<asset>",
    )
    args = p.parse_args(argv)
    build(args.out, args.base_url)


if __name__ == "__main__":
    main()
