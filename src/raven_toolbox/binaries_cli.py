"""``raven-toolbox-binaries``: provision the external command-line tools.

raven-toolbox does not download binaries during ``pip install`` (pip can only
pull PyPI wheels, and BLAST, DIAMOND, HMMER, MAFFT and CD-HIT are not on PyPI).
After installing, running this command once fetches the version-pinned, SHA256-verified
bundles for the current platform::

    raven-toolbox-binaries --set runtime   # end users: blast, diamond, hmmsearch
    raven-toolbox-binaries --set build     # developers: hmmbuild, mafft, cd-hit
    raven-toolbox-binaries --list          # show the sets for this platform

Tools already on ``PATH`` (or pointed at by ``RAVEN_PYTHON_*``) are left as-is
unless ``--force-download`` is given. Tools with no bundle for this OS/arch
(for example MAFFT and CD-HIT on native Windows) are reported as unavailable with a
conda or WSL2 hint; see
https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/maintaining_binaries.md.
"""
from __future__ import annotations

import argparse

from raven_toolbox import binaries

# ASCII markers only — the Windows console (cp1252) garbles Unicode glyphs.
_SYMBOL = {"present": "=", "downloaded": "+", "unavailable": "-", "error": "x"}


def _print_sets() -> None:
    key = binaries.platform_key()
    print(f"Binary sets (platform: {key})\n")
    for name, execs in binaries.BINARY_SETS.items():
        print(f"  {name:<8} {', '.join(execs)}")
    print("  all      = every tool from every set")


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and provision the requested binaries.

    Prints a per-tool status line and returns the process exit code: ``1`` if any
    download failed, otherwise ``0`` (tools reported as unavailable do not change the
    exit code).
    """
    parser = argparse.ArgumentParser(
        prog="raven-toolbox-binaries",
        description="Download raven-toolbox's external binaries for this platform.",
    )
    parser.add_argument(
        "--set", default="runtime",
        help="which set to provision: 'runtime' (default), 'build', or 'all'",
    )
    parser.add_argument("--list", action="store_true", help="list the sets and exit")
    parser.add_argument(
        "--force-download", action="store_true",
        help="download the bundle even if the tool is already on PATH",
    )
    args = parser.parse_args(argv)

    if args.list:
        _print_sets()
        return 0

    try:
        executables = binaries.executables_for_set(args.set)
    except ValueError as exc:
        parser.error(str(exc))

    key = binaries.platform_key()
    print(f"Provisioning '{args.set}' binaries for {key} ...\n")
    results = binaries.provision_binaries(executables, prefer_existing=not args.force_download)

    for r in results:
        if r.status in ("present", "downloaded"):
            print(f"  {_SYMBOL[r.status]} {r.executable:<14} {r.status}: {r.detail}")
        else:
            # 'unavailable'/'error' details can be long; keep the headline tidy.
            print(f"  {_SYMBOL[r.status]} {r.executable:<14} {r.status}")

    unavailable = [r for r in results if r.status == "unavailable"]
    errors = [r for r in results if r.status == "error"]
    if unavailable:
        print(
            f"\n{len(unavailable)} tool(s) have no bundle for {key}: "
            f"{', '.join(r.executable for r in unavailable)}.\n"
            "Install them via conda (`conda install -c bioconda <tool>`) or, on "
            "native Windows for HMMER/MAFFT/CD-HIT, run inside WSL2 (see "
            "https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/maintaining_binaries.md)."
        )
    if errors:
        print(f"\n{len(errors)} download(s) failed:")
        for r in errors:
            print(f"  ✗ {r.executable}: {r.detail}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
