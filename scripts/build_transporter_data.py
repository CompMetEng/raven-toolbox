#!/usr/bin/env python3
"""Build the transporter reference databases for evidence-aware transport scoring.

Produces four toolbox-assets artefacts (host them under a ``transporters-<date>`` release, then run
``make_registry_snippet.py`` + ``publish_to_toolbox_assets.py`` per
https://github.com/CompMetEng/raven-gecko-parity/blob/develop/docs/publishing_artefacts.md):

* ``transporter_pfam.hmm`` — the HMMER database of the transporter Pfam families in
  :data:`raven_toolbox.localization.transporter_tables.PFAM_TRANSPORTERS`, fetched one HMM at a time
  from the InterPro API and concatenated (used directly by ``hmmsearch``; no ``hmmpress`` needed).
* ``tcdb.dmnd`` — a DIAMOND database of the TCDB protein sequences.
* ``tcdb_substrates.tsv`` — TCDB's curated ``TC-ID -> substrate ChEBI(s)`` table (getSubstrates.py),
  with the generic ontology roots (``molecule``, ``ion``, ...) dropped — the substrate-specific layer.
* ``chebi_relations.tsv.gz`` — chebi_lite.obo distilled to ``child <rel> parent`` edges (``is_a`` +
  protonation/tautomer bridges), the ontology graph that rolls a metabolite up to a transporter substrate.

Every family accession is verified against the live InterPro API; a stale/typo'd accession is reported
and skipped rather than silently shipped. Needs network access (InterPro / TCDB / EBI) and the bundled
``diamond`` binary (auto-provisioned via ``raven_toolbox.binaries``).

Usage: ``python scripts/build_transporter_data.py --out dist/transporters``
(``--only-pfam|--only-tcdb|--only-substrates|--only-chebi`` to build a subset). ASCII-only output.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

from raven_toolbox.binaries import resolve_binary
from raven_toolbox.localization.transporter_tables import PFAM_TRANSPORTERS

INTERPRO_HMM = "https://www.ebi.ac.uk/interpro/api/entry/pfam/{acc}?annotation=hmm"
TCDB_FASTA = "https://tcdb.org/public/tcdb"
TCDB_SUBSTRATES = "https://www.tcdb.org/cgi-bin/substrates/getSubstrates.py"
CHEBI_OBO = "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_lite.obo"
UA = {"User-Agent": "raven-toolbox transporter-data build"}

# ChEBI ontology roots / non-specific classes: useless (and dangerous, under roll-up) as substrate
# evidence, since every metabolite is_a "molecule" -- drop them from the transporter substrate table.
GENERIC_CHEBI = frozenset({
    "CHEBI:25367",  # molecule
    "CHEBI:23367",  # molecular entity
    "CHEBI:24431",  # chemical entity
    "CHEBI:36357",  # polyatomic entity
    "CHEBI:33579",  # main-group molecular entity
    "CHEBI:33675",  # p-block molecular entity
    "CHEBI:24870",  # ion
    "CHEBI:14911",  # protein
    "CHEBI:25906",  # peptide
    "CHEBI:33839",  # macromolecule
    "CHEBI:50906",  # role
})

# ChEBI relationships worth traversing when matching a metabolite to a transporter substrate: is_a
# (subtype), the protonation-state bridges (a model anion vs a TCDB acid/base), and the tautomer/
# functional-parent link. All point child -> a more-general or chemically-equivalent term.
CHEBI_RELATIONS = frozenset({
    "is_a", "has_functional_parent", "is_conjugate_base_of", "is_conjugate_acid_of", "is_tautomer_of",
})


def _get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        data = r.read()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data  # InterPro serves gzip


def build_pfam_hmm(out_dir: Path) -> Path:
    hmm_path = out_dir / "transporter_pfam.hmm"
    fetched, missing = 0, []
    with hmm_path.open("wb") as fh:
        for acc, (fam, _classes) in sorted(PFAM_TRANSPORTERS.items()):
            try:
                hmm = _get(INTERPRO_HMM.format(acc=acc))
            except Exception as exc:  # noqa: BLE001
                missing.append(f"{acc} ({fam}): fetch failed - {exc}")
                continue
            if not hmm.lstrip().startswith(b"HMMER3"):
                missing.append(f"{acc} ({fam}): no HMMER3 model returned")
                continue
            fh.write(hmm if hmm.endswith(b"\n") else hmm + b"\n")
            fetched += 1
            print(f"  + {acc} {fam}", flush=True)
    print(f"fetched {fetched}/{len(PFAM_TRANSPORTERS)} transporter HMMs -> {hmm_path}")
    if missing:
        print("MISSING/INVALID (fix the accession in transporter_tables.py):")
        for m in missing:
            print(f"  ! {m}")
    # No hmmpress: the backend uses `hmmsearch` (query = these HMMs, target = proteome), which reads a
    # plain multi-model .hmm directly; pressing is only needed for hmmscan's random access.
    return hmm_path


def build_tcdb_diamond(out_dir: Path) -> Path:
    fasta = out_dir / "tcdb.fasta"
    print(f"downloading TCDB sequences -> {fasta}", flush=True)
    fasta.write_bytes(_get(TCDB_FASTA, timeout=180))
    n = fasta.read_text(errors="ignore").count(">")
    print(f"  {n} sequences")
    diamond = resolve_binary("diamond")
    db = out_dir / "tcdb"
    subprocess.run([diamond, "makedb", "--in", str(fasta), "--db", str(db)], check=True)
    return db.with_suffix(".dmnd")


def build_tcdb_substrates(out_dir: Path) -> Path:
    """TCDB's curated TC-ID -> substrate-ChEBI table (getSubstrates.py), generic classes dropped."""
    raw = _get(TCDB_SUBSTRATES, timeout=120).decode("utf-8", "replace")
    out = out_dir / "tcdb_substrates.tsv"
    n_sys = n_assoc = 0
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for line in raw.splitlines():
            tc, _, subs = line.partition("\t")  # TC-ID \t CHEBI:n;name|CHEBI:m;name
            tc = tc.strip()
            if not tc or not subs:
                continue
            chebis: list[str] = []
            for entry in subs.split("|"):
                cid = entry.split(";", 1)[0].strip()
                if cid.startswith("CHEBI:") and cid not in GENERIC_CHEBI and cid not in chebis:
                    chebis.append(cid)
            if chebis:
                fh.write(f"{tc}\t{';'.join(chebis)}\n")
                n_sys += 1
                n_assoc += len(chebis)
    print(f"tcdb substrates: {n_sys} TC systems, {n_assoc} substrate-ChEBI associations -> {out}")
    return out


def build_chebi_relations(out_dir: Path) -> Path:
    """Distil chebi_lite.obo to child->related edges (is_a + protonation/tautomer) + alt_id normalisation.

    The ``alt_id`` rows (``secondary  alt_id  primary``) are essential: source databases such as TCDB
    annotate substrates with *secondary/deprecated* ChEBI ids, which carry no ``is_a`` edges of their
    own and would never match. Mapping them to the connected primary id makes the roll-up work.
    """
    obo = _get(CHEBI_OBO, timeout=300).decode("utf-8", "replace")
    out = out_dir / "chebi_relations.tsv.gz"
    cur: str | None = None
    n = 0
    with gzip.open(out, "wt", encoding="utf-8", newline="\n") as fh:
        for line in obo.splitlines():
            if line == "[Term]":
                cur = None
            elif line.startswith("id: CHEBI:"):
                cur = line[4:].strip()
            elif cur is None:
                continue
            elif line.startswith("is_a: CHEBI:"):
                fh.write(f"{cur}\tis_a\t{line[6:].split('!', 1)[0].strip()}\n")
                n += 1
            elif line.startswith("alt_id: CHEBI:"):
                fh.write(f"{line[8:].strip()}\talt_id\t{cur}\n")  # secondary -> primary
                n += 1
            elif line.startswith("relationship: "):
                parts = line.split(maxsplit=3)
                if len(parts) >= 3 and parts[1] in CHEBI_RELATIONS and parts[2].startswith("CHEBI:"):
                    fh.write(f"{cur}\t{parts[1]}\t{parts[2]}\n")
                    n += 1
    print(f"chebi relations: {n} edges -> {out}")
    return out


def _checksums(out_dir: Path) -> None:
    lines = []
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.suffix not in {".fasta"}:
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir/'SHA256SUMS'} ({len(lines)} files)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("dist/transporters"))
    ap.add_argument("--only-pfam", action="store_true")
    ap.add_argument("--only-tcdb", action="store_true")
    ap.add_argument("--only-substrates", action="store_true")
    ap.add_argument("--only-chebi", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    run_all = not (args.only_pfam or args.only_tcdb or args.only_substrates or args.only_chebi)
    if run_all or args.only_pfam:
        build_pfam_hmm(args.out)
    if run_all or args.only_tcdb:
        build_tcdb_diamond(args.out)
    if run_all or args.only_substrates:
        build_tcdb_substrates(args.out)
    if run_all or args.only_chebi:
        build_chebi_relations(args.out)
    _checksums(args.out)
    print("done.", file=sys.stderr)


if __name__ == "__main__":
    main()
