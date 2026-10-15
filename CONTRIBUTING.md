# Contributing to raven-toolbox

User documentation lives on [raven-docs](https://github.com/edkerk/raven-docs),
which covers both this package and MATLAB RAVEN. This file is for working on the
package itself, and for the maintainer tasks that have nowhere else to live.

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

Docstrings are NumPy style, which is what the site's generator expects.

---

# Maintaining the published artefacts

raven-toolbox implements the pipeline that builds the large files **both**
toolboxes consume: the KEGG reference data, the relational tables, and the
profile-HMM libraries. MATLAB RAVEN has no builder of its own and only ever
consumes the published output.

End users do none of this. They download the published, version-pinned artefacts
automatically; that side is documented in
[Downloaded data and binaries](https://github.com/edkerk/raven-docs/blob/main/docs/installation/data-and-binaries.md).

Publishing the results, and the versioning model behind the releases, is a shared
convention rather than a raven-toolbox concern, and is documented in
[raven-gecko-parity](https://github.com/SysBioChalmers/raven-gecko-parity/blob/develop/docs/publishing_artefacts.md).

## Rebuilding the KEGG artefacts

Once per KEGG release.

### Prerequisites

The bulk KEGG dump is licensed, and rebuilding needs an active subscription to
`ftp.kegg.net`, which comes with a username and password.

The downloader reads those from a `~/.netrc` file and never takes them on the
command line, so they stay out of shell history and out of `ps` output:

```bash
touch ~/.netrc && chmod 600 ~/.netrc
```

Add one line, with your own credentials:

```text
machine ftp.kegg.net login YOUR_KEGG_USER password YOUR_KEGG_PASSWORD
```

Three things about that file:

- The host must be `ftp.kegg.net`. A `machine` line for anything else is ignored.
- The mode must be `600`. Python's `netrc` parser refuses a file others can read.
- It is the same file `curl`, `wget` and `git` use, so add the line to an
  existing one rather than replacing it.

For credentials kept outside `$HOME`, pass `netrc_path=` instead; the format is
the same.

### Step 1: download and arrange the dump

```python
from raven_toolbox.reconstruction.kegg import download_kegg_dump

download_kegg_dump("keggdb")
```

This fetches the reaction, compound, glycan and ko archives, both proteomes and
the taxonomy file, extracts them, and arranges the flat layout the parser
expects. Files already present are skipped; `force=True` re-fetches for a new
release. `progress=False` turns off the per-file progress bars for a
non-interactive run.

Credentials can also come from elsewhere:

```python
download_kegg_dump("keggdb", netrc_path="/run/secrets/kegg_netrc")

# Only when they come from a secret manager at run time. Never a literal in
# committed code.
download_kegg_dump("keggdb", auth=("YOUR_KEGG_USER", "YOUR_KEGG_PASSWORD"))
```

### Step 2: parse into the published artefacts

```python
from raven_toolbox.reconstruction.kegg import parse_kegg_dump

parse_kegg_dump("keggdb", "artefacts", version="kegg118")
```

This writes the gene-free reference model as gzipped RAVEN/cobra YAML, and the
relational tables as gzipped TSV. With `version=` set, every filename is
version-prefixed to match the published assets. `progress=True` reports each
stage and shows a bar over the streaming pass across the large
`organism_gene_ko` table.

### Step 3: build the HMM libraries

Needs **HMMER**, **MAFFT** and **CD-HIT** on the `PATH`, or pointed at by
`RAVEN_PYTHON_HMMBUILD`, `RAVEN_PYTHON_MAFFT` and `RAVEN_PYTHON_CDHIT`.
`conda install -c bioconda hmmer mafft cd-hit` is the usual way.

> MAFFT and CD-HIT have no native Windows builds, so this step runs on Linux,
> macOS, or inside WSL2. Keep the whole stack inside WSL2 if you go that way:
> raven-toolbox calls the resolved executable directly and does not translate
> paths across the boundary the way RAVEN's `getWSLpath` does.

```python
from raven_toolbox.reconstruction.kegg import build_hmm_library, read_kegg_table

organism_gene_ko = read_kegg_table("artefacts/kegg118_organism_gene_ko.tsv.gz")
for domain in ("prokaryotes", "eukaryotes"):
    build_hmm_library(
        organism_gene_ko,
        "keggdb/genes.pep",
        "keggdb/taxonomy",
        f"hmms/{domain}",
        domain=domain,
        progress=True,
    )
```

For each KO it gathers the member sequences, dereplicates with CD-HIT at about
90% identity, aligns with MAFFT, trains a profile with `hmmbuild`, and
concatenates the result into one `library.hmm`. This is the slow step, hours per
release, and it is resumable: a KO whose `.hmm` already exists is skipped.

The library is published **gzipped ASCII**, not a `hmmpress`-ed binary index.
That keeps the download roughly ten times smaller, lets the same file serve
MATLAB RAVEN, and keeps it readable across HMMER versions, which matters because
the native-Windows `hmmsearch` is an older 3.3.2 build. Keep publishing ASCII.

### All three at once

```bash
python scripts/build_kegg_artefacts.py --keggdb keggdb --out artefacts \
    --version kegg118 --hmms --threads 8
```

Runs step 2, and step 3 with `--hmms`, and lays the output out as publishable,
version-prefixed assets: the core files bundled into `<version>_core.tar.gz`, one
`<version>_<domain>.hmm.gz` per domain, and `<version>_taxonomy.gz`.

The build is idempotent. If it fails partway, and the HMM step can run for hours,
re-run the same command: each stage is skipped when its output exists, and the
per-KO build resumes. `--force` rebuilds from scratch.

### Output format

The tables are gzipped TSV, and the reference model is gzipped RAVEN/cobra YAML.
Gzip rather than xz throughout, including the large `organism_gene_ko` table,
because MATLAB reads gzip with its built-in `gunzip` and has no `xz` dependency;
these artefacts are shared with MATLAB RAVEN, so a format only Python can open
cheaply is not an option. The size cost on a once-per-release download is worth a
dependency-free read on both sides.

`organism_gene_ko` is written sorted by `(organism, gene)`. That puts one
organism's gene ids next to each other, which compresses better and matches the
by-organism query pattern both toolboxes use. The sort is an external merge sort
bounded by `chunk_rows`, so it stays within memory at KEGG's scale.

Parquet would be smaller and typed, and MATLAB has read it natively since R2019a,
but it needs `pyarrow` on the Python side. It is the thing to reconsider if load
*time* rather than size becomes the bottleneck, or if either side starts doing
repeated columnar reads instead of loading once per run. SQLite is not a
candidate: it would need MATLAB's Database Toolbox, which breaks the
"same files, both languages, no extra dependencies" property.

## Maintaining the binary bundles

The BLAST+, DIAMOND and HMMER executables are published as per-platform ZIPs and
resolved at run time by `raven_toolbox.binaries`. The resolution order, the
`RAVEN_PYTHON_*` overrides and the `raven-toolbox-binaries` command are user-facing
and documented on raven-docs; what follows is what a bundle has to look like.

### Ship the minimum

| Bundle | Include | Leave out |
|---|---|---|
| `diamond` | `diamond` | nothing else; it is one static binary |
| `blast` | `blastp`, `makeblastdb` | `blastn`, `tblastn`, `psiblast`, `rpsblast`, `blast_formatter`, the `*_vdb` tools, `doc/`, `ChangeLog`, `README`, and the thirty-odd others |
| `hmmer` | `hmmsearch` | the rest of the suite |

Only `makeblastdb`, `blastp`, and DIAMOND's `makedb` and `blastp` subcommands are
ever invoked, by either toolbox, so the same minimal set serves both. For BLAST+
this is the difference between hundreds of megabytes and a few.

### ZIP convention

`<bundle>-<version>-<os>-<arch>.zip`, flat, executables at the root alongside the
upstream `LICENSE`. No nested `bin/`. Windows ZIPs also carry the `.exe` and any
runtime DLL the build needs at the root, `nghttp2.dll` for BLAST+ and
`cygwin1.dll` for HMMER, because extraction is flat and the executable is looked
up as `<name>.exe`.

### Platforms and licences

Build `linux-x86_64` first, then `macos-arm64`, `macos-x86_64`, `linux-arm64` and
`windows-x86_64` as capacity allows. A platform with no bundle is not a bug:
`ensure_binary` raises an error naming the conda package instead, which is the
documented fallback.

Redistribution terms differ and must be complied with. BLAST+ is NCBI-produced
and public domain; include its `LICENSE` for provenance. DIAMOND is **GPL-3.0**,
so the licence text must be in the ZIP and the binary unmodified, and it must
stay a separate asset rather than going inside the MIT wheel. HMMER is
BSD-3-Clause; include its `LICENSE`.

Record per asset, in the release body or a provenance file: upstream URL,
upstream version, upstream checksum, and the SHA256 published.

### Native-Windows HMMER is 3.3.2

There is no native-Windows HMMER 3.4; upstream targets POSIX. RAVEN `v2.10.5` is
the last release bundling a Cygwin-compiled Windows 3.3.2, and the published
`windows-x86_64` asset is repackaged from it rather than built fresh.

Searching 3.4-built profiles with 3.3.2 is sound here: the published library is
concatenated ASCII in the `HMMER3/f` format, unchanged from 3.1 through 3.4, and
3.4 is a maintenance release over 3.3.2 with no change to the protein scoring
model, so bit scores and the calibrated KEGG cut-offs carry over. That holds only
while the libraries stay ASCII. Ship `cygwin1.dll` alongside the `.exe`.

### Registering a bundle

After building the ZIPs, generate the registry entry rather than writing it by
hand, so the checksums come from the same helper `ensure_binary` verifies with:

```bash
python scripts/make_registry_snippet.py binary --bundle blast --version 2.17.0 \
    --provides blastp makeblastdb --dir zips \
    --base-url https://github.com/CompMetEng/toolbox-assets/releases/download/blast-2.17.0
```

Adding a new tool needs no new provisioning code: a bundle entry with the right
`provides` list, and ZIPs built to the convention above, is enough for the
wrappers to resolve it through `ensure_binary`.

### Never hand-edit the baked registries

`data/manifest.json` is the single source of truth. Change it, then run

```bash
python scripts/make_registry_snippet.py sync
```

which rewrites `_DATA_REGISTRY` and `_REGISTRY` from it.
