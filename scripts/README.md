# Maintainer scripts

Release-time tooling. Not part of the installed package — run them from a checkout
with raven-toolbox installed (`pip install -e .`). End users never need these.

The full publish workflow (build → upload → manifest → sync) is documented in
[artefact_hosting.md](https://github.com/SysBioChalmers/raven-gecko-parity/blob/main/docs/artefact_hosting.md)
(raven-gecko-parity).

The localisation/DeepLoc benchmark scripts (`benchmark_deeploc*.py`,
`compare_predictlocalization.py`, `benchmark_certified_*.py`, `benchmark_vs_carvefungi.py`,
`build_carvefungi_yeast_model.py`, `compare_localization_sources.py`,
`extract_compartment_structure.py`, `finetune_localization_yeast.py`,
`benchmark_replicate_yeast_gem.py`) expect their input/output data under `data/deeploc/`,
which lives in [raven-gecko-parity](https://github.com/SysBioChalmers/raven-gecko-parity/tree/develop/data/deeploc) —
fetch it there first. The write-ups that use these scripts are in
[raven-gecko-parity/docs/localization/](https://github.com/SysBioChalmers/raven-gecko-parity/tree/develop/docs/localization).

## `build_binary_bundles.py`

Build the per-platform binary ZIPs (BLAST+/DIAMOND/HMMER) from RAVEN's vetted
`software/` binaries (pinned commits) into `dist/binaries/`, with checksums and
provenance. See [maintaining_binaries.md](https://github.com/SysBioChalmers/raven-gecko-parity/blob/main/docs/maintaining_binaries.md)
(raven-gecko-parity).

```bash
python scripts/build_binary_bundles.py        # -> dist/binaries/*.zip (+ checksums, PROVENANCE)
```

## `publish_to_toolbox_assets.py`

Upload release assets to the [`toolbox-assets`](https://github.com/CompMetEng/toolbox-assets)
repo with `gh`, idempotently (immutable per-version tags; skips assets already present).

```bash
python scripts/publish_to_toolbox_assets.py binaries --dir dist/binaries
python scripts/publish_to_toolbox_assets.py release --tag kegg118 --dir artefacts
python scripts/publish_to_toolbox_assets.py --dry-run release --tag manifest-v1 data/manifest.json
```

## `build_kegg_artefacts.py`

Build the publishable KEGG artefact set from an arranged KEGG dump (see
`download_kegg_dump`): the gzipped-YAML reference model, the gzipped-TSV tables,
and (with `--hmms`) the per-domain pressed HMM libraries. Output is laid out ready
to upload as release assets. See [maintaining_kegg_data.md](https://github.com/SysBioChalmers/raven-gecko-parity/blob/main/docs/maintaining_kegg_data.md)
(raven-gecko-parity).

```bash
python scripts/build_kegg_artefacts.py --keggdb keggdb --out artefacts          # tables + model
python scripts/build_kegg_artefacts.py --keggdb keggdb --out artefacts --hmms --threads 8
```

## `make_registry_snippet.py`

After uploading the files to a release, compute their SHA256 and print the entry
to merge into the runtime registry — `raven_toolbox.data._DATA_REGISTRY` (data) or
`raven_toolbox.binaries._REGISTRY` (binary ZIP bundles). The checksum helper is shared
with the resolvers, so published checksums always match what `ensure_data` /
`ensure_binary` verify.

```bash
# Data artefacts (--tag is the release tag; the asset URL is built from it):
python scripts/make_registry_snippet.py data --dataset kegg --version kegg116 \
    --dir artefacts --tag v0.3.0

# Binary bundle (ZIPs named <bundle>-<version>-<os>-<arch>.zip):
python scripts/make_registry_snippet.py binary --bundle blast --version 2.16.0 \
    --provides blastp makeblastdb --dir zips --tag blast-2.16.0
```
