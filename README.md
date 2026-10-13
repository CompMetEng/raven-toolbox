# raven-toolbox

[![CI](https://github.com/SysBioChalmers/raven-toolbox/actions/workflows/ci.yml/badge.svg)](https://github.com/SysBioChalmers/raven-toolbox/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/raven-toolbox.svg)](https://pypi.org/project/raven-toolbox/)

**Reconstruction, Analysis and Visualisation of Metabolic Networks — in Python.**

`raven-toolbox` is the Python counterpart of the
[RAVEN Toolbox](https://github.com/SysBioChalmers/RAVEN) (MATLAB). It builds on
[**cobrapy**](https://github.com/opencobra/cobrapy) for everything cobrapy already does
well — simulation, standard analyses, SBML I/O, model manipulation — and adds the
functionality that is unique to RAVEN:

* **De novo reconstruction** from KEGG and protein homology (BLAST / DIAMOND).
* **Context-specific models** from omics data via **ftINIT**, with task-aware gap-filling,
  the linear-merge MILP reduction, and metabolomics-informed reaction scoring.
* **Metabolic-task** validation (`check_tasks`, `find_task_essential_reactions`).
* **Gap-filling** — connectivity gap-filling against template models, plus LP/SWIFTCORE,
  MILP, and topological strategies.
* **Omics integration** — Human Protein Atlas (proteomics + RNA-seq) ingestion.
* **Sub-cellular localisation** prediction by MILP, with partial-update mode and pluggable
  evidence sources (DeepLoc 2, MULocDeep, COMPARTMENTS, UniProt, …).
* **N-model comparison**; **reporter metabolites**; **FSEOF**; **flux sampling** (ACHR, CHRR,
  and the classic random-objective vertex method).
* **YAML I/O** following the cobra standard, plus geckopy's `ec-*` enzyme-constrained
  fields, and **RAVEN-style Excel** export.

The status of every RAVEN function (ported, mapped to a cobra equivalent, or
explicitly not ported) is documented function-by-function in
**[RAVEN 3 and raven-toolbox](https://raven-docs.readthedocs.io/en/latest/raven3-vs-raven-toolbox/)**.

## Design principle

The canonical in-memory object is always a [`cobra.Model`](https://cobrapy.readthedocs.io).
There is no parallel RAVEN struct, no `ravenCobraWrapper`-style adapter. RAVEN-specific
fields that cobra doesn't model natively (`rxnMiriams`, `metDeltaG`,
`rxnConfidenceScores`, …) live in cobra's `annotation` / `notes` dictionaries. This
avoids duplicating cobra's data model and keeps raven-toolbox interoperable with the wider
COBRA ecosystem.

## Status

**raven-toolbox 3.0.0b1** is out — the first beta toward 3.0, tracking the upcoming MATLAB
RAVEN 3.0 beta ([release notes](https://github.com/SysBioChalmers/raven-toolbox/releases/tag/v3.0.0b1),
[changelog](CHANGELOG.md)). It has been validated against MATLAB RAVEN on **Human-GEM** (5 Hart2015
cell-line models, Jaccard 0.975–0.980 — see
[the Human-GEM validation study](https://github.com/edkerk/raven-docs/blob/main/docs/parameter-tuning/studies/humangem-validation.md)
on raven-docs).

Two deliberate scope decisions, not pending work:

* **Classic tINIT is not implemented — only ftINIT.** raven-toolbox is a new
  implementation with no installed base of tINIT-built models to support; MATLAB RAVEN
  keeps both algorithms for backwards compatibility, but ftINIT is the algorithm tINIT
  was superseded by.
* **MetaCyc-based reconstruction** is not implemented and is flagged for removal from
  MATLAB RAVEN as well — see [IMPROVEMENTS.md](IMPROVEMENTS.md) under `R-MetaCyc`.
* **Dynamic FBA** is not implemented — several maintained Python packages already cover
  it ([`dfba`](https://pypi.org/project/dfba/), [`reframed`](https://pypi.org/project/reframed/),
  [`mewpy`](https://pypi.org/project/mewpy/)).

## Installation

```bash
pip install raven-toolbox
```

That installs the latest **stable** release. PyPI pre-releases are opt-in, so get the
3.0.0b1 beta with:

```bash
pip install --pre raven-toolbox
```

To work against the unreleased code on `develop` instead, install from git:

```bash
git clone https://github.com/SysBioChalmers/raven-toolbox
cd raven-toolbox
pip install -e ".[dev]"
```

raven-toolbox requires Python ≥ 3.11. Genome-scale ftINIT MILPs currently require **Gurobi**
([details on solver portability](https://github.com/edkerk/raven-docs/blob/main/docs/parameter-tuning/studies/init-solver-benchmark.md)
on raven-docs); toy and unit-test work runs on the open-source GLPK.

### External command-line tools (BLAST, DIAMOND, HMMER, MAFFT, CD-HIT)

Some workflows call external tools. **For most users there is nothing to do** —
raven-toolbox downloads each tool it needs automatically the first time it's used.

Which tools a workflow uses:

| Workflow | Tools |
|---|---|
| Homology-based reconstruction | `blastp` + `makeblastdb`, or `diamond` |
| KEGG HMM query (`get_kegg_model_for_organism`) | `hmmsearch` |
| Building the KEGG HMM libraries (maintainers) | `hmmbuild`, `mafft`, `cd-hit` |

Optional:

- **Fetch them up front** instead of on first use:
  ```bash
  raven-toolbox-binaries --set runtime   # blastp, makeblastdb, diamond, hmmsearch
  raven-toolbox-binaries --set build     # hmmbuild, mafft, cd-hit
  ```
- **Use your own install** — if a tool is on your `PATH` it's used instead of a
  download, e.g. `conda install -c bioconda blast diamond hmmer mafft cd-hit`.
- **Disable automatic downloads** (air-gapped / conda-only setups): set
  `RAVEN_PYTHON_AUTOFETCH=0`.

**SuppOrth (optional):** homology reconstruction can also start from the consensus
orthologs of [SuppOrth](https://github.com/fma91/SuppOrth). `hits_from_supporth` reads an
existing `suppOrth.tsv` and needs nothing extra. `run_supporth` needs the `suppOrth`
command, which is not downloaded automatically and runs on Linux and macOS:
`pip install 'git+https://github.com/fma91/SuppOrth.git@stand_alone'`, then `suppOrth install`.

**Windows:** homology reconstruction and the KEGG species model work as-is. To
*build* the KEGG HMM libraries (needs MAFFT/CD-HIT), use **WSL2**.

## Documentation

[**raven-docs**](https://raven-docs.readthedocs.io/) is the documentation, and it
covers MATLAB RAVEN and raven-toolbox side by side: the same task shown in both
languages, with every example executed on each commit.

This repository carries no prose documentation of its own. The API reference on the
site is generated from the docstrings here at build time, so a docstring edit
reaches the site without anything being copied across. Contributor and maintainer
notes are in [CONTRIBUTING.md](CONTRIBUTING.md); the roadmap, the backlog and the
MATLAB parity record are in
[raven-gecko-parity](https://github.com/SysBioChalmers/raven-gecko-parity).

## Relationship to MATLAB RAVEN

`raven-toolbox` is an independent Python reimplementation of the
[RAVEN Toolbox](https://github.com/SysBioChalmers/RAVEN), released under the permissive
**MIT** license. If you use it in scientific work, please cite the RAVEN 2 paper:

> Wang H, Marcišauskas S, Sánchez BJ, Domenzain I, Hermansson D, Agren R, Nielsen J,
> Kerkhoven EJ. (2018) RAVEN 2.0: A versatile toolbox for metabolic network
> reconstruction and a case study on *Streptomyces coelicolor*. PLoS Comput Biol 14(10):
> e1006541.

## License

[MIT](LICENSE)
