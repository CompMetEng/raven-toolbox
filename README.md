<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/CompMetEng/raven-toolbox/develop/img/raven-logo-white.png">
  <img src="https://raw.githubusercontent.com/CompMetEng/raven-toolbox/develop/img/raven-logo.png" alt="RAVEN" width="400px">
</picture>

[![CI](https://github.com/CompMetEng/raven-toolbox/actions/workflows/ci.yml/badge.svg)](https://github.com/CompMetEng/raven-toolbox/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/raven-toolbox.svg?include_prereleases)](https://pypi.org/project/raven-toolbox/)
[![Documentation](https://readthedocs.org/projects/raven-docs/badge/?version=latest)](https://raven-docs.readthedocs.io/en/latest/)
[![GitHub Discussions](https://img.shields.io/github/discussions-search?query=repo%3ACompMetEng%2Fraven-toolbox&label=GitHub%20Discussions)](https://github.com/CompMetEng/raven-toolbox/discussions)

**[Documentation](https://raven-docs.readthedocs.io/en/latest/)** · [Installation](https://raven-docs.readthedocs.io/en/latest/installation/python.html) · [Guide](https://raven-docs.readthedocs.io/en/latest/guide/index.html) · [API reference](https://raven-docs.readthedocs.io/en/latest/api/index.html) · [Compare with RAVEN](https://raven-docs.readthedocs.io/en/latest/raven3-vs-raven-toolbox.html) · [Citations](https://raven-docs.readthedocs.io/en/latest/references.html)

# raven-toolbox

Reconstruction, Analysis and Visualisation of Metabolic Networks in Python: the Python counterpart of the [RAVEN Toolbox](https://github.com/CompMetEng/RAVEN) (MATLAB). raven-toolbox builds on [cobrapy](https://github.com/opencobra/cobrapy) for simulation, standard analyses, SBML input and output, and model manipulation, and it adds the functionality that is specific to RAVEN:

- Reconstruction from KEGG and from protein homology (BLAST or DIAMOND).
- Context-specific models from omics data with ftINIT, with task-aware gap-filling, the linear-merge MILP reduction and scoring with metabolomics.
- Validation with metabolic tasks (`check_tasks`, `find_task_essential_reactions`).
- Gap-filling: connectivity gap-filling against template models, and LP, SWIFTCORE, MILP and topological strategies.
- Omics integration with Human Protein Atlas data (proteomics and RNA-seq).
- Prediction of sub-cellular localisation by MILP, with a partial-update mode and exchangeable evidence sources (DeepLoc 2, MULocDeep, COMPARTMENTS, UniProt).
- Comparison of several models, reporter metabolites, FSEOF, and flux sampling (ACHR, CHRR and the random-objective vertex method).
- YAML input and output that follows the cobra standard and adds the `ec-*` fields for enzyme-constrained models of [geckopy](https://github.com/CompMetEng/geckopy), and export to the RAVEN Excel format.

[RAVEN 3 and raven-toolbox](https://raven-docs.readthedocs.io/en/latest/raven3-vs-raven-toolbox.html) gives the status of every RAVEN function: ported, mapped to a cobra equivalent, or not ported.

## Design

The in-memory model is a [`cobra.Model`](https://cobrapy.readthedocs.io). raven-toolbox has no RAVEN struct and no adapter for one. Fields that cobra does not model, such as `rxnMiriams`, `metDeltaG` and `rxnConfidenceScores`, are kept in the `annotation` and `notes` dictionaries of cobra. This keeps raven-toolbox interoperable with the COBRA ecosystem.

## Status

raven-toolbox 3.x tracks the RAVEN 3 pre-releases and is a pre-release itself ([releases](https://github.com/CompMetEng/raven-toolbox/releases), [changelog](CHANGELOG.md)). On Human-GEM, the five cell-line models of Hart2015 extracted with ftINIT have a Jaccard index against MATLAB RAVEN of 0.964 to 0.982 without tasks and 0.981 to 0.988 with tasks ([Human-GEM validation](https://raven-docs.readthedocs.io/en/latest/parameter-tuning/studies/humangem-validation.html)).

Three functions of RAVEN are not implemented:

- Classic tINIT. raven-toolbox has no installed base of tINIT models, and ftINIT supersedes tINIT. MATLAB RAVEN keeps both for backward compatibility.
- MetaCyc-based reconstruction, which is flagged for removal from MATLAB RAVEN as well ([IMPROVEMENTS.md](IMPROVEMENTS.md), `R-MetaCyc`).
- Dynamic FBA, which maintained Python packages cover ([`dfba`](https://pypi.org/project/dfba/), [`reframed`](https://pypi.org/project/reframed/) and [`mewpy`](https://pypi.org/project/mewpy/)).

## Install

The 3.x releases are pre-releases, so pip needs `--pre`. Without it, pip installs the older stable release 0.4.0.

```bash
pip install --pre raven-toolbox
```

Python 3.11 or newer is required. The unreleased code on `develop` installs from git:

```bash
git clone https://github.com/CompMetEng/raven-toolbox
cd raven-toolbox
pip install -e ".[dev]"
```

Genome-scale ftINIT problems need Gurobi ([ftINIT on different solvers](https://raven-docs.readthedocs.io/en/latest/parameter-tuning/studies/init-solver-benchmark.html)), and small models and the unit tests run on the open-source GLPK. [Installation](https://raven-docs.readthedocs.io/en/latest/installation/python.html) describes the solver configuration.

### External command-line tools

BLAST, DIAMOND and HMMER (and MAFFT and CD-HIT for building HMM libraries) are downloaded the first time a workflow needs them. A tool on the `PATH` takes precedence over a download, `raven-toolbox-binaries --set runtime` fetches the tools in advance, and `RAVEN_PYTHON_AUTOFETCH=0` disables downloads for air-gapped systems. [Download data and binaries](https://raven-docs.readthedocs.io/en/latest/installation/data-and-binaries.html) lists the tools per workflow, the cache locations and the platforms. On Windows, homology reconstruction and the KEGG species model run natively, and building the KEGG HMM libraries needs WSL2.

[SuppOrth](https://github.com/fma91/SuppOrth) orthologs can start a homology reconstruction. `hits_from_supporth` reads an existing `suppOrth.tsv` and needs nothing else. `run_supporth` needs the `suppOrth` command, which is not downloaded automatically and runs on Linux and macOS: `pip install 'git+https://github.com/fma91/SuppOrth.git@stand_alone'`, then `suppOrth install`.

## Quick start

```python
from raven_toolbox.io import read_yaml_model

model = read_yaml_model("yeast-GEM.yml")      # a cobra.Model
model.objective = "r_2111"                    # growth
model.reactions.get_by_id("r_1714").lower_bound = -1.0   # glucose uptake
print(model.optimize().objective_value)       # cobrapy
```

[Getting started](https://raven-docs.readthedocs.io/en/latest/guide/getting-started.html) continues from here.

## Documentation and support

[raven-docs](https://raven-docs.readthedocs.io/en/latest/) documents MATLAB RAVEN and raven-toolbox side by side: each task is shown in both languages, and the CI of the documentation runs the examples in both. This repository has no prose documentation of its own. The API reference on the site is generated from the docstrings here when the site is built. [CONTRIBUTING.md](CONTRIBUTING.md) has the notes for contributors and maintainers, and [raven-gecko-parity](https://github.com/CompMetEng/raven-gecko-parity) holds the roadmap, the backlog and the MATLAB parity record. Use [GitHub Discussions](https://github.com/CompMetEng/raven-toolbox/discussions) for questions.

## Cite us

raven-toolbox is an independent Python reimplementation of RAVEN. If you use it in scientific work, please cite the RAVEN 2 paper:

> Wang H, Marcišauskas S, Sánchez BJ, Domenzain I, Hermansson D, Agren R, Nielsen J, Kerkhoven EJ. (2018) RAVEN 2.0: A versatile toolbox for metabolic network reconstruction and a case study on _Streptomyces coelicolor_. PLoS Comput Biol 14(10): e1006541. doi:[10.1371/journal.pcbi.1006541](https://doi.org/10.1371/journal.pcbi.1006541).

## License

[GPL-3.0](LICENSE)
