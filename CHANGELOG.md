# Changelog

Milestones in the raven-toolbox port. For function-level status see
[docs/raven_migration.md](https://github.com/CompMetEng/raven-toolbox/blob/develop/docs/reference/migration.md); for open work see
[docs/todo.md](https://github.com/CompMetEng/raven-toolbox/blob/develop/docs/reference/todo.md).

## 3.0.0b1 — 2026-09-07

First beta toward 3.0, matching the upcoming MATLAB RAVEN 3.0 beta.

* keep `ftinit()` as the only extraction pipeline.
* `analysis.compare_fluxes` replaces `follow_changed` — a `compareFluxes` port, following RAVEN
  dropping `followChanged`/`followFluxes`/`mapCompartments`.
* `comparison.compare_models` gained EC-code / metabolite-name / equation overlap matrices,
  folded from `compareRxnsGenesMetsComps`.
* New: `curation.rename_model_genes`, ported from `renameModelGenes`.
* New: `curation.get_gene_data`, `download_genome_data`, `process_protein_fasta_file` — the
  genome-data cluster.
* New: `manipulation.replace_metabolite`, ported from `replaceMets`.
* New: `gapfilling.find_leak_metabolite`, ported from `findLeakMetabolite`.
* New: `manipulation.remove_bad_reactions`, ported from `removeBadRxns`.
* New: `gapfilling.gap_report`, ported from `gapReport`.
* New: `analysis.trace_flux_path`, ported from `traceFluxPath`.
* New: `analysis.get_flux_z` / `analyze_sampling`, ported from `getFluxZ`/`analyzeSampling`.
* New: `analysis.fit_parameters`, ported from `fitParameters`.
* New: `utils.guess_composition`, ported from `guessComposition`.
* New: `analysis.walk_fluxes` / `FluxWalker`, ported from `walkFluxes`.
* New: `analysis.get_min_nr_fluxes`, ported from `getMinNrFluxes`.
* New: `utils.generate_new_ids`, ported from `generateNewIds`.
* `init.ftinit` implements the metabolomics production-bonus (`metabolomics=`, `prod_weight=`).
* `init.ftinit` reproducibility parameters renamed: `strict_gap`/`canonical` are now
  `prove_abs_gap`/`resolve_ties`, with corrected determinism numbers.
* ftINIT task gap-filling now edits the reference model in place instead of copying it — ~23%
  faster on a genome-scale extraction.
* Fixed three ftINIT/task-essential bugs: a task-essential boundary default, a checkpoint
  file-handle leak, and a non-finite tie-break objective.
* `manipulation.close_model` boundary-reaction detection is now structural, matching RAVEN.
* `manipulation.convert_to_irreversible` now splits exchange reactions too, carries a negative
  objective coefficient onto the reverse reaction, and gained an optional `rxns` restriction.
* `conditions.apply_condition` raises on unknown charge during charge balancing instead of
  silently miscalculating.
* `annotation.load_delta_g_csv` now always records CSV values literally, matching `deltaGCSV`.
* `manipulation.remove_dead_end_reactions` dead-end detection is now bounds-blind, matching RAVEN.
* `analysis.get_min_nr_fluxes` no longer crashes with `GurobiError` reading fluxes from an
  infeasible solve.
* `analysis.fseof` gained an explicit `min_target` scan-floor override.
* Fixed six redundant-calculation hot spots: `merge_linear`, batched `add_reactions` callers,
  `analyse_topology`, `trace_flux_path`, `remove_dead_end_reactions`, `find_task_essential_reactions`.
* `random_sampling` and `fill_gaps_fast_lp` parallelized with `ProcessPool`.
* `io.write_yaml_model` fixed: an infinite/NaN bound wrote as a bare word instead of the YAML
  token; `sort_ids=True` dropped the `!!omap` tag on `compartments`; unknown `metaData` fields
  are now preserved on write; empty `gene_reaction_rule` is now dropped. `read_yaml_model` treats
  a null genes/reactions/metabolites key as empty.
* Parameter defaults measured and unified: BLAST/DIAMOND/HMMER threading, homology
  `min_align_len`/`evalue`, `get_init_model.allow_excretion`.
* Removed `io.export_model_to_sif` — a mislabeled port of a MATLAB function that never existed.

## 0.4.0 — 2026-08-28

A `raven-gecko-parity` cross-validation harness closed six MATLAB-parity divergences and brought
the YAML writer to byte-parity; ftINIT gained an opt-in deterministic-extraction mode; compartment
placement and gap-filling in `assign_compartments` were fixed for determinism and reliability.

* `convert_to_irreversible` now carries a reaction's annotations, subsystem and notes onto its
  `_REV` half, matching `convertToIrrev`.
* `ftinit` gained opt-in deterministic extraction (`strict_gap`, `canonical`) for reproducible
  MILP tie-breaking, default off.
* `assign_compartments`'s gap-fill now uses a flux-based fill (pFBA) instead of cobra's
  unreliable indicator-MILP gapfill.
* `assign_compartments` reaction placement is now deterministic and score-aligned instead of a
  solver tie-break (yeast agreement 52.8% → 72.5%).
* `diff_models` compares GPRs as logic (DNF-normalised), not text — operand order no longer
  registers as a difference.
* `load_delta_g_csv` now stamps every matched value literally by default, matching `deltaGCSV`.
* `confidence.annotate_confidence` runs every applicable scorer in one call; `curation_priority`
  can now drop curator-settled placements from the review queue.
* New `thiele_palsson_score` derives the Thiele & Palsson 2010 confidence score from the
  `gene_association` facet.
* `merge_compartments`/`copy_to_compartment` are now public API; `merge_compartments` no longer
  drops single-metabolite reactions or the model objective.
* Six MATLAB-parity divergences closed via the new `raven-gecko-parity` harness: `check_tasks`
  boundary semantics, `apply_condition` exchange-reset direction, `remove_duplicate_reactions`
  GPR union, `add_reactions_from_equations` metabolite naming, `export_to_excel` default-bound
  cells.
* `write_yaml_model` brought to byte-parity with `writeYAMLmodel.m`: numeric coercion, `metaData`
  field order, MIRIAM collapsing, compartment defaulting, quoting, indentation. `read_yaml_model`
  no longer crashes on a model with no genes.
* BLAST/homology defaults matched to RAVEN and measured: `run_blast.evalue` → `1e-4`,
  `min_align_len` → `100`; new `review_identity` surfaces near-miss homology candidates.
* New `set_exchange_bounds`, RAVEN's `setExchangeBounds` port.
* New cross-language parity testing harness (`tests/parity/`) — exact, set-level, and
  statistical comparison tiers against real MATLAB RAVEN output.
* Fixed a downloaded binary bundle leaving its non-primary executables non-executable.

## 0.3.0 — 2026-07-16

Compartment localisation and per-reaction confidence tracking, new gap-filling and
flux-sampling algorithms, ftINIT brought to parity with RAVEN, and KEGG artefact hosting
moved to the dedicated `raven-data` repository.

* `export_for_git` can pin the MATLAB `.mat` variable name via `varname`.
* `ftinit`/`prep_init_model` now match RAVEN's model preparation, solver parameters, and gap
  schedule; new `manipulation.simplify_model`. Human-GEM prep model sizes now match RAVEN's.
* New confidence facets `equation` and `gene_association` (mass/charge balance, GPR presence)
  plus `facet_summary`; validated against yeast-GEM's own curated confidence levels.
* Fixed `get_elemental_balance` crashing on an unparseable polymer formula (now `unknown`).
* Fixed `score_localization_confidence` vetoing reactions it simply couldn't measure.
* Fixed gap-fill materialising on the wrong compartment metabolite in `assign_compartments`.
* New `localization.transport_evidence` — per-gene transporter evidence lowers a metabolite's
  transport cost in the assignment MILPs.
* `assign_compartments`/`apply_assignment`/`AssignmentProposal` consolidated into raven-toolbox
  from the standalone `assignCompartments` repo.
* Benchmarked the compartment-assignment MILP against CarveFungi's own carve-MILP and against
  RAVEN's `predictLocalization` — more accurate and deterministic on both.
* New `triage_localization` — flags genes/reactions whose localisation call is shakiest, each
  with a plain-English reason.
* Retuned `DEEPLOC_COMPARTMENT_TRUST` and the `min_confidence`/`membrane_threshold` gates from
  real yeast validation data.
* Cross-species DeepLoc benchmarks (yeast, *Arabidopsis*, *Chlamydomonas*, Human-GEM) — DeepLoc
  2.1 generalises across kingdoms.
* `load_deeploc` gained `normalise=False` for raw calibrated probabilities (accuracy-neutral for
  assignment).
* New `combine_scores` (weighted-sum consensus across localisation predictors) and
  `min_confidence=`/`membrane_split=` loader gates.
* New `prepare_deeploc_input`, `fetch_protein_sequences`, `write_fasta` — writes a
  DeepLoc-ready protein FASTA for a model's genes.
* New localisation loaders: `load_mulocdeep`, `load_compartments`, `load_uniprot`,
  `fetch_uniprot_localization`. Removed `load_wolfpsort`.
* Flux sampling: CHRR and ACHR unified under `random_sampling(method=...)`; ACHR is now the
  default. **Breaking:** pass `method="random_objective"` for the previous default.
* New gap-filling algorithms: `fill_gaps_fast_lp` (+ SWIFTCORE variant), `fill_gaps_kumar_milp`,
  `analyse_topology`.
* KEGG artefacts moved from `raven-toolbox` releases to the dedicated `raven-data` repository.
* Refreshed `kegg118` artefact set; fixed 404ing release-asset URLs.
* Binary provisioning split into `runtime`/`build` sets with an explicit
  `raven-toolbox-binaries` fetch CLI; native-Windows HMMER support;
  `RAVEN_PYTHON_AUTOFETCH=0` toggle.
* KEGG artefact build is now resumable (skips completed stages) and reports progress via `tqdm`.
* `export_to_excel` writes populated `model.ec` data (ENZYMES/ENZRXNS sheets).
* CI now runs on macOS and Windows in addition to Linux.
* Docstring rendering fixes for `set_gam` and `add_sbo_terms` (no API change).

## 0.2.0 — 2026-06-14

Project rename plus KEGG-reconstruction and CI improvements.

* Project renamed `raven-python` → `raven-toolbox`; import package is now `raven_toolbox`.
* `get_kegg_model_from_sequences` uses `hmmsearch` instead of `hmmscan`.
* `get_kegg_model_for_organism_from_artefacts` auto-resolves its taxonomy artefact.
* Test data no longer ships real KEGG records — generated at runtime instead.
* Removed the unimplemented `visualization` stub and its extra.
* CI updated to Node 24 / `actions/checkout@v5` / `actions/setup-python@v6`.

## 0.1.0 — 2026-06-10

First release with published, downloadable KEGG artefacts, plus a cobra-aligned hardening pass.

* KEGG artefacts published (`kegg116`): `ensure_kegg_data`/`ensure_kegg_hmm_library` fetch
  version-pinned, SHA256-verified files from the GitHub release; gzip instead of xz so MATLAB
  and Windows can read them without an external tool.
* New `reconstruction.kegg.phyl_dist` (`PhylDist`), a port of RAVEN's `getPhylDist`.
* `raven_toolbox.__version__` now derives from installed package metadata.
* Solver/feasibility failures now raise `cobra.exceptions.OptimizationError` consistently.
* Hardening pass: path-traversal guard on ZIP extraction, `random_sampling` NaN guard,
  `connect_blocked_reactions` non-positive-penalty guard, SHA256 cache re-verification, and
  fixes for a handful of latent edge-case crashes, each with a regression test.

## 0.1.0a1 — 2026-05-30

First alpha release. Covers the functional scope of RAVEN built on cobrapy: de-novo
reconstruction (KEGG / homology), context-specific modeling (tINIT / ftINIT), metabolic-task
validation, connectivity gap-filling, HPA omics ingestion, sub-cellular localisation,
N-model comparison, reporter metabolites, FSEOF, flux sampling, and the RAVEN-style I/O
formats (YAML / SIF / Excel). Validated against MATLAB RAVEN on Human-GEM (Jaccard 0.975–0.980).

* Licensed under MIT (previously GPL-3.0-or-later).
* Sphinx + MyST documentation site.
* Not yet implemented: visualization, metabolomics-based (f)tINIT scoring, published
  binary/KEGG-artefact release bundles.
