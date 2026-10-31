# Changelog

Milestones in the raven-toolbox port. For function-level status see
[docs/raven_migration.md](https://github.com/CompMetEng/raven-toolbox/blob/develop/docs/reference/migration.md); for open work see
[docs/todo.md](https://github.com/CompMetEng/raven-toolbox/blob/develop/docs/reference/todo.md).

## Unreleased

* `manipulation.set_exchange_bounds` is annotated as returning `list[str | Metabolite]`: the entries of `mets` that are not exchanged, as supplied (ids, names or `Metabolite` objects).
* `scripts/analyze_init_params.py` runs the `mip_gap`, `force_on` and `prep_scale` sweeps with the `ftinit` default `big_m` (1000, was: 100) and takes `--big-m` to set it. The `big_m` sweep starts at 1000, which is its reference row. The cached sweep results are keyed on `big_m`, so a resumed pickle from a run with another `big_m` is not reused.
* Models read with `io.read_yaml_model` and `io.read_matlab_model` have `model.tolerance` set to 1e-9, the
  feasibility and optimality tolerance RAVEN's `solveLP` gives the solver (cobrapy's default is 1e-6 for
  Gurobi and 1e-7 for GLPK). `utils.solver.set_raven_tolerance` applies it to any other model.
* `manipulation.remove_genes` defaults to `blocked_reactions="constrain"` (was: `"remove"`), as
  RAVEN's `removeGenes` does: a reaction whose rule loses its last enzyme stays, with the rule
  dropped and bounds `(0, 0)`. Pass `blocked_reactions="remove"` to delete such reactions.
* `manipulation.remove_genes` also removes every gene that no remaining reaction named in its
  rule before the removal, also when not requested (was: only the requested genes), as RAVEN's
  `removeGenes` does, when at least one of the requested genes is in the model; when none is,
  the model is not changed. A gene kept only by a constrained reaction stays.
* `init.merge_linear` numbers the merged groups 1, 2, ... in the order of the first reaction
  of each group in the original model, as RAVEN's `mergeLinear` does, so `group_ids` no
  longer depends on the merge order.
* `manipulation.merge_compartments` gives a merged metabolite the id of the copy already in
  the merged compartment, if any, otherwise the id that sorts first (`ACCOA_c` for `ACCOA_c`
  and `ACCOA_m`), as RAVEN's `mergeCompartments` does. It rewrote the first member's
  compartment suffix to the merged compartment (`ACCOA_s`). The merged metabolite also keeps
  the annotation of that copy (was: dropped), and `drop_single_metabolite_reactions` defaults to
  `False`, as RAVEN's `deleteRxnsWithOneMet` (was: `True`).
* `manipulation.merge_compartments` stores the base id of each merged metabolite in
  `notes["merged_base_id"]` (`ACCOA` for the kept `ACCOA_c`; the own id when two merged
  metabolites would get the same base id). The default metabolite key of
  `localization.assign_compartments`, `apply_assignment`, `relocate_reactions` and
  `curation_priority` is that stored base id, otherwise the id without the suffix of the
  metabolite's own compartment, so the merged `ACCOA_c` in `s` and a universal `ACCOA_m` share
  the key `ACCOA`. The four functions raise `ValueError` when two metabolites in one
  compartment have the same key (was: one of them stood for both).
* `manipulation.merge_compartments` documents where `notes["merged_base_id"]` survives: YAML
  written and read by raven-toolbox, `copy()`, cobra JSON and SBML. RAVEN's `readYAMLmodel`
  shows it in `metNotes` (`merged_base_id: ACCOA note: free text note`) and no RAVEN function
  uses it. A round trip through RAVEN YAML or a .mat file loses it, and the localization
  functions then key a merged `ACCOA_c` in `s` as `ACCOA_c`, without an error.
* `io.read_yaml_model` puts the notes that `write_yaml_model` writes as a `notes` mapping (notes
  without a RAVEN field) back into `notes` (was: nested under `notes["note"]`).
* `reconstruction.homology.run_blast`, `run_diamond` and `run_supporth`, and
  `reconstruction.kegg.run_hmmsearch`, `get_kegg_model_from_sequences`, `build_ko_hmm` and
  `build_hmm_library`, use every CPU the process may run on when `threads` is not given, as
  RAVEN's `getBlast`, `getDiamond`, `runSuppOrth` and `getKEGGModelForOrganism` do (was: the
  machine's CPU count minus one).
* `gapfilling.analyse_topology` takes RAVEN's `gapFillTopological` defaults. Seeds: the
  metabolites of every reaction with only substrates and `lower_bound < 0` or only products
  and `upper_bound > 0`, whatever its number of metabolites (was: one-metabolite reactions with
  `lower_bound < 0`); a `UserWarning` when there is none (was: a message only when `verbose`).
  Targets: the metabolites of the first objective reaction, substrates and products (was: every
  metabolite of the model); `ValueError` when the model has no objective.
* New: `utils.parse_gr_rule` reads a grRule in the grammar of RAVEN's `parseGrRule`
  (`and`/`or` in any case, `&`, `&&`, `|`, `||`, operators as whole words). `manipulation.gpr_to_dnf`
  now also takes a rule string and reads it with this parser, as RAVEN's `grRuleToDNF` does;
  `utils.is_dnf` reads a string with it too, and both raise `ValueError` with the message of
  RAVEN's `RAVEN:badGrRule` for a malformed rule. `G1 && G2`, `G1 || G2` and `G1 Or G2` are
  read as rules, `G1&G2` is one gene id. Whitespace is what MATLAB's `isspace` counts, so a
  no-break space (U+00A0), U+0085, U+2007 and U+202F are part of a word.
* `curation.batch_curate` (and `batch_curate_from_tsv`) stores a gene rule as `parse_gr_rule`
  reads it, so the reaction gets the genes that were checked: `G1&G2 or G3` has the genes
  `G1&G2` and `G3`, as in RAVEN's `curateModelFromTables` (was: `G1`, `G2` and `G3`).
* `io.read_yaml_model` reads gene rules with `parse_gr_rule`, so `G1&G2` written by
  `write_yaml_model` is read back as one gene (was: `G1` and `G2`, cobra's reading). A rule that
  `parse_gr_rule` refuses is read by cobra as before.
* Functions that copy or build a gene rule keep RAVEN's reading of it, so `G1&G2` stays one
  gene (was: the rule was copied as a string and cobra split `G1&G2` into `G1` and `G2`):
  `manipulation.merge_compartments`, `copy_to_compartment`, `convert_to_irreversible`,
  `merge_models`, `add_reactions_from_model`, `expand_model`, `change_gene_reaction_rules`,
  `add_reactions_from_equations`, `localization.apply_assignment`, `init.fill_tasks`,
  `init.remove_low_score_genes`, `reconstruction.homology.get_model_from_homology`,
  `reconstruction.kegg.assemble_model_from_ko_genes` and `io.read_matlab_model`. A rule string
  is read with `parse_gr_rule`; one that `parse_gr_rule` refuses is read by cobra. cobra's JSON
  reader (`cobra.io.load_json_model`) still splits `G1&G2`; cobra's SBML reader keeps it.
* `analysis.fseof` (and so `geckopy.ec_fseof`) solves each level with pFBA that holds the objective at 0.999999 of its optimum at that level, as RAVEN's `solveLP` does, where it held the full value. On the scipy solver (HiGHS) the optimum is only accurate to the solver's feasibility tolerance, so the second step was infeasible at the top level of ecYeastGEM and the scan returned 9 of 10 levels; it now returns all 10, as on Gurobi and GLPK. A level at which the biomass maximisation is infeasible still ends the scan.
* `analysis.fseof` takes the top of the scan as `max_fraction` of the target flux in the pFBA solution that maximises the target, as RAVEN's `FSEOF` and GECKO's `ecFSEOF` do, where it took the plain maximum (about 1e-6 relative difference).
* `analysis.fseof` lists targets of equal score in model order (a stable sort), as MATLAB's `sort` does, where their order was arbitrary.
* `analysis.fseof` gained `equality`: `True` fixes the target flux at each level as GECKO's `ecFSEOF` does, `False` (the default) keeps the lower bound of RAVEN's `FSEOF`.
* `io.read_yaml_model` reads an ec reaction written as `- enzymes: !!omap` with nothing under the tag, the form GECKO and RAVEN write for a reaction without enzymes, as a reaction with no enzymes, as `readYAMLmodel` does. It raised `AttributeError: 'ScalarNode' object has no attribute 'flow_style'`.
* `io.read_yaml_model` reads a scalar written as `""text""` (invalid YAML, written by older RAVEN and GECKO for a note that carried its own quotes) as `"text"`, as `readYAMLmodel` does. Files that parse as YAML are read unchanged.
* `io.read_yaml_model` reads a double-quoted scalar with an unescaped quote inside, such as `- notes: "a "b" c"`, as `a "b" c`: it drops the outer pair of quotes and unescapes `\\` and `\"`, as `readYAMLmodel` does. This repair runs only when the strict parse fails.
* `io.read_yaml_model` reads a single-quoted scalar with an apostrophe inside, such as `- name: 'it's'`, as `it's`: it drops the outer pair of quotes and reads each `''` as `'`, as `readYAMLmodel` does. This repair runs only when the strict parse fails; a comment after the closing quote is not handled.
* `init.ftinit` and `init.run_ftinit` give each scored reaction its own big-M, its flux
  capacity `max(min(ub, big_m), min(-lb, big_m), force_on)` with an infinite bound counted as
  `big_m`, as RAVEN's `ftINITInternalAlg` does. `big_m` is the cap on that capacity; its
  default is 1000 (was 100).
* `tasks.find_task_essential_reactions` (and so `init.prep_init_model`) re-solves a knock-out
  from a cold start when the warm-started solve does not report a feasible status, and
  counts the reaction as essential only if that solve is not feasible either, as RAVEN's
  `solveLP` does. Gurobi could report a feasible knock-out as infeasible when started from
  the previous knock-out's basis, which added reactions to the essential set.
* `init.ftinit` and `init.run_ftinit` with `resolve_ties` take each step's fluxes from the
  least-total-flux solution for the tie-broken on set, so the direction and magnitude at
  which the next step forces a reaction no longer depend on the solver's vertex.
  That phase stops at a relative or absolute gap of 1e-6 and has the full `time_limit`, as
  in RAVEN.
* `init.ftinit` and `init.run_ftinit` with `resolve_ties`: among the least-total-flux
  solutions, each step takes its fluxes from the one with the least id-rank-weighted backward
  flux, so the direction of a flux that carries the same total either way no longer depends
  on the solver. The id-rank phases rank a merged reaction by the smallest original reaction
  id in its group (new `run_ftinit` argument `rank_keys`), so they no longer depend on which
  member names it.
* `init.ftinit` and `init.run_ftinit` with `resolve_ties` raise a `UserWarning` naming the
  phase (`phase2c-minflux`, `phase2d-direction`) when the minimal-flux or the direction phase
  finds no solution, as RAVEN does, and keep the fluxes of the phase before it. A phase that
  stops at the time limit is adopted with a `UserWarning` that its fluxes are an unproven
  incumbent.
* `init.ftinit` with `resolve_ties`: the last step skips the minimal-flux and direction phases,
  as no step reads its fluxes, and reports the fluxes of the id-rank phase. The reaction set
  does not change.
* Docstrings are rewritten in impersonal, present-tense wording with British spelling; the code is unchanged. Statements that differed from the code are corrected, for example `find_good_reactions`, `sum_biomass`, `diff_models`, `compare_models`, `set_gene_rule`, `constrain_reversible_reactions`, `ftinit` (essential forcing, `mip_gap`, determinism figures), `fast_lp` metabolite matching and `batch_curate` index matching.

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
