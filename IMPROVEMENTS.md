# Improvements over RAVEN — decisions and open proposals

This project is **not** a one-to-one MATLAB→Python transcription. Where a RAVEN function can be
made smarter/faster, or where a logical gap in RAVEN's feature set is worth filling, we record the
change — with enough detail that it can **also be back-ported to MATLAB RAVEN** later.

> **Implemented improvements** — and the ones worth carrying upstream into MATLAB RAVEN — now live
> in [matlab_raven_backports.md](https://github.com/CompMetEng/raven-toolbox/blob/develop/docs/reference/matlab_raven_backports.md),
> the MATLAB ↔ raven-toolbox **differences** record (both directions: improvements to back-port, and
> functionality MATLAB has that raven-toolbox deliberately doesn't). **This file** keeps only the
> design decisions behind what was ported *differently* or *not at all*, and the improvements still
> *proposed* (not yet implemented).

Categories:
- **EFFICIENCY** — same behavior, faster/smarter implementation.
- **ERGONOMICS** — same job, less friction / fewer foot-guns / clearer contract.
- **NEW** — functionality RAVEN lacks but that fits naturally alongside what it already has.
- **REMOVAL** — functionality that should be dropped (here *and* in MATLAB RAVEN) because it does
  more harm than good.

Status legend: 💡 proposed · 🗑️ dropped (and to remove from MATLAB RAVEN) · ↩ revised · ❌ rejected

---

## R-MetaCyc: drop MetaCyc reconstruction (REMOVAL — also remove from MATLAB RAVEN)

**Decision 2026-05-24:** MetaCyc-based reconstruction is **not ported to raven-toolbox** and should be
**removed from MATLAB RAVEN**. Status: 🗑️.

**What RAVEN does:** `getMetaCycModelForOrganism` builds a draft by BLAST/DIAMOND of the query
proteome against `protseq.fsa` — MetaCyc's **single representative protein sequence per enzyme**
(~11.6k sequences) — keeping each gene's best hit above a bitscore/positives cutoff and assigning
the linked reaction. With one representative per enzyme there is no profile to tell true family
members from look-alikes.

**Evidence (this repo, real MetaCyc + KEGG 118 data):** a leave-organism-out precision/recall test
(query each representative against the others, excluding its own organism; ground truth =
MetaCyc's own MONOMER→reaction):

| bitscore (ppos≥45) | reaction precision | EC-family precision | EC recall |
|---|---|---|---|
| 50 | 0.34 | 0.55 | 0.33 |
| 100 *(RAVEN default)* | 0.36 | 0.59 | 0.32 |
| 200 | 0.40 | 0.62 | 0.26 |
| 300 | 0.44 | 0.65 | 0.22 |

At the default cutoff **~64 % of reaction assignments are wrong** (~41 % wrong even at EC-family
level); **no cutoff rescues precision** — tightening to bitscore 300 reaches only ~44 %/65 % while
recall halves. Real proteomes (with non-enzyme decoys, not in this test) would be worse. Test
scripts/artifacts: `/home/eduardk/metacyc_test/` (not committed).

**Why drop rather than fix:** the low precision is intrinsic to MetaCyc's one-representative-per-
enzyme data (can't build KEGG-quality HMMs from it). Accurate gene-calling already exists via KEGG
HMMs (3b) and homology-to-template-models (3a). MetaCyc's genuine value (extra reactions/pathways/
compound structures) does not justify a separate, data-heavy, low-precision track.

**MATLAB RAVEN removal list** (`external/metacyc/`): `getMetaCycModelForOrganism.m`,
`getModelFromMetaCyc.m`, `getRxnsFromMetaCyc.m`, `getMetsFromMetaCyc.m`, `getEnzymesFromMetaCyc.m`,
`linkMetaCycKEGGRxns.m`, `addSpontaneousRxns.m`, and data `metaCycEnzymes.mat` / `metaCycMets.mat`
/ `metaCycRxns.mat` / `protseq.fsa`; plus any `combineMetaCycKEGGModels` and MetaCyc references in
tutorials/tests/docs. (`addSpontaneousRxns` could be kept as a small standalone helper if wanted —
it is only incidentally in the MetaCyc folder.)

---

## Still proposed (not yet implemented)

* **A4** (`addRxns`, NEW 💡) — **Infer compartment from a structured metabolite ID** (e.g. `atp_c` →
  `c`) as an alternative to requiring `compartment`. Would reduce boilerplate for SBML-style IDs;
  revisit alongside `addMets`.
* **Y4** (YAML, NEW 💡) — **A first-class home for `deltaG` / `confidence_score`.** They live in
  `notes` because neither cobra nor SBML core has a standard slot; if one emerges (SBML fbc/groups,
  or a cobra attribute), both raven-toolbox and MATLAB RAVEN (`metDeltaG` / `rxnConfidenceScores`)
  should migrate. Watching brief.
* **R4** (`remove_metabolites`, review 💡) — **Deletion candidate.** Its only value over cobra is
  `by_name` cross-compartment deletion, likely rarely used; revisit and possibly drop the wrapper.

MATLAB-RAVEN-only proposals (**G1** hashed lookup, **G5** the `[1 1 1]` mask-vs-index bug, **G7** the
reusable `name[comp]` parser) are in the *getIndexes* decision below.

---

## Design decisions — ported differently, or not ported

### getRxnsInComp / getMetsInComp — not ported

Briefly ported, then **removed** (user review): too thin over cobra (`metabolite.compartment` /
`reaction.compartments` one-liners). Mapped in the §1 migration cheatsheet instead. Reconsider only
if a downstream consumer needs the `include_partial` (fully-contained vs touching) distinction in
several places — and ask before re-adding (see process note: argue pros/cons for marginal WRAPs).

### removeReactions — not ported (orphan cleanup kept coupled)

`remove_metabolites` / `remove_genes` are ported
([manipulation/remove.py](https://github.com/CompMetEng/raven-toolbox/blob/develop/src/raven_toolbox/manipulation/remove.py)).
`removeReactions` was **not** ported: with orphan cleanup kept coupled (decision: don't separate
metabolites from genes), it is identical to `cobra.Model.remove_reactions(remove_orphans=...)`.
Separable orphan-met vs orphan-gene cleanup (**R1**, ❌ rejected) was considered then dropped — keep
them coupled like cobra; the `remove_reactions` wrapper was removed entirely as a result.

### getIndexes — do not port the index resolver

**Decision (raven-toolbox): do NOT port the function.** cobra is object-oriented, so the central
index-resolver that RAVEN's struct-of-parallel-arrays design requires is largely unnecessary.
cobra's `DictList` already covers the use cases more idiomatically — `get_by_any` (mixed
id/object/index → objects), `get_by_id` (O(1)), `query` (name/substring/regex), `index` (position),
list comprehensions for filtering. Porting a 1-based-index resolver would be redundant and
un-Pythonic. **Only** the `name[comp]` composite resolver is kept (G7), as a small internal helper.

The improvement insights below still hold for **MATLAB RAVEN**, where the function remains — flagged
as upstream-only back-port candidates.

| # | Cat | Target | Status | Improvement |
|---|---|---|---|---|
| G1 | EFFICIENCY | MATLAB RAVEN only | 💡 | **Hash-based lookup instead of per-query linear scan.** RAVEN loops `find(strcmp(obj(i), searchIn))` per query → O(n·m). Build a `containers.Map` `{id: position}` once (O(n)), look up in O(1); `metcomps` likewise. (Moot for raven-toolbox — cobra's `DictList` is already hashed.) |
| G5 | ERGONOMICS (bug) | MATLAB RAVEN only | 💡 | **Disambiguate the `[1 1 1]` mask-vs-index bug.** RAVEN's `if all(objects)` conflates a logical all-true mask with the index vector `[1 1 1]` (its own comment: "This gets weird if it's all 1"). Test `islogical(objects)` explicitly instead of `all(objects)`. (Moot for raven-toolbox — input kind is decided by dtype.) |
| G7 | NEW | raven-toolbox helper | 💡 | **Extract a reusable `name[comp]` parser/resolver.** The composite-id parsing buried in `getIndexes`'s `metcomps` branch is the one capability cobra lacks. Expose as a standalone `parse_name_comp` / `resolve_metabolite_by_name_comp`, reused by `addRxns`/`addTransport`/`mergeModels`. This is the only piece carried into raven-toolbox. |

**Obsoleted by cobra (no action — these were earlier raven-toolbox proposals now covered by `DictList`):**
predictable return type, return objects-not-positions, configurable missing-object policy across a
batch, and substring/regex matching — all already provided by `get_by_any` / `get_by_id` / `query`.

### standardizeGrRules — port the lint half only

RAVEN `manipulation/standardizeGrRules.m` normalizes grRule syntax + flags rules not in simple
OR-of-AND-complex (DNF) form (`findPotentialErrors`). **Decision (raven-toolbox): port the lint half
only.** cobra auto-normalizes a GPR on assignment (`"(G1 AND G2)  OR  G3"` is stored as
`"(G1 and G2) or G3"`), so the normalization half is redundant. The non-DNF lint has no cobra
equivalent and was ported as `find_non_dnf_grrules`/`is_dnf`
([utils/gpr.py](https://github.com/CompMetEng/raven-toolbox/blob/develop/src/raven_toolbox/utils/gpr.py)).

### readYAMLmodel — no separate legacy parser needed

**Lens correction (no separate legacy parser).** RAVEN ships a 462-line `parseYAMLLegacy.m` for the
`!!omap` dialect, and geckopy refuses it ("re-save from MATLAB"). But `!!omap` is **cobra's own YAML
format**: `cobra.io.load_yaml_model` reads a real yeast-GEM.yml (4102 rxns) directly. So the
raven-toolbox-unique capability the PLAN imagined (a legacy reader) is unnecessary; the real
cobra-absent value is preserving `metaData` identity and RAVEN-only per-entry fields, which is what
was built.

### setParam — built, then trimmed

A 6-mode keyword `set_parameters` was built then **trimmed** (**P1**, ↩ revised): not Pythonic — it
re-wrapped cobra one-liners for `lb`/`ub`/`eq`/`obj`/`unc`. Only the `var` ±% band, which cobra has
no idiom for, is kept as `set_variance_bounds`; the rest are documented as cobra idioms in the §1
cheatsheet.
