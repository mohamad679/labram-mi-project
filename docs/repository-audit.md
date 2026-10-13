# Repository publication audit — 13 October 2026

**Scope:** Full recursively enumerated tracked repository tree (155 paths before cleanup, including source, notebooks, docs, CI, results, figures and a ~24.5 MB legacy demo recording). All edits are non-destructive commits on a reviewable PR; no Git history rewriting, no new model training, and no hidden Test inference.

## Defects identified and disposition

| Finding | Impact | Treatment |
|---|---|---|
| Front-page README described the project primarily as an unverified LaBraM prototype and reported old incompatible metrics | Readers could miss completed real 2026 Kaggle runs or believe fake/unverified pretrained performance | **Replaced** with a source-evidenced project overview, frozen 75/14/15 split, all completed Phase 3–5.5b model scores and transparent scientific limitations |
| Original flowchart implied a held-out **Test** evaluation fed results figures | False inference: the 15 Test participants were not evaluated in audited phases | **Replaced** with actual split/Train-only CV/one-time Validation pipeline and explicitly isolated LOCKED Test lane |
| No integrated current figure gallery from real checked predictions | Poor accessibility/credibility: historic attention figures looked like performance evidence | Added **seven elegant SVG graphics** generated solely from validated Kaggle outputs and versioned metrics, with class supports, subject heterogeneity, case counts and run provenance |
| Development-Validation numbers and Train-only OOF were not prominently separated | Cross-cohort numerical comparisons can exaggerate generalization | Created **separate** model comparison charts and documented non-comparability of 0.3912 Train OOF vs 0.4191 historic Validation |
| Per-phase CBraMod full adaptation accuracy looked superficially high | Majority-class-only prediction masks total four-class failure | Annotated **rest-only collapse** and Macro-F1 .1302 in all primary result displays |
| Superseded original 12-cell CBraMod notebook plus newer 20-cell S3 notebook | Two launchers without clear authoritative choice | Removed unused duplicate from current checkout; retained actual executed pinned S3 version and immutable Git history |
| Generic root Kaggle training campaign, unused multi-seed launcher and their static tests | Invited further runs after owner froze experimentation; obsolete, outdated dependencies | Removed unused **root notebook, multiseed launcher and their now-inapplicable structural tests** from active tree; all completed phase-specific notebooks and verification tests preserved |
| Large ~24.5 MB old demonstration video and populated-directory `.gitkeep` placeholders | Bloated fresh checkout and visually overprioritized a non-reproducible legacy model | Removed these from HEAD; retained historical demonstration **PNG**, explanation and all original Git history. Git object history remains large until separately rewritten (not undertaken). |
| Old laptop/upload/Kaggle and future-campaign guides contradicted current source-verified S3 process and the stopped research scope | False or unfulfillable run instructions | Retired or marked original documents as **legacy archival context**, added accurate notebooks index and evidence-based usage guidance |
| Scattered phase JSONs, methods and citations lacked a single authoritative narrative | Difficult peer-review and portfolio inspection | Added [final scientific snapshot](final-research-snapshot.md), updated [documentation index](README.md), [model card](model-card.md), [release boundaries](release-readiness.md), and [full visualization gallery](visualization-gallery.md) |
| Lack of explicit repository code license / author-approved citation and distributable final model weights | Public code not safely described as reusable/production-ready | **Not fabricated.** Clearly stated in public-facing README and release-readiness checklist; owner action needed before an actual licensed software/model release |

## Retained intentionally

- **Source, tests and scientific provenance:** `src/`, `scripts/` for actually performed analyses, `configs/phase3_locked_split.json`, exact official pretrained-adapter logic, real-data EEG smoke and safety checks, and `results/validation/` JSON manifests.
- **Actual execution notebooks:** the five owner-run, phase-specific pinned Kaggle notebooks under [notebooks/](../notebooks/README.md), plus the original reusable S3 processing source. HDF5, EDF and checkpoint binaries remain excluded from Git.
- **Scientific historical evidence:** phase protocol documents and legacy attention/demo screenshots with explicit labels; these are **not** advertised as accepted benchmark scores.
- **CI and review:** unit testing, Ruff linting, tracked-sensitive-artifact checks, local Markdown link checks and historically executed real public EEG source smoke evidence.

## Remaining real constraints — not defects silently fixed

1. The 15-person Test was intentionally **never assessed** during the accepted development study. No new seed or Test campaigns are authorized.
2. True clinical/real-world EEG robustness and external replication have not been established.
3. Source license and author/citation metadata need explicit owner choices; absent authority, none was guessed.
4. Published model-weight binaries and an exact machine-image dependency lock are not provided. Genuine runner and model-artifact hashes **are** recorded per phase; do not misstate external model-binary revalidation.
5. Old code paths for descriptive/archival models remain source-reviewable but **not part of a newly evaluated deployable service**. The Gradio screenshot is a historical illustration.
6. This cleanup is a **research freeze**, not automatically a GitHub archived/read-only repository flag: the owner may switch on GitHub's Archive setting separately if the interface permission/action is available.
7. README visuals render as static SVGs and remain accessible by text captions. The gallery intentionally excludes fabricated image previews or generated patient EEG waveforms.

## Audit acceptance

- [x] Real 5.5b final ZIP passed CRC and independently recomputed 2,436-row five-class metrics.
- [x] Final metrics are truthful: EEGNet raw Validation Macro-F1 **.4191** vs 5.5b normalized **.4125**, with **no claimed gain**.
- [x] Groupwise Train vs Validation metrics have separate source-coded chart panels.
- [x] Figures are vector images with explicit labels, supports, cohort scope and negative findings.
- [x] All 15 original Test subjects remain embargoed in the recorded runs.
- [x] Unnecessary legacy assets removed from checkout without rewriting history.
- [x] Current tracked doc links, sensitive-artifact policy and unit tests run via GitHub Actions before merge.
- [ ] Source license / authorship / clean-room GPU checkpoint distribution, outside the owner's requested scope.

**A visually finished GitHub research record is not a certified clinical model, an independently held-out Test benchmark, or a licensed release.** The accountable end state is an attractive, honest and frozen research archive.
