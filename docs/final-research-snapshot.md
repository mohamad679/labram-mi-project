# Final research snapshot — NeuroDecode

**Frozen on 13 October 2026 · Scientific stage: single-seed development Validation · Repository maintenance may continue, but no further seed/model optimization, Phase 6, or original Test evaluation is planned.**

## Executive scientific conclusion

The project produced and independently checked valid, source-pinned EEG recordings and predictions for an exact 104-subject EEGMMIDB motor-imagery cohort (75 Train / 14 development Validation / 15 untouched Test participants). Six models/strategies across EEGNet, LaBraM and CBraMod were evaluated under clearly identified development protocols. **Original EEGNet remains the top development-Validation Macro-F1 at 0.4191.** Pretrained foundation architectures **did not** outperform it in the tested settings. CBraMod Full collapsed entirely to `rest`.

The later **Train-only** three-fold subject cross-validation supported per-epoch EEG standardization as a promising preprocessing candidate: EEGNet standardized **0.3912** versus raw **0.3212** Macro-F1 on pooled Train OOF. The frozen winner was trained on 67 original Train subjects and selected on eight separate Train subjects, then evaluated once on the 14 original development Validation people: **Macro-F1 0.4125**, **lower** than the earlier EEGNet development reference **0.4191**. Hence **no independent Validation gain** is claimed from this step. The final result is a credible negative/neutral finding, not an engineering failure to conceal.

## Quantitative evidence

| Experiment | Scope | Accuracy | Macro-F1 | Balanced acc. |
|---|---|---:|---:|---:|
| Original EEGNet | 14-subject development Validation | .4799 | **.4191** | .4418 |
| EEGNet standardized, nominated on Train OOF | One-time 14-subject development Validation | .4885 | .4125 | .4236 |
| LaBraM pretrained partial | 14-subject development Validation | .3859 | .2546 | .2618 |
| LaBraM pretrained full | 14-subject development Validation | .3485 | .2481 | .2615 |
| CBraMod pretrained frozen probe | 14-subject development Validation | .2779 | .2027 | .2091 |
| CBraMod pretrained full | 14-subject development Validation | .4828 | **.1302** | .2000 |

**Warning:** The CBraMod Full `.4828` accuracy results from predicting `rest` for all 2,436 cases. It is not a competitive five-way model. Comparisons are **descriptive**, because adapters, training data partitions, and early-stop protocols differ. No score in this table comes from the 15 untouched Test participants. All are **single-seed** development observations.

![Complete development Validation comparison](../assets/figures/validation_model_comparison.svg)

## Final nominated EEGNet — details that matter

- [Genuine user-run 5.5b ZIP evidence](../results/validation/phase55b_one_time_validation_seed42_20261013.json): SHA-256 `8e88e4463d1fa96856cb127f57e4a706042ff441095ccb1c1ad5d9109f635391`, ZIP CRC passed, two entries.
- Original EEGMMIDB schema-3 HDF5 SHA-256 `f6d2c7824f6aaa522286ea8ee5253b0220602117fc69d7b49c3af6e6a1324f7a`, **18,096** EEG epochs, shape `(64,400)` per epoch at 160 Hz and five original labels.
- **67** Train participants used for parameter fitting (**11,658** epochs); eight *other Train participants* `[9,25,56,61,70,81,83,90]` selected the checkpoint (**1,392** epochs). Epoch **7** was selected using the **inner Train-only** macro-F1. Training stopped after 12 epochs. FP32, no gradient bypass.
- Original 14-subject development Validation: **2,436** unique EEG row indices, same 14 fixed subject IDs, 174 epochs/person, five finite normalized class probabilities. The model predicted all five labels.
- Independently recomputed final accuracy **.4885057**, Macro-F1 **.41253535**, balanced accuracy **.4236194** and all 25 confusion cells matched owner-executed Kaggle JSON. Per-class F1 (feet/hands/left/rest/right): **.2551 / .3604 / .4414 / .6593 / .3466**. Per-subject F1 ranges **.218–.704**. Exploratory 14-subject bootstrap 95% Macro-F1 interval **[.3486,.4847]**.
- Kaggle reported the selected checkpoint was saved/reloaded and its inner Train probabilities matched. The checkpoint *binary was not in the evidence ZIP*, so **external checkpoint reload has not been verified**.
- The 15 reserved original Test people were **not inferred or scored** in 5.5b; no independent heldout confirmatory test result exists. The historical 5.5a Train CV improvement is **not** a Validation superiority proof.

## What was independently audited, and what was not

| Evidence level | Status |
|---|---|
| Official checkpoint retrieval & strict pretrained LaBraM/CBraMod load | Real public training-signal smoke and strict key/hash audits passed; see [Phase 4](phase4-finetuning-protocol.md) and [Phase 5](phase5-foundation-comparator.md) |
| Phase-2 complete data structure and checksum | Verified by owner-executed Kaggle reports, versioned structural audit and frozen cohort metadata |
| Individual phase prediction files | The users' Kaggle ZIPs were independently opened and classification metrics reconciled; per-phase Git JSONs record SHA and limitations |
| Phase 5.5a 13,050 Train OOF rows per model | Four candidates, three outer folds, exact subject grouping and numerical output integrity audited |
| Phase 5.5b final Validation 2,436 rows | Independent metric, confusion and subject support calculation passed |
| Independent model-weight reload from contributed ZIP | **Not done**: weight binaries not shared; Kaggle-run reload was reported, not repeated externally |
| Multiple seeds, genuinely untouched final Test, external replication | **Not done; deliberately closed** at owner request |
| Licensed and packaged trained model or reproducible live Gradio service | **Not established**; no owner-selected project license or redistributable final weights |

## File-curation decision

- **Retain:** deterministic data/split source, frozen configs, pretrained adapters, scientific smoke tests, code for actually run stages, unit tests, participant-aware scoring, matching notebooks, original phase audit JSONs and all detailed scientific provenance. Deleting those would erase the basis for the findings.
- **Retain but de-emphasize:** legacy LaBraM demo, exploratory attention/BALD and earlier unverified head-only historical results. Their status is separate from independently reviewed new empirical results.
- **Remove from the active snapshot:** oversized obsolete demo recording, empty placeholder files, duplicate superseded notebook and unexecuted multi-seed launcher and its obsolete launcher-only tests; these remain recoverable from Git history where applicable. Do **not** rewrite history or misrepresent that Git history size shrank.
- **Replace:** obsolete landing-page claims and proposed future performance campaigns with a final evidence index, model card, dedicated publication figures and an explicit research-freeze note. No Test was opened for presentation work.

## Interpretation and reproducibility limits

This is a difficult *cross-subject* motor-imagery task containing a large `rest` category (~48.3% of Validation). Motor imagery is a research signal with event/run-specific semantics. Some approaches require a different input adapter: CBraMod resamples to 200 Hz and pads a synthetic segment, and LaBraM uses a patch-compatible prefix crop of the 3-second EEGMMIDB event. Do not describe the network comparison as a controlled architecture-only ablation. The original 14-person Validation cohort has already been repeatedly inspected in the research sequence, so results are **development/exploratory**; the 15-person Test remains held out. Subject-cluster intervals capture one source of uncertainty, not training-seed or external-dataset variance. The shown figures derive from actual versioned metrics; they are not clinical evidence.

## Dataset, upstream credit and rights

- PhysioNet EEG Motor Movement/Imagery dataset (EEGMMIDB), DOI [10.13026/C28G6P](https://doi.org/10.13026/C28G6P); exact signal/run contract [Phase 2](phase2-data-audit.md).
- EEG processing: [MOABB](https://moabb.neurotechx.com/), [MNE](https://mne.tools/), [Braindecode](https://braindecode.org/).
- Official LaBraM: [original research](https://openreview.net/forum?id=QzTpTRVtrP) and Hugging Face `braindecode/labram-pretrained` (revision recorded in [Phase 4](phase4-finetuning-protocol.md)). Official CBraMod: [ICLR 2025 paper](https://openreview.net/forum?id=NPNUHgHF2w) and `braindecode/cbramod-pretrained` (revision recorded in [Phase 5](phase5-foundation-comparator.md)).
- The repository's original code **has no declared owner-approved license**. Copyright/reuse and release tagging cannot be implied by the public GitHub page. The archived attention and Gradio graphics are historical illustrations, not independently audited current predictions.

**Final closure:** No more random seeds, no architecture hunting, no unseen-Test inference, no claimed performance improvement not demonstrated by development evidence. Future work is limited to keeping the record intelligible and correcting provenance/documentation defects.