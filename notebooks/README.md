# Execution notebooks — archival evidence index

These five Kaggle notebooks are retained because they document **actual completed, source-hash-pinned experiments** that led to the final 2026-10-13 NeuroDecode development evidence. They are **not** an invitation to start more Seed campaigns, open the 15-person Test, or tune the 14-person Validation.

| Stage | Retained notebook | What it produced |
|---|---|---|
| Phase 3 | [Original baselines](NeuroDecode_Phase3_Validation_Rerun_Kaggle.ipynb) | Same frozen 14-subject development-Validation baseline comparison |
| Phase 4 | [Pretrained LaBraM fine-tuning](NeuroDecode_Phase4_Finetune_Kaggle.ipynb) | Probe, partial and full adaptation on development Validation |
| Phase 5 | [CBraMod + S3 original data recovery](NeuroDecode_Phase5_CBraMod_AutoS3_PrivateDataset_Kaggle.ipynb) | Genuine pretrained CBraMod and disclosed full-mode class collapse |
| Phase 5.5a | [Subject-only Train CV](NeuroDecode_Phase55_TrainOnly_CV_Kaggle.ipynb) | Three GroupKFold folds and selected normalized EEGNet (no historic Validation read) |
| Phase 5.5b | [Final nominated development Validation](NeuroDecode_Phase55b_OneTime_Validation_Kaggle.ipynb) | Exactly one frozen evaluation of normalized EEGNet on the original 14-subject Validation |

Each notebook's immutable source revision and expected exact code/dataset SHA appear in its own cells and corresponding [phase report](../docs/README.md). The old generic root `run_kaggle.ipynb`, unused multi-seed launcher and replaced 12-cell CBraMod notebook have been **removed from the active checkout**, with their Git history preserved.

**No Kaggle notebook here produced an independent, untouched 15-subject Test result.** No proprietary data or binary checkpoint is bundled in Git. See [scientific closure and actual metrics](../docs/final-research-snapshot.md) and [published vector figures](../docs/visualization-gallery.md).
