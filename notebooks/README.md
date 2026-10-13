# Archived execution notebooks

These are **historical research execution records**. The user ended the experimental study on 2026-10-13 after the final Phase 5.5b Validation audit; **no additional seeds, models, sweeps or held-out Test evaluation are planned**.

| Stage | Notebook | Why it is kept |
| :-- | :-- | :-- |
| Original Phase 3 | [Phase 3 Validation](NeuroDecode_Phase3_Validation_Rerun_Kaggle.ipynb) | Historical source, pretrained/checkpoint provenance and baseline results |
| Authentic LaBraM Phase 4 | [Fine-tuning](NeuroDecode_Phase4_Finetune_Kaggle.ipynb) | Audited pretrained strategy comparison |
| Authentic CBraMod Phase 5 | [S3 and private dataset recovery](NeuroDecode_Phase5_CBraMod_AutoS3_PrivateDataset_Kaggle.ipynb) | Executed source/data and locked runtime context |
| Historical Phase 5 first notebook | [Initial comparator notebook](NeuroDecode_Phase5_CBraMod_Kaggle.ipynb) | Preserves initial source and recovery provenance; superseded for usability |
| Phase 5.5a | [Train-only GroupKFold](NeuroDecode_Phase55_TrainOnly_CV_Kaggle.ipynb) | Fixed candidate screen and fold provenance |
| Phase 5.5b | [One-time Validation](NeuroDecode_Phase55b_OneTime_Validation_Kaggle.ipynb) | Final selected-model fit and one-time development Validation |

Do **not** interpret these notebooks as a new experiment plan. Scientific results are archived in [the evidence registry](../results/validation/README.md); use [the final report](../docs/final-research-summary.md) for the accepted interpretation. The old repository-root [run_kaggle.ipynb](../run_kaggle.ipynb) also remains for historical code comparison, not a prescribed rerun.
