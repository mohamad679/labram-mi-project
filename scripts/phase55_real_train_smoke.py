"""Verify Phase 5.5 scratch EEGNet/EEGNeX gradients using only real Train S003.

A CPU smoke, not a validation benchmark, performance result or Test inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prepare_physionet_s3 import RUNS, edf_path, fetch_run, unsigned_s3
from src.data import load_physionet_mi
from src.phase55_trainonly import epoch_standardize, make_bandpower_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mne-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    import os

    import torch
    from braindecode.models import EEGNet, EEGNeX

    args.mne_root.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(args.mne_root.resolve())
    os.environ["MNE_DATASETS_EEGBCI_PATH"] = str(args.mne_root.resolve())
    os.environ["MOABB_DOWNLOAD_PROVIDER"] = "upstream"
    client = unsigned_s3()
    fetched = []
    for run in RUNS:
        size, _ = fetch_run(client, 3, run, args.mne_root)
        if edf_path(args.mne_root, 3, run).stat().st_size != size:
            raise RuntimeError("Bad original public Train EDF transfer")
        fetched.append(run)
    eeg, labels, meta = load_physionet_mi(subjects=[3])
    if eeg.shape[1:] != (64, 400) or len(eeg) < 2 or len(meta) != len(labels):
        raise ValueError("Original actual Train EEG invalid")
    if set(map(int, meta["subject"].to_numpy())) != {3}:
        raise ValueError("Unexpected participant leakage")
    from src.portable_eeg import LABELS

    y = np.array([LABELS.index(str(x)) for x in labels[:2]], dtype=np.int64)
    samples = np.asarray(eeg[:2], dtype="f4")
    bandpower = make_bandpower_features(samples)
    if bandpower.shape != (2, 256) or not np.isfinite(bandpower).all():
        raise ValueError("Invalid bandpower feature smoke")
    report = {"status": "PASS_REAL_TRAIN_SUBJECT_3_ONLY", "subject": 3,
              "test_eeg_touched": False, "validation_eeg_touched": False,
              "six_source_imagery_runs": fetched, "source_samples": int(len(eeg)),
              "models": {}}
    for name, model, x in (
        ("eegnet_raw", EEGNet(n_chans=64, n_times=400, n_outputs=5, sfreq=160.), samples),
        ("eegnet_epoch_standardized", EEGNet(n_chans=64, n_times=400,
                                               n_outputs=5, sfreq=160.),
         np.stack([epoch_standardize(z) for z in samples])),
        ("eegnex_epoch_standardized", EEGNeX(n_chans=64, n_times=400, n_outputs=5,
                                               sfreq=160., kernel_block_1_2=32,
                                               drop_prob=.5),
         np.stack([epoch_standardize(z) for z in samples])),
    ):
        torch.manual_seed(42)
        model.train()
        logits = model(torch.tensor(x, dtype=torch.float32))
        if logits.shape != (2, 5) or not torch.isfinite(logits).all():
            raise ValueError(f"Invalid {name} logits")
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(y))
        if not torch.isfinite(loss):
            raise ValueError("Invalid model loss")
        loss.backward()
        params = list(model.parameters())
        gradients = [z.grad for z in params if z.grad is not None]
        if not gradients or not any(torch.isfinite(z).all() and torch.any(z != 0)
                                    for z in gradients):
            raise ValueError("No finite nonzero model gradient")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.,
                                                   error_if_nonfinite=True)
        report["models"][name] = {"logits_shape": list(logits.shape),
                                  "finite_nonzero_gradients": True,
                                  "gradient_norm": float(grad_norm)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
