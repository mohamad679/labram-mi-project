"""
Phase 3 — Explainability via attention rollout.

Scope:
  - Attention rollout only.
  - No SHAP / LIME / perturbation explainers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from sklearn.preprocessing import LabelEncoder

from src.data import subject_wise_split
from src.model import load_labram_model
from src.train import CHECKPOINT_PATH, DEVICE, load_dataset


RESULTS_PATH = "/kaggle/working/attention_rollout_results.npz"
REPORT_PATH = "/kaggle/working/attention_rollout_report.md"
PLOT_DIR = Path("/kaggle/working/attention_rollout_plots")

N_SAMPLES = 4
TOP_K_CHANNELS = 16


def prepare_tokens_for_rollout(model, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Prepare embedded tokens before transformer blocks."""
    interp = getattr(model, "interpolation_layer", None)
    if interp is not None:
        x = interp(x)

    batch_size, n_input_chans, _ = x.shape

    input_chans = torch.arange(
        n_input_chans + 1,
        device=x.device,
        dtype=torch.long,
    )

    x = model.patch_embed(x)
    n_patch_tokens = x.shape[1]

    if n_patch_tokens % n_input_chans != 0:
        raise RuntimeError(
            f"Cannot reshape {n_patch_tokens} patch tokens into "
            f"{n_input_chans} channels."
        )

    n_time_patches = n_patch_tokens // n_input_chans

    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    pos_embed_used = model.position_embedding[:, input_chans]
    pos_embed = model._adj_position_embedding(
        pos_embed_used=pos_embed_used,
        batch_size=batch_size,
    )
    x = x + pos_embed

    time_embed = model._adj_temporal_embedding(
        num_ch=n_input_chans,
        batch_size=batch_size,
        dim_embed=model.embed_dim,
    )
    x[:, 1:, :] = x[:, 1:, :] + time_embed

    x = model.pos_drop(x)
    return x, n_input_chans, n_time_patches


def attention_rollout(model, x: torch.Tensor) -> dict:
    """Compute CLS-token attention rollout across all transformer blocks.

    Attention tensors are captured with forward hooks during the normal block
    forward pass. This avoids calling each transformer block twice.
    """
    model.eval()
    x = x.to(DEVICE)

    captured_attentions = []
    handles = []

    def _capture_attention(_module, _inputs, output):
        captured_attentions.append(output.detach())

    with torch.no_grad():
        tokens, n_input_chans, n_time_patches = prepare_tokens_for_rollout(model, x)

        try:
            for block in model.blocks:
                if not hasattr(block, "attn") or not hasattr(block.attn, "attn_drop"):
                    raise RuntimeError(
                        "Expected each LaBraM block to expose block.attn.attn_drop "
                        "for attention hook capture."
                    )

                handle = block.attn.attn_drop.register_forward_hook(_capture_attention)
                handles.append(handle)

            h = tokens
            for block in model.blocks:
                h = block(h)

        finally:
            for handle in handles:
                handle.remove()

        if len(captured_attentions) != len(model.blocks):
            raise RuntimeError(
                f"Captured {len(captured_attentions)} attention tensors, "
                f"but expected {len(model.blocks)} transformer blocks."
            )

        attentions = captured_attentions

        batch_size = attentions[0].shape[0]
        n_tokens = attentions[0].shape[-1]

        eye = (
            torch.eye(n_tokens, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        rollout = eye.clone()

        for attn in attentions:
            attn = attn.mean(dim=1)
            attn = attn + eye
            attn = attn / attn.sum(dim=-1, keepdim=True)
            rollout = torch.bmm(attn, rollout)

        token_importance = rollout[:, 0, 1:]

        patch_grid = token_importance.reshape(
            token_importance.shape[0],
            n_input_chans,
            n_time_patches,
        )

        channel_importance = patch_grid.mean(dim=2)
        time_patch_importance = patch_grid.mean(dim=1)

        return {
            "rollout": rollout.cpu().numpy(),
            "token_importance": token_importance.cpu().numpy(),
            "channel_importance_128": channel_importance.cpu().numpy(),
            "time_patch_importance": time_patch_importance.cpu().numpy(),
        }


def select_representative_test_samples(
    y: np.ndarray,
    test_idx: np.ndarray,
    n_samples: int = N_SAMPLES,
) -> np.ndarray:
    """Select up to n_samples test examples, preferring distinct classes."""
    selected = []
    seen_classes = set()

    for idx in test_idx:
        cls = int(y[idx])
        if cls not in seen_classes:
            selected.append(idx)
            seen_classes.add(cls)
        if len(selected) >= n_samples:
            break

    if len(selected) < n_samples:
        for idx in test_idx:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= n_samples:
                break

    return np.array(selected, dtype=int)


def top_k_channels(
    channel_scores: np.ndarray,
    k: int = TOP_K_CHANNELS,
) -> list[list[tuple[str, float]]]:
    """Return top-k canonical LaBraM channel names and scores per sample."""
    all_top = []

    for sample_scores in channel_scores:
        order = np.argsort(sample_scores)[::-1][:k]
        all_top.append(
            [(LABRAM_CHANNEL_ORDER[i], float(sample_scores[i])) for i in order]
        )

    return all_top


def save_plots(
    channel_scores: np.ndarray,
    time_scores: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: np.ndarray,
    sample_indices: np.ndarray,
) -> list[str]:
    """Save bar plots for top channels and temporal patch importance."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for i in range(channel_scores.shape[0]):
        top = top_k_channels(channel_scores[i : i + 1], k=TOP_K_CHANNELS)[0]
        names = [name for name, _ in top][::-1]
        scores = [score for _, score in top][::-1]

        label_name = str(class_names[labels[i]])
        pred_name = str(class_names[preds[i]])

        fig = plt.figure(figsize=(8, 6))
        plt.barh(names, scores)
        plt.xlabel("Attention rollout importance")
        plt.ylabel("Canonical LaBraM channel")
        plt.title(
            f"Sample {sample_indices[i]} | true={label_name} | pred={pred_name}"
        )
        plt.tight_layout()

        out_path = PLOT_DIR / f"sample_{i}_top_channels.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved_paths.append(str(out_path))

        fig = plt.figure(figsize=(5, 4))
        plt.bar(["patch_0", "patch_1"], time_scores[i])
        plt.xlabel("Temporal patch")
        plt.ylabel("Mean attention rollout importance")
        plt.title(f"Sample {sample_indices[i]} | temporal patch importance")
        plt.tight_layout()

        out_path = PLOT_DIR / f"sample_{i}_time_patches.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved_paths.append(str(out_path))

    return saved_paths


def write_report(
    sample_indices: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: np.ndarray,
    channel_scores: np.ndarray,
    time_scores: np.ndarray,
    plot_paths: list[str],
) -> None:
    """Write a compact Markdown report for Phase 3."""
    top_channels = top_k_channels(channel_scores, k=TOP_K_CHANNELS)

    lines = []
    lines.append("# Phase 3 — Attention Rollout Explainability")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("Attention rollout was computed across all 12 LaBraM transformer blocks.")
    lines.append(
        "For each block, attention weights were averaged across heads, combined "
        "with an identity residual connection, row-normalized, and multiplied "
        "across layers."
    )
    lines.append("")
    lines.append("The CLS-token rollout was summarized into:")
    lines.append("")
    lines.append("- 256 patch-token importance scores")
    lines.append("- 128 canonical LaBraM channel importance scores")
    lines.append("- 2 temporal patch importance scores")
    lines.append("")
    lines.append("## Samples")
    lines.append("")

    for i, idx in enumerate(sample_indices):
        true_name = str(class_names[labels[i]])
        pred_name = str(class_names[preds[i]])

        lines.append(f"### Sample {i} — dataset index {int(idx)}")
        lines.append("")
        lines.append(f"- True label: `{true_name}`")
        lines.append(f"- Predicted label: `{pred_name}`")
        lines.append("")
        lines.append("Top canonical LaBraM channels:")
        lines.append("")
        lines.append("| Rank | Channel | Importance |")
        lines.append("|---:|---|---:|")

        for rank, (name, score) in enumerate(top_channels[i], start=1):
            lines.append(f"| {rank} | {name} | {score:.6f} |")

        lines.append("")
        lines.append("Temporal patch importance:")
        lines.append("")
        lines.append("| Patch | Importance |")
        lines.append("|---:|---:|")

        for patch_idx, score in enumerate(time_scores[i]):
            lines.append(f"| {patch_idx} | {float(score):.6f} |")

        lines.append("")

    lines.append("## Saved Plot Files")
    lines.append("")

    for path in plot_paths:
        lines.append(f"- `{path}`")

    lines.append("")
    lines.append("## Scope Note")
    lines.append("")
    lines.append(
        "This phase uses attention rollout only. SHAP, LIME, and perturbation-based "
        "explainers are intentionally out of scope."
    )
    lines.append("")

    Path(REPORT_PATH).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    X, y_raw, subject_ids = load_dataset()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    n_outputs = len(le.classes_)
    n_times = X.shape[-1]
    n_chans = X.shape[1]

    _, _, test_idx = subject_wise_split(X, y, subject_ids)

    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Run Phase 1 first: python -m src.train"
        )

    model = load_labram_model(
        n_outputs=n_outputs,
        n_times=n_times,
        n_chans=n_chans,
        sfreq=160.0,
    ).to(DEVICE)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    sample_indices = select_representative_test_samples(
        y=y,
        test_idx=test_idx,
        n_samples=N_SAMPLES,
    )

    xb = torch.tensor(X[sample_indices], dtype=torch.float32)
    labels = y[sample_indices]

    with torch.no_grad():
        logits = model(xb.to(DEVICE))
        preds = logits.argmax(dim=1).cpu().numpy()

    results = attention_rollout(model, xb)

    np.savez_compressed(
        RESULTS_PATH,
        rollout=results["rollout"],
        token_importance=results["token_importance"],
        channel_importance_128=results["channel_importance_128"],
        time_patch_importance=results["time_patch_importance"],
        sample_indices=sample_indices,
        labels=labels,
        predictions=preds,
        class_names=np.array([str(c) for c in le.classes_]),
        labram_channel_order=np.array(LABRAM_CHANNEL_ORDER),
    )

    plot_paths = save_plots(
        channel_scores=results["channel_importance_128"],
        time_scores=results["time_patch_importance"],
        labels=labels,
        preds=preds,
        class_names=le.classes_,
        sample_indices=sample_indices,
    )

    write_report(
        sample_indices=sample_indices,
        labels=labels,
        preds=preds,
        class_names=le.classes_,
        channel_scores=results["channel_importance_128"],
        time_scores=results["time_patch_importance"],
        plot_paths=plot_paths,
    )

    print("=" * 60)
    print("ATTENTION ROLLOUT REPORT")
    print(f"Device                : {DEVICE}")
    print(f"Samples               : {len(sample_indices)}")
    print(f"Checkpoint            : {CHECKPOINT_PATH}")
    print(f"Results saved         : {RESULTS_PATH}")
    print(f"Markdown report saved : {REPORT_PATH}")
    print(f"Plots saved in        : {PLOT_DIR}")
    print("-" * 60)

    top_channels = top_k_channels(results["channel_importance_128"], k=5)

    for i, idx in enumerate(sample_indices):
        true_name = str(le.classes_[labels[i]])
        pred_name = str(le.classes_[preds[i]])

        print(f"Sample {i} | index={int(idx)} | true={true_name} | pred={pred_name}")
        print("  Top-5 channels:", top_channels[i])
        print("  Time patches:", results["time_patch_importance"][i])

    print("=" * 60)


if __name__ == "__main__":
    main()
