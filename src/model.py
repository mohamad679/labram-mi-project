"""
Phase 0 — Load the pretrained LaBraM model via braindecode.

License: BSD-3-Clause (braindecode/labram-pretrained on HuggingFace).
No access restrictions.
"""

from braindecode.models import Labram


def load_labram_model(n_outputs: int = 2, **kwargs):
    """
    Load LaBraM and prepare it for fine-tuning on a classification task.

    Args:
        n_outputs: number of classes (2 for binary motor imagery,
                   adjust based on the actual PhysionetMI label set).

    Returns:
        model: the LaBraM model instance, ready for fine-tuning.
    """
    model = Labram(n_outputs=n_outputs, **kwargs)

    # TODO (Phase 0 exit criteria):
    # Run a dummy forward pass with a small batch from data.py here.
    # This confirms input shape compatibility BEFORE any real training.
    return model


if __name__ == "__main__":
    import torch

    model = load_labram_model(n_outputs=2)
    print(model)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
