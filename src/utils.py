"""
utils.py
Shared utilities: device resolution, checkpoint save/load,
per-epoch metric printing, and model summary.
"""

import pathlib
from typing import Optional

import torch
from torch import nn

from config import cfg

def save_checkpoint(model, path):
    dest = pathlib.Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dest)
    print(f"Checkpoint saved → {dest}")


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Checkpoint loaded: {path}")
    return model

def print_epoch_header(epoch, epochs, train_loss, test_loss, lr):
    print(
        f"\nEpoch [{epoch + 1:>3}/{epochs}]  "
        f"train_loss={train_loss:.4f}  "
        f"test_loss={test_loss:.4f}  "
        f"lr={lr:.2e}"
    )


def print_per_class_metrics(classes, p_dict, r_dict, f1_dict):
    header = f"{'Class':<15} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}"
    print(header)
    print("─" * len(header))
    for cls in classes:
        p   = p_dict[f"multiclassprecision_{cls}"].item()
        r   = r_dict[f"multiclassrecall_{cls}"].item()
        f1v = f1_dict[f"multiclassf1score_{cls}"].item()
        print(f"{cls:<15} | {p:>10.4f} | {r:>10.4f} | {f1v:>10.4f}")


def print_model_summary(model):
    try:
        import torchinfo
        torchinfo.summary(model)
    except ImportError:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters — total: {total:,}  trainable: {trainable:,}")