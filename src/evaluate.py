"""
evaluate.py
Standalone evaluation script — loads a saved checkpoint and runs a
full evaluation pass with per-class metrics and a confusion matrix.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pth
    python evaluate.py --checkpoint checkpoints/best.pth --test-dir Test-B
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.wrappers import ClasswiseWrapper
import kagglehub

from config import cfg
from dataset import build_datasets, build_dataloaders, build_gpu_transforms
from model import build_model
from utils import load_checkpoint, resolve_device, print_per_class_metrics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WBC classifier")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=cfg.checkpoint_path,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=cfg.test_a_subdir,
        choices=["Test-A", "Test-B"],
        help="Which test split to evaluate on",
    )
    parser.add_argument(
        "--save-confmat",
        type=str,
        default=None,
        help="If set, save confusion matrix figure to this path",
    )
    return parser.parse_args()


@torch.inference_mode()
def run_evaluation(model, loader, eval_transform,
                   n_classes: int, classes: list[str],
                   device: torch.device) -> None:

    kwargs = dict(num_classes=n_classes, average="none")
    precision = ClasswiseWrapper(MulticlassPrecision(**kwargs), labels=classes).to(device)
    recall    = ClasswiseWrapper(MulticlassRecall(**kwargs),    labels=classes).to(device)
    f1        = ClasswiseWrapper(MulticlassF1Score(**kwargs),   labels=classes).to(device)
    confmat   = MulticlassConfusionMatrix(num_classes=n_classes).to(device)

    model.eval()
    for X, y in loader:
        X, y  = X.to(device), y.to(device)
        X     = eval_transform(X)
        logits = model(X)
        preds  = torch.argmax(logits, dim=1)

        precision.update(preds, y)
        recall.update(preds, y)
        f1.update(preds, y)
        confmat.update(preds, y)

    p_dict  = precision.compute()
    r_dict  = recall.compute()
    f1_dict = f1.compute()

    mean_f1 = sum(f1_dict.values()) / len(f1_dict)

    print(f"\nResults on {len(loader.dataset)} samples")
    print(f"Mean F1: {mean_f1:.4f}\n")
    print_per_class_metrics(classes, p_dict, r_dict, f1_dict)

    return confmat


def main() -> None:
    args   = parse_args()
    device = resolve_device()
    print(f"Using device: {device}")

    cfg.test_a_subdir = args.test_dir

    print("Downloading / locating dataset...")
    data_root = kagglehub.dataset_download(cfg.dataset_id)
    train_dataset, test_dataset = build_datasets(data_root)
    _, test_loader = build_dataloaders(train_dataset, test_dataset)

    classes   = train_dataset.classes
    n_classes = len(classes)

    _, eval_transform = build_gpu_transforms(device)

    model = build_model(device)
    load_checkpoint(model, args.checkpoint, device)

    confmat = run_evaluation(
        model, test_loader, eval_transform,
        n_classes, classes, device,
    )

    fig, ax = confmat.plot(labels=classes)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    if args.save_confmat:
        dest = pathlib.Path(args.save_confmat)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=150)
        print(f"\nConfusion matrix saved → {dest}")
    else:
        plt.show()


if __name__ == "__main__":
    main()