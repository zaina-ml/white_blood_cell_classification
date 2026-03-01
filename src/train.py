"""
train.py
Main training entry point.

Usage:
    python train.py

All hyperparameters are in config.py.
Dataset is downloaded automatically via kagglehub if not already cached.
"""

import torch
from torch import nn, optim
import torchmetrics
from torchmetrics.classification import (
    MulticlassPrecision, MulticlassRecall, MulticlassF1Score,
)
from torchmetrics.wrappers import ClasswiseWrapper
import kagglehub

from config import cfg
from dataset import build_datasets, build_dataloaders, build_gpu_transforms
from model import build_model
from utils import (
    resolve_device, save_checkpoint,
    print_epoch_header, print_per_class_metrics, print_model_summary,
)


def build_optimizer(model: nn.Module):
    return optim.AdamW([
        {"params": model.features.parameters(),    "lr": cfg.backbone_lr},
        {"params": model.classifier.parameters(),  "lr": cfg.classifier_lr},
    ])


def build_scheduler(optimizer):
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=cfg.warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.warmup_epochs,
        eta_min=1e-7,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.warmup_epochs],
    )


def build_metrics(n_classes, classes, device):

    kwargs = dict(num_classes=n_classes, average="none")
    precision = ClasswiseWrapper(MulticlassPrecision(**kwargs), labels=classes).to(device)
    recall    = ClasswiseWrapper(MulticlassRecall(**kwargs),    labels=classes).to(device)
    f1        = ClasswiseWrapper(MulticlassF1Score(**kwargs),   labels=classes).to(device)
    return precision, recall, f1


def train(model, loader, criterion, optimizer, scaler,train_transform, device):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        X = train_transform(X)

        with torch.amp.autocast(device_type=device.type):
            logits = model(X)
            loss   = criterion(logits, y) / cfg.accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % cfg.accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * cfg.accum_steps

    return running_loss / len(loader)


@torch.inference_mode()
def evaluate(model, loader, criterion, eval_transform,
             precision, recall, f1, device) -> float:
    model.eval()
    running_loss = 0.0

    precision.reset()
    recall.reset()
    f1.reset()

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X = eval_transform(X)

        logits = model(X)
        loss   = criterion(logits, y)
        preds  = torch.argmax(logits, dim=1)

        precision.update(preds, y)
        recall.update(preds, y)
        f1.update(preds, y)
        running_loss += loss.item()

    return running_loss / len(loader)

def main() -> None:
    print(f"Using device: {cfg.device}")

    print("Downloading dataset")
    data_root = kagglehub.dataset_download(cfg.dataset_id)
    train_dataset, test_dataset = build_datasets(data_root)
    train_loader, test_loader   = build_dataloaders(train_dataset, test_dataset)

    classes   = train_dataset.classes
    n_classes = len(classes)
    print(f"Classes ({n_classes}): {classes}")

    train_transform, eval_transform = build_gpu_transforms(device)

    model = build_model(device)
    print_model_summary(model)

    class_weights = torch.tensor(cfg.class_loss_weights, dtype=torch.float).to(device)
    criterion     = nn.CrossEntropyLoss(
        label_smoothing=cfg.label_smoothing,
        weight=class_weights,
    )
    optimizer  = build_optimizer(model)
    scheduler  = build_scheduler(optimizer)
    scaler     = torch.amp.GradScaler(device.type)

    precision, recall, f1 = build_metrics(n_classes, classes, device)

    best_f1 = 0.0

    for epoch in range(cfg.epochs):
        if device.type == "cuda":
            torch.cuda.empty_cache()

        train_loss = train(
            model, train_loader, criterion, optimizer,
            scaler, train_transform, device,
        )
        scheduler.step()

        test_loss = evaluate(
            model, test_loader, criterion, eval_transform,
            precision, recall, f1, device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print_epoch_header(epoch, cfg.epochs, train_loss, test_loss, current_lr)

        p_dict  = precision.compute()
        r_dict  = recall.compute()
        f1_dict = f1.compute()
        print_per_class_metrics(classes, p_dict, r_dict, f1_dict)

        mean_f1 = sum(f1_dict.values()) / len(f1_dict)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            save_checkpoint(model, cfg.checkpoint_path)
            print(f"  ✓ New best mean F1: {best_f1:.4f}")

    save_checkpoint(model, cfg.final_model_path)
    print(f"\nTraining complete. Best mean F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()