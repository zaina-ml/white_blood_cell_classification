"""
config.py
Central configuration for all hyperparameters, paths, and model settings.
Edit this file to change any training behaviour without touching training code.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    dataset_id:    str  = "masoudnickparvar/white-blood-cells-dataset"
    train_subdir:  str  = "Train"
    test_a_subdir: str  = "Test-A"
    test_b_subdir: str  = "Test-B"
    n_classes:     int  = 5

    class_loss_weights: list = field(
        default_factory=lambda: [0.5, 3.5, 1.3, 3.0, 0.6]
    )

    backbone:          str  = "densenet121"
    pretrained:        bool = True
    memory_efficient:  bool = True
    image_size:        int  = 512

    hidden_dims: list = field(default_factory=lambda: [512, 256])
    dropout_rates: list = field(default_factory=lambda: [0.5, 0.3])

    epochs:        int   = 40
    warmup_epochs: int   = 5
    batch_size:    int   = 4
    a_steps:   int   = 8

    backbone_lr:    float = 1e-5
    classifier_lr:  float = 1e-3

    label_smoothing: float = 0.1

    num_workers:      int  = 2
    prefetch_factor:  int  = 2
    pin_memory:       bool = True
    persistent_workers: bool = True

    norm_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    norm_std:  list = field(default_factory=lambda: [0.229, 0.224, 0.225])

    checkpoint_path: str = "checkpoints/best.pth"
    final_model_path: str = "checkpoints/model.pth"

    device: str = "cuda"


cfg = Config()