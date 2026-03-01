"""
model.py
Model factory: loads a pretrained DenseNet-121 backbone via timm and
replaces the classifier head with a configurable MLP.
"""

import torch
from torch import nn
import timm

from config import cfg


def build_classifier_head(in_features: int):
    layers: list[nn.Module] = []
    prev_dim = in_features

    for hidden_dim, dropout_rate in zip(cfg.hidden_dims, cfg.dropout_rates):
        layers += [
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        ]
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, cfg.n_classes))
    return nn.Sequential(*layers)


def build_model(device: torch.device):
    model = timm.create_model(
        cfg.backbone,
        pretrained=cfg.pretrained,
        memory_efficient=cfg.memory_efficient,
    )

    in_features = model.classifier.in_features
    model.classifier = build_classifier_head(in_features)
    model.to(device)
    return model