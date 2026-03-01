# White Blood Cell Classifier

A PyTorch image classifier that identifies 5 white blood cell types from microscopy images using a fine-tuned DenseNet-121 backbone.

## Classes

| Label | Cell Type |
|-------|-----------|
| 0 | Eosinophil |
| 1 | Lymphocyte |
| 2 | Monocyte |
| 3 | Neutrophil |
| 4 | Basophil |

## Architecture

- **Backbone**: DenseNet-121 (pretrained on ImageNet via `timm`)
- **Head**: `Linear(1024→512) → ReLU → Dropout(0.5) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→5)`
- **Input size**: 512×512

## Training Strategy

| Technique | Detail |
|-----------|--------|
| Differential LR | Backbone: 1e-5 · Head: 1e-3 |
| Scheduler | Linear warmup (5 epochs) → CosineAnnealing |
| Class imbalance | WeightedRandomSampler + per-class loss weights |
| Mixed precision | `torch.amp.autocast` + `GradScaler` |
| Gradient accumulation | 8 steps (effective batch = 32) |
| Augmentation | HFlip, VFlip, Rotation(15°), AdjustSharpness |
| Label smoothing | 0.1 |

## Setup

```bash
git clone https://github.com/your-username/wbc-classifier.git
cd wbc-classifier
pip install -r requirements.txt
```

You will also need a Kaggle API key to download the dataset. Set up `~/.kaggle/kaggle.json` or export `KAGGLE_USERNAME` and `KAGGLE_KEY` before running.

## Usage

### Train

```bash
python train.py
```

Checkpoints are saved to `checkpoints/`. Edit `config.py` to change any hyperparameter.

### Evaluate

```bash
# Evaluate best checkpoint on Test-A (default)
python evaluate.py --checkpoint checkpoints/best.pth

# Evaluate on Test-B and save confusion matrix
python evaluate.py --checkpoint checkpoints/best.pth --test-dir Test-B --save-confmat results/confmat.png
```

## Project Structure

```
wbc-classifier/
├── config.py        # All hyperparameters and paths
├── dataset.py       # Transforms, samplers, DataLoader factory
├── model.py         # Backbone + classifier head construction
├── train.py         # Training loop entry point
├── evaluate.py      # Standalone evaluation with confusion matrix
├── utils.py         # Device resolution, checkpointing, logging
├── requirements.txt
└── .gitignore
```

## Dataset

[White Blood Cells Dataset](https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset) by Masoud Nickparvar on Kaggle.
Downloaded automatically via `kagglehub` on first run.
