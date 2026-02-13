# SVAMITVA Feature Extraction System

## Overview
AI-powered feature extraction from drone imagery for the SVAMITVA Scheme. Uses DeepLabV3+ architecture with EfficientNet-B4 backbone for multi-class semantic segmentation of drone images, extracting building footprints (RCC, Tiled, Tin, Others), roads, waterbodies, and infrastructure (transformers, tanks, wells). Targets 95%+ accuracy.

## Tech Stack
- **Language**: Python 3.11
- **Framework**: Streamlit (web UI on port 5000)
- **ML**: PyTorch, segmentation-models-pytorch (DeepLabV3+ with EfficientNet-B4)
- **Loss**: Focal + Dice combined loss for class imbalance handling
- **Visualization**: Plotly, Matplotlib, Seaborn

## Project Structure
- `app.py` - Main Streamlit application
- `src/` - Core source code
  - `config.py` - Configuration, class definitions, training hyperparameters
  - `model.py` - DeepLabV3+ model with FocalDiceLoss
  - `inference.py` - Inference with TTA and sliding window
  - `train.py` - Training pipeline with warmup, gradient clipping, accumulation
  - `dataset.py` - Dataset with strong augmentation pipeline
  - `metrics.py` - IoU, Dice, F1, precision, recall metrics
  - `postprocess.py` - Morphological post-processing
  - `vectorize.py` - Mask to shapefile conversion (optional geospatial libs)
  - `utils.py` - Device detection, logging, area/object counting

## Key Design Decisions
- EfficientNet-B4 encoder for better accuracy vs ResNet50
- Focal+Dice loss (0.4/0.6 weights) handles class imbalance
- Heavy augmentation: CLAHE, CoarseDropout, elastic transforms, color jitter
- Gradient accumulation (2 steps) to simulate larger batch size
- LR warmup (5 epochs) + cosine annealing schedule
- TTA (horizontal + vertical flip) at inference

## Running
- Streamlit runs on port 5000, bound to 0.0.0.0
- Configuration in `.streamlit/config.toml`
- Training: `python -m src.train --train_images data/train/images --train_masks data/train/masks --val_images data/val/images --val_masks data/val/masks`

## Recent Changes
- 2026-02-13: Major overhaul - upgraded to EfficientNet-B4, Focal+Dice loss, stronger augmentation, gradient accumulation, LR warmup, cleaned up codebase, removed unnecessary files
