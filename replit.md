# SVAMITVA Feature Extraction System

## Overview
AI model developed by Digital University Kerala (DUK) students for feature extraction from drone imagery for the SVAMITVA Scheme. Uses DeepLabV3+ architecture with EfficientNet-B4 backbone for multi-class semantic segmentation of drone images, extracting building footprints (RCC, Tiled, Tin, Others), roads, waterbodies, and infrastructure (transformers, tanks, wells). Targets 95%+ accuracy.

## Tech Stack
- **Language**: Python 3.11
- **Framework**: Streamlit (web UI on port 5000)
- **ML**: PyTorch, segmentation-models-pytorch (DeepLabV3+ with EfficientNet-B4)
- **Loss**: Focal + Dice combined loss for class imbalance handling
- **Visualization**: Plotly, Matplotlib, Seaborn

## Project Structure
- `app.py` - Main Streamlit application
- `src/` - Core source code
  - `config.py` - Configuration, class definitions, training hyperparameters (GPU + CPU profiles)
  - `model.py` - DeepLabV3+ model with FocalDiceLoss
  - `inference.py` - Inference with TTA and sliding window
  - `train.py` - Training pipeline with warmup, gradient clipping, accumulation
  - `dataset.py` - Dataset with strong augmentation pipeline
  - `metrics.py` - IoU, Dice, F1, precision, recall metrics
  - `postprocess.py` - Morphological post-processing
  - `vectorize.py` - Mask to shapefile conversion (optional geospatial libs)
  - `utils.py` - Device detection, logging, area/object counting
  - `auto_label.py` - Auto-generate segmentation masks from drone images using color/texture heuristics
- `data/` - Training and validation data
  - `train/images/` - 16 training drone images
  - `train/masks/` - Auto-generated training masks
  - `val/images/` - 4 validation drone images
  - `val/masks/` - Auto-generated validation masks

## Key Design Decisions
- EfficientNet-B4 encoder for better accuracy vs ResNet50
- Focal+Dice loss (0.4/0.6 weights) handles class imbalance
- Heavy augmentation: CLAHE, CoarseDropout, elastic transforms, color jitter
- Gradient accumulation (2 steps) to simulate larger batch size
- LR warmup (5 epochs) + cosine annealing schedule
- TTA (horizontal + vertical flip) at inference
- Two training configs: TRAINING_CONFIG (GPU, 512x512) and TRAINING_CONFIG_CPU (CPU, 256x256)

## Running
- Streamlit runs on port 5000, bound to 0.0.0.0
- Configuration in `.streamlit/config.toml`

### Auto-labeling (generate masks from images)
```
python src/auto_label.py data/train/images data/train/masks
```

### Training
- GPU: `python -m src.train`
- CPU (optimized): `python -m src.train --cpu`
- Resume: `python -m src.train --cpu --resume checkpoints/best_model.pth`

## Training Results
- Trained 14 epochs on CPU with 20 drone images (16 train / 4 val)
- Auto-generated masks detected: Background, Building_RCC, Building_Tiled, Building_Other, Road
- Best validation IoU: 0.381, Accuracy: 69.5%
- Note: For 95%+ accuracy, more manually annotated data and GPU training are recommended

## Documentation
- `SVAMITVA_Documentation.pdf` - Full project documentation (23 pages)
- `generate_docs.py` - Script to regenerate the PDF documentation

## Recent Changes
- 2026-02-13: Fixed green output bug (untrained class predictions), added confidence thresholding, class masking, per-class detection controls
- 2026-02-13: Humanized codebase with student-style comments, created PDF documentation
- 2026-02-13: Added auto-labeling script, trained model on 20 drone images, added CPU training config, fixed PyTorch 2.6 weights_only compatibility
- 2026-02-13: Major overhaul - upgraded to EfficientNet-B4, Focal+Dice loss, stronger augmentation, gradient accumulation, LR warmup, cleaned up codebase
