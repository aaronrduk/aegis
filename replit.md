# SVAMITVA Feature Extraction System

## Overview
AI-powered feature extraction from drone imagery for the SVAMITVA Scheme. Uses DeepLabV3+ architecture with PyTorch for multi-class semantic segmentation of drone images, extracting building footprints, roads, waterbodies, and infrastructure.

## Tech Stack
- **Language**: Python 3.11
- **Framework**: Streamlit (web UI)
- **ML**: PyTorch, segmentation-models-pytorch
- **Visualization**: Plotly, Matplotlib, Seaborn

## Project Structure
- `app.py` - Main Streamlit application
- `src/` - Core source code
  - `config.py` - Configuration and class definitions
  - `inference.py` - Model inference logic
  - `model.py` - DeepLabV3+ model architecture
  - `postprocess.py` - Post-processing utilities
  - `vectorize.py` - Mask to shapefile conversion
  - `utils.py` - Utility functions
  - `train.py` - Training pipeline
  - `dataset.py` - Dataset handling
  - `metrics.py` - Evaluation metrics
- `checkpoints/` - Model checkpoints
- `data/` - Input data
- `outputs/` - Generated outputs
- `logs/` - Training logs

## Running
- Streamlit runs on port 5000, bound to 0.0.0.0
- Configuration in `.streamlit/config.toml`

## Recent Changes
- 2026-02-13: Initial Replit setup - installed dependencies, configured Streamlit for port 5000 with proxy support
