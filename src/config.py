"""
Configuration file for SVAMITVA Feature Extraction
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

# Class definitions
CLASS_NAMES = [
    "Background",
    "Building_RCC",
    "Building_Tiled",
    "Building_Tin",
    "Building_Other",
    "Road",
    "Waterbody",
    "Transformer",
    "Tank",
    "Well",
]

# Color mapping for visualization (RGB)
CLASS_COLORS = {
    0: (0, 0, 0),  # Background - Black
    1: (255, 0, 0),  # Building_RCC - Red
    2: (255, 128, 0),  # Building_Tiled - Orange
    3: (128, 128, 128),  # Building_Tin - Gray
    4: (255, 255, 0),  # Building_Other - Yellow
    5: (128, 64, 0),  # Road - Brown
    6: (0, 0, 255),  # Waterbody - Blue
    7: (255, 0, 255),  # Transformer - Magenta
    8: (0, 255, 255),  # Tank - Cyan
    9: (0, 255, 0),  # Well - Green
}

# Training hyperparameters
TRAINING_CONFIG = {
    "num_classes": 10,
    "input_size": (512, 512),
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "num_workers": 4,
    # Model architecture
    "encoder": "resnet50",
    "encoder_weights": "imagenet",
    "activation": None,  # No activation for logits
    # Loss function
    "loss_weights": {"ce_weight": 0.5, "dice_weight": 0.5},
    # Class weights (for imbalanced data)
    "class_weights": [
        0.5,  # Background
        2.0,  # Building_RCC
        2.0,  # Building_Tiled
        2.0,  # Building_Tin
        2.0,  # Building_Other
        1.5,  # Road
        1.5,  # Waterbody
        3.0,  # Transformer
        3.0,  # Tank
        3.0,  # Well
    ],
    # Learning rate scheduler
    "scheduler": "cosine",
    "min_lr": 1e-6,
    # Early stopping
    "patience": 15,
    "min_delta": 0.001,
    # Checkpoint saving
    "save_best_only": False,
    "save_frequency": 5,
}

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    "train": {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.5,
        "rotate_limit": 45,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "scale_limit": 0.2,
        "shift_limit": 0.1,
        "elastic_transform": True,
        "grid_distortion": True,
        "optical_distortion": True,
    },
    "val": {
        # No augmentation for validation
    },
}

# Inference parameters
INFERENCE_CONFIG = {
    "use_tta": True,  # Test-time augmentation
    "tta_transforms": ["original", "hflip", "vflip", "rotate90"],
    "sliding_window": True,
    "window_size": 512,
    "overlap": 128,
    "batch_size": 4,
}

# Post-processing parameters
POSTPROCESS_CONFIG = {
    "min_area": {
        "building": 50,  # pixels²
        "road": 100,
        "waterbody": 200,
        "infrastructure": 20,
    },
    "simplify_tolerance": 1.0,  # Douglas-Peucker tolerance
    "morphology": {"opening_kernel": 3, "closing_kernel": 3},
    "smooth_iterations": 2,
}

# Evaluation metrics
METRICS = ["iou", "f1", "precision", "recall", "accuracy"]

# Device configuration
DEVICE = "cuda"  # Will auto-detect in code
MIXED_PRECISION = True  # Use automatic mixed precision
