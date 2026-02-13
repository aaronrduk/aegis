"""
Configuration file for the SVAMITVA project.

All the hyperparameters, class definitions, and paths live here.
We kept tweaking these values during the hackathon — the current ones
are what gave us the best results after a LOT of experimentation.

Digital University Kerala (DUK)
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

# 10 classes total — we only have good training data for ~5 of them right now
# TODO: collect more labeled samples for Tin roofs, waterbodies, and infrastructure
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

# colors for visualization — picked these to be visually distinct
# red/orange shades for buildings, brown for roads, blue for water, etc.
CLASS_COLORS = {
    0: (0, 0, 0),         # background — black
    1: (255, 0, 0),       # RCC — red
    2: (255, 128, 0),     # Tiled — orange
    3: (128, 128, 128),   # Tin — gray
    4: (255, 255, 0),     # Other buildings — yellow
    5: (128, 64, 0),      # Road — brown
    6: (0, 0, 255),       # Water — blue
    7: (255, 0, 255),     # Transformer — magenta
    8: (0, 255, 255),     # Tank — cyan
    9: (0, 255, 0),       # Well — green
}

TRAINING_CONFIG = {
    "num_classes": 10,
    "input_size": (512, 512),
    "batch_size": 4,
    "num_epochs": 150,
    # we tried resnet50 first but efficientnet-b4 gave way better results
    # ~3% higher mIoU on our val set
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "encoder": "efficientnet-b4",
    "encoder_weights": "imagenet",
    "activation": None,
    # focal + dice combo handles class imbalance much better than plain CE
    "loss_weights": {"focal_weight": 0.4, "dice_weight": 0.6},
    # higher weights for rare classes so the model doesn't just predict background everywhere
    "class_weights": [
        0.3,   # Background — super common, downweight it
        2.5,   # Building_RCC
        2.5,   # Building_Tiled
        2.5,   # Building_Tin
        2.5,   # Building_Other
        1.5,   # Road
        2.0,   # Waterbody
        4.0,   # Transformer — very rare, needs high weight
        4.0,   # Tank
        4.0,   # Well
    ],
    "scheduler": "cosine",
    "min_lr": 1e-7,
    "warmup_epochs": 5,   # warmup helps a lot with pretrained encoders
    "patience": 20,
    "min_delta": 0.001,
    "save_best_only": False,
    "save_frequency": 10,
    "accumulation_steps": 2,  # simulates batch_size of 8 on limited VRAM
    "gradient_clip_max_norm": 1.0,
}

# CPU config for development/testing — smaller images, fewer epochs
# NOTE: training on CPU takes forever, only use this for debugging
TRAINING_CONFIG_CPU = {
    **TRAINING_CONFIG,
    "input_size": (256, 256),
    "batch_size": 2,
    "num_epochs": 30,
    "learning_rate": 1e-3,
    "num_workers": 0,   # multiprocessing on CPU causes issues sometimes
    "accumulation_steps": 1,
    "warmup_epochs": 3,
    "patience": 10,
    "save_frequency": 5,
}

# augmentation pipeline — these settings gave us good generalization
# without making the training too slow
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
        "use_mixup": True,  # TODO: actually implement mixup in the dataloader
    },
    "val": {},
}

INFERENCE_CONFIG = {
    "use_tta": True,  # test-time augmentation — flip + average for better predictions
    "tta_transforms": ["original", "hflip", "vflip"],
    "sliding_window": True,
    "window_size": 512,
    "overlap": 128,  # 25% overlap seemed to be the sweet spot
    "batch_size": 4,
}

# post-processing thresholds — tuned these by looking at outputs manually
POSTPROCESS_CONFIG = {
    "min_area": {
        "building": 50,
        "road": 100,
        "waterbody": 200,
        "infrastructure": 20,  # infra objects are tiny, keep threshold low
    },
    "simplify_tolerance": 1.0,
    "morphology": {"opening_kernel": 3, "closing_kernel": 3},
    "smooth_iterations": 2,
}

METRICS = ["iou", "f1", "precision", "recall", "accuracy"]

DEVICE = "cpu"  # auto-detected at runtime, this is just a fallback
MIXED_PRECISION = True
