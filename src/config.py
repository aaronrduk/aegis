import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

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

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (255, 128, 0),
    3: (128, 128, 128),
    4: (255, 255, 0),
    5: (128, 64, 0),
    6: (0, 0, 255),
    7: (255, 0, 255),
    8: (0, 255, 255),
    9: (0, 255, 0),
}

TRAINING_CONFIG = {
    "num_classes": 10,
    "input_size": (512, 512),
    "batch_size": 4,
    "num_epochs": 150,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "encoder": "efficientnet-b4",
    "encoder_weights": "imagenet",
    "activation": None,
    "loss_weights": {"focal_weight": 0.4, "dice_weight": 0.6},
    "class_weights": [
        0.3,   # Background
        2.5,   # Building_RCC
        2.5,   # Building_Tiled
        2.5,   # Building_Tin
        2.5,   # Building_Other
        1.5,   # Road
        2.0,   # Waterbody
        4.0,   # Transformer
        4.0,   # Tank
        4.0,   # Well
    ],
    "scheduler": "cosine",
    "min_lr": 1e-7,
    "warmup_epochs": 5,
    "patience": 20,
    "min_delta": 0.001,
    "save_best_only": False,
    "save_frequency": 10,
    "accumulation_steps": 2,
    "gradient_clip_max_norm": 1.0,
}

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
        "use_mixup": True,
    },
    "val": {},
}

INFERENCE_CONFIG = {
    "use_tta": True,
    "tta_transforms": ["original", "hflip", "vflip"],
    "sliding_window": True,
    "window_size": 512,
    "overlap": 128,
    "batch_size": 4,
}

POSTPROCESS_CONFIG = {
    "min_area": {
        "building": 50,
        "road": 100,
        "waterbody": 200,
        "infrastructure": 20,
    },
    "simplify_tolerance": 1.0,
    "morphology": {"opening_kernel": 3, "closing_kernel": 3},
    "smooth_iterations": 2,
}

METRICS = ["iou", "f1", "precision", "recall", "accuracy"]

DEVICE = "cpu"  # auto-detected in code
MIXED_PRECISION = True
