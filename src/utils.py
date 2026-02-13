"""
Utility functions used across the SVAMITVA project.

Logger setup, device detection, GeoTIFF loading, area/object counting,
and the AverageMeter class for tracking training metrics.

Nothing fancy here, just helper stuff we kept needing in multiple files
so we pulled it into one place.

Digital University Kerala (DUK)
"""

import cv2
import numpy as np
import torch
import logging
from typing import Tuple, Optional, Dict

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None


def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Setup a logger with console output and optional file logging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_device() -> torch.device:
    """Auto-detect the best available device.
    
    Priority: CUDA GPU > Apple MPS > CPU
    We always end up running on CPU during the hackathon anyway lol
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def visualize_prediction(image: np.ndarray, mask: np.ndarray, prediction: np.ndarray,
                         class_colors: Dict[int, Tuple[int, int, int]], alpha: float = 0.5) -> np.ndarray:
    """Overlay prediction on the original image — useful for visual debugging."""
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    h, w = prediction.shape
    pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in class_colors.items():
        pred_colored[prediction == class_idx] = color

    # alpha blend: lower alpha = more transparent overlay
    return cv2.addWeighted(image, 1 - alpha, pred_colored, alpha, 0)


def load_geotiff(file_path: str) -> Tuple[np.ndarray, dict]:
    """Load a GeoTIFF file and return the image array + geospatial metadata.
    
    The metadata (transform, CRS, bounds) is needed later when we
    create shapefiles so the polygons have real-world coordinates.
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio not installed. Cannot load GeoTIFF.")
    with rasterio.open(file_path) as src:
        array = src.read()
        metadata = {
            "transform": src.transform,
            "crs": src.crs,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
        }
    return array, metadata


def calculate_area(mask: np.ndarray, class_idx: int, pixel_size: float = 1.0) -> float:
    """Calculate the area of a specific class in square meters.
    
    pixel_size is the ground sampling distance (GSD) — depends on drone altitude.
    For our test flights at 60m altitude it was about 0.03m/pixel.
    """
    return float(np.sum(mask == class_idx) * pixel_size)


def count_objects(mask: np.ndarray, class_idx: int) -> int:
    """Count the number of separate objects of a given class using connected components.
    
    We subtract 1 because connectedComponents counts the background as a label.
    """
    binary_mask = (mask == class_idx).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(binary_mask)
    return num_labels - 1


class AverageMeter:
    """Keeps a running average — we use this to track loss during training.
    
    Borrowed this pattern from PyTorch examples, it's super handy.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
