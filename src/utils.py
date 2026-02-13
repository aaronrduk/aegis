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
    """Setup logger with console and optional file handler."""
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
    """Get best available device (CUDA > MPS > CPU)."""
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
    """Visualize prediction overlay on image."""
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    h, w = prediction.shape
    pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in class_colors.items():
        pred_colored[prediction == class_idx] = color

    return cv2.addWeighted(image, 1 - alpha, pred_colored, alpha, 0)


def load_geotiff(file_path: str) -> Tuple[np.ndarray, dict]:
    """Load GeoTIFF and return array with metadata."""
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
    """Calculate area of a specific class in mask (in square meters)."""
    return float(np.sum(mask == class_idx) * pixel_size)


def count_objects(mask: np.ndarray, class_idx: int) -> int:
    """Count disconnected objects of a specific class."""
    binary_mask = (mask == class_idx).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(binary_mask)
    return num_labels - 1


class AverageMeter:
    """Computes and stores the running average."""

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
