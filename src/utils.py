"""
Utility functions for SVAMITVA Feature Extraction
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import logging

# Optional imports
try:
    import rasterio
    from rasterio.transform import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None
    from_bounds = None


def setup_logger(
    name: str, log_file: Optional[str] = None, level=logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU)

    Returns:
        torch.device
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


def visualize_prediction(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    class_colors: Dict[int, Tuple[int, int, int]],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Visualize prediction overlay on image

    Args:
        image: Original image (H, W, 3)
        mask: Ground truth mask (H, W)
        prediction: Predicted mask (H, W)
        class_colors: Dictionary mapping class indices to RGB colors
        alpha: Overlay transparency

    Returns:
        Visualization image (H, W, 3)
    """
    # Denormalize image if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Create colored masks
    h, w = prediction.shape
    pred_colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in class_colors.items():
        pred_colored[prediction == class_idx] = color

    # Overlay prediction on image
    overlay = cv2.addWeighted(image, 1 - alpha, pred_colored, alpha, 0)

    return overlay


def create_color_legend(
    class_names: list,
    class_colors: Dict[int, Tuple[int, int, int]],
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create color legend image

    Args:
        class_names: List of class names
        class_colors: Dictionary mapping class indices to RGB colors
        save_path: Optional path to save legend

    Returns:
        Legend image
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, len(class_names) * 0.5))

    # Hide axes
    ax.axis("off")

    # Add color patches
    for idx, name in enumerate(class_names):
        color = np.array(class_colors[idx]) / 255.0
        rect = plt.Rectangle((0, idx), 1, 0.8, facecolor=color)
        ax.add_patch(rect)
        ax.text(1.2, idx + 0.4, name, va="center", fontsize=12)

    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, len(class_names))
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # Convert to image
    fig.canvas.draw()
    legend_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    legend_img = legend_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return legend_img


def save_geotiff(
    array: np.ndarray,
    output_path: str,
    transform: Optional[object] = None,
    crs: Optional[str] = None,
    nodata: Optional[int] = None,
):
    """
    Save numpy array as GeoTIFF

    Args:
        array: Array to save (H, W) or (C, H, W)
        output_path: Output file path
        transform: Affine transform
        crs: Coordinate reference system
        nodata: NoData value
    """
    if not HAS_RASTERIO:
        print("Warning: rasterio not installed. Cannot save GeoTIFF.")
        return

    # Ensure 3D array
    if array.ndim == 2:
        array = array[np.newaxis, :, :]

    count, height, width = array.shape

    # Default transform if not provided
    if transform is None and from_bounds is not None:
        transform = from_bounds(0, 0, width, height, width, height)

    # Write GeoTIFF
    if rasterio:
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=array.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
            compress="lzw",
        ) as dst:
            dst.write(array)


def load_geotiff(file_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load GeoTIFF and return array with metadata

    Args:
        file_path: Path to GeoTIFF file

    Returns:
        Tuple of (array, metadata dict)
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
    """
    Calculate area of a specific class in mask

    Args:
        mask: Segmentation mask (H, W)
        class_idx: Class index to calculate area for
        pixel_size: Size of one pixel in square meters

    Returns:
        Area in square meters
    """
    num_pixels = np.sum(mask == class_idx)
    area = num_pixels * pixel_size
    return area


def count_objects(mask: np.ndarray, class_idx: int) -> int:
    """
    Count number of disconnected objects of a specific class

    Args:
        mask: Segmentation mask (H, W)
        class_idx: Class index to count

    Returns:
        Number of objects
    """
    # Create binary mask for class
    binary_mask = (mask == class_idx).astype(np.uint8)

    # Find connected components
    num_labels, _ = cv2.connectedComponents(binary_mask)

    # Subtract 1 for background
    return num_labels - 1


class AverageMeter:
    """Computes and stores the average and current value"""

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



