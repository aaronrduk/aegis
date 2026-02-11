"""
Post-processing utilities for segmentation masks
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple


def morphological_operations(
    mask: np.ndarray, opening_kernel: int = 3, closing_kernel: int = 3
) -> np.ndarray:
    """
    Apply morphological operations to clean up mask

    Args:
        mask: Input binary mask
        opening_kernel: Kernel size for opening (remove noise)
        closing_kernel: Kernel size for closing (fill holes)

    Returns:
        Cleaned mask
    """
    # Opening (remove small noise)
    if opening_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Closing (fill small holes)
    if closing_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Remove small connected components

    Args:
        mask: Input binary mask
        min_size: Minimum object size in pixels

    Returns:
        Filtered mask
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # Create output mask
    output = np.zeros_like(mask)

    # Keep only components larger than min_size
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            output[labels == i] = 255

    return output


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in binary mask

    Args:
        mask: Input binary mask

    Returns:
        Mask with filled holes
    """
    # Invert mask
    mask_inv = cv2.bitwise_not(mask)

    # Find contours
    contours, _ = cv2.findContours(mask_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Fill contours
    for contour in contours:
        cv2.drawContours(mask, [contour], 0, 255, -1)

    return mask


def smooth_boundaries(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Smooth mask boundaries

    Args:
        mask: Input binary mask
        iterations: Number of smoothing iterations

    Returns:
        Smoothed mask
    """
    for _ in range(iterations):
        # Apply Gaussian blur
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Threshold back to binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


def postprocess_mask(
    mask: np.ndarray,
    class_idx: int,
    min_area: int = 50,
    simplify: bool = True,
    fill: bool = True,
    smooth: bool = True,
) -> np.ndarray:
    """
    Complete post-processing pipeline for a single class

    Args:
        mask: Full segmentation mask (H, W)
        class_idx: Class index to process
        min_area: Minimum object area in pixels
        simplify: Apply morphological operations
        fill: Fill holes
        smooth: Smooth boundaries

    Returns:
        Processed binary mask for the class
    """
    # Extract binary mask for class
    binary_mask = (mask == class_idx).astype(np.uint8) * 255

    # Apply morphological operations
    if simplify:
        binary_mask = morphological_operations(binary_mask, 3, 3)

    # Fill holes
    if fill:
        binary_mask = fill_holes(binary_mask)

    # Remove small objects
    if min_area > 0:
        binary_mask = remove_small_objects(binary_mask, min_area)

    # Smooth boundaries
    if smooth:
        binary_mask = smooth_boundaries(binary_mask, iterations=2)

    return binary_mask


def postprocess_multiclass_mask(
    mask: np.ndarray, min_areas: dict, num_classes: int
) -> np.ndarray:
    """
    Post-process multi-class segmentation mask

    Args:
        mask: Segmentation mask (H, W)
        min_areas: Dictionary mapping class groups to minimum areas
        num_classes: Number of classes

    Returns:
        Post-processed mask
    """
    output_mask = np.zeros_like(mask)

    for class_idx in range(1, num_classes):  # Skip background
        # Determine minimum area for this class
        if 1 <= class_idx <= 4:  # Buildings
            min_area = min_areas.get("building", 50)
        elif class_idx == 5:  # Roads
            min_area = min_areas.get("road", 100)
        elif class_idx == 6:  # Waterbodies
            min_area = min_areas.get("waterbody", 200)
        else:  # Infrastructure (7, 8, 9)
            min_area = min_areas.get("infrastructure", 20)

        # Process class
        processed_binary = postprocess_mask(
            mask, class_idx, min_area=min_area, simplify=True, fill=True, smooth=True
        )

        # Add to output (later classes overwrite earlier ones in overlaps)
        output_mask[processed_binary > 0] = class_idx

    return output_mask


if __name__ == "__main__":
    # Test post-processing
    from config import POSTPROCESS_CONFIG

    # Create dummy mask
    mask = np.random.randint(0, 10, size=(512, 512), dtype=np.uint8)

    # Post-process
    processed = postprocess_multiclass_mask(
        mask, min_areas=POSTPROCESS_CONFIG["min_area"], num_classes=10
    )

    print(f"Original unique classes: {np.unique(mask)}")
    print(f"Processed unique classes: {np.unique(processed)}")
