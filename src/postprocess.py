import cv2
import numpy as np


def morphological_operations(mask: np.ndarray, opening_kernel: int = 3, closing_kernel: int = 3) -> np.ndarray:
    """Apply morphological opening and closing to clean up mask."""
    if opening_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if closing_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than min_size pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255
    return output


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in binary mask."""
    mask_inv = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(mask_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(mask, [contour], 0, 255, -1)
    return mask


def smooth_boundaries(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Smooth mask boundaries with Gaussian blur and re-threshold."""
    for _ in range(iterations):
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def postprocess_mask(mask: np.ndarray, class_idx: int, min_area: int = 50,
                     simplify: bool = True, fill: bool = True, smooth: bool = True) -> np.ndarray:
    """Post-processing pipeline for a single class binary mask."""
    binary_mask = (mask == class_idx).astype(np.uint8) * 255
    if simplify:
        binary_mask = morphological_operations(binary_mask, 3, 3)
    if fill:
        binary_mask = fill_holes(binary_mask)
    if min_area > 0:
        binary_mask = remove_small_objects(binary_mask, min_area)
    if smooth:
        binary_mask = smooth_boundaries(binary_mask, iterations=2)
    return binary_mask


def postprocess_multiclass_mask(mask: np.ndarray, min_areas: dict, num_classes: int) -> np.ndarray:
    """Post-process multi-class segmentation mask."""
    output_mask = np.zeros_like(mask)
    for class_idx in range(1, num_classes):
        if 1 <= class_idx <= 4:
            min_area = min_areas.get("building", 50)
        elif class_idx == 5:
            min_area = min_areas.get("road", 100)
        elif class_idx == 6:
            min_area = min_areas.get("waterbody", 200)
        else:
            min_area = min_areas.get("infrastructure", 20)

        processed = postprocess_mask(mask, class_idx, min_area=min_area, simplify=True, fill=True, smooth=True)
        output_mask[processed > 0] = class_idx

    return output_mask
