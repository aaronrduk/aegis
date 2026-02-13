"""
Auto-labeling script for drone imagery using color/texture heuristics.

This was our quick-and-dirty solution for bootstrapping training data
when we didn't have any manually labeled masks yet. It uses HSV color
thresholds to roughly classify pixels into building types, roads, etc.

It's NOT perfect — the masks are noisy and need manual cleanup — but it
gave us enough data to start training the neural network, which then
produces much better results.

We spent a whole night tweaking these HSV ranges with trial and error lol.

Team SVAMITVA - SIH Hackathon 2026
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


def classify_pixel_hsv(hsv: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Classify each pixel based on HSV color ranges.
    
    These thresholds were tuned on our specific drone imagery — they might
    not work well for other datasets with different lighting/cameras.
    """
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = np.zeros(h.shape, dtype=np.uint8)

    # roads tend to be grayish with low saturation
    road_mask = (s < 40) & (v > 80) & (v < 200) & (gray > 100) & (gray < 200)

    # tiled roofs have a distinctive orange-red hue
    tiled_roof = (
        ((h >= 5) & (h <= 25) & (s > 80) & (v > 100)) |
        ((h >= 0) & (h <= 10) & (s > 100) & (v > 120))
    )

    # RCC roofs are usually dark gray/concrete colored
    dark_roof = (s < 50) & (v < 100) & (v > 30)
    # other buildings — light gray roofs
    gray_roof = (s < 30) & (v >= 100) & (v < 160)

    # vegetation and bare ground — we classify these as background
    green_veg = (h >= 30) & (h <= 85) & (s > 30) & (v > 40)
    brown_ground = (h >= 15) & (h <= 35) & (s > 20) & (s < 80) & (v > 60) & (v < 180)

    # order matters here — later assignments override earlier ones
    mask[green_veg | brown_ground] = 0       # background
    mask[road_mask & ~tiled_roof & ~dark_roof] = 5  # road
    mask[tiled_roof] = 2                     # tiled roof
    mask[dark_roof & ~road_mask] = 1         # RCC roof
    mask[gray_roof & ~road_mask & ~green_veg] = 4   # other building

    return mask


def refine_mask(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    """Clean up the noisy pixel-level classification with morphological ops.
    
    Without this step, the masks are basically unusable — too many isolated
    pixels and fragmented regions.
    """
    refined = mask.copy()

    # clean up building classes
    for class_id in [1, 2, 3, 4]:
        binary = (mask == class_id).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(binary)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                cv2.drawContours(clean, [cnt], -1, 1, -1)
        refined[mask == class_id] = 0
        refined[clean == 1] = class_id

    # roads need a bigger kernel — they're long and thin
    road_binary = (mask == 5).astype(np.uint8)
    road_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    road_binary = cv2.morphologyEx(road_binary, cv2.MORPH_CLOSE, road_kernel, iterations=3)
    road_binary = cv2.morphologyEx(road_binary, cv2.MORPH_OPEN, road_kernel, iterations=1)
    contours, _ = cv2.findContours(road_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_road = np.zeros_like(road_binary)
    for cnt in contours:
        # 2000 pixel minimum for roads — smaller than that is probably noise
        if cv2.contourArea(cnt) > 2000:
            cv2.drawContours(clean_road, [cnt], -1, 1, -1)
    refined[mask == 5] = 0
    refined[clean_road == 1] = 5

    return refined


def generate_mask(image_path: str, output_path: str) -> np.ndarray:
    """Generate a segmentation mask for a single drone image.
    
    Also does convex hull approximation on buildings to make them look
    more like actual building footprints instead of blobby shapes.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read: {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = classify_pixel_hsv(hsv, gray)
    mask = refine_mask(mask, min_area=300)

    # use edge detection to help define building boundaries
    edges = cv2.Canny(gray, 50, 150)
    edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # convex hull for building contours — makes them look more realistic
    building_classes = [1, 2, 3, 4]
    building_mask = np.isin(mask, building_classes)
    edge_near_building = edge_dilated & building_mask.astype(np.uint8)

    for cls in building_classes:
        cls_mask = (mask == cls).astype(np.uint8)
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(mask, [hull], -1, int(cls), -1)

    cv2.imwrite(str(output_path), mask)
    return mask


def auto_label_directory(image_dir: str, mask_dir: str):
    """Process an entire directory of images and generate masks.
    
    Prints out class distribution stats at the end so we can see
    how balanced (or unbalanced) our auto-generated labels are.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"]
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
    image_files = sorted(image_files)

    print(f"Found {len(image_files)} images to label")

    class_pixel_counts = np.zeros(10, dtype=np.int64)
    for img_path in image_files:
        mask_path = mask_dir / (img_path.stem + ".png")
        print(f"  Labeling: {img_path.name} -> {mask_path.name}")
        mask = generate_mask(str(img_path), str(mask_path))
        for c in range(10):
            class_pixel_counts[c] += np.sum(mask == c)

    class_names = [
        "Background", "Building_RCC", "Building_Tiled", "Building_Tin",
        "Building_Other", "Road", "Waterbody", "Transformer", "Tank", "Well"
    ]
    total = class_pixel_counts.sum()
    print("\nClass distribution:")
    for i, name in enumerate(class_names):
        pct = class_pixel_counts[i] / total * 100 if total > 0 else 0
        print(f"  {name}: {class_pixel_counts[i]:,} pixels ({pct:.1f}%)")


if __name__ == "__main__":
    import sys
    img_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train/images"
    msk_dir = sys.argv[2] if len(sys.argv) > 2 else "data/train/masks"
    auto_label_directory(img_dir, msk_dir)
