"""
Visualization utilities for debugging and analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

from src.config import CLASS_NAMES, CLASS_COLORS
from src.utils import load_geotiff


def visualize_sample(image_path, mask_path=None, output_path=None):
    """
    Visualize a training sample (image + mask)

    Args:
        image_path: Path to image
        mask_path: Path to mask (optional)
        output_path: Path to save visualization
    """
    # Load image
    if Path(image_path).suffix.lower() in [".tif", ".tiff"]:
        image, _ = load_geotiff(image_path)
        image = image.transpose(1, 2, 0)
        if image.max() > 255:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(
                np.uint8
            )
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if mask_path is None:
        # Show only image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title(f"Image: {Path(image_path).name}")
        plt.axis("off")
    else:
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Create colored mask
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, color in CLASS_COLORS.items():
            colored_mask[mask == class_idx] = color

        # Create overlay
        overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(colored_mask)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        # Add legend
        unique_classes = np.unique(mask)
        legend_elements = []
        for class_idx in unique_classes:
            if class_idx < len(CLASS_NAMES):
                color = np.array(CLASS_COLORS[class_idx]) / 255.0
                from matplotlib.patches import Patch

                legend_elements.append(
                    Patch(facecolor=color, label=CLASS_NAMES[class_idx])
                )

        fig.legend(handles=legend_elements, loc="lower center", ncol=5)
        plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_prediction(image_path, mask_path, pred_path, output_path=None):
    """
    Visualize prediction vs ground truth

    Args:
        image_path: Path to original image
        mask_path: Path to ground truth mask
        pred_path: Path to predicted mask
        output_path: Path to save visualization
    """
    # Load image
    if Path(image_path).suffix.lower() in [".tif", ".tiff"]:
        image, _ = load_geotiff(image_path)
        image = image.transpose(1, 2, 0)
        if image.max() > 255:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(
                np.uint8
            )
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load masks
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    # Create colored masks
    h, w = gt_mask.shape
    gt_colored = np.zeros((h, w, 3), dtype=np.uint8)
    pred_colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in CLASS_COLORS.items():
        gt_colored[gt_mask == class_idx] = color
        pred_colored[pred_mask == class_idx] = color

    # Create difference map
    diff = (gt_mask != pred_mask).astype(np.uint8) * 255
    diff_colored = np.zeros((h, w, 3), dtype=np.uint8)
    diff_colored[diff > 0] = [255, 0, 0]  # Red for errors
    diff_colored[diff == 0] = [0, 255, 0]  # Green for correct

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_colored)
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pred_colored)
    axes[1, 0].set_title("Prediction")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(diff_colored)
    accuracy = (diff == 0).sum() / diff.size
    axes[1, 1].set_title(f"Difference Map (Accuracy: {accuracy:.2%})")
    axes[1, 1].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def check_data_quality(data_dir, num_samples=5):
    """
    Check quality of training data

    Args:
        data_dir: Directory containing images/ and masks/ subdirectories
        num_samples: Number of samples to check
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / "images"
    mask_dir = data_dir / "masks"

    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return

    if not mask_dir.exists():
        print(f"❌ Mask directory not found: {mask_dir}")
        return

    # Get image files
    image_files = []
    for ext in [".tif", ".tiff", ".jpg", ".jpeg", ".png"]:
        image_files.extend(image_dir.glob(f"*{ext}"))

    print(f"✅ Found {len(image_files)} images in {image_dir}")

    # Check samples
    for idx, image_path in enumerate(image_files[:num_samples]):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}: {image_path.name}")
        print(f"{'='*60}")

        # Check image
        if image_path.suffix.lower() in [".tif", ".tiff"]:
            image, _ = load_geotiff(str(image_path))
            image = image.transpose(1, 2, 0)
        else:
            image = cv2.imread(str(image_path))

        print(f"Image shape: {image.shape}")

        # Check mask
        mask_path = mask_dir / (image_path.stem + ".png")
        if not mask_path.exists():
            print(f"❌ Mask not found: {mask_path}")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        print(f"Mask shape: {mask.shape}")

        # Check dimensions match
        if image.shape[:2] != mask.shape:
            print(f"❌ Dimension mismatch!")
            print(f"   Image: {image.shape[:2]}")
            print(f"   Mask:  {mask.shape}")
        else:
            print(f"✅ Dimensions match")

        # Check mask values
        unique_values = np.unique(mask)
        print(f"Mask unique values: {unique_values.tolist()}")

        invalid_values = unique_values[unique_values > 9]
        if len(invalid_values) > 0:
            print(f"❌ Invalid mask values found: {invalid_values.tolist()}")
        else:
            print(f"✅ All mask values valid (0-9)")

        # Class distribution
        print("\nClass distribution:")
        for class_idx in range(10):
            count = (mask == class_idx).sum()
            percentage = count / mask.size * 100
            if count > 0:
                print(
                    f"  {CLASS_NAMES[class_idx]:20s}: {count:8d} pixels ({percentage:5.2f}%)"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualization utilities")
    parser.add_argument(
        "--mode",
        choices=["sample", "prediction", "check"],
        required=True,
        help="Visualization mode",
    )
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--mask", type=str, help="Path to mask")
    parser.add_argument("--pred", type=str, help="Path to prediction mask")
    parser.add_argument("--data_dir", type=str, help="Data directory for checking")
    parser.add_argument("--output", type=str, help="Output path for visualization")

    args = parser.parse_args()

    if args.mode == "sample":
        visualize_sample(args.image, args.mask, args.output)
    elif args.mode == "prediction":
        visualize_prediction(args.image, args.mask, args.pred, args.output)
    elif args.mode == "check":
        check_data_quality(args.data_dir)
