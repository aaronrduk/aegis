"""
Evaluation script for testing model performance
"""

import argparse
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

from src.model import create_model
from src.dataset import SVAMITVADataset, get_validation_augmentation
from src.metrics import calculate_all_metrics, MetricsTracker
from src.config import TRAINING_CONFIG, CLASS_NAMES
from src.utils import get_device, setup_logger
from torch.utils.data import DataLoader


def evaluate_model(
    checkpoint_path: str,
    test_images: str,
    test_masks: str,
    output_dir: str = "outputs/evaluation",
):
    """
    Evaluate model on test set

    Args:
        checkpoint_path: Path to model checkpoint
        test_images: Directory with test images
        test_masks: Directory with test masks
        output_dir: Directory for evaluation outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger("Evaluation", log_file=str(output_dir / "evaluation.log"))
    logger.info("Starting evaluation...")

    # Get device
    device = get_device()

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Create model
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded (Epoch {checkpoint['epoch']})")

    # Create dataset
    test_dataset = SVAMITVADataset(
        image_dir=test_images,
        mask_dir=test_masks,
        transform=get_validation_augmentation(config["input_size"]),
        image_size=config["input_size"],
    )

    logger.info(f"Test dataset: {len(test_dataset)} images")

    # Create data loader
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    # Metrics tracker
    metrics_tracker = MetricsTracker(config["num_classes"], CLASS_NAMES)

    # Store all predictions and targets for confusion matrix
    all_preds = []
    all_targets = []

    # Evaluate
    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Update metrics
            metrics_tracker.update(preds, masks)

            # Store for confusion matrix
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    # Get metrics
    avg_metrics = metrics_tracker.get_metrics()

    # Print metrics
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Overall Accuracy: {avg_metrics['mean_accuracy']:.4f}")
    logger.info(f"Mean IoU:         {avg_metrics['mean_iou']:.4f}")
    logger.info(f"Mean Dice:        {avg_metrics['mean_dice']:.4f}")
    logger.info(f"Mean F1:          {avg_metrics['mean_f1']:.4f}")
    logger.info(f"Mean Precision:   {avg_metrics['mean_precision']:.4f}")
    logger.info(f"Mean Recall:      {avg_metrics['mean_recall']:.4f}")
    logger.info("=" * 50)

    # Per-class metrics
    logger.info("\nPer-Class Metrics:")

    # Calculate per-class metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    per_class_metrics = []
    for class_idx in range(1, config["num_classes"]):  # Skip background
        class_name = CLASS_NAMES[class_idx]

        # Calculate metrics for this class
        pred_mask = all_preds == class_idx
        target_mask = all_targets == class_idx

        if target_mask.sum() == 0:
            logger.info(f"{class_name}: No samples in test set")
            continue

        tp = np.logical_and(pred_mask, target_mask).sum()
        fp = np.logical_and(pred_mask, ~target_mask).sum()
        fn = np.logical_and(~pred_mask, target_mask).sum()

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        per_class_metrics.append(
            {
                "Class": class_name,
                "IoU": f"{iou:.4f}",
                "F1": f"{f1:.4f}",
                "Precision": f"{precision:.4f}",
                "Recall": f"{recall:.4f}",
            }
        )

        logger.info(
            f"{class_name:20s} - IoU: {iou:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}"
        )

    # Save metrics to CSV
    metrics_df = pd.DataFrame(per_class_metrics)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"\nMetrics saved to: {metrics_path}")

    # Create confusion matrix
    logger.info("\nGenerating confusion matrix...")
    cm = confusion_matrix(
        all_targets.flatten(), all_preds.flatten(), labels=range(config["num_classes"])
    )

    # Normalize confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Normalized Count"},
    )
    plt.title("Confusion Matrix (Normalized)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    logger.info(f"Confusion matrix saved to: {cm_path}")

    # Plot metrics bar chart
    if per_class_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        metrics_to_plot = ["IoU", "F1", "Precision", "Recall"]
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]

            classes = [m["Class"] for m in per_class_metrics]
            values = [float(m[metric]) for m in per_class_metrics]

            bars = ax.bar(classes, values, color="steelblue", alpha=0.7)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} per Class")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        metrics_chart_path = output_dir / "metrics_chart.png"
        plt.savefig(metrics_chart_path, dpi=300, bbox_inches="tight")
        logger.info(f"Metrics chart saved to: {metrics_chart_path}")

    # Create summary report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("SVAMITVA Feature Extraction - Evaluation Report\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Model Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Images: {test_images}\n")
        f.write(f"Test Samples: {len(test_dataset)}\n\n")

        f.write("Overall Metrics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:  {avg_metrics['mean_accuracy']:.4f}\n")
        f.write(f"Mean IoU:  {avg_metrics['mean_iou']:.4f}\n")
        f.write(f"Mean Dice: {avg_metrics['mean_dice']:.4f}\n")
        f.write(f"Mean F1:   {avg_metrics['mean_f1']:.4f}\n")
        f.write(f"Precision: {avg_metrics['mean_precision']:.4f}\n")
        f.write(f"Recall:    {avg_metrics['mean_recall']:.4f}\n\n")

        f.write("Per-Class Metrics:\n")
        f.write("-" * 70 + "\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n")

    logger.info(f"Full report saved to: {report_path}")
    logger.info("\nEvaluation complete!")

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SVAMITVA model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test_images", type=str, required=True, help="Path to test images directory"
    )
    parser.add_argument(
        "--test_masks", type=str, required=True, help="Path to test masks directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for evaluation results",
    )

    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        test_images=args.test_images,
        test_masks=args.test_masks,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
