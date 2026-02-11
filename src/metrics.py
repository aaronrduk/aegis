"""
Evaluation metrics for segmentation
"""

import numpy as np
import torch
from typing import Dict, List
from sklearn.metrics import confusion_matrix


def calculate_iou(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> Dict[str, float]:
    """
    Calculate Intersection over Union (IoU) for each class

    Args:
        pred: Predicted mask (H, W)
        target: Ground truth mask (H, W)
        num_classes: Number of classes

    Returns:
        Dictionary with IoU per class and mean IoU
    """
    ious = {}

    for class_idx in range(num_classes):
        pred_mask = pred == class_idx
        target_mask = target == class_idx

        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        if union == 0:
            iou = float("nan")
        else:
            iou = intersection / union

        ious[f"iou_class_{class_idx}"] = iou

    # Calculate mean IoU (excluding NaN values)
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    ious["mean_iou"] = np.mean(valid_ious) if len(valid_ious) > 0 else 0.0

    return ious


def calculate_dice(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> Dict[str, float]:
    """
    Calculate Dice coefficient for each class

    Args:
        pred: Predicted mask (H, W)
        target: Ground truth mask (H, W)
        num_classes: Number of classes

    Returns:
        Dictionary with Dice per class and mean Dice
    """
    dice_scores = {}

    for class_idx in range(num_classes):
        pred_mask = pred == class_idx
        target_mask = target == class_idx

        intersection = np.logical_and(pred_mask, target_mask).sum()

        if pred_mask.sum() + target_mask.sum() == 0:
            dice = float("nan")
        else:
            dice = 2 * intersection / (pred_mask.sum() + target_mask.sum())

        dice_scores[f"dice_class_{class_idx}"] = dice

    # Calculate mean Dice (excluding NaN values)
    valid_dice = [v for v in dice_scores.values() if not np.isnan(v)]
    dice_scores["mean_dice"] = np.mean(valid_dice) if len(valid_dice) > 0 else 0.0

    return dice_scores


def calculate_precision_recall(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> Dict[str, float]:
    """
    Calculate Precision and Recall for each class

    Args:
        pred: Predicted mask (H, W)
        target: Ground truth mask (H, W)
        num_classes: Number of classes

    Returns:
        Dictionary with Precision and Recall per class
    """
    metrics = {}

    for class_idx in range(num_classes):
        pred_mask = pred == class_idx
        target_mask = target == class_idx

        tp = np.logical_and(pred_mask, target_mask).sum()
        fp = np.logical_and(pred_mask, ~target_mask).sum()
        fn = np.logical_and(~pred_mask, target_mask).sum()

        # Precision
        if tp + fp == 0:
            precision = float("nan")
        else:
            precision = tp / (tp + fp)
        metrics[f"precision_class_{class_idx}"] = precision

        # Recall
        if tp + fn == 0:
            recall = float("nan")
        else:
            recall = tp / (tp + fn)
        metrics[f"recall_class_{class_idx}"] = recall

        # F1 Score
        if precision + recall == 0 or np.isnan(precision) or np.isnan(recall):
            f1 = float("nan")
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        metrics[f"f1_class_{class_idx}"] = f1

    # Calculate mean metrics (excluding NaN)
    for metric_name in ["precision", "recall", "f1"]:
        values = [
            v
            for k, v in metrics.items()
            if k.startswith(metric_name) and not np.isnan(v)
        ]
        metrics[f"mean_{metric_name}"] = np.mean(values) if len(values) > 0 else 0.0

    return metrics


def calculate_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate overall pixel accuracy

    Args:
        pred: Predicted mask (H, W)
        target: Ground truth mask (H, W)

    Returns:
        Accuracy value
    """
    correct = (pred == target).sum()
    total = pred.size
    accuracy = correct / total
    return accuracy


def calculate_all_metrics(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> Dict[str, float]:
    """
    Calculate all metrics

    Args:
        pred: Predicted mask (H, W)
        target: Ground truth mask (H, W)
        num_classes: Number of classes

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # IoU
    metrics.update(calculate_iou(pred, target, num_classes))

    # Dice
    metrics.update(calculate_dice(pred, target, num_classes))

    # Precision, Recall, F1
    metrics.update(calculate_precision_recall(pred, target, num_classes))

    # Accuracy
    metrics["accuracy"] = calculate_accuracy(pred, target)

    return metrics


class MetricsTracker:
    """Track metrics during training"""

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            "iou": [],
            "dice": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "accuracy": [],
        }

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new batch

        Args:
            pred: Predicted logits (B, C, H, W) or predictions (B, H, W)
            target: Ground truth (B, H, W)
        """
        # Convert to numpy
        if pred.ndim == 4:  # Logits
            pred = torch.argmax(pred, dim=1)

        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        # Calculate metrics for each image in batch
        for p, t in zip(pred, target):
            batch_metrics = calculate_all_metrics(p, t, self.num_classes)

            self.metrics["iou"].append(batch_metrics["mean_iou"])
            self.metrics["dice"].append(batch_metrics["mean_dice"])
            self.metrics["precision"].append(batch_metrics["mean_precision"])
            self.metrics["recall"].append(batch_metrics["mean_recall"])
            self.metrics["f1"].append(batch_metrics["mean_f1"])
            self.metrics["accuracy"].append(batch_metrics["accuracy"])

    def get_metrics(self) -> Dict[str, float]:
        """Get average metrics"""
        avg_metrics = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                avg_metrics[f"mean_{key}"] = np.mean(values)
            else:
                avg_metrics[f"mean_{key}"] = 0.0

        return avg_metrics

    def print_metrics(self, prefix: str = ""):
        """Print metrics in readable format"""
        avg_metrics = self.get_metrics()

        print(f"\n{prefix} Metrics:")
        print(f"  Accuracy:  {avg_metrics['mean_accuracy']:.4f}")
        print(f"  Mean IoU:  {avg_metrics['mean_iou']:.4f}")
        print(f"  Mean Dice: {avg_metrics['mean_dice']:.4f}")
        print(f"  Mean F1:   {avg_metrics['mean_f1']:.4f}")
        print(f"  Precision: {avg_metrics['mean_precision']:.4f}")
        print(f"  Recall:    {avg_metrics['mean_recall']:.4f}")


if __name__ == "__main__":
    # Test metrics
    num_classes = 10

    # Create dummy predictions and targets
    pred = np.random.randint(0, num_classes, size=(512, 512))
    target = np.random.randint(0, num_classes, size=(512, 512))

    # Calculate metrics
    metrics = calculate_all_metrics(pred, target, num_classes)

    print("Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  Mean Dice: {metrics['mean_dice']:.4f}")
    print(f"  Mean F1: {metrics['mean_f1']:.4f}")
