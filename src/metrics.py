"""
Metrics calculation for segmentation evaluation.

We track IoU, Dice, Precision, Recall, F1, and pixel accuracy.
IoU (Intersection over Union) is the main metric we care about —
it's the standard for segmentation tasks and what most papers report.

Digital University Kerala (DUK)
"""

import numpy as np
import torch
from typing import Dict, List


def calculate_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Calculate IoU (Jaccard index) for each class.
    
    IoU = intersection / union — simple but tells you exactly how well
    the predicted mask overlaps with the ground truth.
    """
    ious = {}
    for class_idx in range(num_classes):
        pred_mask = pred == class_idx
        target_mask = target == class_idx
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        # if the class doesn't exist in either pred or target, it's NaN (not 0)
        ious[f"iou_class_{class_idx}"] = intersection / union if union > 0 else float("nan")

    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    ious["mean_iou"] = np.mean(valid_ious) if valid_ious else 0.0
    return ious


def calculate_dice(pred: np.ndarray, target: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Calculate Dice coefficient — similar to F1 score but for pixels.
    
    Dice = 2 * intersection / (|pred| + |target|)
    It's more forgiving than IoU for small objects.
    """
    dice_scores = {}
    for class_idx in range(num_classes):
        pred_mask = pred == class_idx
        target_mask = target == class_idx
        intersection = np.logical_and(pred_mask, target_mask).sum()
        denom = pred_mask.sum() + target_mask.sum()
        dice_scores[f"dice_class_{class_idx}"] = 2 * intersection / denom if denom > 0 else float("nan")

    valid = [v for v in dice_scores.values() if not np.isnan(v)]
    dice_scores["mean_dice"] = np.mean(valid) if valid else 0.0
    return dice_scores


def calculate_precision_recall(pred: np.ndarray, target: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Calculate Precision, Recall, and F1 for each class.
    
    We compute these per-class so we can see which classes the model struggles with.
    Usually roads are easier than small infrastructure objects.
    """
    metrics = {}
    for class_idx in range(num_classes):
        pred_mask = pred == class_idx
        target_mask = target == class_idx
        tp = np.logical_and(pred_mask, target_mask).sum()
        fp = np.logical_and(pred_mask, ~target_mask).sum()
        fn = np.logical_and(~pred_mask, target_mask).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            f1 = float("nan")
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        metrics[f"precision_class_{class_idx}"] = precision
        metrics[f"recall_class_{class_idx}"] = recall
        metrics[f"f1_class_{class_idx}"] = f1

    # compute mean across classes (ignoring NaN)
    for metric_name in ["precision", "recall", "f1"]:
        values = [v for k, v in metrics.items() if k.startswith(metric_name) and not np.isnan(v)]
        metrics[f"mean_{metric_name}"] = np.mean(values) if values else 0.0

    return metrics


def calculate_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Overall pixel accuracy — not the best metric for segmentation
    (because background dominates) but nice to have as a sanity check.
    """
    return (pred == target).sum() / pred.size


def calculate_all_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Calculate everything at once — used in the training loop."""
    metrics = {}
    metrics.update(calculate_iou(pred, target, num_classes))
    metrics.update(calculate_dice(pred, target, num_classes))
    metrics.update(calculate_precision_recall(pred, target, num_classes))
    metrics["accuracy"] = calculate_accuracy(pred, target)
    return metrics


class MetricsTracker:
    """Accumulates metrics over batches during training/validation.
    
    We compute per-sample metrics and average them at the end of each epoch.
    This is slightly different from computing metrics over the full dataset
    at once, but it's good enough and way more memory efficient.
    """

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.metrics = {k: [] for k in ["iou", "dice", "precision", "recall", "f1", "accuracy"]}

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update with a batch of predictions and targets."""
        if pred.ndim == 4:
            pred = torch.argmax(pred, dim=1)
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()

        for p, t in zip(pred_np, target_np):
            m = calculate_all_metrics(p, t, self.num_classes)
            self.metrics["iou"].append(m["mean_iou"])
            self.metrics["dice"].append(m["mean_dice"])
            self.metrics["precision"].append(m["mean_precision"])
            self.metrics["recall"].append(m["mean_recall"])
            self.metrics["f1"].append(m["mean_f1"])
            self.metrics["accuracy"].append(m["accuracy"])

    def get_metrics(self) -> Dict[str, float]:
        """Return averaged metrics across all samples seen so far."""
        return {f"mean_{k}": np.mean(v) if v else 0.0 for k, v in self.metrics.items()}

    def print_metrics(self, prefix: str = ""):
        """Pretty-print current metrics — useful for debugging."""
        avg = self.get_metrics()
        print(f"\n{prefix} Metrics:")
        print(f"  Accuracy:  {avg['mean_accuracy']:.4f}")
        print(f"  Mean IoU:  {avg['mean_iou']:.4f}")
        print(f"  Mean Dice: {avg['mean_dice']:.4f}")
        print(f"  Mean F1:   {avg['mean_f1']:.4f}")
        print(f"  Precision: {avg['mean_precision']:.4f}")
        print(f"  Recall:    {avg['mean_recall']:.4f}")
