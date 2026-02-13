"""
Model definitions for the SVAMITVA segmentation pipeline.

We're using DeepLabV3+ with an EfficientNet-B4 backbone from the
segmentation_models_pytorch library. Tried UNet and FPN too but
DeepLabV3+ handled the multi-scale building detection best.

The loss function is a combo of Focal + Dice which works really well
for our imbalanced dataset (way too much background, not enough buildings).

Team SVAMITVA - SIH Hackathon 2026
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional


class SVAMITVASegmentationModel(nn.Module):
    """Our main segmentation model — DeepLabV3+ with EfficientNet-B4 encoder.
    
    We picked this combo because:
    - EfficientNet-B4 has good accuracy/speed tradeoff
    - DeepLabV3+ handles multi-scale features well (important for buildings of different sizes)
    - smp makes it super easy to swap encoders if we want to experiment later
    """

    def __init__(
        self,
        num_classes: int = 10,
        encoder: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        activation: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=activation,
        )

    def forward(self, x):
        """Forward pass — returns raw logits (B, C, H, W)."""
        return self.model(x)

    def predict(self, x):
        """Get hard class predictions (B, H, W) — just argmax over softmax."""
        logits = self.forward(x)
        return torch.argmax(torch.softmax(logits, dim=1), dim=1)

    def predict_proba(self, x):
        """Get probability maps (B, C, H, W) — useful for TTA averaging."""
        return torch.softmax(self.forward(x), dim=1)


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss for handling class imbalance.
    
    Plain cross-entropy was terrible for us because >80% of pixels are background.
    Focal loss downweights easy examples, and Dice directly optimizes the overlap metric.
    We also add a small CE term with class weights for extra stability.
    """

    def __init__(
        self,
        focal_weight: float = 0.4,
        dice_weight: float = 0.6,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = smp.losses.FocalLoss(mode="multiclass")
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=list(range(10)))
        self.class_weights = class_weights

    def forward(self, predictions, targets):
        """Compute the combined loss."""
        focal = self.focal_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        total = self.focal_weight * focal + self.dice_weight * dice

        # the 0.1 weight on CE was found empirically — too high and it dominates
        if self.class_weights is not None:
            ce = nn.functional.cross_entropy(predictions, targets, weight=self.class_weights)
            total = total + 0.1 * ce
        return total


def create_model(config: dict) -> SVAMITVASegmentationModel:
    """Factory function to build the model from config dict."""
    return SVAMITVASegmentationModel(
        num_classes=config["num_classes"],
        encoder=config["encoder"],
        encoder_weights=config["encoder_weights"],
        activation=config["activation"],
    )


def create_loss(config: dict, device: torch.device) -> FocalDiceLoss:
    """Factory function to build the loss from config dict."""
    class_weights = None
    if "class_weights" in config:
        class_weights = torch.tensor(config["class_weights"], dtype=torch.float32).to(device)

    return FocalDiceLoss(
        focal_weight=config["loss_weights"]["focal_weight"],
        dice_weight=config["loss_weights"]["dice_weight"],
        class_weights=class_weights,
    )
