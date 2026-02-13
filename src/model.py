import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional


class SVAMITVASegmentationModel(nn.Module):
    """Multi-class segmentation model using DeepLabV3+ for SVAMITVA drone imagery."""

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
        """Forward pass returning logits (B, C, H, W)."""
        return self.model(x)

    def predict(self, x):
        """Return class predictions (B, H, W)."""
        logits = self.forward(x)
        return torch.argmax(torch.softmax(logits, dim=1), dim=1)

    def predict_proba(self, x):
        """Return probability maps (B, C, H, W)."""
        return torch.softmax(self.forward(x), dim=1)


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss for segmentation with class imbalance."""

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
        """Compute combined focal + dice loss, with optional class weighting."""
        focal = self.focal_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        total = self.focal_weight * focal + self.dice_weight * dice
        if self.class_weights is not None:
            ce = nn.functional.cross_entropy(predictions, targets, weight=self.class_weights)
            total = total + 0.1 * ce
        return total


def create_model(config: dict) -> SVAMITVASegmentationModel:
    """Create model from config."""
    return SVAMITVASegmentationModel(
        num_classes=config["num_classes"],
        encoder=config["encoder"],
        encoder_weights=config["encoder_weights"],
        activation=config["activation"],
    )


def create_loss(config: dict, device: torch.device) -> FocalDiceLoss:
    """Create loss function from config."""
    class_weights = None
    if "class_weights" in config:
        class_weights = torch.tensor(config["class_weights"], dtype=torch.float32).to(device)

    return FocalDiceLoss(
        focal_weight=config["loss_weights"]["focal_weight"],
        dice_weight=config["loss_weights"]["dice_weight"],
        class_weights=class_weights,
    )
