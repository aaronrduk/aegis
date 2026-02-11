"""
DeepLabV3+ Model for Multi-class Segmentation
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional


class SVAMITVASegmentationModel(nn.Module):
    """
    Multi-class segmentation model for SVAMITVA drone imagery
    Uses DeepLabV3+ architecture with ResNet-50 backbone
    """

    def __init__(
        self,
        num_classes: int = 10,
        encoder: str = "resnet50",
        encoder_weights: str = "imagenet",
        activation: Optional[str] = None,
    ):
        super(SVAMITVASegmentationModel, self).__init__()

        self.num_classes = num_classes

        # DeepLabV3+ model
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=activation,
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Output logits of shape (B, num_classes, H, W)
        """
        return self.model(x)

    def predict(self, x):
        """
        Generate predictions with softmax

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Class predictions of shape (B, H, W)
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds

    def predict_proba(self, x):
        """
        Generate probability maps

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Probability maps of shape (B, num_classes, H, W)
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy and Dice Loss for better segmentation
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super(CombinedLoss, self).__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Dice loss from segmentation_models_pytorch
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, predictions, targets):
        """
        Calculate combined loss

        Args:
            predictions: Model output logits (B, num_classes, H, W)
            targets: Ground truth masks (B, H, W)

        Returns:
            Combined loss value
        """
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)

        total_loss = self.ce_weight * ce + self.dice_weight * dice

        return total_loss


def create_model(config: dict) -> SVAMITVASegmentationModel:
    """
    Factory function to create model from config

    Args:
        config: Configuration dictionary

    Returns:
        Initialized model
    """
    model = SVAMITVASegmentationModel(
        num_classes=config["num_classes"],
        encoder=config["encoder"],
        encoder_weights=config["encoder_weights"],
        activation=config["activation"],
    )

    return model


def create_loss(config: dict, device: torch.device) -> CombinedLoss:
    """
    Factory function to create loss from config

    Args:
        config: Configuration dictionary
        device: Device to place weights on

    Returns:
        Initialized loss function
    """
    class_weights = None
    if "class_weights" in config:
        class_weights = torch.tensor(config["class_weights"], dtype=torch.float32).to(
            device
        )

    loss_fn = CombinedLoss(
        num_classes=config["num_classes"],
        ce_weight=config["loss_weights"]["ce_weight"],
        dice_weight=config["loss_weights"]["dice_weight"],
        class_weights=class_weights,
    )

    return loss_fn


if __name__ == "__main__":
    # Test model creation
    from config import TRAINING_CONFIG

    model = create_model(TRAINING_CONFIG)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test prediction
    preds = model.predict(dummy_input)
    print(f"Prediction shape: {preds.shape}")
    print(f"Unique classes in prediction: {torch.unique(preds).tolist()}")
