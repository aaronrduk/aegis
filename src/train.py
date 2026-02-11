"""
Training script for SVAMITVA Feature Extraction Model
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np

from config import TRAINING_CONFIG, CLASS_NAMES
from model import create_model, create_loss
from dataset import (
    SVAMITVADataset,
    get_training_augmentation,
    get_validation_augmentation,
)
from metrics import MetricsTracker
from utils import setup_logger, get_device, AverageMeter


class Trainer:
    """Trainer class for SVAMITVA segmentation model"""

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device

        # Create directories
        self.checkpoint_dir = Path("checkpoints")
        self.log_dir = Path("logs")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # Setup logger
        self.logger = setup_logger(
            "SVAMITVA_Trainer", log_file=str(self.log_dir / "training.log")
        )

        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Create model
        self.logger.info("Creating model...")
        self.model = create_model(config)
        self.model = self.model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        # Create loss function
        self.criterion = create_loss(config, device)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        # Create learning rate scheduler
        if config["scheduler"] == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config["num_epochs"], eta_min=config["min_lr"]
            )
        else:
            self.scheduler = None

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        # Best metrics tracking
        self.best_iou = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()

        loss_meter = AverageMeter()
        metrics_tracker = MetricsTracker(self.config["num_classes"], CLASS_NAMES)

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']} [Train]"
        )

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            metrics_tracker.update(outputs.detach(), masks.detach())

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        # Get epoch metrics
        epoch_metrics = metrics_tracker.get_metrics()
        epoch_metrics["loss"] = loss_meter.avg

        return epoch_metrics

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> dict:
        """Validate for one epoch"""
        self.model.eval()

        loss_meter = AverageMeter()
        metrics_tracker = MetricsTracker(self.config["num_classes"], CLASS_NAMES)

        pbar = tqdm(
            val_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']} [Val]  "
        )

        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # Update metrics
                loss_meter.update(loss.item(), images.size(0))
                metrics_tracker.update(outputs, masks)

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        # Get epoch metrics
        epoch_metrics = metrics_tracker.get_metrics()
        epoch_metrics["loss"] = loss_meter.avg

        return epoch_metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "best_iou": self.best_iou,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")

        for epoch in range(1, self.config["num_epochs"] + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate_epoch(val_loader, epoch)

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Log metrics
            self.logger.info(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            self.logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['mean_iou']:.4f}"
            )
            self.logger.info(
                f"Val   Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['mean_iou']:.4f}"
            )

            # TensorBoard logging
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)
            self.writer.add_scalar(
                "learning_rate", self.optimizer.param_groups[0]["lr"], epoch
            )

            # Check for improvement
            is_best = val_metrics["mean_iou"] > self.best_iou
            if is_best:
                self.best_iou = val_metrics["mean_iou"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self.logger.info(f"New best IoU: {self.best_iou:.4f}")
            else:
                self.patience_counter += 1

            # Save checkpoint
            if epoch % self.config["save_frequency"] == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if self.patience_counter >= self.config["patience"]:
                self.logger.info(f"Early stopping at epoch {epoch}")
                self.logger.info(
                    f"Best IoU: {self.best_iou:.4f} at epoch {self.best_epoch}"
                )
                break

        self.logger.info("Training completed!")
        self.logger.info(f"Best IoU: {self.best_iou:.4f} at epoch {self.best_epoch}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train SVAMITVA Feature Extraction Model"
    )
    parser.add_argument(
        "--train_images",
        type=str,
        default="data/train/images",
        help="Path to training images",
    )
    parser.add_argument(
        "--train_masks",
        type=str,
        default="data/train/masks",
        help="Path to training masks",
    )
    parser.add_argument(
        "--val_images",
        type=str,
        default="data/val/images",
        help="Path to validation images",
    )
    parser.add_argument(
        "--val_masks",
        type=str,
        default="data/val/masks",
        help="Path to validation masks",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (overrides config)"
    )

    args = parser.parse_args()

    # Override config with command line arguments
    config = TRAINING_CONFIG.copy()
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.lr is not None:
        config["learning_rate"] = args.lr

    # Get device
    device = get_device()

    # Create datasets
    print("Loading datasets...")
    train_dataset = SVAMITVADataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        transform=get_training_augmentation(config["input_size"]),
        image_size=config["input_size"],
    )

    val_dataset = SVAMITVADataset(
        image_dir=args.val_images,
        mask_dir=args.val_masks,
        transform=get_validation_augmentation(config["input_size"]),
        image_size=config["input_size"],
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
    )

    # Create trainer and start training
    trainer = Trainer(config, device)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
