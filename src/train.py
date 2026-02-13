"""
Training script for the SVAMITVA segmentation model.

This handles the full training loop — data loading, forward/backward pass,
gradient accumulation, LR warmup, early stopping, checkpointing, etc.
We kept it all in one class to keep things organized during the hackathon.

Usage:
    python -m src.train --cpu           # for testing on laptop
    python -m src.train                 # for full GPU training
    python -m src.train --epochs 50     # override specific params

Team SVAMITVA - SIH Hackathon 2026
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

try:
    from .config import TRAINING_CONFIG, TRAINING_CONFIG_CPU, CLASS_NAMES
    from .model import create_model, create_loss
    from .dataset import SVAMITVADataset, get_training_augmentation, get_validation_augmentation
    from .metrics import MetricsTracker
    from .utils import setup_logger, get_device, AverageMeter
except ImportError:
    # this happens when running the script directly (not as a module)
    from config import TRAINING_CONFIG, TRAINING_CONFIG_CPU, CLASS_NAMES
    from model import create_model, create_loss
    from dataset import SVAMITVADataset, get_training_augmentation, get_validation_augmentation
    from metrics import MetricsTracker
    from utils import setup_logger, get_device, AverageMeter


class Trainer:
    """Handles the full training pipeline for our segmentation model.
    
    Includes gradient accumulation (to simulate larger batch sizes on limited VRAM),
    LR warmup, cosine annealing, mixed precision training, and early stopping.
    """

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.accumulation_steps = config.get("accumulation_steps", 2)
        self.warmup_epochs = config.get("warmup_epochs", 5)
        self.max_norm = config.get("gradient_clip_max_norm", 1.0)

        self.checkpoint_dir = Path("checkpoints")
        self.log_dir = Path("logs")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        self.logger = setup_logger("SVAMITVA_Trainer", log_file=str(self.log_dir / "training.log"))
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.logger.info("Creating model...")
        self.model = create_model(config).to(device)

        # nice to see how big the model is
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        self.criterion = create_loss(config, device)

        # AdamW > Adam for us, the weight decay regularization helps
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        if config["scheduler"] == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config["num_epochs"], eta_min=config["min_lr"]
            )
        else:
            self.scheduler = None

        # mixed precision for faster training on GPU
        self.scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        self.best_iou = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def _get_warmup_lr(self, epoch: int) -> float:
        """Linear warmup — start low and ramp up to target LR."""
        if epoch <= self.warmup_epochs:
            return self.config["learning_rate"] * epoch / max(self.warmup_epochs, 1)
        return self.config["learning_rate"]

    def _apply_warmup(self, epoch: int):
        """Apply warmup LR for the first few epochs.
        
        Without this, training was really unstable in the first 3-4 epochs.
        """
        if epoch <= self.warmup_epochs:
            lr = self._get_warmup_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """Train for one epoch with gradient accumulation and clipping."""
        self.model.train()
        loss_meter = AverageMeter()
        metrics_tracker = MetricsTracker(self.config["num_classes"], CLASS_NAMES)

        self.optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']} [Train]")

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            if self.scaler is not None:
                # mixed precision path (GPU only)
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks) / self.accumulation_steps
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # regular precision path (CPU)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks) / self.accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # gradient clipping prevents exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            loss_meter.update(loss.item() * self.accumulation_steps, images.size(0))
            metrics_tracker.update(outputs.detach(), masks.detach())
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        epoch_metrics = metrics_tracker.get_metrics()
        epoch_metrics["loss"] = loss_meter.avg
        return epoch_metrics

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> dict:
        """Validate for one epoch — no gradients needed here."""
        self.model.eval()
        loss_meter = AverageMeter()
        metrics_tracker = MetricsTracker(self.config["num_classes"], CLASS_NAMES)

        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']} [Val]  ")

        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                loss_meter.update(loss.item(), images.size(0))
                metrics_tracker.update(outputs, masks)
                pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        epoch_metrics = metrics_tracker.get_metrics()
        epoch_metrics["loss"] = loss_meter.avg
        return epoch_metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint — always save periodic + best separately."""
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

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop — this is where the magic happens 🚀"""
        self.logger.info("Starting training...")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")

        for epoch in range(1, self.config["num_epochs"] + 1):
            self._apply_warmup(epoch)

            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate_epoch(val_loader, epoch)

            # don't step the scheduler during warmup — they'd conflict
            if self.scheduler is not None and epoch > self.warmup_epochs:
                self.scheduler.step()

            self.logger.info(
                f"Epoch {epoch}/{self.config['num_epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['mean_iou']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['mean_iou']:.4f}"
            )

            # log everything to tensorboard
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)
            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], epoch)

            # check if this is a new best
            current_iou = val_metrics["mean_iou"]
            is_best = current_iou > (self.best_iou + self.config.get("min_delta", 0))
            if is_best:
                self.best_iou = current_iou
                self.best_epoch = epoch
                self.patience_counter = 0
                self.logger.info(f"New best IoU: {self.best_iou:.4f}")
            else:
                self.patience_counter += 1

            if epoch % self.config["save_frequency"] == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)

            # early stopping — no point training if we're not improving
            if self.patience_counter >= self.config["patience"]:
                self.logger.info(f"Early stopping at epoch {epoch}. Best IoU: {self.best_iou:.4f} at epoch {self.best_epoch}")
                break

        self.logger.info(f"Training completed! Best IoU: {self.best_iou:.4f} at epoch {self.best_epoch}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train SVAMITVA Feature Extraction Model")
    parser.add_argument("--train_images", type=str, default="data/train/images")
    parser.add_argument("--train_masks", type=str, default="data/train/masks")
    parser.add_argument("--val_images", type=str, default="data/val/images")
    parser.add_argument("--val_masks", type=str, default="data/val/masks")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--cpu", action="store_true", help="Use CPU-optimized config (256x256, smaller batch)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    args = parser.parse_args()

    # pick the right config based on whether we have a GPU or not
    config = TRAINING_CONFIG_CPU.copy() if args.cpu else TRAINING_CONFIG.copy()
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.lr is not None:
        config["learning_rate"] = args.lr

    device = get_device()

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",  # only useful with GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    trainer = Trainer(config, device)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
