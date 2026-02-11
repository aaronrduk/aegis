"""
Inference module for SVAMITVA Feature Extraction
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm import tqdm

# Optional import
try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None

from src.model import SVAMITVASegmentationModel
from src.dataset import get_inference_augmentation
from src.utils import get_device, load_geotiff
import albumentations as A


class SVAMITVAInference:
    """Inference class for SVAMITVA segmentation model"""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        use_tta: bool = True,
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            use_tta: Use test-time augmentation
        """
        self.device = device if device is not None else get_device()
        self.use_tta = use_tta

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.config = checkpoint["config"]

            # Create model
            self.model = SVAMITVASegmentationModel(
                num_classes=self.config["num_classes"],
                encoder=self.config["encoder"],
                encoder_weights=None,  # Don't load pretrained weights
                activation=self.config["activation"],
            )

            # Load weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"Model loaded successfully (Epoch {checkpoint['epoch']})")
            print(f"Best IoU: {checkpoint.get('best_iou', 'N/A')}")

        except FileNotFoundError:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Creating initialized model for testing/demo purposes...")
            # Create dummy model for demo if checkpoint missing
            self.config = {
                "num_classes": 10,
                "encoder": "resnet50",
                "activation": "softmax2d",
                "input_size": (512, 512),
            }
            self.model = SVAMITVASegmentationModel(
                num_classes=10,
                encoder="resnet50",
                encoder_weights="imagenet",
                activation="softmax2d",
            )
            self.model = self.model.to(self.device)
            self.model.eval()

    def predict_image(
        self,
        image: np.ndarray,
        use_sliding_window: bool = False,
        window_size: int = 512,
        overlap: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict segmentation mask for an image

        Args:
            image: Input image (H, W, 3)
            use_sliding_window: Use sliding window for large images
            window_size: Window size for sliding window
            overlap: Overlap between windows

        Returns:
            Tuple of (predicted mask, probability map)
        """
        h, w = image.shape[:2]

        if use_sliding_window and (h > window_size or w > window_size):
            # Use sliding window for large images
            mask, probs = self._predict_sliding_window(image, window_size, overlap)
        else:
            # Direct prediction for small images
            mask, probs = self._predict_direct(image)

        return mask, probs

    def _predict_direct(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Direct prediction without sliding window"""
        original_h, original_w = image.shape[:2]

        # Prepare transforms
        transform = get_inference_augmentation(self.config["input_size"])

        if self.use_tta:
            # Test-time augmentation
            predictions = []

            # Original
            aug = transform(image=image)
            img_tensor = aug["image"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model.predict_proba(img_tensor)
            predictions.append(pred)

            # Horizontal flip
            img_flipped = cv2.flip(image, 1)
            aug = transform(image=img_flipped)
            img_tensor = aug["image"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model.predict_proba(img_tensor)
                pred = torch.flip(pred, dims=[3])  # Flip back
            predictions.append(pred)

            # Vertical flip
            img_flipped = cv2.flip(image, 0)
            aug = transform(image=img_flipped)
            img_tensor = aug["image"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model.predict_proba(img_tensor)
                pred = torch.flip(pred, dims=[2])  # Flip back
            predictions.append(pred)

            # Average predictions
            probs = torch.stack(predictions).mean(dim=0)
        else:
            # Single prediction
            aug = transform(image=image)
            img_tensor = aug["image"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = self.model.predict_proba(img_tensor)

        # Get final prediction
        probs = probs.squeeze(0).cpu().numpy()
        mask = np.argmax(probs, axis=0)

        # Resize back to original size
        if (mask.shape[0] != original_h) or (mask.shape[1] != original_w):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Resize probability maps
            probs_resized = np.zeros(
                (probs.shape[0], original_h, original_w), dtype=np.float32
            )
            for i in range(probs.shape[0]):
                probs_resized[i] = cv2.resize(
                    probs[i], (original_w, original_h), interpolation=cv2.INTER_LINEAR
                )
            probs = probs_resized

        return mask, probs

    def _predict_sliding_window(
        self, image: np.ndarray, window_size: int, overlap: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using sliding window approach"""
        h, w = image.shape[:2]
        stride = window_size - overlap

        # Initialize output arrays
        mask_final = np.zeros((h, w), dtype=np.uint8)
        probs_final = np.zeros((self.config["num_classes"], h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        # Prepare transform
        transform = get_inference_augmentation((window_size, window_size))

        # Calculate window positions
        y_positions = list(range(0, h - window_size + 1, stride))
        x_positions = list(range(0, w - window_size + 1, stride))

        # Add last positions if needed
        if y_positions[-1] + window_size < h:
            y_positions.append(h - window_size)
        if x_positions[-1] + window_size < w:
            x_positions.append(w - window_size)

        # Process each window
        total_windows = len(y_positions) * len(x_positions)
        pbar = tqdm(total=total_windows, desc="Processing windows")

        for y in y_positions:
            for x in x_positions:
                # Extract window
                window = image[y : y + window_size, x : x + window_size]

                # Predict
                aug = transform(image=window)
                img_tensor = aug["image"].unsqueeze(0).to(self.device)

                with torch.no_grad():
                    window_probs = self.model.predict_proba(img_tensor)
                    window_probs = window_probs.squeeze(0).cpu().numpy()

                # Accumulate predictions
                probs_final[:, y : y + window_size, x : x + window_size] += window_probs
                count_map[y : y + window_size, x : x + window_size] += 1

                pbar.update(1)

        pbar.close()

        # Average overlapping predictions
        count_map = np.maximum(count_map, 1)  # Avoid division by zero
        probs_final = probs_final / count_map

        # Get final mask
        mask_final = np.argmax(probs_final, axis=0).astype(np.uint8)

        return mask_final, probs_final

    def predict_file(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        save_probs: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Predict segmentation for an image file

        Args:
            image_path: Path to input image
            output_path: Path to save output mask
            save_probs: Save probability maps

        Returns:
            Tuple of (mask, probs, metadata)
        """
        print(f"Processing: {image_path}")

        # Load image
        image_path = Path(image_path)
        metadata = {}

        if image_path.suffix.lower() in [".tif", ".tiff"] and HAS_RASTERIO:
            # Load geospatial image
            try:
                image, metadata = load_geotiff(str(image_path))
                image = image.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

                # Normalize if needed
                if image.max() > 255:
                    image = (
                        (image - image.min()) / (image.max() - image.min()) * 255
                    ).astype(np.uint8)
            except ImportError:
                # Fallback to OpenCV if load_geotiff fails
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Load regular image or if rasterio missing
            img_read = cv2.imread(str(image_path))
            if img_read is None:
                raise ValueError(f"Could not load image at {image_path}")
            image = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)

        # Predict
        mask, probs = self.predict_image(
            image, use_sliding_window=True, window_size=512, overlap=128
        )

        # Save output
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save mask
            cv2.imwrite(str(output_path), mask)
            print(f"Saved mask to: {output_path}")

            # Save probability maps
            if save_probs:
                probs_path = output_path.parent / (output_path.stem + "_probs.npy")
                np.save(probs_path, probs)
                print(f"Saved probabilities to: {probs_path}")

        return mask, probs, metadata


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SVAMITVA Inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to output mask")
    parser.add_argument(
        "--use_tta", action="store_true", help="Use test-time augmentation"
    )
    parser.add_argument(
        "--save_probs", action="store_true", help="Save probability maps"
    )

    args = parser.parse_args()

    # Create inference object
    inference = SVAMITVAInference(checkpoint_path=args.checkpoint, use_tta=args.use_tta)

    # Predict
    mask, probs, metadata = inference.predict_file(
        image_path=args.image, output_path=args.output, save_probs=args.save_probs
    )

    print(f"Prediction shape: {mask.shape}")
    print(f"Unique classes: {np.unique(mask).tolist()}")


if __name__ == "__main__":
    main()
