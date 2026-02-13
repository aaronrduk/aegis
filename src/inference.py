"""
Inference pipeline for the SVAMITVA segmentation model.

Handles loading a trained checkpoint and running predictions on new images.
Supports test-time augmentation (TTA) and sliding window for large images.

NOTE: on CPU, a single 4000x3000 drone image takes about 2 minutes with
sliding window + TTA. On a GPU it's like 5 seconds. We really need GPU access lol.

Team SVAMITVA - SIH Hackathon 2026
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None

from src.model import SVAMITVASegmentationModel
from src.dataset import get_inference_augmentation
from src.utils import get_device, load_geotiff


class SVAMITVAInference:
    """Main inference class — load a checkpoint and predict on images.
    
    We mask out untrained classes so the model doesn't hallucinate predictions
    for classes we haven't actually trained on yet.
    """

    # classes we've actually trained well — others get masked out
    DEFAULT_VALID_CLASSES = [0, 1, 2, 4, 5]
    # this threshold worked best after testing different values (0.2 was too low, 0.5 too aggressive)
    CONFIDENCE_THRESHOLD = 0.3

    def __init__(self, checkpoint_path: str, device: Optional[torch.device] = None,
                 use_tta: bool = True, valid_classes: Optional[List[int]] = None):
        self.device = device if device is not None else get_device()
        self.use_tta = use_tta

        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            # had to add weights_only=False because pytorch 2.6 changed the default
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.config = checkpoint["config"]
            self.model = SVAMITVASegmentationModel(
                num_classes=self.config["num_classes"],
                encoder=self.config["encoder"],
                encoder_weights=None,  # don't download imagenet weights, we have our own
                activation=self.config["activation"],
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully (Epoch {checkpoint['epoch']})")
            print(f"Best IoU: {checkpoint.get('best_iou', 'N/A')}")
        except FileNotFoundError:
            # no checkpoint? just create a demo model with random weights
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Creating demo model with efficientnet-b4 encoder...")
            self.config = {
                "num_classes": 10,
                "encoder": "efficientnet-b4",
                "activation": None,
                "input_size": (512, 512),
            }
            self.model = SVAMITVASegmentationModel(
                num_classes=10,
                encoder="efficientnet-b4",
                encoder_weights="imagenet",
                activation=None,
            )
            self.model = self.model.to(self.device)
            self.model.eval()

        self.valid_classes = valid_classes if valid_classes is not None else self.DEFAULT_VALID_CLASSES
        self._invalid_classes = [c for c in range(self.config["num_classes"]) if c not in self.valid_classes]
        print(f"Valid classes: {self.valid_classes}")
        if self._invalid_classes:
            print(f"Masking out untrained classes: {self._invalid_classes}")

    def predict_image(self, image: np.ndarray, use_sliding_window: bool = False,
                      window_size: int = 512, overlap: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """Run prediction on a numpy image. Uses sliding window for large images."""
        h, w = image.shape[:2]
        if use_sliding_window and (h > window_size or w > window_size):
            return self._predict_sliding_window(image, window_size, overlap)
        return self._predict_direct(image)

    def _apply_class_masking(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Zero out probabilities for classes we haven't trained on."""
        for c in self._invalid_classes:
            probs[c] = -1e9  # effectively zero after softmax
        mask = np.argmax(probs, axis=0)
        # if the model isn't confident enough, just call it background
        max_probs = np.max(probs, axis=0)
        mask[max_probs < self.CONFIDENCE_THRESHOLD] = 0
        return mask, probs

    def _predict_direct(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Direct prediction — resize to input_size, predict, resize back."""
        original_h, original_w = image.shape[:2]
        transform = get_inference_augmentation(self.config["input_size"])

        def _infer(img):
            aug = transform(image=img)
            tensor = aug["image"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.model.predict_proba(tensor)

        if self.use_tta:
            # TTA: average predictions from original + horizontal flip + vertical flip
            # this gives ~1-2% better IoU for almost free
            predictions = [_infer(image)]

            hflip = cv2.flip(image, 1)
            pred_h = _infer(hflip)
            predictions.append(torch.flip(pred_h, dims=[3]))

            vflip = cv2.flip(image, 0)
            pred_v = _infer(vflip)
            predictions.append(torch.flip(pred_v, dims=[2]))

            probs = torch.stack(predictions).mean(dim=0)
        else:
            probs = _infer(image)

        probs = probs.squeeze(0).cpu().numpy()

        # resize probabilities back to original image dimensions
        if probs.shape[1] != original_h or probs.shape[2] != original_w:
            probs_resized = np.zeros((probs.shape[0], original_h, original_w), dtype=np.float32)
            for i in range(probs.shape[0]):
                probs_resized[i] = cv2.resize(probs[i], (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            probs = probs_resized

        mask, probs = self._apply_class_masking(probs)
        mask = mask.astype(np.uint8)

        return mask, probs

    def _predict_sliding_window(self, image: np.ndarray, window_size: int, overlap: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding window prediction for images larger than the model's input size.
        
        We accumulate probability maps and average overlapping regions.
        This avoids edge artifacts that you get with naive tiling.
        """
        h, w = image.shape[:2]
        stride = window_size - overlap
        probs_final = np.zeros((self.config["num_classes"], h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        transform = get_inference_augmentation((window_size, window_size))

        y_positions = list(range(0, max(1, h - window_size + 1), stride))
        x_positions = list(range(0, max(1, w - window_size + 1), stride))

        # make sure we cover the edges too
        if y_positions[-1] + window_size < h:
            y_positions.append(h - window_size)
        if x_positions[-1] + window_size < w:
            x_positions.append(w - window_size)

        for y in y_positions:
            for x in x_positions:
                window = image[y:y + window_size, x:x + window_size]
                aug = transform(image=window)
                img_tensor = aug["image"].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    window_probs = self.model.predict_proba(img_tensor)
                    window_probs = window_probs.squeeze(0).cpu().numpy()
                probs_final[:, y:y + window_size, x:x + window_size] += window_probs
                count_map[y:y + window_size, x:x + window_size] += 1

        # average the overlapping predictions
        count_map = np.maximum(count_map, 1)  # avoid division by zero
        probs_final = probs_final / count_map
        mask_final, probs_final = self._apply_class_masking(probs_final)
        mask_final = mask_final.astype(np.uint8)
        return mask_final, probs_final

    def predict_file(self, image_path: str, output_path: Optional[str] = None,
                     save_probs: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Predict on an image file. Handles both GeoTIFF and regular images."""
        print(f"Processing: {image_path}")
        image_path = Path(image_path)
        metadata = {}

        # try loading as GeoTIFF first (preserves geospatial info for shapefiles)
        try:
            if image_path.suffix.lower() in [".tif", ".tiff"] and HAS_RASTERIO:
                try:
                    image, metadata = load_geotiff(str(image_path))
                    image = image.transpose(1, 2, 0)
                    # normalize 16-bit images to 8-bit
                    if image.max() > 255:
                        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                except Exception:
                    # fallback to opencv if rasterio chokes on the file
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise ValueError(f"Could not load image: {image_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image: {e}")
            raise

        mask, probs = self.predict_image(image, use_sliding_window=True, window_size=512, overlap=128)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), mask)
            print(f"Saved mask to: {output_path}")
            if save_probs:
                probs_path = output_path.parent / (output_path.stem + "_probs.npy")
                np.save(probs_path, probs)
                print(f"Saved probabilities to: {probs_path}")

        return mask, probs, metadata
