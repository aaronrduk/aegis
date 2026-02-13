import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple

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
    """Inference class for SVAMITVA segmentation model."""

    def __init__(self, checkpoint_path: str, device: Optional[torch.device] = None, use_tta: bool = True):
        self.device = device if device is not None else get_device()
        self.use_tta = use_tta

        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.config = checkpoint["config"]
            self.model = SVAMITVASegmentationModel(
                num_classes=self.config["num_classes"],
                encoder=self.config["encoder"],
                encoder_weights=None,
                activation=self.config["activation"],
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully (Epoch {checkpoint['epoch']})")
            print(f"Best IoU: {checkpoint.get('best_iou', 'N/A')}")
        except FileNotFoundError:
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

    def predict_image(self, image: np.ndarray, use_sliding_window: bool = False,
                      window_size: int = 512, overlap: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """Predict segmentation mask for an image."""
        h, w = image.shape[:2]
        if use_sliding_window and (h > window_size or w > window_size):
            return self._predict_sliding_window(image, window_size, overlap)
        return self._predict_direct(image)

    def _predict_direct(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Direct prediction with optional TTA (horizontal + vertical flip)."""
        original_h, original_w = image.shape[:2]
        transform = get_inference_augmentation(self.config["input_size"])

        def _infer(img):
            aug = transform(image=img)
            tensor = aug["image"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.model.predict_proba(tensor)

        if self.use_tta:
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
        mask = np.argmax(probs, axis=0)

        if mask.shape[0] != original_h or mask.shape[1] != original_w:
            mask = cv2.resize(mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            probs_resized = np.zeros((probs.shape[0], original_h, original_w), dtype=np.float32)
            for i in range(probs.shape[0]):
                probs_resized[i] = cv2.resize(probs[i], (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            probs = probs_resized

        return mask, probs

    def _predict_sliding_window(self, image: np.ndarray, window_size: int, overlap: int) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using sliding window approach."""
        h, w = image.shape[:2]
        stride = window_size - overlap
        probs_final = np.zeros((self.config["num_classes"], h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        transform = get_inference_augmentation((window_size, window_size))

        y_positions = list(range(0, max(1, h - window_size + 1), stride))
        x_positions = list(range(0, max(1, w - window_size + 1), stride))

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

        count_map = np.maximum(count_map, 1)
        probs_final = probs_final / count_map
        mask_final = np.argmax(probs_final, axis=0).astype(np.uint8)
        return mask_final, probs_final

    def predict_file(self, image_path: str, output_path: Optional[str] = None,
                     save_probs: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Predict segmentation for an image file."""
        print(f"Processing: {image_path}")
        image_path = Path(image_path)
        metadata = {}

        try:
            if image_path.suffix.lower() in [".tif", ".tiff"] and HAS_RASTERIO:
                try:
                    image, metadata = load_geotiff(str(image_path))
                    image = image.transpose(1, 2, 0)
                    if image.max() > 255:
                        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                except Exception:
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
