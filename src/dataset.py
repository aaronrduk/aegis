"""
Dataset class for SVAMITVA drone imagery
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset

# Optional import
try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None


class SVAMITVADataset(Dataset):
    """
    PyTorch Dataset for SVAMITVA drone imagery and segmentation masks
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (512, 512),
    ):
        """
        Args:
            image_dir: Directory containing drone images (TIF/JPEG)
            mask_dir: Directory containing segmentation masks (PNG)
            transform: Albumentations transformation pipeline
            image_size: Target image size (H, W)
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.image_size = image_size

        # Supported image formats
        self.image_extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]

        # Get list of image files
        self.image_files = self._get_files(self.image_dir)

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"Found {len(self.image_files)} images in {self.image_dir}")

    def _get_files(self, directory: Path) -> List[Path]:
        """Get all image files from directory"""
        files = []
        for ext in self.image_extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))
        return sorted(files)

    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load image from file
        Supports TIF (with rasterio) and JPEG/PNG (with OpenCV)
        """
        if image_path.suffix.lower() in [".tif", ".tiff"] and HAS_RASTERIO:
            # Load with rasterio for geospatial TIF
            try:
                with rasterio.open(image_path) as src:
                    # Read RGB bands (assuming bands 1,2,3 are RGB)
                    image = src.read([1, 2, 3]).transpose(1, 2, 0)

                    # Normalize to 0-255 if needed
                    if image.max() > 255:
                        image = (
                            (image - image.min()) / (image.max() - image.min()) * 255
                        ).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
            except Exception as e:
                print(f"Error loading TIF with rasterio: {e}. Fallback to OpenCV.")
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Load with OpenCV for JPEG/PNG or if rasterio missing
            image = cv2.imread(str(image_path))
            # Handle TIFs loaded by OpenCV which might be BGR or BGRA
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Check channels
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """
        Load segmentation mask
        Expects single-channel PNG with class indices (0-9)
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return mask

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        """
        Get item by index

        Returns:
            Dictionary with 'image' and optionally 'mask'
        """
        # Load image
        image_path = self.image_files[idx]
        image = self._load_image(image_path)

        # Load mask if available
        mask = None
        if self.mask_dir is not None:
            # Construct mask filename (same name as image but .png)
            mask_name = image_path.stem + ".png"
            mask_path = self.mask_dir / mask_name

            if mask_path.exists():
                mask = self._load_mask(mask_path)
            else:
                # Create dummy mask if not found (for inference)
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Apply transformations
        if self.transform is not None:
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            else:
                transformed = self.transform(image=image)
                image = transformed["image"]

        # Prepare output
        sample = {"image": image, "filename": image_path.name}

        if mask is not None:
            sample["mask"] = mask

        return sample


def get_training_augmentation(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """
    Get training augmentation pipeline

    Args:
        image_size: Target image size (H, W)

    Returns:
        Albumentations composition
    """
    train_transform = A.Compose(
        [
            # Resize
            A.Resize(height=image_size[0], width=image_size[1]),
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            # Elastic deformation
            A.ElasticTransform(
                alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3
            ),
            # Distortions
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(p=0.2),
            # Color augmentations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
            ),
            # Blur and noise
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=7, p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                ],
                p=0.2,
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            # Normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Convert to tensor
            ToTensorV2(),
        ]
    )

    return train_transform


def get_validation_augmentation(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """
    Get validation augmentation pipeline (resize + normalize only)

    Args:
        image_size: Target image size (H, W)

    Returns:
        Albumentations composition
    """
    val_transform = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    return val_transform


def get_inference_augmentation(
    image_size: Optional[Tuple[int, int]] = None,
) -> A.Compose:
    """
    Get inference augmentation pipeline

    Args:
        image_size: Target image size (H, W), None to keep original size

    Returns:
        Albumentations composition
    """
    transforms = []

    if image_size is not None:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1]))

    transforms.extend(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)



