import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None


class SVAMITVADataset(Dataset):
    """PyTorch Dataset for SVAMITVA drone imagery and segmentation masks."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (512, 512),
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.image_size = image_size
        self.image_extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]
        self.image_files = self._get_files(self.image_dir)

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        print(f"Found {len(self.image_files)} images in {self.image_dir}")

    def _get_files(self, directory: Path) -> List[Path]:
        files = []
        for ext in self.image_extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))
        return sorted(files)

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image with rasterio (TIF) or OpenCV fallback."""
        if image_path.suffix.lower() in [".tif", ".tiff"] and HAS_RASTERIO:
            try:
                with rasterio.open(image_path) as src:
                    image = src.read([1, 2, 3]).transpose(1, 2, 0)
                    if image.max() > 255:
                        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                return image
            except Exception:
                pass

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load single-channel segmentation mask with class indices."""
        return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_files[idx]
        image = self._load_image(image_path)

        mask = None
        if self.mask_dir is not None:
            mask_name = image_path.stem + ".png"
            mask_path = self.mask_dir / mask_name
            if mask_path.exists():
                mask = self._load_mask(mask_path)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.transform is not None:
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            else:
                transformed = self.transform(image=image)
                image = transformed["image"]

        sample = {"image": image, "filename": image_path.name}
        if mask is not None:
            sample["mask"] = mask
        return sample


def get_training_augmentation(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """Training augmentation pipeline with strong augmentations."""
    return A.Compose([
        A.Resize(height=576, width=576),
        A.RandomCrop(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5,
        ),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        A.GridDistortion(p=0.2),
        A.OpticalDistortion(p=0.2),
        A.CLAHE(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_validation_augmentation(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """Validation augmentation: resize + normalize only."""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_inference_augmentation(image_size: Optional[Tuple[int, int]] = None) -> A.Compose:
    """Inference augmentation pipeline."""
    transforms = []
    if image_size is not None:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1]))
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return A.Compose(transforms)
