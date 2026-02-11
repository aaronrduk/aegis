"""
Batch processing script for multiple images
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from src.inference import SVAMITVAInference
from src.postprocess import postprocess_multiclass_mask
from src.vectorize import mask_to_shapefiles
from src.config import CLASS_NAMES, POSTPROCESS_CONFIG


def batch_process(
    checkpoint_path: str,
    input_dir: str,
    output_dir: str,
    use_tta: bool = True,
    apply_postprocess: bool = True,
    generate_shapefiles: bool = True,
):
    """
    Process multiple images in batch

    Args:
        checkpoint_path: Path to model checkpoint
        input_dir: Directory containing input images
        output_dir: Directory for outputs
        use_tta: Use test-time augmentation
        apply_postprocess: Apply post-processing
        generate_shapefiles: Generate shapefiles for each image
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directories
    masks_dir = output_dir / "masks"
    shapefiles_dir = output_dir / "shapefiles"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if generate_shapefiles:
        shapefiles_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = SVAMITVAInference(checkpoint_path, use_tta=use_tta)

    # Find all images
    image_extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)

    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process each image
    results = []

    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Predict
            mask, probs, metadata = model.predict_file(
                str(image_path), output_path=None
            )

            # Apply post-processing
            if apply_postprocess:
                mask = postprocess_multiclass_mask(
                    mask, POSTPROCESS_CONFIG["min_area"], len(CLASS_NAMES)
                )

            # Save mask
            import cv2

            mask_path = masks_dir / f"{image_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), mask)

            # Generate shapefiles
            if generate_shapefiles:
                mask_to_shapefiles(
                    mask,
                    output_dir=str(shapefiles_dir / image_path.stem),
                    base_name=image_path.stem,
                    class_names=CLASS_NAMES,
                    transform=metadata.get("transform", None),
                    crs=metadata.get("crs", None),
                    simplify_tolerance=1.0,
                    separate_classes=True,
                )

            # Collect statistics
            import numpy as np

            unique, counts = np.unique(mask, return_counts=True)
            stats = {
                "image": image_path.name,
                "status": "success",
                "classes_detected": len(unique) - 1,  # Exclude background
            }

            for class_idx, count in zip(unique, counts):
                if class_idx > 0:
                    stats[f"{CLASS_NAMES[class_idx]}_pixels"] = int(count)

            results.append(stats)

        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
            results.append(
                {"image": image_path.name, "status": "failed", "error": str(e)}
            )

    # Save results summary
    results_df = pd.DataFrame(results)
    results_path = output_dir / "processing_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nProcessing complete!")
    print(f"Processed: {len(image_files)} images")
    print(f"Successful: {len([r for r in results if r['status'] == 'success'])}")
    print(f"Failed: {len([r for r in results if r['status'] == 'failed'])}")
    print(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch process drone images")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory for outputs"
    )
    parser.add_argument(
        "--use_tta", action="store_true", help="Use test-time augmentation"
    )
    parser.add_argument(
        "--no_postprocess", action="store_true", help="Skip post-processing"
    )
    parser.add_argument(
        "--no_shapefiles", action="store_true", help="Skip shapefile generation"
    )

    args = parser.parse_args()

    batch_process(
        checkpoint_path=args.checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_tta=args.use_tta,
        apply_postprocess=not args.no_postprocess,
        generate_shapefiles=not args.no_shapefiles,
    )


if __name__ == "__main__":
    main()
