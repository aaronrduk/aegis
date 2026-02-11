"""
Vectorization module - Convert raster masks to vector shapefiles
"""

import numpy as np
from typing import List, Optional

# Optional imports
try:
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon, Point
    # from shapely.ops import unary_union
    import fiona

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None
    Polygon = None
    MultiPolygon = None
    Point = None
    fiona = None

try:
    import rasterio
    from rasterio.features import shapes

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None
    shapes = None

try:
    from .config import CLASS_NAMES
except ImportError:
    # Fallback for direct script execution from the src/ directory
    from config import CLASS_NAMES


def simplify_polygon(polygon, tolerance: float = 1.0):
    """
    Simplify polygon using Douglas-Peucker algorithm
    """
    if polygon is None:
        return None
    return polygon.simplify(tolerance, preserve_topology=True)


def mask_to_polygons(
    mask: np.ndarray,
    class_idx: int,
    transform: Optional[object] = None,
    simplify_tolerance: float = 1.0,
) -> List:
    """
    Convert binary mask to polygons
    """
    if not HAS_GEOPANDAS or not HAS_RASTERIO:
        print("Warning: geopandas/rasterio not installed. Cannot vectorize.")
        return []

    # Create binary mask for class
    binary_mask = (mask == class_idx).astype(np.uint8)

    if transform is None:
        # Default pixel-based transform
        h, w = mask.shape
        transform = rasterio.transform.from_bounds(0, 0, w, h, w, h)

    # Extract shapes from raster
    polygons = []
    for geom, value in shapes(binary_mask, transform=transform):
        if value == 1:  # Only extract class regions
            poly = Polygon(geom["coordinates"][0])

            # Simplify polygon
            if simplify_tolerance > 0:
                poly = simplify_polygon(poly, simplify_tolerance)

            # Only keep valid polygons with area > 0
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)

    return polygons


def create_geodataframe(
    mask: np.ndarray,
    class_names: List[str],
    transform: Optional[object] = None,
    crs: Optional[str] = None,
    simplify_tolerance: float = 1.0,
) -> Optional[object]:
    """
    Create GeoDataFrame from segmentation mask
    """
    if not HAS_GEOPANDAS:
        return None

    features = []

    # Process each class
    unique_classes = np.unique(mask)

    for class_idx in unique_classes:
        if class_idx == 0:  # Skip background
            continue

        # Extract polygons for this class
        polygons = mask_to_polygons(
            mask, class_idx, transform=transform, simplify_tolerance=simplify_tolerance
        )

        # Add to features
        for poly_idx, poly in enumerate(polygons):
            # Safe checking for shapely geometry in case simplified to None
            if poly is None or poly.is_empty:
                continue

            features.append(
                {
                    "geometry": poly,
                    "class_id": int(class_idx),
                    "class_name": (
                        class_names[class_idx]
                        if class_idx < len(class_names)
                        else f"Class_{class_idx}"
                    ),
                    "area": poly.area,
                    "perimeter": poly.length,
                    "feature_id": (
                        f"{class_names[class_idx]}_{poly_idx}"
                        if class_idx < len(class_names)
                        else f"Class_{class_idx}_{poly_idx}"
                    ),
                }
            )

    # Create GeoDataFrame
    if len(features) > 0:
        gdf = gpd.GeoDataFrame(features, crs=crs)
    else:
        # Empty GeoDataFrame
        gdf = gpd.GeoDataFrame(
            columns=[
                "geometry",
                "class_id",
                "class_name",
                "area",
                "perimeter",
                "feature_id",
            ],
            crs=crs,
        )

    return gdf


def save_shapefile(gdf, output_path: str, driver: str = "ESRI Shapefile"):
    """
    Save GeoDataFrame as shapefile
    """
    if gdf is None:
        return

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save shapefile
    try:
        gdf.to_file(output_path, driver=driver)
        print(f"Saved shapefile: {output_path}")
        print(f"Features: {len(gdf)}")
    except Exception as e:
        print(f"Error saving shapefile: {e}")


def create_class_shapefile(
    mask: np.ndarray,
    class_idx: int,
    class_name: str,
    output_path: str,
    transform: Optional[object] = None,
    crs: Optional[str] = None,
    simplify_tolerance: float = 1.0,
):
    """
    Create shapefile for a single class
    """
    if not HAS_GEOPANDAS:
        return

    # Extract polygons
    polygons = mask_to_polygons(
        mask, class_idx, transform=transform, simplify_tolerance=simplify_tolerance
    )

    if len(polygons) == 0:
        # print(f"No features found for class {class_name}")
        return

    # Create features
    features = []
    for poly_idx, poly in enumerate(polygons):
        features.append(
            {
                "geometry": poly,
                "class_id": int(class_idx),
                "class_name": class_name,
                "area": poly.area,
                "perimeter": poly.length,
                "feature_id": f"{class_name}_{poly_idx}",
            }
        )

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(features, crs=crs)

    # Save shapefile
    save_shapefile(gdf, output_path)


def mask_to_shapefiles(
    mask: np.ndarray,
    output_dir: str,
    base_name: str,
    class_names: Optional[list] = None,
    transform: Optional[object] = None,
    crs: Optional[str] = None,
    simplify_tolerance: float = 1.0,
    separate_classes: bool = True,
):
    """
    Convert mask to shapefiles (one per class or combined)
    """
    if not HAS_GEOPANDAS or not HAS_RASTERIO:
        print("Error: Missing geospatial libraries (geopandas/rasterio).")
        print("Skipping shapefile generation.")
        return

    import os
    os.makedirs(output_dir, exist_ok=True)

    if class_names is None:
        class_names = CLASS_NAMES

    if separate_classes:
        # Create separate shapefile for each class
        unique_classes = np.unique(mask)

        for class_idx in unique_classes:
            if class_idx == 0:  # Skip background
                continue

            class_name = (
                class_names[class_idx]
                if class_idx < len(class_names)
                else f"Class_{class_idx}"
            )
            output_path = os.path.join(output_dir, f"{base_name}_{class_name}.shp")

            create_class_shapefile(
                mask,
                class_idx,
                class_name,
                str(output_path),
                transform=transform,
                crs=crs,
                simplify_tolerance=simplify_tolerance,
            )
    else:
        # Create combined shapefile
        gdf = create_geodataframe(
            mask,
            class_names,
            transform=transform,
            crs=crs,
            simplify_tolerance=simplify_tolerance,
        )

        output_path = os.path.join(output_dir, f"{base_name}_combined.shp")
        save_shapefile(gdf, str(output_path))



