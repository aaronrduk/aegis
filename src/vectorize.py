"""
Vectorization module — converts raster segmentation masks to vector shapefiles.

This is what makes our output actually useful for GIS software.
The pipeline: pixel mask -> polygons (via rasterio) -> simplify -> GeoDataFrame -> .shp

Shapefiles are the standard format that government agencies use, which is
why we're outputting these instead of just PNGs.

TODO: add GeoJSON output option too — it's more modern and easier to work with
TODO: try with larger simplify_tolerance for faster loading in QGIS

Digital University Kerala (DUK)
"""

import os
import numpy as np
from typing import List, Optional

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon, Point
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
    try:
        from config import CLASS_NAMES
    except ImportError:
        CLASS_NAMES = [f"Class_{i}" for i in range(10)]


def simplify_polygon(polygon, tolerance: float = 1.0):
    """Simplify polygon geometry using Douglas-Peucker algorithm.
    
    This reduces the number of vertices which makes the shapefiles
    way smaller and faster to render in GIS software.
    """
    if polygon is None:
        return None
    return polygon.simplify(tolerance, preserve_topology=True)


def mask_to_polygons(mask: np.ndarray, class_idx: int, transform: Optional[object] = None,
                     simplify_tolerance: float = 1.0) -> List:
    """Convert a binary class mask to a list of polygons.
    
    rasterio.features.shapes does the heavy lifting here — it traces
    the boundaries of connected regions and returns them as GeoJSON-like dicts.
    """
    if not HAS_GEOPANDAS or not HAS_RASTERIO:
        print("Warning: geopandas/rasterio not installed. Cannot vectorize.")
        return []

    binary_mask = (mask == class_idx).astype(np.uint8)

    # if no geospatial transform is given, just use pixel coordinates
    if transform is None:
        h, w = mask.shape
        transform = rasterio.transform.from_bounds(0, 0, w, h, w, h)

    polygons = []
    for geom, value in shapes(binary_mask, transform=transform):
        if value == 1:
            poly = Polygon(geom["coordinates"][0])
            if simplify_tolerance > 0:
                poly = simplify_polygon(poly, simplify_tolerance)
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)

    return polygons


def create_geodataframe(mask: np.ndarray, class_names: List[str], transform: Optional[object] = None,
                        crs: Optional[str] = None, simplify_tolerance: float = 1.0) -> Optional[object]:
    """Create a GeoDataFrame with all detected features from the mask."""
    if not HAS_GEOPANDAS:
        return None

    features = []
    for class_idx in np.unique(mask):
        if class_idx == 0:
            continue  # skip background
        polygons = mask_to_polygons(mask, class_idx, transform=transform, simplify_tolerance=simplify_tolerance)
        for poly_idx, poly in enumerate(polygons):
            if poly is None or poly.is_empty:
                continue
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
            features.append({
                "geometry": poly,
                "class_id": int(class_idx),
                "class_name": class_name,
                "area": poly.area,
                "perimeter": poly.length,
                "feature_id": f"{class_name}_{poly_idx}",
            })

    if features:
        return gpd.GeoDataFrame(features, crs=crs)

    # return empty GeoDataFrame with correct schema if nothing was found
    return gpd.GeoDataFrame(
        columns=["geometry", "class_id", "class_name", "area", "perimeter", "feature_id"],
        crs=crs,
    )


def save_shapefile(gdf, output_path: str, driver: str = "ESRI Shapefile"):
    """Save a GeoDataFrame as a shapefile."""
    if gdf is None:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        gdf.to_file(output_path, driver=driver)
        print(f"Saved shapefile: {output_path} ({len(gdf)} features)")
    except Exception as e:
        print(f"Error saving shapefile: {e}")


def create_class_shapefile(mask: np.ndarray, class_idx: int, class_name: str, output_path: str,
                           transform: Optional[object] = None, crs: Optional[str] = None,
                           simplify_tolerance: float = 1.0):
    """Create a separate shapefile for a single class — cleaner for analysis."""
    if not HAS_GEOPANDAS:
        return

    polygons = mask_to_polygons(mask, class_idx, transform=transform, simplify_tolerance=simplify_tolerance)
    if not polygons:
        return

    features = [{
        "geometry": poly,
        "class_id": int(class_idx),
        "class_name": class_name,
        "area": poly.area,
        "perimeter": poly.length,
        "feature_id": f"{class_name}_{i}",
    } for i, poly in enumerate(polygons)]

    gdf = gpd.GeoDataFrame(features, crs=crs)
    save_shapefile(gdf, output_path)


def mask_to_shapefiles(mask: np.ndarray, output_dir: str, base_name: str,
                       class_names: Optional[list] = None, transform: Optional[object] = None,
                       crs: Optional[str] = None, simplify_tolerance: float = 1.0,
                       separate_classes: bool = True):
    """Convert a full segmentation mask to shapefiles.
    
    Can output either one shapefile per class (easier to work with in QGIS)
    or a single combined shapefile with a class_name column.
    """
    if not HAS_GEOPANDAS or not HAS_RASTERIO:
        print("Error: Missing geospatial libraries (geopandas/rasterio). Skipping shapefile generation.")
        return

    os.makedirs(output_dir, exist_ok=True)
    if class_names is None:
        class_names = CLASS_NAMES

    if separate_classes:
        for class_idx in np.unique(mask):
            if class_idx == 0:
                continue
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
            output_path = os.path.join(output_dir, f"{base_name}_{class_name}.shp")
            create_class_shapefile(mask, class_idx, class_name, output_path,
                                   transform=transform, crs=crs, simplify_tolerance=simplify_tolerance)
    else:
        gdf = create_geodataframe(mask, class_names, transform=transform, crs=crs,
                                  simplify_tolerance=simplify_tolerance)
        output_path = os.path.join(output_dir, f"{base_name}_combined.shp")
        save_shapefile(gdf, output_path)
