"""Utility functions for artmap2dem."""

import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path
from typing import Union, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def load_geotiff(input_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """
    Load a GeoTIFF file.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the GeoTIFF file
        
    Returns
    -------
    Tuple[np.ndarray, Dict]
        Image array and rasterio profile
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        
        # Read all bands
        if src.count == 1:
            image = src.read(1)
        else:
            image = src.read()
    
    return image, profile


def save_dem(
    output_path: Union[str, Path],
    dem: np.ndarray,
    profile: Optional[Dict] = None,
    crs: Optional[str] = None,
    transform: Optional[Affine] = None,
    dtype: Optional[np.dtype] = None,
    nodata: float = -9999
) -> None:
    """
    Save a DEM to a GeoTIFF file.
    
    Parameters
    ----------
    output_path : str or Path
        Output file path
    dem : np.ndarray
        DEM array
    profile : Dict, optional
        Rasterio profile (if None, will create default)
    crs : str, optional
        Coordinate reference system
    transform : Affine, optional
        Affine transform
    dtype : np.dtype, optional
        Output data type
    nodata : float
        NoData value
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine data type
    if dtype is None:
        dtype = np.float32
    
    # Create or update profile
    if profile is None:
        profile = {
            'driver': 'GTiff',
            'height': dem.shape[0],
            'width': dem.shape[1],
            'count': 1,
            'dtype': dtype,
            'compress': 'lzw',
            'nodata': nodata,
        }
        
        if crs is not None:
            profile['crs'] = crs
        if transform is not None:
            profile['transform'] = transform
    else:
        profile = profile.copy()
        profile.update({
            'driver': 'GTiff',
            'height': dem.shape[0],
            'width': dem.shape[1],
            'count': 1,
            'dtype': dtype,
            'nodata': nodata,
        })
    
    # Write file
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(dem.astype(dtype), 1)
    
    logger.info(f"Saved DEM to {output_path}")


def get_profile(input_path: Union[str, Path]) -> Dict:
    """
    Get the profile of a GeoTIFF without loading data.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the GeoTIFF file
        
    Returns
    Returns
    -------
    Dict
        Rasterio profile
    """
    with rasterio.open(input_path) as src:
        return src.profile.copy()


def calculate_hillshade(
    dem: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    transform: Optional[Affine] = None,
    z_factor: float = 1.0
) -> np.ndarray:
    """
    Calculate hillshade from DEM.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM
    azimuth : float
        Sun azimuth angle in degrees (0-360, default: 315)
    altitude : float
        Sun altitude angle in degrees (0-90, default: 45)
    transform : Affine, optional
        Affine transform for pixel size calculation
    z_factor : float
        Vertical exaggeration factor
        
    Returns
    -------
    np.ndarray
        Hillshade array (0-255)
    """
    # Convert angles to radians
    azimuth_rad = np.radians(360.0 - azimuth + 90.0)
    altitude_rad = np.radians(altitude)
    
    # Calculate pixel size
    if transform is not None:
        dx = abs(transform[0])
        dy = abs(transform[4])
    else:
        dx = 1.0
        dy = 1.0
    
    # Calculate gradients
    gy, gx = np.gradient(dem * z_factor, dy, dx)
    
    # Slope and aspect
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gx, gy)
    
    # Hillshade calculation
    hillshade = 255.0 * (
        np.sin(altitude_rad) * np.cos(slope) +
        np.cos(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
    )
    
    # Clip to 0-255
    hillshade = np.clip(hillshade, 0, 255)
    
    return hillshade.astype(np.uint8)


def calculate_slope(
    dem: np.ndarray,
    transform: Optional[Affine] = None,
    unit: str = 'degree'
) -> np.ndarray:
    """
    Calculate slope from DEM.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM
    transform : Affine, optional
        Affine transform for pixel size calculation
    unit : str
        Output unit ('degree' or 'percent')
        
    Returns
    -------
    np.ndarray
        Slope array
    """
    # Calculate pixel size
    if transform is not None:
        dx = abs(transform[0])
        dy = abs(transform[4])
    else:
        dx = 1.0
        dy = 1.0
    
    # Calculate gradients
    gy, gx = np.gradient(dem, dy, dx)
    
    # Slope
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    
    if unit == 'degree':
        return np.degrees(slope)
    elif unit == 'percent':
        return np.tan(slope) * 100
    else:
        raise ValueError(f"Unknown unit: {unit}")


def calculate_aspect(
    dem: np.ndarray,
    transform: Optional[Affine] = None
) -> np.ndarray:
    """
    Calculate aspect (direction of slope) from DEM.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM
    transform : Affine, optional
        Affine transform for pixel size calculation
        
    Returns
    -------
    np.ndarray
        Aspect array in degrees (0-360, -1 for flat)
    """
    # Calculate pixel size
    if transform is not None:
        dx = abs(transform[0])
        dy = abs(transform[4])
    else:
        dx = 1.0
        dy = 1.0
    
    # Calculate gradients
    gy, gx = np.gradient(dem, dy, dx)
    
    # Aspect
    aspect = np.degrees(np.arctan2(-gx, -gy))
    
    # Convert to 0-360 range
    aspect = np.where(aspect < 0, 90.0 - aspect, 360.0 - aspect + 90.0)
    aspect = np.where(aspect >= 360.0, aspect - 360.0, aspect)
    
    # Flat areas have aspect -1
    slope = calculate_slope(dem, transform)
    aspect = np.where(slope < 0.1, -1, aspect)
    
    return aspect


def calculate_curvature(
    dem: np.ndarray,
    transform: Optional[Affine] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate terrain curvature.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM
    transform : Affine, optional
        Affine transform for pixel size calculation
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Profile curvature, plan curvature, and total curvature
    """
    # Calculate pixel size
    if transform is not None:
        dx = abs(transform[0])
        dy = abs(transform[4])
    else:
        dx = 1.0
        dy = 1.0
    
    # First derivatives
    gy, gx = np.gradient(dem, dy, dx)
    
    # Second derivatives
    gyy, gyx = np.gradient(gy, dy, dx)
    gxy, gxx = np.gradient(gx, dy, dx)
    
    # Slope
    p = gx**2 + gy**2
    
    # Curvatures (from Zevenbergen and Thorne, 1987)
    # Profile curvature (curvature in direction of steepest slope)
    with np.errstate(divide='ignore', invalid='ignore'):
        profile_curv = (gxx * gx**2 + 2 * gxy * gx * gy + gyy * gy**2) / \
                       (p * np.sqrt(1 + p))
    
    # Plan curvature (curvature perpendicular to direction of steepest slope)
    with np.errstate(divide='ignore', invalid='ignore'):
        plan_curv = (gxx * gy**2 - 2 * gxy * gx * gy + gyy * gx**2) / \
                    (p * np.sqrt(1 + p))
    
    # Total curvature
    total_curv = gxx + gyy
    
    # Handle flat areas
    profile_curv = np.where(p < 1e-10, 0, profile_curv)
    plan_curv = np.where(p < 1e-10, 0, plan_curv)
    
    return profile_curv, plan_curv, total_curv


def resample_dem(
    dem: np.ndarray,
    src_transform: Affine,
    dst_transform: Affine,
    dst_shape: Tuple[int, int],
    method: str = 'bilinear'
) -> np.ndarray:
    """
    Resample DEM to new resolution.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM
    src_transform : Affine
        Source affine transform
    dst_transform : Affine
        Destination affine transform
    dst_shape : Tuple[int, int]
        Destination shape (height, width)
    method : str
        Resampling method ('bilinear', 'cubic', 'nearest')
        
    Returns
    -------
    np.ndarray
        Resampled DEM
    """
    from scipy.ndimage import map_coordinates
    
    dst_h, dst_w = dst_shape
    src_h, src_w = dem.shape
    
    # Create coordinate grids for destination
    dst_y, dst_x = np.mgrid[0:dst_h, 0:dst_w]
    
    # Convert to world coordinates
    world_x = dst_transform[2] + dst_x * dst_transform[0] + dst_y * dst_transform[1]
    world_y = dst_transform[5] + dst_x * dst_transform[3] + dst_y * dst_transform[4]
    
    # Convert to source pixel coordinates
    src_x = (world_x - src_transform[2]) / src_transform[0]
    src_y = (world_y - src_transform[5]) / src_transform[4]
    
    # Resample
    order = {'nearest': 0, 'bilinear': 1, 'cubic': 3}.get(method, 1)
    resampled = map_coordinates(dem, [src_y, src_x], order=order, mode='nearest')
    
    return resampled


def fill_nodata(
    dem: np.ndarray,
    nodata_value: float = -9999,
    max_search_dist: int = 100
) -> np.ndarray:
    """
    Fill NoData values using interpolation.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM with NoData values
    nodata_value : float
        Value indicating NoData
    max_search_dist : int
        Maximum distance to search for valid values
        
    Returns
    -------
    np.ndarray
        DEM with NoData filled
    """
    from scipy.interpolate import griddata
    
    result = dem.copy()
    
    # Find NoData mask
    if np.isnan(nodata_value):
        nodata_mask = np.isnan(dem)
    else:
        nodata_mask = dem == nodata_value
    
    if not nodata_mask.any():
        return result
    
    # Get valid and invalid coordinates
    valid_y, valid_x = np.where(~nodata_mask)
    invalid_y, invalid_x = np.where(nodata_mask)
    
    # Interpolate
    valid_values = dem[valid_y, valid_x]
    
    filled = griddata(
        (valid_y, valid_x),
        valid_values,
        (invalid_y, invalid_x),
        method='linear'
    )
    
    # Fill values
    result[invalid_y, invalid_x] = filled
    
    # Handle any remaining NaN with nearest neighbor
    remaining_nan = np.isnan(result)
    if remaining_nan.any():
        remaining_y, remaining_x = np.where(remaining_nan)
        filled_nearest = griddata(
            (valid_y, valid_x),
            valid_values,
            (remaining_y, remaining_x),
            method='nearest'
        )
        result[remaining_y, remaining_x] = filled_nearest
    
    return result


def smooth_dem(
    dem: np.ndarray,
    sigma: float = 1.0,
    preserve_edges: bool = True
) -> np.ndarray:
    """
    Smooth DEM while optionally preserving edges.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM
    sigma : float
        Smoothing factor
    preserve_edges : bool
        Whether to preserve sharp edges
        
    Returns
    -------
    np.ndarray
        Smoothed DEM
    """
    from scipy.ndimage import gaussian_filter
    
    if preserve_edges:
        # Bilateral filter approximation
        smoothed = gaussian_filter(dem, sigma=sigma)
        
        # Calculate edge weights
        grad_y, grad_x = np.gradient(dem)
        edge_mag = np.sqrt(grad_y**2 + grad_x**2)
        edge_weight = np.exp(-edge_mag / edge_mag.mean())
        
        # Blend based on edge weight
        result = dem * edge_weight + smoothed * (1 - edge_weight)
        
        return result
    else:
        return gaussian_filter(dem, sigma=sigma)


def create_color_relief(
    dem: np.ndarray,
    colormap: Optional[Dict[float, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Create color relief visualization of DEM.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM
    colormap : Dict, optional
        Custom colormap as {elevation: (R, G, B)}
        
    Returns
    -------
    np.ndarray
        Color relief image (H, W, 3)
    """
    if colormap is None:
        # Default elevation colormap
        colormap = {
            -500: (0, 0, 128),      # Deep water
            0: (0, 0, 255),         # Water
            1: (0, 128, 255),       # Shallow water
            10: (0, 255, 255),      # Wetland
            50: (0, 128, 0),        # Lowland
            200: (0, 255, 0),       # Plain
            500: (128, 255, 0),     # Foothills
            1000: (255, 255, 0),    # Hills
            2000: (255, 128, 0),    # Mountains
            3000: (255, 0, 0),      # High mountains
            4000: (255, 255, 255),  # Snow
        }
    
    # Sort colormap by elevation
    elevations = sorted(colormap.keys())
    colors = [colormap[e] for e in elevations]
    
    # Create output
    h, w = dem.shape
    color_relief = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Interpolate colors
    for i in range(len(elevations) - 1):
        e1, e2 = elevations[i], elevations[i + 1]
        c1, c2 = np.array(colors[i]), np.array(colors[i + 1])
        
        # Mask for this elevation range
        mask = (dem >= e1) & (dem < e2)
        
        # Interpolate
        if mask.any():
            t = (dem[mask] - e1) / (e2 - e1)
            color_relief[mask] = (c1 * (1 - t[:, None]) + c2 * t[:, None]).astype(np.uint8)
    
    # Handle values outside range
    color_relief[dem < elevations[0]] = colors[0]
    color_relief[dem >= elevations[-1]] = colors[-1]
    
    return color_relief


def reproject_dem(
    dem: np.ndarray,
    src_crs: str,
    dst_crs: str,
    src_transform: Affine,
    dst_transform: Affine,
    dst_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Reproject DEM to new coordinate reference system.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM
    src_crs : str
        Source CRS
    dst_crs : str
        Destination CRS
    src_transform : Affine
        Source affine transform
    dst_transform : Affine
        Destination affine transform
    dst_shape : Tuple[int, int]
        Destination shape (height, width)
        
    Returns
    -------
    np.ndarray
        Reprojected DEM
    """
    try:
        from rasterio.warp import reproject, Resampling
        
        dst_dem = np.empty(dst_shape, dtype=dem.dtype)
        
        reproject(
            source=dem,
            destination=dst_dem,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        
        return dst_dem
    except ImportError:
        raise ImportError("rasterio is required for reprojection")


def get_statistics(dem: np.ndarray, nodata: Optional[float] = None) -> Dict[str, float]:
    """
    Calculate statistics for DEM.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM
    nodata : float, optional
        NoData value to exclude
        
    Returns
    -------
    Dict[str, float]
        Statistics dictionary
    """
    # Mask NoData
    if nodata is not None:
        if np.isnan(nodata):
            valid = dem[~np.isnan(dem)]
        else:
            valid = dem[dem != nodata]
    else:
        valid = dem.flatten()
    
    return {
        'min': float(valid.min()),
        'max': float(valid.max()),
        'mean': float(valid.mean()),
        'std': float(valid.std()),
        'median': float(np.median(valid)),
        'range': float(valid.max() - valid.min()),
    }
