"""Core processor for converting artistic maps to DEMs."""

import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import logging

from .terrain_analyzer import TerrainAnalyzer
from .dem_generator import DEMGenerator
from .feature_extractor import FeatureExtractor
from .hydrology import HydrologyProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArtMapProcessor:
    """
    Main processor for converting artistic georeferenced maps to DEMs.
    
    This class orchestrates the entire pipeline:
    1. Load and preprocess the artistic map
    2. Extract visual features (colors, textures, patterns)
    3. Analyze terrain characteristics
    4. Generate a believable DEM
    5. Apply hydrological corrections
    
    Parameters
    ----------
    min_elevation : float
        Minimum elevation value in meters (default: 0)
    max_elevation : float
        Maximum elevation value in meters (default: 4000)
    water_level : float
        Elevation for water bodies in meters (default: 0)
    smoothness : float
        Smoothing factor for terrain (0-1, default: 0.5)
    preserve_features : bool
        Whether to preserve sharp terrain features (default: True)
    """
    
    def __init__(
        self,
        min_elevation: float = 0.0,
        max_elevation: float = 4000.0,
        water_level: float = 0.0,
        smoothness: float = 0.5,
        preserve_features: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        self.min_elevation = min_elevation
        self.max_elevation = max_elevation
        self.water_level = water_level
        self.smoothness = smoothness
        self.preserve_features = preserve_features
        self.config = config or {}
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.terrain_analyzer = TerrainAnalyzer()
        self.dem_generator = DEMGenerator(
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            water_level=water_level,
            smoothness=smoothness
        )
        self.hydrology = HydrologyProcessor()
        
        # Store intermediate results
        self._input_image = None
        self._features = None
        self._terrain_map = None
        self._dem = None
        self._profile = None
        self._transform = None
        self._crs = None
        
    def load_map(
        self,
        input_path: Union[str, Path],
        band: Optional[int] = None
    ) -> np.ndarray:
        """
        Load an artistic map from a GeoTIFF file.
        
        Parameters
        ----------
        input_path : str or Path
            Path to the input GeoTIFF file
        band : int, optional
            Specific band to load (None loads all bands)
            
        Returns
        -------
        np.ndarray
            Loaded image array
        """
        input_path = Path(input_path)
        logger.info(f"Loading artistic map from {input_path}")
        
        with rasterio.open(input_path) as src:
            self._profile = src.profile.copy()
            self._transform = src.transform
            self._crs = src.crs
            
            if band is not None:
                self._input_image = src.read(band)
            else:
                # Read all bands and combine if necessary
                image = src.read()
                if image.shape[0] == 1:
                    self._input_image = image[0]
                else:
                    # For multi-band, keep as (bands, height, width)
                    self._input_image = image
                    
        logger.info(f"Loaded image with shape {self._input_image.shape}")
        return self._input_image
    
    def process(
        self,
        input_path: Optional[Union[str, Path]] = None,
        apply_hydrology: bool = True,
        river_channels: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Process an artistic map and generate a DEM.
        
        Parameters
        ----------
        input_path : str or Path, optional
            Path to input GeoTIFF (if not already loaded)
        apply_hydrology : bool
            Whether to apply hydrological corrections
        river_channels : np.ndarray, optional
            Binary mask of known river channels
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Generated DEM
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Load map if path provided
        if input_path is not None:
            self.load_map(input_path)
            
        if self._input_image is None:
            raise ValueError("No input image loaded. Provide input_path or call load_map() first.")
        
        # Step 1: Extract visual features
        logger.info("Extracting visual features...")
        self._features = self.feature_extractor.extract(self._input_image)
        
        # Step 2: Analyze terrain from features
        logger.info("Analyzing terrain characteristics...")
        self._terrain_map = self.terrain_analyzer.analyze(
            self._input_image,
            self._features,
            config=self.config.get('terrain_analysis', {})
        )
        
        # Step 3: Generate initial DEM
        logger.info("Generating DEM...")
        self._dem = self.dem_generator.generate(
            self._terrain_map,
            self._features,
            preserve_features=self.preserve_features
        )
        
        # Step 4: Apply hydrological corrections if requested
        if apply_hydrology:
            logger.info("Applying hydrological corrections...")
            self._dem = self.hydrology.process(
                self._dem,
                river_mask=river_channels,
                water_level=self.water_level,
                terrain_map=self._terrain_map
            )
        
        logger.info("DEM generation complete!")
        return self._dem
    
    def save_dem(
        self,
        output_path: Union[str, Path],
        dem: Optional[np.ndarray] = None,
        dtype: Optional[np.dtype] = None
    ) -> None:
        """
        Save the generated DEM to a GeoTIFF file.
        
        Parameters
        ----------
        output_path : str or Path
            Path for output GeoTIFF
        dem : np.ndarray, optional
            DEM to save (uses stored DEM if None)
        dtype : np.dtype, optional
            Output data type (default: float32)
        """
        if dem is None:
            dem = self._dem
            
        if dem is None:
            raise ValueError("No DEM to save. Run process() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine output dtype
        if dtype is None:
            dtype = np.float32
            
        # Update profile for output
        profile = self._profile.copy() if self._profile else {}
        profile.update({
            'driver': 'GTiff',
            'height': dem.shape[0],
            'width': dem.shape[1],
            'count': 1,
            'dtype': dtype,
            'crs': self._crs,
            'transform': self._transform,
            'nodata': -9999,
            'compress': 'lzw'
        })
        
        logger.info(f"Saving DEM to {output_path}")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dem.astype(dtype), 1)
            
        logger.info(f"DEM saved successfully!")
    
    def get_hillshade(
        self,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        dem: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate a hillshade from the DEM.
        
        Parameters
        ----------
        azimuth : float
            Sun azimuth angle in degrees (default: 315)
        altitude : float
            Sun altitude angle in degrees (default: 45)
        dem : np.ndarray, optional
            Input DEM (uses stored DEM if None)
            
        Returns
        -------
        np.ndarray
            Hillshade array
        """
        if dem is None:
            dem = self._dem
            
        if dem is None:
            raise ValueError("No DEM available. Run process() first.")
        
        from .utils import calculate_hillshade
        return calculate_hillshade(dem, azimuth, altitude, self._transform)
    
    def get_slope(self, dem: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate slope from the DEM.
        
        Parameters
        ----------
        dem : np.ndarray, optional
            Input DEM (uses stored DEM if None)
            
        Returns
        -------
        np.ndarray
            Slope in degrees
        """
        if dem is None:
            dem = self._dem
            
        if dem is None:
            raise ValueError("No DEM available. Run process() first.")
        
        from .utils import calculate_slope
        return calculate_slope(dem, self._transform)
    
    @property
    def dem(self) -> Optional[np.ndarray]:
        """Get the generated DEM."""
        return self._dem
    
    @property
    def terrain_map(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the terrain analysis results."""
        return self._terrain_map
    
    @property
    def features(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the extracted features."""
        return self._features
