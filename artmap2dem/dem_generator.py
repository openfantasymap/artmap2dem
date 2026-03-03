"""DEM generation module for creating elevation models."""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter, distance_transform_edt
from scipy.interpolate import griddata
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import gaussian
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DEMGenerator:
    """
    Generate Digital Elevation Models from terrain analysis.
    
    This class creates realistic DEMs by:
    1. Using terrain probabilities to guide elevation
    2. Adding fractal noise for natural variation
    3. Enforcing hydrological consistency
    4. Preserving sharp terrain features
    """
    
    def __init__(
        self,
        min_elevation: float = 0.0,
        max_elevation: float = 4000.0,
        water_level: float = 0.0,
        smoothness: float = 0.5,
        noise_octaves: int = 6,
        noise_persistence: float = 0.5,
        seed: Optional[int] = None
    ):
        self.min_elevation = min_elevation
        self.max_elevation = max_elevation
        self.water_level = water_level
        self.smoothness = smoothness
        self.noise_octaves = noise_octaves
        self.noise_persistence = noise_persistence
        self.seed = seed
        
    def generate(
        self,
        terrain_map: Dict[str, np.ndarray],
        features: Dict[str, Dict[str, np.ndarray]],
        preserve_features: bool = True
    ) -> np.ndarray:
        """
        Generate a DEM from terrain analysis.
        
        Parameters
        ----------
        terrain_map : Dict[str, np.ndarray]
            Terrain analysis results from TerrainAnalyzer
        features : Dict[str, Dict[str, np.ndarray]]
            Extracted features from FeatureExtractor
        preserve_features : bool
            Whether to preserve sharp terrain features
            
        Returns
        -------
        np.ndarray
            Generated DEM
        """
        h, w = terrain_map['base_elevation'].shape
        
        logger.info(f"Generating DEM of size {h}x{w}")
        
        # Start with base elevation
        dem = terrain_map['base_elevation'].copy()
        
        # Add terrain-specific detail
        dem = self._add_terrain_detail(dem, terrain_map, features)
        
        # Add fractal noise for natural variation
        dem = self._add_fractal_noise(dem, terrain_map)
        
        # Enforce feature preservation if requested
        if preserve_features:
            dem = self._preserve_features(dem, terrain_map, features)
        
        # Apply smoothness
        dem = self._apply_smoothness(dem, terrain_map)
        
        # Scale to elevation range
        dem = self._scale_elevation(dem, terrain_map)
        
        # Set water areas
        dem = self._apply_water_level(dem, terrain_map)
        
        return dem
    
    def _add_terrain_detail(
        self,
        dem: np.ndarray,
        terrain_map: Dict[str, np.ndarray],
        features: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Add detail specific to each terrain type."""
        h, w = dem.shape
        
        # Get terrain probabilities
        probs = terrain_map['terrain_probabilities']
        
        # Mountain detail - add peaks and ridges
        mountain_mask = probs['peak'] + probs['ridge']
        if mountain_mask.max() > 0:
            mountain_detail = self._generate_mountain_detail(
                mountain_mask, 
                features['edges'].get('ridges', np.zeros((h, w)))
            )
            dem += mountain_detail * mountain_mask * 0.3
        
        # Valley detail - create depressions
        valley_mask = probs['valley']
        if valley_mask.max() > 0:
            valley_detail = -self._generate_valley_detail(valley_mask)
            dem += valley_detail * valley_mask * 0.2
        
        # Cliff detail - sharp changes
        cliff_mask = probs['cliff']
        if cliff_mask.max() > 0:
            cliff_detail = self._generate_cliff_detail(cliff_mask, features)
            dem += cliff_detail * cliff_mask * 0.25
        
        # Slope variation
        slope_mask = probs['steep_slope'] + probs['moderate_slope'] + probs['gentle_slope']
        if slope_mask.max() > 0:
            slope_detail = self._generate_slope_detail(slope_mask, features)
            dem += slope_detail * slope_mask * 0.15
        
        return np.clip(dem, 0, 1)
    
    def _generate_mountain_detail(
        self,
        mask: np.ndarray,
        ridges: np.ndarray
    ) -> np.ndarray:
        """Generate mountain peak and ridge detail."""
        h, w = mask.shape
        
        # Enhance ridges
        ridge_enhanced = ridges / (ridges.max() + 1e-10)
        
        # Add spiky noise for peaks
        peak_noise = np.random.exponential(0.5, (h, w))
        peak_noise = gaussian_filter(peak_noise, sigma=2)
        
        # Combine
        detail = ridge_enhanced * 0.6 + peak_noise * 0.4
        
        return gaussian_filter(detail, sigma=1)
    
    def _generate_valley_detail(self, mask: np.ndarray) -> np.ndarray:
        """Generate valley depression detail."""
        h, w = mask.shape
        
        # Create channel-like depressions
        distance = distance_transform_edt(mask < 0.5)
        
        # Valley depth decreases away from center
        depth = np.exp(-distance / 20)
        
        return gaussian_filter(depth, sigma=2)
    
    def _generate_cliff_detail(
        self,
        mask: np.ndarray,
        features: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Generate cliff/step detail."""
        h, w = mask.shape
        
        # Use edge information
        edges = features['edges']['sobel_magnitude']
        
        # Create step-like features
        step = np.zeros((h, w))
        step[edges > edges.mean() + edges.std()] = 1
        
        # Smooth steps
        step = gaussian_filter(step, sigma=0.5)
        
        return step
    
    def _generate_slope_detail(
        self,
        mask: np.ndarray,
        features: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Generate slope variation detail."""
        h, w = mask.shape
        
        # Use texture for slope variation
        texture = features['texture'].get('complexity', np.zeros((h, w)))
        
        # Normalize
        texture = texture / (texture.max() + 1e-10)
        
        return gaussian_filter(texture, sigma=3)
    
    def _add_fractal_noise(
        self,
        dem: np.ndarray,
        terrain_map: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Add fractal (multi-octave) noise for natural terrain."""
        h, w = dem.shape
        
        # Generate fractal noise
        noise = self._generate_fractal_noise(h, w)
        
        # Modulate noise by terrain type
        # Mountains get more high-frequency noise
        # Plains get smoother noise
        probs = terrain_map['terrain_probabilities']
        
        roughness = (
            probs['peak'] * 1.0 +
            probs['ridge'] * 0.8 +
            probs['cliff'] * 0.7 +
            probs['steep_slope'] * 0.6 +
            probs['moderate_slope'] * 0.4 +
            probs['gentle_slope'] * 0.2 +
            probs['flat'] * 0.1 +
            probs['valley'] * 0.3 +
            probs['water'] * 0.0
        )
        
        # Apply modulated noise
        noise_modulated = noise * (0.3 + 0.7 * roughness)
        
        # Blend with DEM
        result = dem * 0.8 + noise_modulated * 0.2
        
        return np.clip(result, 0, 1)
    
    def _generate_fractal_noise(self, h: int, w: int) -> np.ndarray:
        """Generate fractal Brownian motion noise."""
        noise = np.zeros((h, w), dtype=np.float32)
        
        amplitude = 1.0
        frequency = 1.0
        
        for octave in range(self.noise_octaves):
            # Generate noise at this octave
            octave_h = int(h * frequency)
            octave_w = int(w * frequency)
            
            if octave_h < 2 or octave_w < 2:
                break
            
            # Random noise
            np.random.seed(self.seed + octave if self.seed else None)
            octave_noise = np.random.randn(octave_h, octave_w)
            
            # Upsample to full size
            from scipy.ndimage import zoom
            zoom_factor = (h / octave_h, w / octave_w)
            upsampled = zoom(octave_noise, zoom_factor, order=1)
            
            # Add to accumulation
            noise += upsampled[:h, :w] * amplitude
            
            # Update for next octave
            amplitude *= self.noise_persistence
            frequency *= 2
        
        # Normalize
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
        
        return noise
    
    def _preserve_features(
        self,
        dem: np.ndarray,
        terrain_map: Dict[str, np.ndarray],
        features: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Preserve sharp terrain features like ridges and cliffs."""
        h, w = dem.shape
        
        # Identify feature locations
        edges = features['edges']
        
        # Ridge preservation
        ridges = edges.get('ridges', np.zeros((h, w)))
        ridge_mask = ridges > ridges.mean() + ridges.std()
        
        # Cliff preservation
        cliff_mask = terrain_map['terrain_probabilities']['cliff'] > 0.5
        
        # Combine feature masks
        feature_mask = ridge_mask | cliff_mask
        
        # Create feature-enhanced DEM
        feature_dem = dem.copy()
        
        # Enhance gradients at feature locations
        grad_y, grad_x = np.gradient(dem)
        grad_mag = np.sqrt(grad_y ** 2 + grad_x ** 2)
        
        # Sharpen features
        sharpening = feature_mask.astype(float) * 0.3
        feature_dem += grad_mag * sharpening
        
        # Blend original and feature-enhanced
        result = dem * (1 - feature_mask.astype(float) * 0.3) + \
                 feature_dem * feature_mask.astype(float) * 0.3
        
        return np.clip(result, 0, 1)
    
    def _apply_smoothness(
        self,
        dem: np.ndarray,
        terrain_map: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Apply adaptive smoothing based on terrain type."""
        probs = terrain_map['terrain_probabilities']
        
        # Different smoothness for different terrain
        # Water - very smooth
        # Flat - smooth
        # Mountains - less smooth
        
        smooth_sigma = (
            probs['water'] * 4.0 +
            probs['flat'] * 3.0 +
            probs['gentle_slope'] * 2.0 +
            probs['moderate_slope'] * 1.0 +
            probs['steep_slope'] * 0.5 +
            probs['cliff'] * 0.3 +
            probs['peak'] * 0.5 +
            probs['ridge'] * 0.5 +
            probs['valley'] * 1.5
        )
        
        # Apply variable smoothing
        result = np.zeros_like(dem)
        
        # Multiple smoothing levels
        for sigma in [0.5, 1.0, 2.0, 4.0]:
            smoothed = gaussian_filter(dem, sigma=sigma)
            weight = np.exp(-((smooth_sigma - sigma) ** 2) / (2 * (sigma ** 2)))
            result += smoothed * weight
        
        # Normalize by weights
        weight_sum = sum(
            np.exp(-((smooth_sigma - s) ** 2) / (2 * (s ** 2)))
            for s in [0.5, 1.0, 2.0, 4.0]
        )
        
        result = result / (weight_sum + 1e-10)
        
        # Blend with original based on global smoothness parameter
        result = dem * (1 - self.smoothness) + result * self.smoothness
        
        return result
    
    def _scale_elevation(
        self,
        dem: np.ndarray,
        terrain_map: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Scale normalized elevation to actual elevation range."""
        # Get elevation ranges for each terrain type
        ranges = terrain_map['elevation_ranges']
        probs = terrain_map['terrain_probabilities']
        
        # Calculate weighted elevation range
        min_elev = sum(
            probs[t] * r[0] for t, r in ranges.items()
        )
        max_elev = sum(
            probs[t] * r[1] for t, r in ranges.items()
        )
        
        # Blend with global range
        global_min = 0.0
        global_max = 1.0
        
        # Scale DEM
        scaled = dem * (self.max_elevation - self.min_elevation) + self.min_elevation
        
        return scaled
    
    def _apply_water_level(
        self,
        dem: np.ndarray,
        terrain_map: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Apply water level to water bodies."""
        water_mask = terrain_map['water_mask']
        
        # Set water areas to water level
        dem = np.where(water_mask, self.water_level, dem)
        
        # Add slight variation to water (depth)
        # Deeper water slightly lower
        water_prob = terrain_map['terrain_probabilities']['water']
        depth_variation = water_prob * 2.0  # Up to 2m depth variation
        dem = np.where(water_mask, self.water_level - depth_variation, dem)
        
        return dem
    
    def refine_dem(
        self,
        dem: np.ndarray,
        terrain_map: Dict[str, np.ndarray],
        iterations: int = 3
    ) -> np.ndarray:
        """
        Refine DEM through iterative improvement.
        
        Parameters
        ----------
        dem : np.ndarray
            Initial DEM
        terrain_map : Dict[str, np.ndarray]
            Terrain analysis
        iterations : int
            Number of refinement iterations
            
        Returns
        -------
        np.ndarray
            Refined DEM
        """
        refined = dem.copy()
        
        for i in range(iterations):
            # Smooth while preserving features
            refined = self._feature_preserving_smoothing(refined, terrain_map)
            
            # Enforce slope constraints
            refined = self._enforce_slope_constraints(refined, terrain_map)
        
        return refined
    
    def _feature_preserving_smoothing(
        self,
        dem: np.ndarray,
        terrain_map: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Apply smoothing that preserves terrain features."""
        # Bilateral filter approximation
        from scipy.ndimage import gaussian_filter
        
        # Smooth elevation
        smoothed = gaussian_filter(dem, sigma=1.0)
        
        # Smooth gradient magnitude (edge indicator)
        grad_y, grad_x = np.gradient(dem)
        grad_mag = np.sqrt(grad_y ** 2 + grad_x ** 2)
        smoothed_grad = gaussian_filter(grad_mag, sigma=1.0)
        
        # Weight by gradient (less smoothing at edges)
        edge_weight = np.exp(-smoothed_grad / smoothed_grad.mean())
        
        result = dem * edge_weight + smoothed * (1 - edge_weight)
        
        return result
    
    def _enforce_slope_constraints(
        self,
        dem: np.ndarray,
        terrain_map: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Enforce maximum slope constraints."""
        # Calculate current slopes
        grad_y, grad_x = np.gradient(dem)
        slope = np.sqrt(grad_y ** 2 + grad_x ** 2)
        
        # Maximum slope per terrain type (in elevation units per pixel)
        max_slopes = {
            'flat': 0.5,
            'gentle_slope': 2.0,
            'moderate_slope': 5.0,
            'steep_slope': 15.0,
            'cliff': 50.0,
            'peak': 20.0,
            'ridge': 15.0,
            'valley': 10.0,
            'water': 0.1,
        }
        
        probs = terrain_map['terrain_probabilities']
        
        # Weighted maximum slope
        max_slope = sum(
            probs.get(t, 0) * s for t, s in max_slopes.items()
        )
        
        # Scale slopes that exceed maximum
        scale = np.minimum(1.0, max_slope / (slope + 1e-10))
        
        # Apply scaling to gradients
        grad_x_scaled = grad_x * scale
        grad_y_scaled = grad_y * scale
        
        # Reconstruct DEM from scaled gradients (simplified)
        # This is a rough approximation - full Poisson reconstruction would be better
        result = dem.copy()
        result[1:, :] -= (grad_y_scaled[1:, :] - grad_y[1:, :]) * 0.5
        result[:, 1:] -= (grad_x_scaled[:, 1:] - grad_x[:, 1:]) * 0.5
        
        return result
