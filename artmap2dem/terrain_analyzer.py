"""Terrain analysis module for interpreting artistic map features."""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TerrainAnalyzer:
    """
    Analyze artistic map features to identify terrain characteristics.
    
    This class interprets visual features (colors, edges, textures) as terrain types:
    - Mountains (dark colors, rough textures, ridge lines)
    - Hills (medium colors, moderate textures)
    - Plains (light colors, smooth textures)
    - Valleys (linear features, converging patterns)
    - Water bodies (uniform colors, smooth textures)
    - Forests (textured patterns, specific colors)
    """
    
    def __init__(self):
        self.terrain_classes = {
            'water': 0,
            'flat': 1,
            'gentle_slope': 2,
            'moderate_slope': 3,
            'steep_slope': 4,
            'cliff': 5,
            'peak': 6,
            'ridge': 7,
            'valley': 8,
        }
        
    def analyze(
        self,
        image: np.ndarray,
        features: Dict[str, Dict[str, np.ndarray]],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Analyze terrain from extracted features.
        
        Parameters
        ----------
        image : np.ndarray
            Original normalized image
        features : Dict
            Extracted features from FeatureExtractor
        config : Dict, optional
            Analysis configuration parameters
            
        Returns
        -------
        Dict[str, np.ndarray]
            Terrain analysis results
        """
        config = config or {}
        
        h, w = image.shape[:2]
        
        # Initialize terrain probability maps
        terrain_probs = {}
        for name in self.terrain_classes.keys():
            terrain_probs[name] = np.zeros((h, w), dtype=np.float32)
        
        # Extract color features
        color_feats = features['color']
        
        # Extract edge features
        edge_feats = features['edges']
        
        # Extract texture features
        texture_feats = features['texture']
        
        # Extract pattern features
        pattern_feats = features['patterns']
        
        # Extract region features
        region_feats = features['regions']
        
        # === Water detection ===
        terrain_probs['water'] = self._detect_water(
            color_feats, texture_feats, edge_feats
        )
        
        # === Mountain/Peak detection ===
        terrain_probs['peak'], terrain_probs['ridge'] = self._detect_mountains(
            color_feats, edge_feats, texture_feats, pattern_feats
        )
        
        # === Valley detection ===
        terrain_probs['valley'] = self._detect_valleys(
            color_feats, edge_feats, pattern_feats
        )
        
        # === Slope detection ===
        (
            terrain_probs['cliff'],
            terrain_probs['steep_slope'],
            terrain_probs['moderate_slope'],
            terrain_probs['gentle_slope']
        ) = self._detect_slopes(
            color_feats, edge_feats, texture_feats
        )
        
        # === Flat areas ===
        terrain_probs['flat'] = self._detect_flat_areas(
            color_feats, texture_feats, edge_feats
        )
        
        # Normalize probabilities
        total_prob = sum(terrain_probs.values())
        total_prob = np.maximum(total_prob, 1e-10)  # Avoid division by zero
        
        for key in terrain_probs:
            terrain_probs[key] /= total_prob
        
        # Create terrain classification map
        terrain_class_map = self._create_class_map(terrain_probs)
        
        # Estimate elevation ranges for each terrain type
        elevation_ranges = self._estimate_elevation_ranges(
            terrain_probs, color_feats, edge_feats
        )
        
        # Generate base elevation estimate
        base_elevation = self._estimate_base_elevation(
            terrain_probs, elevation_ranges, color_feats
        )
        
        # Detect drainage patterns
        drainage = self._detect_drainage(
            terrain_probs, edge_feats, pattern_feats
        )
        
        return {
            'terrain_probabilities': terrain_probs,
            'terrain_class_map': terrain_class_map,
            'elevation_ranges': elevation_ranges,
            'base_elevation': base_elevation,
            'drainage_pattern': drainage,
            'water_mask': terrain_probs['water'] > 0.5,
            'mountain_mask': (terrain_probs['peak'] + terrain_probs['ridge']) > 0.3,
            'valley_mask': terrain_probs['valley'] > 0.3,
        }
    
    def _detect_water(
        self,
        color_feats: Dict[str, np.ndarray],
        texture_feats: Dict[str, np.ndarray],
        edge_feats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Detect water bodies based on color and texture."""
        h, w = color_feats['gray'].shape
        water_prob = np.zeros((h, w), dtype=np.float32)
        
        # Blue/cyan colors are typical for water
        hue = color_feats['hue']
        saturation = color_feats['saturation']
        value = color_feats['value']
        
        # Water typically has hue in blue range (0.5-0.7 in HSV)
        blue_mask = ((hue > 0.45) & (hue < 0.75)) | (hue < 0.15)  # Include gray/white
        
        # Water has moderate to high saturation (not too gray)
        sat_mask = saturation > 0.1
        
        # Water is typically smooth (low texture)
        roughness = texture_feats.get('roughness_scale_2', np.zeros((h, w)))
        smooth_mask = roughness < roughness.mean()
        
        # Water has low edge density
        edge_density = edge_feats['edges_canny']
        low_edge_mask = ~edge_density
        
        # Dark water (deep) vs light water (shallow)
        dark_mask = value < 0.6
        
        # Combine indicators
        water_prob = (
            blue_mask.astype(float) * 0.3 +
            sat_mask.astype(float) * 0.1 +
            smooth_mask.astype(float) * 0.25 +
            low_edge_mask.astype(float) * 0.2 +
            dark_mask.astype(float) * 0.15
        )
        
        # Smooth the result
        water_prob = gaussian_filter(water_prob, sigma=2)
        
        return np.clip(water_prob, 0, 1)
    
    def _detect_mountains(
        self,
        color_feats: Dict[str, np.ndarray],
        edge_feats: Dict[str, np.ndarray],
        texture_feats: Dict[str, np.ndarray],
        pattern_feats: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect mountain peaks and ridges."""
        h, w = color_feats['gray'].shape
        
        # Peaks are often darker (shadowed) or have specific colors
        value = color_feats['value']
        gray = color_feats['gray']
        
        # Dark areas might be mountains
        dark_areas = value < value.mean()
        
        # High texture indicates rough terrain
        complexity = texture_feats.get('complexity', np.zeros((h, w)))
        high_texture = complexity > complexity.mean() + complexity.std()
        
        # Ridge lines
        ridges = edge_feats.get('ridges', np.zeros((h, w)))
        strong_ridges = ridges > ridges.mean() + ridges.std()
        
        # Converging patterns indicate peaks
        pattern_strength = pattern_feats.get('pattern_strength', np.zeros((h, w)))
        
        # Peak detection - local maxima in ridge strength
        peak_prob = np.zeros((h, w), dtype=np.float32)
        
        # Find local maxima in ridge image
        from skimage.feature import peak_local_max
        ridge_maxima = peak_local_max(
            ridges,
            min_distance=10,
            threshold_abs=ridges.mean() + 0.5 * ridges.std()
        )
        
        # Create peak probability map
        for y, x in ridge_maxima:
            peak_prob[y, x] = 1.0
        
        # Dilate peaks
        peak_prob = gaussian_filter(peak_prob, sigma=5)
        peak_prob *= dark_areas.astype(float) * high_texture.astype(float)
        
        # Ridge probability
        ridge_prob = strong_ridges.astype(float) * 0.5 + ridges / ridges.max() * 0.5
        ridge_prob = gaussian_filter(ridge_prob, sigma=1)
        
        return np.clip(peak_prob, 0, 1), np.clip(ridge_prob, 0, 1)
    
    def _detect_valleys(
        self,
        color_feats: Dict[str, np.ndarray],
        edge_feats: Dict[str, np.ndarray],
        pattern_feats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Detect valley patterns."""
        h, w = color_feats['gray'].shape
        
        # Valleys often appear as linear features
        edges = edge_feats['edges_canny_coarse']
        
        # Distance from edges (valleys are often between features)
        edge_dist = distance_transform_edt(~edges)
        
        # Valley patterns - converging lines
        sobel_dir = edge_feats['sobel_direction']
        
        # Detect linear patterns
        from skimage import morphology
        skeleton = skeletonize(edges)
        
        # Valleys are typically lower areas between higher terrain
        # Use distance from ridges as proxy
        ridges = edge_feats.get('ridges', np.zeros((h, w)))
        ridge_dist = distance_transform_edt(ridges < ridges.mean())
        
        # Valley probability increases with distance from ridges
        # but decreases very far from any features
        valley_prob = np.exp(-ridge_dist / 50) * (ridge_dist > 5).astype(float)
        
        # Enhance with linear features
        valley_prob += skeleton.astype(float) * 0.3
        
        return np.clip(gaussian_filter(valley_prob, sigma=2), 0, 1)
    
    def _detect_slopes(
        self,
        color_feats: Dict[str, np.ndarray],
        edge_feats: Dict[str, np.ndarray],
        texture_feats: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Detect different slope categories."""
        h, w = color_feats['gray'].shape
        
        # Edge density correlates with slope
        edge_density = gaussian_filter(
            edge_feats['sobel_magnitude'],
            sigma=2
        )
        
        # Texture complexity
        complexity = texture_feats.get('complexity', np.zeros((h, w)))
        
        # Normalize
        edge_density_norm = edge_density / edge_density.max() if edge_density.max() > 0 else edge_density
        complexity_norm = complexity / complexity.max() if complexity.max() > 0 else complexity
        
        # Combine indicators
        slope_indicator = (edge_density_norm + complexity_norm) / 2
        
        # Define thresholds for slope categories
        cliff_thresh = 0.7
        steep_thresh = 0.5
        moderate_thresh = 0.3
        
        # Cliff - very high edge density
        cliff_prob = np.maximum(0, (slope_indicator - cliff_thresh) / (1 - cliff_thresh))
        
        # Steep slope
        steep_prob = np.maximum(0, np.minimum(1, (slope_indicator - steep_thresh) / (cliff_thresh - steep_thresh)))
        steep_prob = np.maximum(0, steep_prob - cliff_prob)
        
        # Moderate slope
        moderate_prob = np.maximum(0, np.minimum(1, (slope_indicator - moderate_thresh) / (steep_thresh - moderate_thresh)))
        moderate_prob = np.maximum(0, moderate_prob - steep_prob - cliff_prob)
        
        # Gentle slope
        gentle_prob = np.maximum(0, np.minimum(1, slope_indicator / moderate_thresh))
        gentle_prob = np.maximum(0, gentle_prob - moderate_prob - steep_prob - cliff_prob)
        
        return (
            np.clip(cliff_prob, 0, 1),
            np.clip(steep_prob, 0, 1),
            np.clip(moderate_prob, 0, 1),
            np.clip(gentle_prob, 0, 1)
        )
    
    def _detect_flat_areas(
        self,
        color_feats: Dict[str, np.ndarray],
        texture_feats: Dict[str, np.ndarray],
        edge_feats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Detect flat areas."""
        h, w = color_feats['gray'].shape
        
        # Low edge density
        edges = edge_feats['sobel_magnitude']
        low_edges = edges < edges.mean()
        
        # Low texture
        roughness = texture_feats.get('roughness_scale_2', np.zeros((h, w)))
        smooth = roughness < roughness.mean()
        
        # Uniform color
        local_std = color_feats['local_std']
        uniform = local_std < local_std.mean()
        
        # Combine
        flat_prob = (
            low_edges.astype(float) * 0.4 +
            smooth.astype(float) * 0.35 +
            uniform.astype(float) * 0.25
        )
        
        return np.clip(gaussian_filter(flat_prob, sigma=2), 0, 1)
    
    def _create_class_map(
        self,
        terrain_probs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Create a hard classification map from probabilities."""
        h, w = list(terrain_probs.values())[0].shape
        class_map = np.zeros((h, w), dtype=np.int32)
        
        # Stack probabilities and find max
        prob_stack = np.stack(list(terrain_probs.values()), axis=-1)
        class_map = np.argmax(prob_stack, axis=-1)
        
        return class_map
    
    def _estimate_elevation_ranges(
        self,
        terrain_probs: Dict[str, np.ndarray],
        color_feats: Dict[str, np.ndarray],
        edge_feats: Dict[str, np.ndarray]
    ) -> Dict[str, Tuple[float, float]]:
        """Estimate elevation ranges for each terrain type."""
        # These are relative ranges (0-1) that will be scaled later
        ranges = {
            'water': (0.0, 0.0),
            'flat': (0.0, 0.1),
            'gentle_slope': (0.05, 0.2),
            'moderate_slope': (0.15, 0.4),
            'steep_slope': (0.35, 0.7),
            'cliff': (0.5, 0.9),
            'peak': (0.7, 1.0),
            'ridge': (0.6, 0.95),
            'valley': (0.0, 0.3),
        }
        
        return ranges
    
    def _estimate_base_elevation(
        self,
        terrain_probs: Dict[str, np.ndarray],
        elevation_ranges: Dict[str, Tuple[float, float]],
        color_feats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Estimate base elevation from terrain probabilities."""
        h, w = list(terrain_probs.values())[0].shape
        
        base_elev = np.zeros((h, w), dtype=np.float32)
        
        # Weighted average of terrain type elevations
        for terrain_type, prob in terrain_probs.items():
            if terrain_type in elevation_ranges:
                min_elev, max_elev = elevation_ranges[terrain_type]
                mean_elev = (min_elev + max_elev) / 2
                base_elev += prob * mean_elev
        
        # Modulate by brightness (darker = higher in many artistic maps)
        value = color_feats['value']
        # Invert: darker areas tend to be higher in many artistic conventions
        height_mod = 1.0 - value
        
        # Blend with terrain-based estimate
        base_elev = 0.7 * base_elev + 0.3 * height_mod
        
        return np.clip(base_elev, 0, 1)
    
    def _detect_drainage(
        self,
        terrain_probs: Dict[str, np.ndarray],
        edge_feats: Dict[str, np.ndarray],
        pattern_feats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Detect drainage patterns."""
        h, w = list(terrain_probs.values())[0].shape
        
        # Valleys often contain drainage
        valley_prob = terrain_probs['valley']
        
        # Linear features
        edges = edge_feats['edges_canny_coarse']
        
        # Converging patterns
        pattern_strength = pattern_feats.get('pattern_strength', np.zeros((h, w)))
        
        # Combine
        drainage = valley_prob * 0.5 + edges.astype(float) * 0.3 + pattern_strength * 0.2
        
        # Skeletonize to get centerlines
        from skimage.morphology import skeletonize
        drainage_binary = drainage > drainage.mean() + drainage.std()
        drainage_skel = skeletonize(drainage_binary)
        
        return np.clip(drainage + drainage_skel.astype(float) * 0.5, 0, 1)
