"""Basic tests for artmap2dem."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from artmap2dem import ArtMapProcessor
from artmap2dem.feature_extractor import FeatureExtractor
from artmap2dem.terrain_analyzer import TerrainAnalyzer
from artmap2dem.dem_generator import DEMGenerator


class TestFeatureExtractor:
    """Test feature extraction."""
    
    def test_extract_from_grayscale(self):
        """Test feature extraction from grayscale image."""
        extractor = FeatureExtractor()
        image = np.random.rand(100, 100).astype(np.float32)
        
        features = extractor.extract(image)
        
        assert 'color' in features
        assert 'edges' in features
        assert 'texture' in features
        assert 'patterns' in features
        assert 'regions' in features
    
    def test_extract_from_rgb(self):
        """Test feature extraction from RGB image."""
        extractor = FeatureExtractor()
        image = np.random.rand(3, 100, 100).astype(np.float32)
        
        features = extractor.extract(image)
        
        assert 'color' in features
        assert features['color']['rgb'].shape == (100, 100, 3)


class TestTerrainAnalyzer:
    """Test terrain analysis."""
    
    def test_analyze_returns_terrain_map(self):
        """Test that analyze returns terrain map."""
        analyzer = TerrainAnalyzer()
        
        # Create mock features
        h, w = 100, 100
        features = {
            'color': {
                'gray': np.random.rand(h, w),
                'hue': np.random.rand(h, w),
                'saturation': np.random.rand(h, w),
                'value': np.random.rand(h, w),
                'local_std': np.random.rand(h, w) * 0.1,
            },
            'edges': {
                'edges_canny': np.random.rand(h, w) > 0.9,
                'edges_canny_coarse': np.random.rand(h, w) > 0.9,
                'sobel_magnitude': np.random.rand(h, w),
                'sobel_direction': np.random.rand(h, w) * 2 * np.pi,
                'ridges': np.random.rand(h, w),
            },
            'texture': {
                'roughness_scale_2': np.random.rand(h, w) * 0.1,
                'complexity': np.random.rand(h, w),
            },
            'patterns': {
                'pattern_strength': np.random.rand(h, w),
            },
            'regions': {
                'segments': np.random.randint(0, 10, (h, w)),
            }
        }
        
        image = np.random.rand(3, h, w)
        terrain_map = analyzer.analyze(image, features)
        
        assert 'terrain_probabilities' in terrain_map
        assert 'terrain_class_map' in terrain_map
        assert 'base_elevation' in terrain_map
        assert 'water_mask' in terrain_map


class TestDEMGenerator:
    """Test DEM generation."""
    
    def test_generate_dem(self):
        """Test DEM generation."""
        generator = DEMGenerator(
            min_elevation=0,
            max_elevation=1000,
            seed=42
        )
        
        h, w = 100, 100
        terrain_map = {
            'base_elevation': np.random.rand(h, w),
            'terrain_probabilities': {
                'water': np.zeros((h, w)),
                'flat': np.random.rand(h, w) * 0.3,
                'gentle_slope': np.random.rand(h, w) * 0.3,
                'moderate_slope': np.random.rand(h, w) * 0.2,
                'steep_slope': np.random.rand(h, w) * 0.1,
                'cliff': np.zeros((h, w)),
                'peak': np.random.rand(h, w) * 0.05,
                'ridge': np.random.rand(h, w) * 0.05,
                'valley': np.random.rand(h, w) * 0.1,
            },
            'elevation_ranges': {
                'water': (0.0, 0.0),
                'flat': (0.0, 0.1),
                'gentle_slope': (0.05, 0.2),
                'moderate_slope': (0.15, 0.4),
                'steep_slope': (0.35, 0.7),
                'cliff': (0.5, 0.9),
                'peak': (0.7, 1.0),
                'ridge': (0.6, 0.95),
                'valley': (0.0, 0.3),
            },
            'water_mask': np.zeros((h, w), dtype=bool),
        }
        
        features = {
            'color': {'gray': np.random.rand(h, w)},
            'edges': {'sobel_magnitude': np.random.rand(h, w)},
            'texture': {'complexity': np.random.rand(h, w)},
        }
        
        dem = generator.generate(terrain_map, features)
        
        assert dem.shape == (h, w)
        assert dem.min() >= 0
        assert dem.max() <= 1000


class TestArtMapProcessor:
    """Test main processor."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = ArtMapProcessor(
            min_elevation=0,
            max_elevation=4000,
            water_level=0
        )
        
        assert processor.min_elevation == 0
        assert processor.max_elevation == 4000
        assert processor.water_level == 0
    
    def test_process_synthetic_map(self):
        """Test processing a synthetic map."""
        processor = ArtMapProcessor(
            min_elevation=0,
            max_elevation=1000,
            seed=42
        )
        
        # Create synthetic map
        h, w = 100, 100
        synthetic_map = np.random.rand(3, h, w).astype(np.float32)
        
        # Save temporarily
        import tempfile
        import rasterio
        from rasterio.transform import Affine
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        profile = {
            'driver': 'GTiff',
            'height': h,
            'width': w,
            'count': 3,
            'dtype': synthetic_map.dtype,
            'crs': 'EPSG:4326',
            'transform': Affine.identity(),
        }
        
        with rasterio.open(tmp_path, 'w', **profile) as dst:
            dst.write(synthetic_map)
        
        # Process
        dem = processor.process(tmp_path, seed=42)
        
        assert dem.shape == (h, w)
        assert not np.isnan(dem).any()
        
        # Cleanup
        Path(tmp_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
