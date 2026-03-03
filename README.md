# ArtMap2DEM

Convert artistic georeferenced maps into believable Digital Elevation Models (DEMs).

## Overview

ArtMap2DEM is a Python library that analyzes artistic maps (hand-drawn, painted, or stylized) and generates realistic DEMs based on visual cues like color, texture, and pattern. It uses computer vision and machine learning techniques to interpret artistic conventions and translate them into elevation data.

## Features

- **Visual Feature Extraction**: Analyzes colors, textures, edges, and patterns
- **Terrain Classification**: Identifies mountains, valleys, plains, water bodies, etc.
- **Realistic DEM Generation**: Creates elevation models with natural variation
- **Hydrological Processing**: Ensures water flows correctly and sinks are filled
- **Georeferenced Output**: Maintains spatial reference information

## Installation

```bash
pip install artmap2dem
```

Or install from source:

```bash
git clone https://github.com/yourusername/artmap2dem.git
cd artmap2dem
pip install -e .
```

## Quick Start

```python
from artmap2dem import ArtMapProcessor

# Initialize processor
processor = ArtMapProcessor(
    min_elevation=0,
    max_elevation=4000,
    water_level=0
)

# Process artistic map
processor.process('input_map.tif')

# Save DEM
processor.save_dem('output_dem.tif')

# Get hillshade
hillshade = processor.get_hillshade()
```

## Usage Examples

### Basic Usage

```python
from artmap2dem import ArtMapProcessor

# Simple conversion
processor = ArtMapProcessor()
dem = processor.process('artistic_map.tif')
processor.save_dem('dem_output.tif')
```

### Advanced Configuration

```python
from artmap2dem import ArtMapProcessor

# Configure for specific terrain characteristics
processor = ArtMapProcessor(
    min_elevation=-100,      # Below sea level areas
    max_elevation=8000,      # High mountains
    water_level=-50,         # Water body elevation
    smoothness=0.3,          # Less smoothing for rugged terrain
    preserve_features=True   # Keep sharp features
)

# Process with custom settings
dem = processor.process(
    input_path='artistic_map.tif',
    apply_hydrology=True,
    seed=42  # For reproducibility
)

# Save and visualize
processor.save_dem('dem_output.tif')
hillshade = processor.get_hillshade(azimuth=315, altitude=45)
```

### Batch Processing

```python
from artmap2dem import ArtMapProcessor
from pathlib import Path

processor = ArtMapProcessor()

input_dir = Path('input_maps')
output_dir = Path('output_dems')
output_dir.mkdir(exist_ok=True)

for map_file in input_dir.glob('*.tif'):
    print(f"Processing {map_file.name}...")
    dem = processor.process(map_file)
    processor.save_dem(output_dir / f"{map_file.stem}_dem.tif")
```

### Accessing Intermediate Results

```python
from artmap2dem import ArtMapProcessor

processor = ArtMapProcessor()
processor.process('artistic_map.tif')

# Access terrain classification
terrain_map = processor.terrain_map
water_mask = terrain_map['water_mask']
mountain_mask = terrain_map['mountain_mask']

# Access extracted features
features = processor.features
color_features = features['color']
edge_features = features['edges']

# Access terrain probabilities
terrain_probs = terrain_map['terrain_probabilities']
peak_probability = terrain_probs['peak']
```

### Visualization

```python
from artmap2dem import ArtMapProcessor
import matplotlib.pyplot as plt

processor = ArtMapProcessor()
processor.process('artistic_map.tif')

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Original map (if RGB)
# axes[0, 0].imshow(...)
# axes[0, 0].set_title('Input Map')

# Generated DEM
dem = processor.dem
im1 = axes[0, 1].imshow(dem, cmap='terrain')
axes[0, 1].set_title('Generated DEM')
plt.colorbar(im1, ax=axes[0, 1])

# Hillshade
hillshade = processor.get_hillshade()
axes[1, 0].imshow(hillshade, cmap='gray')
axes[1, 0].set_title('Hillshade')

# Slope
slope = processor.get_slope()
im2 = axes[1, 1].imshow(slope, cmap='YlOrRd')
axes[1, 1].set_title('Slope')
plt.colorbar(im2, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('dem_analysis.png', dpi=150)
```

## How It Works

1. **Feature Extraction**: The library analyzes the input image to extract:
   - Color features (dominant colors, HSV components)
   - Edge features (Canny edges, Sobel gradients, ridge detection)
   - Texture features (roughness, local binary patterns, Gabor filters)
   - Pattern features (Fourier analysis, periodic patterns)

2. **Terrain Analysis**: Features are interpreted as terrain characteristics:
   - Water detection (blue colors, smooth textures)
   - Mountain detection (dark colors, rough textures, ridge lines)
   - Valley detection (linear features, converging patterns)
   - Slope classification (edge density, texture complexity)

3. **DEM Generation**: A realistic elevation model is created:
   - Base elevation from terrain classification
   - Fractal noise for natural variation
   - Terrain-specific detail enhancement
   - Feature preservation for sharp terrain elements

4. **Hydrological Correction**: The DEM is refined for hydrological consistency:
   - Sink filling
   - River channel processing
   - Drainage enforcement
   - Water body flattening

## Configuration Options

### ArtMapProcessor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_elevation` | float | 0.0 | Minimum elevation in meters |
| `max_elevation` | float | 4000.0 | Maximum elevation in meters |
| `water_level` | float | 0.0 | Elevation for water bodies |
| `smoothness` | float | 0.5 | Smoothing factor (0-1) |
| `preserve_features` | bool | True | Preserve sharp terrain features |

### Terrain Analysis Configuration

```python
config = {
    'terrain_analysis': {
        'water_hue_range': (0.45, 0.75),
        'mountain_texture_threshold': 0.6,
        'valley_convergence_threshold': 0.5,
    }
}

processor = ArtMapProcessor(config=config)
```

## API Reference

### ArtMapProcessor

Main class for converting artistic maps to DEMs.

#### Methods

- `load_map(input_path, band=None)`: Load an artistic map
- `process(input_path=None, apply_hydrology=True, river_channels=None, seed=None)`: Process map and generate DEM
- `save_dem(output_path, dem=None, dtype=None)`: Save DEM to GeoTIFF
- `get_hillshade(azimuth=315, altitude=45, dem=None)`: Generate hillshade
- `get_slope(dem=None)`: Calculate slope

#### Properties

- `dem`: Generated DEM array
- `terrain_map`: Terrain analysis results
- `features`: Extracted visual features

## Dependencies

- numpy >= 1.20.0
- rasterio >= 1.3.0
- scipy >= 1.7.0
- scikit-image >= 0.19.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- opencv-python >= 4.5.0
- Pillow >= 8.0.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use ArtMap2DEM in your research, please cite:

```
@software{artmap2dem,
  title={ArtMap2DEM: Converting Artistic Maps to Digital Elevation Models},
  author={ArtMap2DEM Team},
  year={2024}
}
```

## Acknowledgments

This library was inspired by the need to digitize historical and artistic maps for GIS applications. It combines techniques from computer vision, geomorphometry, and hydrological modeling.
