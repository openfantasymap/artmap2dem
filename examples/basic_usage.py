"""Basic usage example for artmap2dem."""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from artmap2dem import ArtMapProcessor
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Run basic example."""
    print("ArtMap2DEM - Basic Usage Example")
    print("=" * 50)
    
    # Initialize processor with default settings
    print("\n1. Initializing processor...")
    processor = ArtMapProcessor(
        min_elevation=0,
        max_elevation=4000,
        water_level=0,
        smoothness=0.5,
        preserve_features=True
    )
    
    # For this example, we'll create a synthetic artistic map
    # In real usage, you would load an actual GeoTIFF
    print("\n2. Creating synthetic artistic map for demonstration...")
    synthetic_map = create_synthetic_artistic_map()
    
    # Save synthetic map as GeoTIFF for demonstration
    from artmap2dem.utils import save_dem
    save_dem(
        'synthetic_artistic_map.tif',
        synthetic_map,
        crs='EPSG:4326',
        transform=None
    )
    print("   Saved synthetic map to: synthetic_artistic_map.tif")
    
    # Process the map
    print("\n3. Processing artistic map...")
    processor.load_map('synthetic_artistic_map.tif')
    dem = processor.process(apply_hydrology=True, seed=42)
    print(f"   Generated DEM shape: {dem.shape}")
    print(f"   DEM elevation range: {dem.min():.2f} to {dem.max():.2f} meters")
    
    # Save DEM
    print("\n4. Saving DEM...")
    processor.save_dem('output_dem.tif')
    print("   Saved DEM to: output_dem.tif")
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    hillshade = processor.get_hillshade(azimuth=315, altitude=45)
    slope = processor.get_slope()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Synthetic artistic map
    if synthetic_map.ndim == 3:
        axes[0, 0].imshow(synthetic_map.transpose(1, 2, 0))
    else:
        axes[0, 0].imshow(synthetic_map, cmap='terrain')
    axes[0, 0].set_title('Input Artistic Map (Synthetic)', fontsize=12)
    axes[0, 0].axis('off')
    
    # Generated DEM
    im1 = axes[0, 1].imshow(dem, cmap='terrain', vmin=dem.min(), vmax=dem.max())
    axes[0, 1].set_title('Generated DEM', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], label='Elevation (m)')
    
    # Hillshade
    axes[1, 0].imshow(hillshade, cmap='gray')
    axes[1, 0].set_title('Hillshade (Azimuth 315°, Altitude 45°)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Slope
    im2 = axes[1, 1].imshow(slope, cmap='YlOrRd', vmin=0, vmax=60)
    axes[1, 1].set_title('Slope', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], label='Slope (degrees)')
    
    plt.tight_layout()
    plt.savefig('dem_visualization.png', dpi=150, bbox_inches='tight')
    print("   Saved visualization to: dem_visualization.png")
    
    # Print terrain statistics
    print("\n6. Terrain Statistics:")
    terrain_map = processor.terrain_map
    water_coverage = terrain_map['water_mask'].mean() * 100
    mountain_coverage = terrain_map['mountain_mask'].mean() * 100
    valley_coverage = terrain_map['valley_mask'].mean() * 100
    
    print(f"   Water coverage: {water_coverage:.1f}%")
    print(f"   Mountain coverage: {mountain_coverage:.1f}%")
    print(f"   Valley coverage: {valley_coverage:.1f}%")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    
    return processor, dem


def create_synthetic_artistic_map(size: int = 512) -> np.ndarray:
    """
    Create a synthetic artistic map for demonstration.
    
    This creates a map with:
    - Blue water body in the center
    - Green lowlands around water
    - Brown hills
    - Gray/white mountains with ridge lines
    """
    h, w = size, size
    
    # Create coordinate grids
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    
    # Create base terrain pattern
    # Distance from center
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Create mountain peaks pattern
    peaks = np.zeros((h, w))
    np.random.seed(42)
    n_peaks = 8
    for _ in range(n_peaks):
        px = np.random.randint(50, w - 50)
        py = np.random.randint(50, h - 50)
        peak_dist = np.sqrt((x - px)**2 + (y - py)**2)
        peaks += np.exp(-peak_dist**2 / (2 * (30 + np.random.randint(20))**2))
    
    # Create ridge lines
    ridges = np.zeros((h, w))
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        ridge_x = cx + np.cos(angle) * (x - cx) - np.sin(angle) * (y - cy)
        ridge_y = np.sin(angle) * (x - cx) + np.cos(angle) * (y - cy)
        ridges += np.exp(-ridge_y**2 / 100) * (1 + 0.3 * np.sin(ridge_x / 20))
    
    # Combine patterns
    terrain_pattern = peaks + ridges * 0.5
    terrain_pattern = (terrain_pattern - terrain_pattern.min()) / (terrain_pattern.max() - terrain_pattern.min())
    
    # Create RGB artistic map
    artistic_map = np.zeros((3, h, w), dtype=np.float32)
    
    # Water (center lake) - blue
    lake_mask = dist < size * 0.15
    artistic_map[0, lake_mask] = 0.2  # R
    artistic_map[1, lake_mask] = 0.4  # G
    artistic_map[2, lake_mask] = 0.8  # B
    
    # Lowlands - green
    lowland_mask = (dist >= size * 0.15) & (dist < size * 0.3) & (terrain_pattern < 0.3)
    artistic_map[0, lowland_mask] = 0.2
    artistic_map[1, lowland_mask] = 0.6
    artistic_map[2, lowland_mask] = 0.2
    
    # Hills - yellow/brown
    hill_mask = (dist >= size * 0.3) & (terrain_pattern < 0.6)
    artistic_map[0, hill_mask] = 0.6
    artistic_map[1, hill_mask] = 0.5
    artistic_map[2, hill_mask] = 0.2
    
    # Mountains - gray/white with texture
    mountain_mask = terrain_pattern >= 0.6
    mountain_intensity = terrain_pattern[mountain_mask]
    artistic_map[0, mountain_mask] = 0.3 + mountain_intensity * 0.5
    artistic_map[1, mountain_mask] = 0.3 + mountain_intensity * 0.5
    artistic_map[2, mountain_mask] = 0.3 + mountain_intensity * 0.5
    
    # Add noise for artistic texture
    noise = np.random.randn(3, h, w) * 0.05
    artistic_map = np.clip(artistic_map + noise, 0, 1)
    
    # Add some "hand-drawn" line artifacts
    from scipy.ndimage import gaussian_filter
    line_noise = np.random.randn(h, w)
    line_noise = gaussian_filter(line_noise, sigma=1)
    line_mask = np.abs(line_noise) > 2
    artistic_map[:, line_mask] *= 0.9
    
    return artistic_map


if __name__ == '__main__':
    main()
