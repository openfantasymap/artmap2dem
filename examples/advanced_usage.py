"""Advanced usage example for artmap2dem."""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from artmap2dem import ArtMapProcessor
from artmap2dem.utils import (
    calculate_hillshade, calculate_slope, calculate_aspect,
    calculate_curvature, create_color_relief, get_statistics
)
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def main():
    """Run advanced example."""
    print("ArtMap2DEM - Advanced Usage Example")
    print("=" * 50)
    
    # Example 1: Custom configuration for different terrain types
    print("\n1. Custom Configuration for Mountainous Terrain")
    print("-" * 50)
    
    processor_mountains = ArtMapProcessor(
        min_elevation=500,       # Higher minimum for mountain region
        max_elevation=6000,      # Very high peaks
        water_level=400,         # Alpine lakes
        smoothness=0.2,          # Less smoothing for rugged terrain
        preserve_features=True,
        config={
            'terrain_analysis': {
                'mountain_texture_threshold': 0.5,  # More sensitive mountain detection
            }
        }
    )
    print("   Configured for mountainous terrain (500-6000m)")
    
    # Example 2: Custom configuration for coastal terrain
    print("\n2. Custom Configuration for Coastal Terrain")
    print("-" * 50)
    
    processor_coastal = ArtMapProcessor(
        min_elevation=-100,      # Below sea level
        max_elevation=500,       # Low coastal hills
        water_level=0,
        smoothness=0.7,          # Smoother for coastal plains
        preserve_features=False, # Less feature preservation for gentle terrain
    )
    print("   Configured for coastal terrain (-100 to 500m)")
    
    # Example 3: Processing with river channel mask
    print("\n3. Processing with Known River Channels")
    print("-" * 50)
    
    # Create synthetic map with rivers
    artistic_map, river_mask = create_map_with_rivers()
    
    # Save for demonstration
    from artmap2dem.utils import save_dem
    save_dem('map_with_rivers.tif', artistic_map)
    print("   Created synthetic map with river channels")
    
    processor = ArtMapProcessor(
        min_elevation=0,
        max_elevation=2000,
        water_level=0
    )
    
    # Process with river mask
    processor.load_map('map_with_rivers.tif')
    dem_with_rivers = processor.process(
        apply_hydrology=True,
        river_channels=river_mask,
        seed=42
    )
    processor.save_dem('dem_with_rivers.tif')
    print("   Generated DEM with river channel enforcement")
    
    # Example 4: Batch processing
    print("\n4. Batch Processing Multiple Maps")
    print("-" * 50)
    
    # Create multiple synthetic maps
    map_configs = [
        ('lowlands', 0, 500),
        ('hills', 100, 1500),
        ('mountains', 500, 4000),
    ]
    
    processor = ArtMapProcessor()
    
    for name, min_elev, max_elev in map_configs:
        # Create synthetic map
        synthetic_map = create_synthetic_map_by_type(name)
        save_dem(f'{name}_map.tif', synthetic_map)
        
        # Process
        processor = ArtMapProcessor(min_elevation=min_elev, max_elevation=max_elev)
        dem = processor.process(f'{name}_map.tif')
        processor.save_dem(f'{name}_dem.tif')
        
        stats = get_statistics(dem)
        print(f"   {name}: range={stats['min']:.0f}-{stats['max']:.0f}m, "
              f"mean={stats['mean']:.0f}m, std={stats['std']:.0f}m")
    
    # Example 5: Detailed terrain analysis
    print("\n5. Detailed Terrain Analysis")
    print("-" * 50)
    
    processor = ArtMapProcessor()
    processor.load_map('hills_map.tif')
    dem = processor.process()
    
    # Access terrain probabilities
    terrain_probs = processor.terrain_map['terrain_probabilities']
    
    print("   Terrain type probabilities:")
    for terrain_type, prob_map in terrain_probs.items():
        avg_prob = prob_map.mean()
        max_prob = prob_map.max()
        coverage = (prob_map > 0.5).mean() * 100
        print(f"     {terrain_type:15s}: avg={avg_prob:.3f}, max={max_prob:.3f}, coverage={coverage:.1f}%")
    
    # Example 6: Advanced visualization
    print("\n6. Advanced Visualization")
    print("-" * 50)
    
    create_advanced_visualization(processor, dem)
    print("   Created advanced visualization: advanced_visualization.png")
    
    # Example 7: Terrain metrics calculation
    print("\n7. Terrain Metrics Calculation")
    print("-" * 50)
    
    # Calculate various terrain metrics
    slope = calculate_slope(dem)
    aspect = calculate_aspect(dem)
    profile_curv, plan_curv, total_curv = calculate_curvature(dem)
    
    print(f"   Slope: mean={slope.mean():.1f}°, max={slope.max():.1f}°")
    print(f"   Aspect: dominant direction={calculate_dominant_aspect(aspect):.0f}°")
    print(f"   Curvature: mean={total_curv.mean():.4f}")
    
    # Example 8: Creating color relief
    print("\n8. Creating Color Relief")
    print("-" * 50)
    
    # Custom colormap
    custom_colormap = {
        0: (0, 0, 255),       # Water
        10: (0, 128, 255),    # Wetland
        100: (0, 200, 0),     # Low vegetation
        500: (128, 255, 0),   # Hills
        1000: (255, 200, 0),  # High hills
        1500: (255, 128, 0),  # Mountains
        2000: (255, 0, 0),    # High mountains
    }
    
    color_relief = create_color_relief(dem, custom_colormap)
    
    # Save color relief
    plt.imsave('color_relief.png', color_relief)
    print("   Saved color relief: color_relief.png")
    
    print("\n" + "=" * 50)
    print("Advanced example completed successfully!")


def create_map_with_rivers(size: int = 512) -> tuple:
    """Create a synthetic map with river channels."""
    h, w = size, size
    
    # Create base terrain
    y, x = np.mgrid[0:h, 0:w]
    
    # Create elevation-like pattern
    elevation = (
        np.sin(x / 50) * 100 +
        np.cos(y / 50) * 100 +
        np.sin((x + y) / 100) * 200
    )
    
    # Normalize to 0-1
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    
    # Create river channels (following elevation gradients)
    river_mask = np.zeros((h, w), dtype=bool)
    
    # Main river
    for t in np.linspace(0, 1, 1000):
        ry = int(h * 0.2 + h * 0.6 * t + 50 * np.sin(t * 4 * np.pi))
        rx = int(w * 0.1 + w * 0.8 * t)
        if 0 <= ry < h and 0 <= rx < w:
            river_mask[ry-2:ry+3, rx-2:rx+3] = True
    
    # Tributary
    for t in np.linspace(0, 1, 500):
        ry = int(h * 0.5 + h * 0.3 * t)
        rx = int(w * 0.4 + w * 0.3 * t + 30 * np.sin(t * 3 * np.pi))
        if 0 <= ry < h and 0 <= rx < w:
            river_mask[ry-1:ry+2, rx-1:rx+2] = True
    
    # Create RGB artistic map
    artistic_map = np.zeros((3, h, w), dtype=np.float32)
    
    # Terrain colors based on elevation
    artistic_map[0] = 0.3 + elevation * 0.4  # R
    artistic_map[1] = 0.5 + elevation * 0.3  # G
    artistic_map[2] = 0.2 + elevation * 0.3  # B
    
    # Rivers in blue
    artistic_map[0, river_mask] = 0.1
    artistic_map[1, river_mask] = 0.3
    artistic_map[2, river_mask] = 0.8
    
    # Add texture
    noise = np.random.randn(3, h, w) * 0.05
    artistic_map = np.clip(artistic_map + noise, 0, 1)
    
    return artistic_map, river_mask


def create_synthetic_map_by_type(map_type: str, size: int = 512) -> np.ndarray:
    """Create synthetic map based on terrain type."""
    h, w = size, size
    y, x = np.mgrid[0:h, 0:w]
    
    if map_type == 'lowlands':
        # Smooth, low variation
        pattern = np.sin(x / 100) * 0.3 + np.cos(y / 100) * 0.3
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        # Green/brown colors
        artistic_map = np.zeros((3, h, w))
        artistic_map[0] = 0.3 + pattern * 0.2
        artistic_map[1] = 0.5 + pattern * 0.3
        artistic_map[2] = 0.2 + pattern * 0.1
        
    elif map_type == 'hills':
        # Moderate variation
        pattern = (
            np.sin(x / 50) * 0.5 +
            np.cos(y / 50) * 0.5 +
            np.sin((x + y) / 80) * 0.3
        )
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        # Yellow/brown colors
        artistic_map = np.zeros((3, h, w))
        artistic_map[0] = 0.5 + pattern * 0.3
        artistic_map[1] = 0.4 + pattern * 0.2
        artistic_map[2] = 0.2 + pattern * 0.1
        
    elif map_type == 'mountains':
        # High variation with peaks
        pattern = np.zeros((h, w))
        np.random.seed(42)
        for _ in range(10):
            px = np.random.randint(0, w)
            py = np.random.randint(0, h)
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            pattern += np.exp(-dist**2 / (2 * 40**2))
        
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        # Gray/white colors
        artistic_map = np.zeros((3, h, w))
        artistic_map[0] = 0.3 + pattern * 0.5
        artistic_map[1] = 0.3 + pattern * 0.5
        artistic_map[2] = 0.3 + pattern * 0.5
    
    else:
        # Default
        artistic_map = np.random.rand(3, h, w) * 0.5 + 0.25
    
    # Add noise
    noise = np.random.randn(3, h, w) * 0.05
    return np.clip(artistic_map + noise, 0, 1)


def calculate_dominant_aspect(aspect: np.ndarray) -> float:
    """Calculate dominant aspect direction."""
    # Exclude flat areas (-1)
    valid_aspect = aspect[aspect >= 0]
    
    if len(valid_aspect) == 0:
        return 0.0
    
    # Convert to radians and calculate circular mean
    aspect_rad = np.radians(valid_aspect)
    sin_mean = np.sin(aspect_rad).mean()
    cos_mean = np.cos(aspect_rad).mean()
    
    dominant = np.degrees(np.arctan2(sin_mean, cos_mean))
    if dominant < 0:
        dominant += 360
    
    return dominant


def create_advanced_visualization(processor, dem):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # DEM
    ax1 = fig.add_subplot(gs[0, :2])
    im1 = ax1.imshow(dem, cmap='terrain')
    ax1.set_title('Digital Elevation Model', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)')
    
    # Hillshade
    ax2 = fig.add_subplot(gs[0, 2:])
    hillshade = processor.get_hillshade(azimuth=315, altitude=45)
    ax2.imshow(hillshade, cmap='gray')
    ax2.set_title('Hillshade', fontsize=12, fontweight='bold')
    
    # Slope
    ax3 = fig.add_subplot(gs[1, 0])
    slope = processor.get_slope()
    im3 = ax3.imshow(slope, cmap='YlOrRd', vmin=0, vmax=60)
    ax3.set_title('Slope', fontsize=10)
    plt.colorbar(im3, ax=ax3, label='Degrees')
    
    # Aspect
    ax4 = fig.add_subplot(gs[1, 1])
    aspect = calculate_aspect(dem)
    aspect_vis = aspect.copy()
    aspect_vis[aspect < 0] = np.nan
    im4 = ax4.imshow(aspect_vis, cmap='hsv', vmin=0, vmax=360)
    ax4.set_title('Aspect', fontsize=10)
    plt.colorbar(im4, ax=ax4, label='Degrees')
    
    # Terrain classification
    ax5 = fig.add_subplot(gs[1, 2])
    terrain_class = processor.terrain_map['terrain_class_map']
    im5 = ax5.imshow(terrain_class, cmap='tab10')
    ax5.set_title('Terrain Classification', fontsize=10)
    
    # Water mask
    ax6 = fig.add_subplot(gs[1, 3])
    water_mask = processor.terrain_map['water_mask']
    ax6.imshow(water_mask, cmap='Blues')
    ax6.set_title('Water Mask', fontsize=10)
    
    # Terrain probabilities
    terrain_probs = processor.terrain_map['terrain_probabilities']
    prob_names = ['peak', 'ridge', 'valley', 'flat', 'water']
    
    for i, name in enumerate(prob_names):
        ax = fig.add_subplot(gs[2, i])
        prob_map = terrain_probs[name]
        im = ax.imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'{name.capitalize()} Probability', fontsize=9)
        plt.colorbar(im, ax=ax)
    
    plt.savefig('advanced_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
