"""
ArtMap2DEM - Convert artistic georeferenced maps into believable Digital Elevation Models.

This library analyzes artistic maps (hand-drawn, painted, or stylized) and generates
realistic DEMs based on visual cues like color, texture, and pattern.
"""

__version__ = "0.1.0"

from .core import ArtMapProcessor
from .dem_generator import DEMGenerator
from .terrain_analyzer import TerrainAnalyzer
from .utils import save_dem, load_geotiff, get_profile

__all__ = [
    "ArtMapProcessor",
    "DEMGenerator", 
    "TerrainAnalyzer",
    "save_dem",
    "load_geotiff",
    "get_profile",
]
