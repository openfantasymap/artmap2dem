"""Hydrology processing module for water flow and drainage."""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, distance_transform_edt, maximum_filter
from skimage.morphology import skeletonize, remove_small_objects
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class HydrologyProcessor:
    """
    Process hydrological features for DEM correction.
    
    This class ensures:
    - Water flows downhill
    - Rivers have continuous gradients
    - Sinks are filled
    - Drainage networks are consistent
    """
    
    def __init__(self):
        self.flow_directions = np.array([
            [64, 128, 1],
            [32,  0,  2],
            [16,   8,  4]
        ])
        
    def process(
        self,
        dem: np.ndarray,
        river_mask: Optional[np.ndarray] = None,
        water_level: float = 0.0,
        terrain_map: Optional[Dict[str, np.ndarray]] = None,
        fill_sinks: bool = True,
        enforce_drainage: bool = True
    ) -> np.ndarray:
        """
        Apply hydrological corrections to DEM.
        
        Parameters
        ----------
        dem : np.ndarray
            Input DEM
        river_mask : np.ndarray, optional
            Binary mask of known river channels
        water_level : float
            Elevation for water bodies
        terrain_map : Dict, optional
            Terrain analysis results
        fill_sinks : bool
            Whether to fill sinks in the DEM
        enforce_drainage : bool
            Whether to enforce continuous drainage
            
        Returns
        -------
        np.ndarray
            Hydrologically corrected DEM
        """
        result = dem.copy()
        
        # Fill sinks
        if fill_sinks:
            logger.info("Filling sinks...")
            result = self._fill_sinks(result)
        
        # Process rivers if provided
        if river_mask is not None:
            logger.info("Processing river channels...")
            result = self._process_rivers(result, river_mask, water_level)
        
        # Enforce drainage from terrain map
        if enforce_drainage and terrain_map is not None:
            logger.info("Enforcing drainage patterns...")
            result = self._enforce_drainage(result, terrain_map)
        
        # Ensure water bodies are flat
        if terrain_map is not None:
            water_mask = terrain_map.get('water_mask', np.zeros_like(result, dtype=bool))
            result = self._flatten_water_bodies(result, water_mask, water_level)
        
        return result
    
    def _fill_sinks(
        self,
        dem: np.ndarray,
        epsilon: float = 0.001
    ) -> np.ndarray:
        """
        Fill sinks in the DEM using priority flood algorithm.
        
        Parameters
        ----------
        dem : np.ndarray
            Input DEM
        epsilon : float
            Small value to ensure drainage
            
        Returns
        -------
        np.ndarray
            DEM with sinks filled
        """
        from scipy.ndimage import minimum_filter
        
        result = dem.copy()
        h, w = result.shape
        
        # Identify sinks (local minima)
        neighborhood = np.ones((3, 3))
        local_min = minimum_filter(result, footprint=neighborhood, mode='nearest')
        sinks = (result == local_min) & (result < np.roll(result, 1, axis=0))
        
        # Simple sink filling - raise each sink to its spill point
        max_iterations = 100
        for iteration in range(max_iterations):
            # Find unfilled sinks
            local_min = minimum_filter(result, footprint=neighborhood, mode='nearest')
            sinks = result <= local_min + epsilon
            
            if not sinks.any():
                break
            
            # Raise sink elevations slightly
            result[sinks] += epsilon
        
        return result
    
    def _process_rivers(
        self,
        dem: np.ndarray,
        river_mask: np.ndarray,
        water_level: float,
        min_slope: float = 0.001
    ) -> np.ndarray:
        """
        Process river channels to ensure continuous downstream flow.
        
        Parameters
        ----------
        dem : np.ndarray
            Input DEM
        river_mask : np.ndarray
            Binary mask of river channels
        water_level : float
            Base water level
        min_slope : float
            Minimum slope along rivers
            
        Returns
        -------
        np.ndarray
            DEM with processed rivers
        """
        result = dem.copy()
        
        # Skeletonize river mask to get centerlines
        river_skel = skeletonize(river_mask)
        
        # Label river segments
        labeled_rivers, n_rivers = ndimage.label(river_skel)
        
        if n_rivers == 0:
            return result
        
        # Process each river segment
        for river_id in range(1, n_rivers + 1):
            river_pixels = np.argwhere(labeled_rivers == river_id)
            
            if len(river_pixels) < 3:
                continue
            
            # Find river endpoints (highest and lowest)
            elevations = result[river_pixels[:, 0], river_pixels[:, 1]]
            
            # Sort by elevation to find flow direction
            sorted_indices = np.argsort(elevations)[::-1]  # High to low
            
            # Ensure monotonic decrease
            for i in range(len(sorted_indices) - 1):
                idx_current = sorted_indices[i]
                idx_next = sorted_indices[i + 1]
                
                y_current, x_current = river_pixels[idx_current]
                y_next, x_next = river_pixels[idx_next]
                
                elev_current = result[y_current, x_current]
                elev_next = result[y_next, x_next]
                
                # Ensure minimum slope
                if elev_next >= elev_current - min_slope:
                    result[y_next, x_next] = elev_current - min_slope
        
        # Smooth river channels
        river_area = ndimage.binary_dilation(river_mask, iterations=2)
        result_rivers = result.copy()
        result_rivers[~river_area] = 0
        result_rivers = gaussian_filter(result_rivers, sigma=1)
        result[river_area] = result_rivers[river_area] * 0.7 + result[river_area] * 0.3
        
        return result
    
    def _enforce_drainage(
        self,
        dem: np.ndarray,
        terrain_map: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Enforce drainage patterns based on terrain analysis.
        
        Parameters
        ----------
        dem : np.ndarray
            Input DEM
        terrain_map : Dict[str, np.ndarray]
            Terrain analysis results
            
        Returns
        -------
        np.ndarray
            DEM with enforced drainage
        """
        result = dem.copy()
        
        # Get drainage pattern
        drainage = terrain_map.get('drainage_pattern', np.zeros_like(result))
        
        # Identify valley networks
        valley_mask = terrain_map.get('valley_mask', np.zeros_like(result, dtype=bool))
        
        # Calculate flow accumulation
        flow_dir = self._calculate_flow_direction(result)
        flow_acc = self._calculate_flow_accumulation(flow_dir)
        
        # Identify main drainage channels
        main_channels = flow_acc > np.percentile(flow_acc, 95)
        
        # Ensure valleys have decreasing elevation downstream
        result = self._correct_valley_gradients(result, valley_mask, flow_dir)
        
        return result
    
    def _calculate_flow_direction(self, dem: np.ndarray) -> np.ndarray:
        """
        Calculate D8 flow direction for each cell.
        
        Returns
        -------
        np.ndarray
            Flow direction (1, 2, 4, 8, 16, 32, 64, 128)
        """
        h, w = dem.shape
        flow_dir = np.zeros((h, w), dtype=np.uint8)
        
        # Calculate slopes to all 8 neighbors
        slopes = np.zeros((h, w, 8))
        
        # Neighbor offsets: E, SE, S, SW, W, NW, N, NE
        dy = [0, 1, 1, 1, 0, -1, -1, -1]
        dx = [1, 1, 0, -1, -1, -1, 0, 1]
        
        for i in range(8):
            # Roll array to get neighbor values
            neighbor = np.roll(np.roll(dem, dy[i], axis=0), dx[i], axis=1)
            
            # Calculate slope (elevation difference / distance)
            distance = np.sqrt(dy[i]**2 + dx[i]**2)
            slopes[:, :, i] = (dem - neighbor) / distance
        
        # Flow direction is toward steepest descent
        # D8 encoding: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
        d8_codes = [1, 2, 4, 8, 16, 32, 64, 128]
        
        # Find direction of maximum slope
        max_slope_idx = np.argmax(slopes, axis=2)
        
        # Only assign flow direction if slope is positive (downhill)
        max_slope = np.max(slopes, axis=2)
        has_flow = max_slope > 0
        
        for i, code in enumerate(d8_codes):
            flow_dir[(max_slope_idx == i) & has_flow] = code
        
        return flow_dir
    
    def _calculate_flow_accumulation(self, flow_dir: np.ndarray) -> np.ndarray:
        """
        Calculate flow accumulation from flow directions.
        
        Returns
        -------
        np.ndarray
            Flow accumulation (number of upstream cells)
        """
        h, w = flow_dir.shape
        flow_acc = np.ones((h, w), dtype=np.int32)
        
        # Invert flow direction to find upstream cells
        # This is a simplified approach - full algorithm would process in topological order
        
        # Multiple passes for convergence
        for _ in range(10):
            new_acc = flow_acc.copy()
            
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if flow_dir[y, x] > 0:
                        # Find downstream cell
                        dy, dx = self._flow_dir_to_offset(flow_dir[y, x])
                        ny, nx = y + dy, x + dx
                        
                        if 0 <= ny < h and 0 <= nx < w:
                            new_acc[ny, nx] += flow_acc[y, x]
            
            flow_acc = new_acc
        
        return flow_acc
    
    def _flow_dir_to_offset(self, direction: int) -> Tuple[int, int]:
        """Convert D8 flow direction to row/col offset."""
        offsets = {
            1: (0, 1),    # E
            2: (1, 1),    # SE
            4: (1, 0),    # S
            8: (1, -1),   # SW
            16: (0, -1),  # W
            32: (-1, -1), # NW
            64: (-1, 0),  # N
            128: (-1, 1), # NE
        }
        return offsets.get(direction, (0, 0))
    
    def _correct_valley_gradients(
        self,
        dem: np.ndarray,
        valley_mask: np.ndarray,
        flow_dir: np.ndarray
    ) -> np.ndarray:
        """Correct elevation gradients in valleys to ensure flow."""
        result = dem.copy()
        
        # Find valley pixels
        valley_pixels = np.argwhere(valley_mask)
        
        if len(valley_pixels) == 0:
            return result
        
        # Sort by elevation (highest first)
        elevations = result[valley_pixels[:, 0], valley_pixels[:, 1]]
        sorted_indices = np.argsort(elevations)[::-1]
        
        # Process from high to low
        min_slope = 0.001
        
        for idx in sorted_indices:
            y, x = valley_pixels[idx]
            
            # Find downstream neighbor
            if flow_dir[y, x] > 0:
                dy, dx = self._flow_dir_to_offset(flow_dir[y, x])
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < result.shape[0] and 0 <= nx < result.shape[1]:
                    # Ensure minimum slope
                    if result[ny, nx] >= result[y, x] - min_slope:
                        result[ny, nx] = result[y, x] - min_slope
        
        return result
    
    def _flatten_water_bodies(
        self,
        dem: np.ndarray,
        water_mask: np.ndarray,
        water_level: float
    ) -> np.ndarray:
        """Ensure water bodies have consistent elevation."""
        result = dem.copy()
        
        if not water_mask.any():
            return result
        
        # Label separate water bodies
        labeled_water, n_water = ndimage.label(water_mask)
        
        # Flatten each water body
        for water_id in range(1, n_water + 1):
            water_pixels = labeled_water == water_id
            
            # Set to water level with small depth variation
            n_pixels = water_pixels.sum()
            if n_pixels > 0:
                # Larger water bodies can have slight depth variation
                depth_variation = np.random.rand(*result.shape) * 0.5  # Up to 0.5m variation
                depth_variation = gaussian_filter(depth_variation, sigma=3)
                depth_variation[~water_pixels] = 0
                
                result[water_pixels] = water_level - depth_variation[water_pixels]
        
        return result
    
    def extract_watersheds(
        self,
        dem: np.ndarray,
        flow_acc: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract watershed boundaries from DEM.
        
        Parameters
        ----------
        dem : np.ndarray
            Input DEM
        flow_acc : np.ndarray, optional
            Pre-calculated flow accumulation
            
        Returns
        -------
        np.ndarray
            Watershed labels
        """
        if flow_acc is None:
            flow_dir = self._calculate_flow_direction(dem)
            flow_acc = self._calculate_flow_accumulation(flow_dir)
        
        # Find pour points (local maxima in flow accumulation)
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(flow_acc, size=3)
        pour_points = (flow_acc == local_max) & (flow_acc > flow_acc.mean())
        
        # Label pour points
        labeled_pours, n_pours = ndimage.label(pour_points)
        
        # Watershed from pour points
        from scipy.ndimage import watershed_ift
        
        # Invert DEM for watershed (we want drainage basins)
        inverted = dem.max() - dem
        
        # Watershed
        watersheds = watershed_ift(inverted.astype(np.uint16), labeled_pours)
        
        return watersheds
    
    def calculate_stream_order(
        self,
        flow_acc: np.ndarray,
        threshold: int = 100
    ) -> np.ndarray:
        """
        Calculate Strahler stream order.
        
        Parameters
        ----------
        flow_acc : np.ndarray
            Flow accumulation
        threshold : int
            Minimum accumulation to be considered a stream
            
        Returns
        -------
        np.ndarray
            Stream order for each cell
        """
        stream_mask = flow_acc >= threshold
        
        # Initialize stream order
        stream_order = np.zeros_like(flow_acc)
        stream_order[stream_mask] = 1
        
        # Propagate stream order downstream
        # This is a simplified version
        h, w = flow_acc.shape
        
        for _ in range(20):  # Multiple passes
            new_order = stream_order.copy()
            
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if stream_mask[y, x]:
                        # Find upstream cells
                        upstream_orders = []
                        
                        for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                                       (0, -1), (0, 1),
                                       (1, -1), (1, 0), (1, 1)]:
                            ny, nx = y + dy, x + dx
                            if stream_mask[ny, nx]:
                                upstream_orders.append(stream_order[ny, nx])
                        
                        if len(upstream_orders) >= 2:
                            # Strahler order rules
                            max_order = max(upstream_orders)
                            count_max = upstream_orders.count(max_order)
                            
                            if count_max >= 2:
                                new_order[y, x] = max_order + 1
                            else:
                                new_order[y, x] = max_order
            
            stream_order = new_order
        
        return stream_order
