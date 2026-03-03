"""Feature extraction module for analyzing artistic maps."""

import numpy as np
from skimage import feature, filters, morphology, segmentation
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from typing import Dict, Tuple, Optional
import cv2


class FeatureExtractor:
    """
    Extract visual features from artistic maps.
    
    This class analyzes the input image to extract:
    - Color features (dominant colors, color clusters)
    - Texture features (roughness, patterns)
    - Edge features (contours, boundaries)
    - Pattern features (repetitive structures)
    """
    
    def __init__(
        self,
        n_color_clusters: int = 6,
        edge_sigma: float = 2.0,
        texture_scales: Tuple[int, ...] = (1, 2),
        min_region_size: int = 100
    ):
        self.n_color_clusters = n_color_clusters
        self.edge_sigma = edge_sigma
        self.texture_scales = texture_scales
        self.min_region_size = min_region_size
        
    def extract(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all features from the input image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (H, W) grayscale or (C, H, W) or (H, W, C) RGB
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of extracted features
        """
        # Normalize and prepare image
        img_normalized = self._normalize_image(image)
        
        features = {
            'color': self._extract_color_features(img_normalized),
            'edges': self._extract_edge_features(img_normalized),
            'texture': self._extract_texture_features(img_normalized),
            'patterns': self._extract_pattern_features(img_normalized),
            'regions': self._extract_region_features(img_normalized),
        }
        
        return features
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to standard format (H, W, C) float32."""
        # Handle different input formats
        if image.ndim == 2:
            # Grayscale - convert to RGB
            img = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3:
            if image.shape[0] == 3 or image.shape[0] == 4:
                # (C, H, W) format - transpose to (H, W, C)
                img = np.transpose(image, (1, 2, 0))
            else:
                # Already (H, W, C)
                img = image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Normalize to 0-1 range
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1.0:
            img = img / img.max()
            
        # Take only first 3 channels (RGB)
        img = img[:, :, :3]
        
        return img
    
    def _extract_color_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract color-based features."""
        h, w = image.shape[:2]
        
        # Convert to different color spaces
        hsv = rgb2hsv(image)
        gray = rgb2gray(image)
        
        # HSV components
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Color clustering using K-means - subsample for large images
        pixels = image.reshape(-1, 3)
        
        # Subsample if too many pixels
        max_samples = 10000
        if len(pixels) > max_samples:
            indices = np.random.choice(len(pixels), max_samples, replace=False)
            sample_pixels = pixels[indices]
        else:
            sample_pixels = pixels
        
        # Simple color clustering
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_color_clusters,
            random_state=42,
            batch_size=min(1000, len(sample_pixels)),
            max_iter=50
        )
        kmeans.fit(sample_pixels)
        
        # Predict labels for all pixels
        labels = kmeans.predict(pixels)
        color_labels = labels.reshape(h, w)
        
        # Calculate color distances from cluster centers (simplified)
        color_distances = np.zeros((h, w, self.n_color_clusters))
        for i, center in enumerate(kmeans.cluster_centers_):
            dist = np.abs(image - center.reshape(1, 1, 3)).sum(axis=2)
            color_distances[:, :, i] = dist
        
        # Dominant color map
        dominant_colors = kmeans.cluster_centers_[color_labels]
        
        # Brightness and contrast
        local_mean = filters.gaussian(gray, sigma=5)
        local_std = np.sqrt(filters.gaussian((gray - local_mean) ** 2, sigma=5))
        
        return {
            'rgb': image,
            'hsv': hsv,
            'hue': hue,
            'saturation': saturation,
            'value': value,
            'gray': gray,
            'color_labels': color_labels,
            'color_distances': color_distances,
            'dominant_colors': dominant_colors,
            'local_mean': local_mean,
            'local_std': local_std,
            'cluster_centers': kmeans.cluster_centers_,
        }
    
    def _extract_edge_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract edge and contour features."""
        gray = rgb2gray(image)
        
        # Multi-scale edge detection
        edges_canny = feature.canny(gray, sigma=self.edge_sigma)
        edges_canny_fine = feature.canny(gray, sigma=1.0)
        edges_canny_coarse = feature.canny(gray, sigma=4.0)
        
        # Sobel edges
        sobel_x = sobel(gray, axis=1)
        sobel_y = sobel(gray, axis=0)
        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_dir = np.arctan2(sobel_y, sobel_x)
        
        # Laplacian for second derivatives (ridges/valleys)
        laplacian = filters.laplace(gray)
        
        # Ridge detection
        ridges = self._detect_ridges(gray)
        
        # Contour detection using OpenCV
        gray_uint8 = (gray * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            gray_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create contour mask
        contour_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(contour_mask, contours, -1, 1, 1)
        
        return {
            'edges_canny': edges_canny,
            'edges_canny_fine': edges_canny_fine,
            'edges_canny_coarse': edges_canny_coarse,
            'sobel_magnitude': sobel_mag,
            'sobel_direction': sobel_dir,
            'sobel_x': sobel_x,
            'sobel_y': sobel_y,
            'laplacian': laplacian,
            'ridges': ridges,
            'contour_mask': contour_mask.astype(bool),
            'contours': contours,
        }
    
    def _detect_ridges(self, image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Detect ridge lines using Hessian matrix eigenvalues."""
        # Compute Hessian matrix elements
        Ixx = ndimage.gaussian_filter(image, sigma=sigma, order=(2, 0))
        Iyy = ndimage.gaussian_filter(image, sigma=sigma, order=(0, 2))
        Ixy = ndimage.gaussian_filter(image, sigma=sigma, order=(1, 1))
        
        # Eigenvalues of Hessian
        trace = Ixx + Iyy
        determinant = Ixx * Iyy - Ixy ** 2
        
        # Ridge measure (negative eigenvalue indicates ridge)
        sqrt_term = np.sqrt(np.maximum(trace**2 / 4 - determinant, 0))
        eigenvalue1 = trace / 2 + sqrt_term
        eigenvalue2 = trace / 2 - sqrt_term
        
        # Ridge strength
        ridge_strength = np.maximum(-eigenvalue2, 0)
        
        return ridge_strength
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract texture features at multiple scales."""
        gray = rgb2gray(image)
        
        texture_features = {}
        
        for scale in self.texture_scales:
            # Local standard deviation (roughness)
            smoothed = filters.gaussian(gray, sigma=scale)
            local_var = filters.gaussian((gray - smoothed) ** 2, sigma=scale)
            roughness = np.sqrt(local_var)
            
            # Local binary pattern (texture pattern)
            lbp = self._local_binary_pattern(gray, P=8 * scale, R=scale)
            
            # Gabor filter responses
            gabor_real, gabor_imag = filters.gabor(
                gray, 
                frequency=0.1 / scale,
                theta=0
            )
            gabor_mag = np.sqrt(gabor_real ** 2 + gabor_imag ** 2)
            
            texture_features[f'roughness_scale_{scale}'] = roughness
            texture_features[f'lbp_scale_{scale}'] = lbp
            texture_features[f'gabor_scale_{scale}'] = gabor_mag
        
        # Overall texture complexity
        texture_complexity = np.zeros_like(gray)
        for key in texture_features:
            if 'roughness' in key:
                texture_complexity += texture_features[key]
        
        texture_features['complexity'] = texture_complexity / len(self.texture_scales)
        
        # Fractal dimension estimate (using box counting approximation)
        texture_features['fractal_dimension'] = self._estimate_fractal_dimension(gray)
        
        return texture_features
    
    def _local_binary_pattern(
        self, 
        image: np.ndarray, 
        P: int = 8, 
        R: int = 1
    ) -> np.ndarray:
        """Compute Local Binary Pattern using scikit-image."""
        from skimage.feature import local_binary_pattern as skimage_lbp
        
        # Use scikit-image's implementation
        lbp = skimage_lbp(image, P=P, R=R, method='uniform')
        
        return lbp
    
    def _estimate_fractal_dimension(self, image: np.ndarray) -> np.ndarray:
        """Estimate local fractal dimension using variance at multiple scales."""
        h, w = image.shape
        
        # Use local variance as proxy for fractal dimension
        from scipy.ndimage import gaussian_filter
        
        # Calculate local roughness at different scales
        scales = [1, 2, 4, 8]
        roughness = np.zeros_like(image)
        
        for scale in scales:
            smoothed = gaussian_filter(image, sigma=scale)
            residual = np.abs(image - smoothed)
            roughness += residual / scale
        
        # Normalize
        roughness = roughness / len(scales)
        
        return roughness
    
    def _extract_pattern_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract repetitive pattern features."""
        gray = rgb2gray(image)
        
        # Fourier analysis for periodic patterns
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Dominant orientations
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        
        # Compute orientation histogram from FFT
        Y, X = np.ogrid[:h, :w]
        angles = np.arctan2(Y - cy, X - cx)
        distances = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        
        # Mask center (DC component)
        mask = distances > 5
        
        # Orientation strength
        orientation_bins = np.linspace(-np.pi, np.pi, 37)
        orientation_strength = np.zeros(36)
        
        for i in range(36):
            angle_mask = (angles >= orientation_bins[i]) & (angles < orientation_bins[i + 1])
            orientation_strength[i] = magnitude[mask & angle_mask].mean()
        
        # Pattern periodicity map
        pattern_strength = np.zeros_like(gray)
        
        # Use local autocorrelation for pattern detection
        for dy in range(-5, 6, 2):
            for dx in range(-5, 6, 2):
                if dx == 0 and dy == 0:
                    continue
                shifted = np.roll(gray, (dy, dx), axis=(0, 1))
                correlation = gray * shifted
                pattern_strength += correlation
        
        pattern_strength /= 25  # Normalize
        
        return {
            'fft_magnitude': magnitude,
            'orientation_strength': orientation_strength,
            'pattern_strength': pattern_strength,
            'dominant_orientation': angles,
        }
    
    def _extract_region_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract region/segment features."""
        gray = rgb2gray(image)
        h, w = gray.shape
        
        # Simple grid-based segmentation (fast and memory efficient)
        segments = np.zeros((h, w), dtype=np.int32)
        grid_size = max(20, min(h, w) // 20)
        label = 1
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                segments[i:min(i+grid_size, h), j:min(j+grid_size, w)] = label
                label += 1
        
        # Simple region statistics
        region_mean = gray.copy()
        region_std = np.zeros_like(gray)
        
        # Calculate local std in small windows
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(gray, size=5)
        local_sqr_mean = uniform_filter(gray**2, size=5)
        region_std = np.sqrt(np.maximum(0, local_sqr_mean - local_mean**2))
        
        return {
            'segments': segments,
            'region_size': np.ones_like(gray) * grid_size**2,
            'region_mean': region_mean,
            'region_std': region_std,
            'n_regions': label - 1,
        }
