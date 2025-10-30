"""
Feature Extraction Module
Extracts structural and quantitative features from preprocessed MRI scans.
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import ndimage
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts features from brain MRI scans for comparative analysis.
    
    Features include:
    - Volume measurements
    - Intensity statistics
    - Texture features
    - Morphological features
    - Regional features
    """
    
    def __init__(self, n_pca_components: int = 50):
        """
        Initialize Feature Extractor.
        
        Args:
            n_pca_components: Number of PCA components for dimensionality reduction
        """
        self.n_pca_components = n_pca_components
        self.pca_model = None
        
    def extract_volume_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract volume-based features.
        
        Args:
            data: Preprocessed MRI scan data
            
        Returns:
            Dictionary of volume features
        """
        total_voxels = data.size
        brain_voxels = np.count_nonzero(data)
        brain_fraction = brain_voxels / total_voxels if total_voxels > 0 else 0
        
        features = {
            'total_volume': float(total_voxels),
            'brain_volume': float(brain_voxels),
            'brain_fraction': float(brain_fraction),
            'non_brain_volume': float(total_voxels - brain_voxels)
        }
        
        return features
    
    def extract_intensity_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract intensity-based statistical features.
        
        Args:
            data: Preprocessed MRI scan data
            
        Returns:
            Dictionary of intensity features
        """
        # Focus on non-zero voxels (brain tissue)
        brain_data = data[data > 0]
        
        if len(brain_data) == 0:
            return {
                'mean_intensity': 0.0,
                'std_intensity': 0.0,
                'median_intensity': 0.0,
                'min_intensity': 0.0,
                'max_intensity': 0.0,
                'intensity_range': 0.0,
                'q25_intensity': 0.0,
                'q75_intensity': 0.0,
                'iqr_intensity': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
        
        from scipy import stats
        
        features = {
            'mean_intensity': float(np.mean(brain_data)),
            'std_intensity': float(np.std(brain_data)),
            'median_intensity': float(np.median(brain_data)),
            'min_intensity': float(np.min(brain_data)),
            'max_intensity': float(np.max(brain_data)),
            'intensity_range': float(np.max(brain_data) - np.min(brain_data)),
            'q25_intensity': float(np.percentile(brain_data, 25)),
            'q75_intensity': float(np.percentile(brain_data, 75)),
            'iqr_intensity': float(np.percentile(brain_data, 75) - np.percentile(brain_data, 25)),
            'skewness': float(stats.skew(brain_data)),
            'kurtosis': float(stats.kurtosis(brain_data))
        }
        
        return features
    
    def extract_texture_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features using statistical methods.
        
        Args:
            data: Preprocessed MRI scan data
            
        Returns:
            Dictionary of texture features
        """
        # Gray Level Co-occurrence Matrix (GLCM) inspired features
        # Simplified version for demonstration
        
        brain_data = data[data > 0]
        
        if len(brain_data) == 0:
            return {
                'entropy': 0.0,
                'energy': 0.0,
                'homogeneity': 0.0,
                'contrast_measure': 0.0
            }
        
        # Histogram-based texture features
        hist, _ = np.histogram(brain_data, bins=50, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        
        features = {
            'entropy': float(-np.sum(hist * np.log2(hist))),
            'energy': float(np.sum(hist ** 2)),
            'homogeneity': float(np.sum(hist / (1 + np.arange(len(hist))))),
            'contrast_measure': float(np.std(brain_data) / (np.mean(brain_data) + 1e-10))
        }
        
        return features
    
    def extract_morphological_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features from brain structure.
        
        Args:
            data: Preprocessed MRI scan data
            
        Returns:
            Dictionary of morphological features
        """
        # Create binary mask
        brain_mask = data > 0
        
        if not np.any(brain_mask):
            return {
                'surface_area': 0.0,
                'compactness': 0.0,
                'sphericity': 0.0,
                'elongation': 0.0
            }
        
        # Surface area estimation (boundary voxels)
        from scipy import ndimage
        eroded = ndimage.binary_erosion(brain_mask)
        boundary = brain_mask & ~eroded
        surface_area = float(np.sum(boundary))
        
        # Volume
        volume = float(np.sum(brain_mask))
        
        # Compactness: ratio of volume to surface area
        compactness = volume / (surface_area + 1e-10)
        
        # Estimate shape characteristics using moments
        coords = np.argwhere(brain_mask)
        if len(coords) > 0:
            centroid = coords.mean(axis=0)
            centered_coords = coords - centroid
            cov_matrix = np.cov(centered_coords.T)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
            
            # Sphericity: measure of how sphere-like the shape is
            sphericity = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
            
            # Elongation: ratio of largest to second largest eigenvalue
            elongation = eigenvalues[0] / (eigenvalues[1] + 1e-10)
        else:
            sphericity = 0.0
            elongation = 0.0
        
        features = {
            'surface_area': surface_area,
            'compactness': float(compactness),
            'sphericity': float(sphericity),
            'elongation': float(elongation)
        }
        
        return features
    
    def extract_regional_features(self, data: np.ndarray, n_regions: int = 8) -> Dict[str, float]:
        """
        Extract features from different brain regions.
        
        Args:
            data: Preprocessed MRI scan data
            n_regions: Number of regions to divide the brain into
            
        Returns:
            Dictionary of regional features
        """
        # Divide brain into regions along each axis
        shape = data.shape
        features = {}
        
        # Axial slices (top to bottom)
        for i in range(n_regions):
            start = i * shape[2] // n_regions
            end = (i + 1) * shape[2] // n_regions
            region_data = data[:, :, start:end]
            region_mean = float(np.mean(region_data[region_data > 0])) if np.any(region_data > 0) else 0.0
            features[f'axial_region_{i}_mean'] = region_mean
        
        # Sagittal slices (left to right)
        for i in range(n_regions):
            start = i * shape[0] // n_regions
            end = (i + 1) * shape[0] // n_regions
            region_data = data[start:end, :, :]
            region_mean = float(np.mean(region_data[region_data > 0])) if np.any(region_data > 0) else 0.0
            features[f'sagittal_region_{i}_mean'] = region_mean
        
        return features
    
    def extract_all_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract all available features from MRI scan.
        
        Args:
            data: Preprocessed MRI scan data
            
        Returns:
            Dictionary containing all extracted features
        """
        logger.info("Extracting all features from MRI scan")
        
        features = {}
        
        # Extract different feature types
        features.update(self.extract_volume_features(data))
        features.update(self.extract_intensity_features(data))
        features.update(self.extract_texture_features(data))
        features.update(self.extract_morphological_features(data))
        features.update(self.extract_regional_features(data))
        
        logger.info(f"Extracted {len(features)} features")
        return features
    
    def extract_features_from_dataset(
        self,
        preprocessed_scans: List[Dict]
    ) -> List[Dict[str, float]]:
        """
        Extract features from multiple preprocessed scans.
        
        Args:
            preprocessed_scans: List of preprocessed scan dictionaries
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for scan in preprocessed_scans:
            try:
                features = self.extract_all_features(scan['data'])
                features['filepath'] = scan['filepath']
                features_list.append(features)
            except Exception as e:
                logger.error(f"Failed to extract features from {scan.get('filepath', 'unknown')}: {e}")
                continue
        
        logger.info(f"Extracted features from {len(features_list)} scans")
        return features_list
    
    def apply_pca(self, features_matrix: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            features_matrix: Matrix of features (n_samples x n_features)
            fit: Whether to fit the PCA model or use existing one
            
        Returns:
            Transformed features
        """
        if fit:
            self.pca_model = PCA(n_components=min(self.n_pca_components, features_matrix.shape[1]))
            transformed = self.pca_model.fit_transform(features_matrix)
            logger.info(f"PCA fitted with {self.pca_model.n_components_} components")
            logger.info(f"Explained variance ratio: {np.sum(self.pca_model.explained_variance_ratio_):.3f}")
        else:
            if self.pca_model is None:
                raise ValueError("PCA model not fitted yet. Set fit=True first.")
            transformed = self.pca_model.transform(features_matrix)
        
        return transformed
