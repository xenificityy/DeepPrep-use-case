"""
MRI Preprocessing Module
Handles preprocessing of brain MRI scans following DeepPrep pipeline principles.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRIPreprocessor:
    """
    Preprocessor for brain MRI scans following DeepPrep pipeline.
    
    Performs standardization, normalization, and quality checks on MRI data.
    """
    
    def __init__(self, target_shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize the MRI Preprocessor.
        
        Args:
            target_shape: Target shape for resizing images. If None, keeps original shape.
        """
        self.target_shape = target_shape
        self.preprocessing_stats = {}
        
    def load_mri_scan(self, filepath: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """
        Load MRI scan from file.
        
        Args:
            filepath: Path to the MRI scan file (NIfTI format)
            
        Returns:
            Tuple of (numpy array of image data, nibabel image object)
        """
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            logger.info(f"Loaded MRI scan from {filepath} with shape {data.shape}")
            return data, img
        except Exception as e:
            logger.error(f"Error loading MRI scan from {filepath}: {e}")
            raise
    
    def normalize_intensity(self, data: np.ndarray, method: str = 'z-score') -> np.ndarray:
        """
        Normalize intensity values of MRI scan.
        
        Args:
            data: MRI scan data
            method: Normalization method ('z-score', 'min-max', or 'percentile')
            
        Returns:
            Normalized MRI data
        """
        if method == 'z-score':
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                normalized = (data - mean) / std
            else:
                normalized = data - mean
        elif method == 'min-max':
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val > min_val:
                normalized = (data - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(data)
        elif method == 'percentile':
            p1, p99 = np.percentile(data, [1, 99])
            normalized = np.clip((data - p1) / (p99 - p1), 0, 1)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        logger.info(f"Applied {method} normalization")
        return normalized
    
    def remove_background(self, data: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Remove background noise from MRI scan.
        
        Args:
            data: MRI scan data
            threshold: Threshold for background removal
            
        Returns:
            MRI data with background removed
        """
        # Create binary mask
        mask = data > threshold
        cleaned_data = data * mask
        logger.info(f"Removed background with threshold {threshold}")
        return cleaned_data
    
    def skull_stripping_simulation(self, data: np.ndarray) -> np.ndarray:
        """
        Simulate skull stripping to extract brain tissue.
        
        Args:
            data: MRI scan data
            
        Returns:
            Skull-stripped MRI data
        """
        # Simple threshold-based skull stripping simulation
        # In real scenarios, use tools like FSL BET or FreeSurfer
        threshold = np.percentile(data[data > 0], 20)
        brain_mask = data > threshold
        
        # Apply morphological operations to clean up the mask
        from scipy import ndimage
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        brain_mask = ndimage.binary_erosion(brain_mask, iterations=2)
        brain_mask = ndimage.binary_dilation(brain_mask, iterations=2)
        
        stripped_data = data * brain_mask
        logger.info("Applied skull stripping simulation")
        return stripped_data
    
    def preprocess_scan(
        self,
        filepath: str,
        normalize: bool = True,
        remove_bg: bool = True,
        skull_strip: bool = True,
        normalization_method: str = 'z-score'
    ) -> Dict:
        """
        Complete preprocessing pipeline for a single MRI scan.
        
        Args:
            filepath: Path to MRI scan file
            normalize: Whether to normalize intensity
            remove_bg: Whether to remove background
            skull_strip: Whether to perform skull stripping
            normalization_method: Method for intensity normalization
            
        Returns:
            Dictionary containing preprocessed data and metadata
        """
        # Load scan
        data, img = self.load_mri_scan(filepath)
        original_shape = data.shape
        
        # Preprocessing steps
        if remove_bg:
            data = self.remove_background(data)
        
        if skull_strip:
            data = self.skull_stripping_simulation(data)
        
        if normalize:
            data = self.normalize_intensity(data, method=normalization_method)
        
        # Store preprocessing statistics
        stats = {
            'original_shape': original_shape,
            'final_shape': data.shape,
            'mean_intensity': float(np.mean(data)),
            'std_intensity': float(np.std(data)),
            'min_intensity': float(np.min(data)),
            'max_intensity': float(np.max(data)),
            'non_zero_voxels': int(np.count_nonzero(data))
        }
        
        result = {
            'data': data,
            'affine': img.affine,
            'header': img.header,
            'filepath': filepath,
            'stats': stats
        }
        
        logger.info(f"Completed preprocessing for {filepath}")
        return result
    
    def preprocess_dataset(
        self,
        filepaths: List[str],
        **kwargs
    ) -> List[Dict]:
        """
        Preprocess multiple MRI scans.
        
        Args:
            filepaths: List of paths to MRI scan files
            **kwargs: Additional arguments passed to preprocess_scan
            
        Returns:
            List of dictionaries containing preprocessed data and metadata
        """
        results = []
        for filepath in filepaths:
            try:
                result = self.preprocess_scan(filepath, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to preprocess {filepath}: {e}")
                continue
        
        logger.info(f"Preprocessed {len(results)}/{len(filepaths)} scans successfully")
        return results
