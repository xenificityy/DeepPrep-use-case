"""
Data Generator Module
Generates synthetic MRI data for testing and demonstration purposes.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticMRIGenerator:
    """
    Generates synthetic brain MRI data for testing.
    
    Creates realistic-looking 3D brain scans with different characteristics
    for healthy and disorder groups.
    """
    
    def __init__(self, image_shape: Tuple[int, int, int] = (64, 64, 64)):
        """
        Initialize Synthetic MRI Generator.
        
        Args:
            image_shape: Shape of the 3D MRI volume
        """
        self.image_shape = image_shape
        
    def create_brain_mask(self) -> np.ndarray:
        """
        Create a brain-like ellipsoidal mask.
        
        Returns:
            3D binary mask representing brain shape
        """
        x, y, z = np.ogrid[
            -self.image_shape[0]//2:self.image_shape[0]//2,
            -self.image_shape[1]//2:self.image_shape[1]//2,
            -self.image_shape[2]//2:self.image_shape[2]//2
        ]
        
        # Create ellipsoid (brain-like shape)
        a, b, c = self.image_shape[0]//2.5, self.image_shape[1]//2.5, self.image_shape[2]//2.5
        mask = (x**2 / a**2 + y**2 / b**2 + z**2 / c**2) <= 1
        
        return mask.astype(float)
    
    def add_brain_structures(self, base_image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """
        Add simulated brain structures (ventricles, gray matter, white matter).
        
        Args:
            base_image: Base brain mask
            intensity: Overall intensity scale
            
        Returns:
            MRI image with simulated structures
        """
        image = base_image.copy() * intensity
        
        # Add ventricles (darker regions in center)
        x, y, z = np.ogrid[
            -self.image_shape[0]//2:self.image_shape[0]//2,
            -self.image_shape[1]//2:self.image_shape[1]//2,
            -self.image_shape[2]//2:self.image_shape[2]//2
        ]
        
        ventricles = (x**2 / 5**2 + y**2 / 5**2 + z**2 / 10**2) <= 1
        image[ventricles] *= 0.3
        
        # Add gray matter variations
        from scipy import ndimage
        gray_matter_variation = ndimage.gaussian_filter(
            np.random.randn(*self.image_shape), sigma=3
        )
        image = image * (1 + 0.2 * gray_matter_variation)
        
        # Ensure non-negative
        image = np.maximum(image, 0)
        
        return image
    
    def add_disorder_characteristics(
        self,
        image: np.ndarray,
        disorder_type: str = 'atrophy'
    ) -> np.ndarray:
        """
        Add characteristics typical of neurological disorders.
        
        Args:
            image: Base MRI image
            disorder_type: Type of disorder characteristics to add
                          ('atrophy', 'lesions', 'intensity_change')
            
        Returns:
            Modified MRI image with disorder characteristics
        """
        modified_image = image.copy()
        
        if disorder_type == 'atrophy':
            # Simulate brain atrophy (reduced volume)
            from scipy import ndimage
            modified_image = ndimage.zoom(modified_image, 0.95, order=1)
            # Pad to original size
            pad_width = [(
                (self.image_shape[i] - modified_image.shape[i]) // 2,
                (self.image_shape[i] - modified_image.shape[i]) - 
                (self.image_shape[i] - modified_image.shape[i]) // 2
            ) for i in range(3)]
            modified_image = np.pad(modified_image, pad_width, mode='constant')
            
        elif disorder_type == 'lesions':
            # Add random lesions (hyperintense spots)
            n_lesions = np.random.randint(5, 15)
            for _ in range(n_lesions):
                center = tuple(np.random.randint(10, s-10) for s in self.image_shape)
                radius = np.random.randint(2, 5)
                
                x, y, z = np.ogrid[:self.image_shape[0], :self.image_shape[1], :self.image_shape[2]]
                lesion_mask = (
                    (x - center[0])**2 + 
                    (y - center[1])**2 + 
                    (z - center[2])**2
                ) <= radius**2
                
                modified_image[lesion_mask] *= 1.5
                
        elif disorder_type == 'intensity_change':
            # Global intensity changes
            modified_image *= 0.85  # Reduced overall intensity
            
        return modified_image
    
    def add_noise(self, image: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        Add Gaussian noise to simulate MRI acquisition noise.
        
        Args:
            image: MRI image
            noise_level: Standard deviation of noise relative to signal
            
        Returns:
            Noisy MRI image
        """
        noise = np.random.randn(*image.shape) * noise_level * np.max(image)
        noisy_image = image + noise
        return np.maximum(noisy_image, 0)
    
    def generate_healthy_mri(self, subject_id: int = 0) -> np.ndarray:
        """
        Generate a synthetic healthy brain MRI scan.
        
        Args:
            subject_id: Subject identifier for random seed
            
        Returns:
            3D MRI image
        """
        np.random.seed(subject_id)
        
        # Create base brain mask
        brain_mask = self.create_brain_mask()
        
        # Add structures with healthy characteristics
        intensity = np.random.uniform(0.8, 1.0)
        mri_image = self.add_brain_structures(brain_mask, intensity)
        
        # Add noise
        mri_image = self.add_noise(mri_image, noise_level=0.03)
        
        logger.debug(f"Generated healthy MRI for subject {subject_id}")
        return mri_image
    
    def generate_disorder_mri(
        self,
        subject_id: int = 0,
        disorder_type: str = 'mixed'
    ) -> np.ndarray:
        """
        Generate a synthetic MRI scan with disorder characteristics.
        
        Args:
            subject_id: Subject identifier for random seed
            disorder_type: Type of disorder ('atrophy', 'lesions', 'intensity_change', 'mixed')
            
        Returns:
            3D MRI image
        """
        np.random.seed(subject_id + 10000)  # Different seed range
        
        # Create base brain mask
        brain_mask = self.create_brain_mask()
        
        # Add structures
        intensity = np.random.uniform(0.7, 0.9)
        mri_image = self.add_brain_structures(brain_mask, intensity)
        
        # Add disorder characteristics
        if disorder_type == 'mixed':
            # Randomly select disorder type
            disorder_types = ['atrophy', 'lesions', 'intensity_change']
            selected_type = np.random.choice(disorder_types)
        else:
            selected_type = disorder_type
        
        mri_image = self.add_disorder_characteristics(mri_image, selected_type)
        
        # Add more noise
        mri_image = self.add_noise(mri_image, noise_level=0.05)
        
        logger.debug(f"Generated disorder MRI for subject {subject_id} with {selected_type}")
        return mri_image
    
    def save_as_nifti(
        self,
        data: np.ndarray,
        filepath: str,
        affine: np.ndarray = None
    ):
        """
        Save MRI data as NIfTI file.
        
        Args:
            data: 3D MRI data
            filepath: Output file path
            affine: Affine transformation matrix (if None, uses identity)
        """
        if affine is None:
            affine = np.eye(4)
        
        nifti_img = nib.Nifti1Image(data, affine)
        nib.save(nifti_img, filepath)
        logger.info(f"Saved NIfTI file to {filepath}")
    
    def generate_dataset(
        self,
        n_healthy: int = 20,
        n_disorder: int = 20,
        output_dir: str = 'data'
    ):
        """
        Generate a complete synthetic dataset.
        
        Args:
            n_healthy: Number of healthy subjects
            n_disorder: Number of disorder subjects
            output_dir: Output directory for data files
        """
        output_path = Path(output_dir)
        healthy_dir = output_path / 'healthy'
        disorder_dir = output_path / 'disorder'
        
        healthy_dir.mkdir(parents=True, exist_ok=True)
        disorder_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {n_healthy} healthy and {n_disorder} disorder MRI scans")
        
        # Generate healthy scans
        for i in range(n_healthy):
            mri_data = self.generate_healthy_mri(subject_id=i)
            filepath = healthy_dir / f'healthy_subject_{i:03d}.nii.gz'
            self.save_as_nifti(mri_data, str(filepath))
        
        # Generate disorder scans
        for i in range(n_disorder):
            mri_data = self.generate_disorder_mri(subject_id=i)
            filepath = disorder_dir / f'disorder_subject_{i:03d}.nii.gz'
            self.save_as_nifti(mri_data, str(filepath))
        
        logger.info(f"Dataset generation complete. Data saved to {output_path}")
