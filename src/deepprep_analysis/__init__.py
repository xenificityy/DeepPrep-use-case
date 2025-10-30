"""
DeepPrep Analysis Package
Comparative study of brain MRI scans between healthy individuals 
and patients with neurological disorders.
"""

__version__ = "0.1.0"

from .preprocessing import MRIPreprocessor
from .feature_extraction import FeatureExtractor
from .comparative_analysis import ComparativeAnalyzer
from .visualization import Visualizer

__all__ = [
    "MRIPreprocessor",
    "FeatureExtractor",
    "ComparativeAnalyzer",
    "Visualizer",
]
