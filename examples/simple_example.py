"""
Simple Example Script
Quick demonstration of the brain MRI comparative analysis pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from deepprep_analysis import (
    MRIPreprocessor,
    FeatureExtractor,
    ComparativeAnalyzer,
    Visualizer
)
from deepprep_analysis.data_generator import SyntheticMRIGenerator

print("=" * 80)
print("Brain MRI Comparative Analysis - Simple Example")
print("=" * 80)

# Step 1: Generate synthetic data
print("\n1. Generating synthetic data...")
generator = SyntheticMRIGenerator(image_shape=(64, 64, 64))

# Generate small samples for quick demonstration
healthy_data = [generator.generate_healthy_mri(i) for i in range(5)]
disorder_data = [generator.generate_disorder_mri(i) for i in range(5)]

print(f"   Generated {len(healthy_data)} healthy and {len(disorder_data)} disorder samples")

# Step 2: Create preprocessed data structures
print("\n2. Preparing preprocessed data...")
import numpy as np

healthy_preprocessed = [
    {
        'data': data,
        'affine': np.eye(4),
        'header': None,
        'filepath': f'healthy_{i}.nii.gz',
        'stats': {}
    }
    for i, data in enumerate(healthy_data)
]

disorder_preprocessed = [
    {
        'data': data,
        'affine': np.eye(4),
        'header': None,
        'filepath': f'disorder_{i}.nii.gz',
        'stats': {}
    }
    for i, data in enumerate(disorder_data)
]

# Step 3: Extract features
print("\n3. Extracting features...")
feature_extractor = FeatureExtractor()

healthy_features = feature_extractor.extract_features_from_dataset(healthy_preprocessed)
disorder_features = feature_extractor.extract_features_from_dataset(disorder_preprocessed)

print(f"   Extracted {len(healthy_features[0]) - 1} features per sample")

# Step 4: Perform comparative analysis
print("\n4. Performing comparative analysis...")
analyzer = ComparativeAnalyzer(alpha=0.05)

analysis_results = analyzer.analyze_group_differences(
    healthy_features,
    disorder_features
)

# Step 5: Display results
print("\n5. Results:")
print("=" * 80)

t_tests = analysis_results['t_tests']
classification = analysis_results['classification']

print(f"\nSample Sizes:")
print(f"  Healthy: {analysis_results['n_healthy']}")
print(f"  Disorder: {analysis_results['n_disorder']}")

print(f"\nFeature Analysis:")
print(f"  Total features: {len(t_tests)}")
print(f"  Significant features (p < 0.05): {t_tests['significant'].sum()}")
print(f"  Percentage: {t_tests['significant'].sum() / len(t_tests) * 100:.1f}%")

print(f"\nClassification Performance:")
print(f"  Accuracy: {classification['cv_accuracy_mean']:.3f} ± {classification['cv_accuracy_std']:.3f}")

print(f"\nTop 5 Discriminative Features:")
top_5 = t_tests.nsmallest(5, 'p_value')
for idx, row in top_5.iterrows():
    print(f"  {idx+1}. {row['feature']}")
    print(f"     p-value: {row['p_value']:.6f}")
    print(f"     Cohen's d: {row['cohens_d']:.3f}")

print("\n" + "=" * 80)
print("Example complete!")
print("For full pipeline with visualizations, run: python src/main_pipeline.py --generate-data")
print("=" * 80)
