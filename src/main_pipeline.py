"""
Main Pipeline Script
Complete pipeline for comparative analysis of brain MRI scans.
"""

import argparse
import logging
from pathlib import Path
import sys
import json
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from deepprep_analysis import (
    MRIPreprocessor,
    FeatureExtractor,
    ComparativeAnalyzer,
    Visualizer
)
from deepprep_analysis.data_generator import SyntheticMRIGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argparser():
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Comparative Analysis of Brain MRI Scans using DeepPrep Pipeline'
    )
    
    parser.add_argument(
        '--healthy-dir',
        type=str,
        default='data/healthy',
        help='Directory containing healthy MRI scans'
    )
    
    parser.add_argument(
        '--disorder-dir',
        type=str,
        default='data/disorder',
        help='Directory containing disorder MRI scans'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Generate synthetic data for testing'
    )
    
    parser.add_argument(
        '--n-healthy',
        type=int,
        default=20,
        help='Number of healthy subjects to generate'
    )
    
    parser.add_argument(
        '--n-disorder',
        type=int,
        default=20,
        help='Number of disorder subjects to generate'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing step (use if data is already preprocessed)'
    )
    
    parser.add_argument(
        '--normalization-method',
        type=str,
        default='z-score',
        choices=['z-score', 'min-max', 'percentile'],
        help='Method for intensity normalization'
    )
    
    return parser


def get_nifti_files(directory: str):
    """Get all NIfTI files from a directory."""
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory {directory} does not exist")
        return []
    
    # Look for .nii and .nii.gz files
    nifti_files = list(directory.glob('*.nii')) + list(directory.glob('*.nii.gz'))
    return [str(f) for f in nifti_files]


def main():
    """Main pipeline execution."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Starting Comparative Brain MRI Analysis Pipeline")
    logger.info("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate synthetic data if requested
    if args.generate_data:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Generating Synthetic Data")
        logger.info("="*80)
        
        generator = SyntheticMRIGenerator(image_shape=(64, 64, 64))
        generator.generate_dataset(
            n_healthy=args.n_healthy,
            n_disorder=args.n_disorder,
            output_dir='data'
        )
        logger.info("Synthetic data generation complete")
    
    # Step 2: Load data
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Loading MRI Scans")
    logger.info("="*80)
    
    healthy_files = get_nifti_files(args.healthy_dir)
    disorder_files = get_nifti_files(args.disorder_dir)
    
    logger.info(f"Found {len(healthy_files)} healthy scans")
    logger.info(f"Found {len(disorder_files)} disorder scans")
    
    if len(healthy_files) == 0 or len(disorder_files) == 0:
        logger.error("No MRI scans found. Please provide data or use --generate-data flag.")
        return
    
    # Step 3: Preprocessing
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Preprocessing MRI Scans")
    logger.info("="*80)
    
    preprocessor = MRIPreprocessor()
    
    if not args.skip_preprocessing:
        logger.info("Preprocessing healthy scans...")
        healthy_preprocessed = preprocessor.preprocess_dataset(
            healthy_files,
            normalize=True,
            remove_bg=True,
            skull_strip=True,
            normalization_method=args.normalization_method
        )
        
        logger.info("Preprocessing disorder scans...")
        disorder_preprocessed = preprocessor.preprocess_dataset(
            disorder_files,
            normalize=True,
            remove_bg=True,
            skull_strip=True,
            normalization_method=args.normalization_method
        )
    else:
        logger.info("Skipping preprocessing (loading raw data)")
        healthy_preprocessed = []
        disorder_preprocessed = []
        # Load raw data without preprocessing
        for filepath in healthy_files:
            data, img = preprocessor.load_mri_scan(filepath)
            healthy_preprocessed.append({
                'data': data,
                'affine': img.affine,
                'header': img.header,
                'filepath': filepath
            })
        for filepath in disorder_files:
            data, img = preprocessor.load_mri_scan(filepath)
            disorder_preprocessed.append({
                'data': data,
                'affine': img.affine,
                'header': img.header,
                'filepath': filepath
            })
    
    logger.info(f"Preprocessed {len(healthy_preprocessed)} healthy scans")
    logger.info(f"Preprocessed {len(disorder_preprocessed)} disorder scans")
    
    # Step 4: Feature Extraction
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Extracting Features")
    logger.info("="*80)
    
    feature_extractor = FeatureExtractor(n_pca_components=20)
    
    logger.info("Extracting features from healthy scans...")
    healthy_features = feature_extractor.extract_features_from_dataset(healthy_preprocessed)
    
    logger.info("Extracting features from disorder scans...")
    disorder_features = feature_extractor.extract_features_from_dataset(disorder_preprocessed)
    
    logger.info(f"Extracted features from {len(healthy_features)} healthy scans")
    logger.info(f"Extracted features from {len(disorder_features)} disorder scans")
    
    # Save features to CSV
    healthy_df = pd.DataFrame(healthy_features)
    disorder_df = pd.DataFrame(disorder_features)
    healthy_df.to_csv(output_dir / 'healthy_features.csv', index=False)
    disorder_df.to_csv(output_dir / 'disorder_features.csv', index=False)
    logger.info(f"Features saved to {output_dir}")
    
    # Step 5: Comparative Analysis
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Performing Comparative Analysis")
    logger.info("="*80)
    
    analyzer = ComparativeAnalyzer(alpha=0.05)
    analysis_results = analyzer.analyze_group_differences(
        healthy_features,
        disorder_features
    )
    
    # Save analysis results
    logger.info("Saving analysis results...")
    
    # Save t-test results
    analysis_results['t_tests'].to_csv(
        output_dir / 't_test_results.csv',
        index=False
    )
    
    # Save Mann-Whitney results
    analysis_results['mann_whitney'].to_csv(
        output_dir / 'mann_whitney_results.csv',
        index=False
    )
    
    # Save classification results
    classification_results = analysis_results['classification']
    with open(output_dir / 'classification_results.json', 'w') as f:
        json.dump({
            'cv_accuracy_mean': float(classification_results['cv_accuracy_mean']),
            'cv_accuracy_std': float(classification_results['cv_accuracy_std']),
            'n_features': classification_results['n_features']
        }, f, indent=2)
    
    classification_results['feature_importances'].to_csv(
        output_dir / 'feature_importances.csv',
        index=False
    )
    
    # Save top discriminative features
    top_features = analyzer.get_top_discriminative_features(n_top=20, method='both')
    top_features.to_csv(output_dir / 'top_discriminative_features.csv', index=False)
    
    logger.info(f"Analysis results saved to {output_dir}")
    
    # Step 6: Visualization
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Generating Visualizations")
    logger.info("="*80)
    
    visualizer = Visualizer(output_dir=str(output_dir))
    
    # Get sample MRI data for visualization
    sample_mri_data = {
        'healthy': healthy_preprocessed[0]['data'] if healthy_preprocessed else None,
        'disorder': disorder_preprocessed[0]['data'] if disorder_preprocessed else None
    }
    
    visualizer.generate_all_visualizations(
        analysis_results,
        sample_mri_data
    )
    
    logger.info(f"Visualizations saved to {output_dir}")
    
    # Step 7: Summary Report
    logger.info("\n" + "="*80)
    logger.info("STEP 7: Summary Report")
    logger.info("="*80)
    
    # Print key findings
    t_tests = analysis_results['t_tests']
    n_significant = t_tests['significant'].sum()
    
    logger.info(f"\nKey Findings:")
    logger.info(f"  - Total subjects: {analysis_results['n_healthy']} healthy, "
                f"{analysis_results['n_disorder']} disorder")
    logger.info(f"  - Total features analyzed: {len(t_tests)}")
    logger.info(f"  - Significant features (p < 0.05): {n_significant} ({n_significant/len(t_tests)*100:.1f}%)")
    logger.info(f"  - Classification accuracy: {classification_results['cv_accuracy_mean']:.3f} "
                f"± {classification_results['cv_accuracy_std']:.3f}")
    
    logger.info(f"\nTop 5 Most Discriminative Features:")
    top_5_features = t_tests.nsmallest(5, 'p_value')
    for idx, row in top_5_features.iterrows():
        logger.info(f"  {idx+1}. {row['feature']}")
        logger.info(f"     - p-value: {row['p_value']:.6f}")
        logger.info(f"     - Cohen's d: {row['cohens_d']:.3f}")
        logger.info(f"     - Mean difference: {row['mean_difference']:.3f}")
    
    logger.info("\n" + "="*80)
    logger.info("Pipeline Complete!")
    logger.info(f"All results saved to: {output_dir.absolute()}")
    logger.info("="*80)
    
    return analysis_results


if __name__ == '__main__':
    main()
