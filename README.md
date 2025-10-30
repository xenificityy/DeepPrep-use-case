# DeepPrep-use-case: Comparative Analysis of Brain MRI Scans

A comprehensive pipeline for performing comparative studies between healthy individuals and patients with neurological disorders using brain MRI scans. This project implements preprocessing, feature extraction, statistical analysis, and visualization following DeepPrep pipeline principles.

## Overview

This project provides a complete framework for:
- Preprocessing brain MRI scans (normalization, skull stripping, background removal)
- Extracting structural and quantitative features from brain images
- Performing statistical comparisons between healthy and disorder groups
- Visualizing results and generating comprehensive reports
- Testing with synthetic data generation

## Features

### 1. MRI Preprocessing (`preprocessing.py`)
- **Intensity Normalization**: Z-score, min-max, and percentile-based normalization
- **Background Removal**: Threshold-based background noise removal
- **Skull Stripping**: Automated brain tissue extraction
- **Quality Checks**: Preprocessing statistics and validation

### 2. Feature Extraction (`feature_extraction.py`)
- **Volume Features**: Total volume, brain volume, brain fraction
- **Intensity Features**: Mean, std, median, range, skewness, kurtosis
- **Texture Features**: Entropy, energy, homogeneity, contrast
- **Morphological Features**: Surface area, compactness, sphericity, elongation
- **Regional Features**: Axial and sagittal region-wise analysis
- **PCA**: Dimensionality reduction for high-dimensional features

### 3. Comparative Analysis (`comparative_analysis.py`)
- **Statistical Tests**: Independent t-tests and Mann-Whitney U tests
- **Effect Sizes**: Cohen's d calculation for group differences
- **Multiple Testing Correction**: Bonferroni correction
- **Classification**: Random Forest-based group classification
- **Feature Importance**: Identification of most discriminative features

### 4. Visualization (`visualization.py`)
- MRI slice visualizations
- Feature distribution comparisons
- Statistical test results (t-tests, effect sizes)
- Feature importance plots
- Correlation heatmaps
- PCA scatter plots
- Comprehensive summary reports

### 5. Synthetic Data Generation (`data_generator.py`)
- Generate realistic synthetic brain MRI scans
- Simulate healthy brain characteristics
- Simulate disorder characteristics (atrophy, lesions, intensity changes)
- NIfTI format output

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/xenificityy/DeepPrep-use-case.git
cd DeepPrep-use-case
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with Synthetic Data

Generate synthetic data and run the complete analysis:

```bash
python src/main_pipeline.py --generate-data --n-healthy 20 --n-disorder 20
```

This will:
1. Generate 20 healthy and 20 disorder synthetic MRI scans
2. Preprocess all scans
3. Extract features
4. Perform comparative analysis
5. Generate visualizations
6. Save results to the `results/` directory

### Using Your Own Data

If you have your own MRI data in NIfTI format (.nii or .nii.gz):

```bash
python src/main_pipeline.py \
    --healthy-dir path/to/healthy/scans \
    --disorder-dir path/to/disorder/scans \
    --output-dir path/to/results
```

### Advanced Options

```bash
python src/main_pipeline.py \
    --healthy-dir data/healthy \
    --disorder-dir data/disorder \
    --output-dir results \
    --normalization-method z-score \
    --skip-preprocessing  # if data is already preprocessed
```

Available normalization methods:
- `z-score`: Zero mean and unit variance
- `min-max`: Scale to [0, 1] range
- `percentile`: Based on 1st and 99th percentiles

## Project Structure

```
DeepPrep-use-case/
├── src/
│   ├── deepprep_analysis/
│   │   ├── __init__.py
│   │   ├── preprocessing.py          # MRI preprocessing
│   │   ├── feature_extraction.py     # Feature extraction
│   │   ├── comparative_analysis.py   # Statistical analysis
│   │   ├── visualization.py          # Visualization tools
│   │   └── data_generator.py         # Synthetic data generation
│   └── main_pipeline.py              # Main execution script
├── data/
│   ├── healthy/                      # Healthy MRI scans
│   └── disorder/                     # Disorder MRI scans
├── results/                          # Output results and visualizations
├── notebooks/                        # Jupyter notebooks (optional)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Output Files

After running the pipeline, the following files are generated in the results directory:

### Data Files
- `healthy_features.csv`: Extracted features from healthy group
- `disorder_features.csv`: Extracted features from disorder group
- `t_test_results.csv`: T-test results for all features
- `mann_whitney_results.csv`: Mann-Whitney U test results
- `feature_importances.csv`: Feature importance scores
- `top_discriminative_features.csv`: Top features distinguishing groups
- `classification_results.json`: Classification performance metrics

### Visualization Files
- `mri_slices_healthy.png`: Sample healthy MRI slices
- `mri_slices_disorder.png`: Sample disorder MRI slices
- `t_test_results.png`: Statistical test results
- `feature_importance.png`: Most important features
- `feature_distributions.png`: Distribution comparisons
- `correlation_heatmap.png`: Feature correlations
- `summary_report.png`: Comprehensive summary visualization

## Example Results

The pipeline identifies significant differences between groups, including:
- **Volume changes**: Brain atrophy indicators
- **Intensity variations**: Signal abnormalities
- **Texture differences**: Tissue heterogeneity
- **Morphological changes**: Structural alterations
- **Regional patterns**: Localized differences

Classification accuracy typically ranges from 70-95% depending on:
- Sample size
- Quality of preprocessing
- Severity of disorder characteristics
- Feature selection

## Scientific Background

### DeepPrep Pipeline Principles

This implementation follows neuroimaging preprocessing best practices:

1. **Quality Control**: Visual and automated checks
2. **Standardization**: Consistent coordinate systems and orientations
3. **Artifact Removal**: Background and skull removal
4. **Normalization**: Intensity standardization across subjects
5. **Feature Engineering**: Extraction of biologically meaningful features

### Statistical Analysis

The comparative analysis uses established methods:

- **Parametric Tests**: t-tests for normally distributed features
- **Non-parametric Tests**: Mann-Whitney U for non-normal distributions
- **Effect Sizes**: Cohen's d for practical significance
- **Multiple Comparison Correction**: Bonferroni method
- **Machine Learning**: Random Forest for non-linear relationships

## Applications

This pipeline can be applied to various neurological conditions:

- **Alzheimer's Disease**: Brain atrophy patterns
- **Multiple Sclerosis**: White matter lesions
- **Parkinson's Disease**: Subcortical volume changes
- **Brain Tumors**: Structural abnormalities
- **Stroke**: Tissue damage assessment
- **Traumatic Brain Injury**: Regional alterations

## Limitations

- Synthetic data is simplified and may not capture all biological complexity
- Real MRI preprocessing may require additional tools (FSL, FreeSurfer, SPM)
- Feature extraction is basic; advanced methods may improve results
- Sample size affects statistical power
- Cross-sectional analysis; longitudinal studies require different approaches

## Future Enhancements

Potential improvements:
- Integration with deep learning models
- Advanced registration and normalization
- Longitudinal analysis support
- Multi-modal imaging integration
- Population-level atlases
- Real-time processing pipeline
- Web-based visualization dashboard

## Contributing

Contributions are welcome! Areas for improvement:
- Additional feature extraction methods
- More preprocessing options
- Enhanced visualization techniques
- Documentation and examples
- Performance optimization
- Testing and validation

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{deepprep_use_case,
  title={DeepPrep Use Case: Comparative Analysis of Brain MRI Scans},
  author={xenificityy},
  year={2025},
  url={https://github.com/xenificityy/DeepPrep-use-case}
}
```

## Acknowledgments

- DeepPrep framework for preprocessing principles
- NiBabel for NIfTI file handling
- Nilearn for neuroimaging utilities
- scikit-learn for machine learning tools
- matplotlib and seaborn for visualization

## Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Visit: https://github.com/xenificityy/DeepPrep-use-case

## References

Key papers and resources:
- [DeepPrep: A Deep Learning Framework for Preprocessing Neuroimaging Data](https://doi.org/10.1016/j.neuroimage.2021.118759)
- [NiBabel: Access to neuroimaging file formats](https://nipy.org/nibabel/)
- [Nilearn: Machine learning for neuroimaging](https://nilearn.github.io/)

---

**Note**: This is a demonstration project. For clinical applications, please consult with domain experts and follow appropriate validation procedures.
