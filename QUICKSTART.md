# Quick Start Guide

This guide will help you get started with the Brain MRI Comparative Analysis pipeline in just a few minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/xenificityy/DeepPrep-use-case.git
cd DeepPrep-use-case
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Your First Analysis

### Option 1: Quick Example (2 minutes)

Run the simple example script with minimal data:

```bash
python examples/simple_example.py
```

This will:
- Generate 5 healthy and 5 disorder synthetic brain MRI scans
- Extract features from each scan
- Perform statistical analysis
- Display key results in the terminal

**Expected Output:**
```
Sample Sizes:
  Healthy: 5
  Disorder: 5

Feature Analysis:
  Total features: 39
  Significant features (p < 0.05): 26
  Percentage: 66.7%

Classification Performance:
  Accuracy: 1.000 ± 0.000
```

### Option 2: Full Pipeline with Visualizations (5 minutes)

Run the complete analysis pipeline:

```bash
python src/main_pipeline.py --generate-data --n-healthy 20 --n-disorder 20
```

This will:
1. Generate 20 healthy and 20 disorder synthetic MRI scans
2. Preprocess all scans (normalization, skull stripping, background removal)
3. Extract 39 features from each scan
4. Perform comprehensive statistical analysis
5. Generate multiple visualizations
6. Save all results to the `results/` directory

**What to expect:**
- Processing time: ~2-5 minutes
- Output: 16 files in `results/` directory
- Visualizations: 7 PNG images showing different analyses
- Data files: CSV and JSON files with detailed results

**Check your results:**
```bash
ls -la results/
```

You should see files like:
- `summary_report.png` - Comprehensive overview
- `t_test_results.png` - Statistical comparisons
- `feature_importance.png` - Most discriminative features
- `mri_slices_healthy.png` - Sample healthy brain
- `mri_slices_disorder.png` - Sample disorder brain
- And more...

### Option 3: Interactive Jupyter Notebook (15 minutes)

For a step-by-step interactive experience:

```bash
jupyter notebook notebooks/example_analysis.ipynb
```

This notebook includes:
- Detailed explanations of each step
- Code you can modify and experiment with
- Inline visualizations
- Interpretation of results

## Using Your Own Data

If you have your own brain MRI scans in NIfTI format (.nii or .nii.gz):

1. Organize your data:
```
your_project/
├── healthy/
│   ├── subject_001.nii.gz
│   ├── subject_002.nii.gz
│   └── ...
└── disorder/
    ├── patient_001.nii.gz
    ├── patient_002.nii.gz
    └── ...
```

2. Run the pipeline:
```bash
python src/main_pipeline.py \
    --healthy-dir your_project/healthy \
    --disorder-dir your_project/disorder \
    --output-dir your_project/results
```

## Understanding the Results

### Key Output Files

1. **summary_report.png** - Start here! This shows:
   - Sample sizes
   - Number of significant features
   - Top discriminative features
   - Classification performance

2. **t_test_results.csv** - Statistical test results for each feature:
   - `p_value`: Statistical significance (< 0.05 is significant)
   - `cohens_d`: Effect size (0.2=small, 0.5=medium, 0.8=large)
   - `mean_difference`: Average difference between groups

3. **feature_importances.csv** - Most important features for classification:
   - `importance`: Relative importance (higher = more discriminative)

4. **classification_results.json** - Overall classification accuracy:
   - `cv_accuracy_mean`: Cross-validated accuracy (0-1 scale)

### Interpreting Results

**High Classification Accuracy (>0.8):**
- Strong differences between groups
- Features are highly discriminative
- Good separability

**Many Significant Features (>30%):**
- Multiple aspects differ between groups
- Comprehensive differences detected
- Strong evidence of group differences

**Large Effect Sizes (|Cohen's d| > 0.8):**
- Practically significant differences
- Not just statistically significant
- Meaningful clinical relevance

## Next Steps

1. **Explore Different Normalization Methods:**
```bash
python src/main_pipeline.py --generate-data --normalization-method min-max
```

2. **Increase Sample Size for More Robust Results:**
```bash
python src/main_pipeline.py --generate-data --n-healthy 50 --n-disorder 50
```

3. **Modify the Code:**
   - Add new feature extraction methods in `src/deepprep_analysis/feature_extraction.py`
   - Customize visualizations in `src/deepprep_analysis/visualization.py`
   - Try different statistical tests in `src/deepprep_analysis/comparative_analysis.py`

4. **Read the Full Documentation:**
   - See `README.md` for comprehensive documentation
   - Check `METHODOLOGY.md` for scientific background
   - Explore the code with inline comments

## Troubleshooting

**Import Errors:**
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt
```

**Memory Issues with Large Datasets:**
```bash
# Reduce sample size or use smaller images
python src/main_pipeline.py --generate-data --n-healthy 10 --n-disorder 10
```

**Missing Visualizations:**
```bash
# Check if matplotlib backend is working
python -c "import matplotlib.pyplot as plt; print('OK')"
```

## Getting Help

- Open an issue on GitHub
- Check the examples in `examples/` directory
- Review the notebook in `notebooks/` directory
- Read the comprehensive `README.md`

## What's Next?

After completing this quick start, you can:

1. Apply the pipeline to real MRI data
2. Customize feature extraction for your specific use case
3. Add new statistical tests or machine learning models
4. Integrate with your existing neuroimaging workflow
5. Extend the visualization capabilities

Happy analyzing! 🧠📊
