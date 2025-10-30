"""
Visualization Module
Creates visualizations for comparative analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class Visualizer:
    """
    Creates visualizations for brain MRI comparative analysis.
    
    Includes:
    - MRI slice visualizations
    - Statistical comparison plots
    - Feature importance plots
    - Distribution comparisons
    """
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize Visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_mri_slices(
        self,
        data: np.ndarray,
        n_slices: int = 9,
        title: str = 'MRI Slices',
        save_path: Optional[str] = None
    ):
        """
        Plot multiple slices from an MRI scan.
        
        Args:
            data: 3D MRI scan data
            n_slices: Number of slices to display
            title: Plot title
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        # Select evenly spaced slices
        slice_indices = np.linspace(0, data.shape[2] - 1, n_slices, dtype=int)
        
        for idx, slice_idx in enumerate(slice_indices):
            axes[idx].imshow(data[:, :, slice_idx].T, cmap='gray', origin='lower')
            axes[idx].set_title(f'Slice {slice_idx}')
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved MRI slices plot to {save_path}")
        
        plt.close()
    
    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        features: List[str],
        group_col: str = 'group',
        save_path: Optional[str] = None
    ):
        """
        Plot distribution comparisons for multiple features.
        
        Args:
            df: DataFrame with features and group labels
            features: List of feature names to plot
            group_col: Name of the column containing group labels
            save_path: Path to save the figure
        """
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.ravel() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if feature in df.columns:
                sns.boxplot(data=df, x=group_col, y=feature, ax=axes[idx])
                axes[idx].set_title(f'{feature}')
                axes[idx].set_xlabel('Group')
                axes[idx].set_ylabel('Value')
        
        # Hide extra subplots
        for idx in range(len(features), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Distribution Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved feature distributions plot to {save_path}")
        
        plt.close()
    
    def plot_t_test_results(
        self,
        t_test_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot t-test results showing significant differences.
        
        Args:
            t_test_df: DataFrame with t-test results
            top_n: Number of top features to display
            save_path: Path to save the figure
        """
        # Sort by p-value and take top N
        top_features = t_test_df.nsmallest(top_n, 'p_value')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Mean differences
        colors = ['red' if sig else 'gray' for sig in top_features['significant']]
        ax1.barh(range(len(top_features)), top_features['mean_difference'], color=colors)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=10)
        ax1.set_xlabel('Mean Difference (Healthy - Disorder)', fontsize=12)
        ax1.set_title('Top Features by Mean Difference', fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.invert_yaxis()
        
        # Plot 2: Effect sizes (Cohen's d)
        colors = ['red' if sig else 'gray' for sig in top_features['significant']]
        ax2.barh(range(len(top_features)), top_features['cohens_d'], color=colors)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature'], fontsize=10)
        ax2.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
        ax2.set_title('Effect Sizes', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax2.invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label=f'Significant (p < 0.05)'),
            Patch(facecolor='gray', label='Not significant')
        ]
        ax1.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved t-test results plot to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance from classification model.
        
        Args:
            importance_df: DataFrame with feature importances
            top_n: Number of top features to display
            save_path: Path to save the figure
        """
        # Get top N features
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Top Features by Importance (Random Forest)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.close()
    
    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        top_features: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot correlation heatmap of features.
        
        Args:
            df: DataFrame with features
            top_features: List of features to include (if None, use all)
            save_path: Path to save the figure
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if top_features:
            features_to_plot = [f for f in top_features if f in numeric_cols]
        else:
            features_to_plot = numeric_cols[:20]  # Limit to 20 for readability
        
        correlation_matrix = df[features_to_plot].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            correlation_matrix,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved correlation heatmap to {save_path}")
        
        plt.close()
    
    def plot_pca_scatter(
        self,
        pca_data: np.ndarray,
        labels: np.ndarray,
        explained_variance: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot PCA scatter plot for the first two components.
        
        Args:
            pca_data: PCA-transformed data
            labels: Group labels
            explained_variance: Explained variance ratio for each component
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(
                pca_data[mask, 0],
                pca_data[mask, 1],
                label=label,
                alpha=0.7,
                s=100
            )
        
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)', fontsize=12)
        ax.set_title('PCA: Group Separation', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved PCA scatter plot to {save_path}")
        
        plt.close()
    
    def create_summary_report(
        self,
        analysis_results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Create a comprehensive summary visualization.
        
        Args:
            analysis_results: Dictionary with all analysis results
            save_path: Path to save the figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Sample sizes
        ax1 = fig.add_subplot(gs[0, 0])
        groups = ['Healthy', 'Disorder']
        counts = [analysis_results['n_healthy'], analysis_results['n_disorder']]
        ax1.bar(groups, counts, color=['green', 'red'], alpha=0.7)
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Sample Sizes', fontweight='bold')
        for i, v in enumerate(counts):
            ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        # 2. Significant features
        ax2 = fig.add_subplot(gs[0, 1])
        t_test_df = analysis_results['t_tests']
        sig_count = t_test_df['significant'].sum()
        total_count = len(t_test_df)
        categories = ['Significant', 'Not Significant']
        counts = [sig_count, total_count - sig_count]
        ax2.pie(counts, labels=categories, autopct='%1.1f%%', colors=['red', 'gray'])
        ax2.set_title('Feature Significance (p < 0.05)', fontweight='bold')
        
        # 3. Top discriminative features
        ax3 = fig.add_subplot(gs[1, :])
        top_features = t_test_df.nsmallest(10, 'p_value')
        colors = ['red' if sig else 'gray' for sig in top_features['significant']]
        ax3.barh(range(len(top_features)), top_features['cohens_d'], color=colors)
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels(top_features['feature'])
        ax3.set_xlabel("Cohen's d")
        ax3.set_title('Top 10 Discriminative Features (Effect Size)', fontweight='bold')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax3.invert_yaxis()
        
        # 4. Classification performance
        ax4 = fig.add_subplot(gs[2, 0])
        class_results = analysis_results['classification']
        metrics = ['Accuracy']
        values = [class_results['cv_accuracy_mean']]
        errors = [class_results['cv_accuracy_std']]
        ax4.bar(metrics, values, yerr=errors, capsize=10, color='steelblue', alpha=0.7)
        ax4.set_ylabel('Score')
        ax4.set_ylim([0, 1])
        ax4.set_title('Classification Performance (5-fold CV)', fontweight='bold')
        ax4.axhline(y=0.5, color='red', linestyle='--', label='Chance level')
        ax4.legend()
        for i, v in enumerate(values):
            ax4.text(i, v + 0.05, f'{v:.3f}±{errors[i]:.3f}', ha='center', fontweight='bold')
        
        # 5. Feature importance distribution
        ax5 = fig.add_subplot(gs[2, 1])
        importance_vals = class_results['feature_importances']['importance'].values
        ax5.hist(importance_vals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Feature Importance')
        ax5.set_ylabel('Count')
        ax5.set_title('Distribution of Feature Importances', fontweight='bold')
        ax5.axvline(x=importance_vals.mean(), color='red', linestyle='--', 
                    label=f'Mean: {importance_vals.mean():.4f}')
        ax5.legend()
        
        plt.suptitle('Comparative Analysis Summary Report', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved summary report to {save_path}")
        
        plt.close()
    
    def generate_all_visualizations(
        self,
        analysis_results: Dict,
        sample_mri_data: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Generate all visualizations for the analysis.
        
        Args:
            analysis_results: Dictionary with all analysis results
            sample_mri_data: Optional dictionary with sample MRI data 
                            {'healthy': data, 'disorder': data}
        """
        logger.info("Generating all visualizations")
        
        # Sample MRI slices
        if sample_mri_data:
            for group_name, data in sample_mri_data.items():
                save_path = self.output_dir / f'mri_slices_{group_name}.png'
                self.plot_mri_slices(data, title=f'MRI Slices - {group_name.capitalize()}', 
                                    save_path=str(save_path))
        
        # T-test results
        save_path = self.output_dir / 't_test_results.png'
        self.plot_t_test_results(analysis_results['t_tests'], save_path=str(save_path))
        
        # Feature importance
        save_path = self.output_dir / 'feature_importance.png'
        self.plot_feature_importance(
            analysis_results['classification']['feature_importances'],
            save_path=str(save_path)
        )
        
        # Feature distributions for top features
        top_features = analysis_results['t_tests'].nsmallest(9, 'p_value')['feature'].tolist()
        save_path = self.output_dir / 'feature_distributions.png'
        self.plot_feature_distributions(
            analysis_results['data'],
            top_features,
            save_path=str(save_path)
        )
        
        # Correlation heatmap
        save_path = self.output_dir / 'correlation_heatmap.png'
        top_features_for_corr = analysis_results['t_tests'].nsmallest(15, 'p_value')['feature'].tolist()
        self.plot_correlation_heatmap(
            analysis_results['data'],
            top_features_for_corr,
            save_path=str(save_path)
        )
        
        # Summary report
        save_path = self.output_dir / 'summary_report.png'
        self.create_summary_report(analysis_results, save_path=str(save_path))
        
        logger.info(f"All visualizations saved to {self.output_dir}")
