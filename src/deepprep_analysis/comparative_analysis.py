"""
Comparative Analysis Module
Performs statistical comparisons between healthy and disorder groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComparativeAnalyzer:
    """
    Performs comparative analysis between healthy individuals and patients
    with neurological disorders.
    
    Includes:
    - Statistical hypothesis testing
    - Effect size calculations
    - Group differences analysis
    - Classification analysis
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize Comparative Analyzer.
        
        Args:
            alpha: Significance level for hypothesis testing
        """
        self.alpha = alpha
        self.results = {}
        
    def create_dataframe(
        self,
        healthy_features: List[Dict],
        disorder_features: List[Dict]
    ) -> pd.DataFrame:
        """
        Create a combined DataFrame from feature dictionaries.
        
        Args:
            healthy_features: List of feature dictionaries for healthy group
            disorder_features: List of feature dictionaries for disorder group
            
        Returns:
            Combined DataFrame with group labels
        """
        # Create DataFrames for each group
        df_healthy = pd.DataFrame(healthy_features)
        df_healthy['group'] = 'healthy'
        
        df_disorder = pd.DataFrame(disorder_features)
        df_disorder['group'] = 'disorder'
        
        # Combine
        df_combined = pd.concat([df_healthy, df_disorder], ignore_index=True)
        
        logger.info(f"Created DataFrame with {len(df_healthy)} healthy and {len(df_disorder)} disorder samples")
        return df_combined
    
    def compute_descriptive_statistics(
        self,
        df: pd.DataFrame,
        group_col: str = 'group'
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute descriptive statistics for each group.
        
        Args:
            df: DataFrame with features and group labels
            group_col: Name of the column containing group labels
            
        Returns:
            Dictionary of descriptive statistics DataFrames
        """
        # Exclude non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats_dict = {}
        
        for group in df[group_col].unique():
            group_data = df[df[group_col] == group][numeric_cols]
            
            stats_df = pd.DataFrame({
                'mean': group_data.mean(),
                'std': group_data.std(),
                'median': group_data.median(),
                'min': group_data.min(),
                'max': group_data.max(),
                'q25': group_data.quantile(0.25),
                'q75': group_data.quantile(0.75)
            })
            
            stats_dict[group] = stats_df
        
        logger.info(f"Computed descriptive statistics for {len(stats_dict)} groups")
        return stats_dict
    
    def perform_t_tests(
        self,
        df: pd.DataFrame,
        group_col: str = 'group',
        healthy_label: str = 'healthy',
        disorder_label: str = 'disorder'
    ) -> pd.DataFrame:
        """
        Perform independent t-tests for each feature.
        
        Args:
            df: DataFrame with features and group labels
            group_col: Name of the column containing group labels
            healthy_label: Label for healthy group
            disorder_label: Label for disorder group
            
        Returns:
            DataFrame with t-test results
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = []
        
        healthy_data = df[df[group_col] == healthy_label]
        disorder_data = df[df[group_col] == disorder_label]
        
        for col in numeric_cols:
            healthy_vals = healthy_data[col].dropna()
            disorder_vals = disorder_data[col].dropna()
            
            if len(healthy_vals) > 1 and len(disorder_vals) > 1:
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(healthy_vals, disorder_vals)
                
                # Calculate Cohen's d (effect size)
                pooled_std = np.sqrt(
                    ((len(healthy_vals) - 1) * healthy_vals.std() ** 2 +
                     (len(disorder_vals) - 1) * disorder_vals.std() ** 2) /
                    (len(healthy_vals) + len(disorder_vals) - 2)
                )
                cohens_d = (healthy_vals.mean() - disorder_vals.mean()) / (pooled_std + 1e-10)
                
                results.append({
                    'feature': col,
                    'healthy_mean': healthy_vals.mean(),
                    'disorder_mean': disorder_vals.mean(),
                    'mean_difference': healthy_vals.mean() - disorder_vals.mean(),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < self.alpha
                })
        
        results_df = pd.DataFrame(results)
        
        # Apply Bonferroni correction
        if len(results_df) > 0:
            results_df['p_value_bonferroni'] = results_df['p_value'] * len(results_df)
            results_df['significant_bonferroni'] = results_df['p_value_bonferroni'] < self.alpha
        
        logger.info(f"Performed t-tests for {len(results_df)} features")
        logger.info(f"Found {results_df['significant'].sum()} significant differences (p < {self.alpha})")
        
        self.results['t_tests'] = results_df
        return results_df
    
    def perform_mann_whitney_tests(
        self,
        df: pd.DataFrame,
        group_col: str = 'group',
        healthy_label: str = 'healthy',
        disorder_label: str = 'disorder'
    ) -> pd.DataFrame:
        """
        Perform Mann-Whitney U tests (non-parametric alternative to t-test).
        
        Args:
            df: DataFrame with features and group labels
            group_col: Name of the column containing group labels
            healthy_label: Label for healthy group
            disorder_label: Label for disorder group
            
        Returns:
            DataFrame with Mann-Whitney test results
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = []
        
        healthy_data = df[df[group_col] == healthy_label]
        disorder_data = df[df[group_col] == disorder_label]
        
        for col in numeric_cols:
            healthy_vals = healthy_data[col].dropna()
            disorder_vals = disorder_data[col].dropna()
            
            if len(healthy_vals) > 1 and len(disorder_vals) > 1:
                # Perform Mann-Whitney U test
                u_stat, p_value = stats.mannwhitneyu(healthy_vals, disorder_vals, alternative='two-sided')
                
                results.append({
                    'feature': col,
                    'healthy_median': healthy_vals.median(),
                    'disorder_median': disorder_vals.median(),
                    'median_difference': healthy_vals.median() - disorder_vals.median(),
                    'u_statistic': u_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Performed Mann-Whitney tests for {len(results_df)} features")
        
        self.results['mann_whitney'] = results_df
        return results_df
    
    def classify_groups(
        self,
        df: pd.DataFrame,
        group_col: str = 'group',
        n_folds: int = 5
    ) -> Dict:
        """
        Train a classifier to distinguish between groups.
        
        Args:
            df: DataFrame with features and group labels
            group_col: Name of the column containing group labels
            n_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with classification results
        """
        # Prepare data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_cols].fillna(0).values
        y = (df[group_col] == 'disorder').astype(int).values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_scaled, y, cv=n_folds, scoring='accuracy')
        
        # Fit on all data to get feature importances
        clf.fit(X_scaled, y)
        
        # Get feature importances
        feature_importances = pd.DataFrame({
            'feature': numeric_cols,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'feature_importances': feature_importances,
            'n_features': len(numeric_cols)
        }
        
        logger.info(f"Classification accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        logger.info(f"Top 5 important features: {feature_importances.head()['feature'].tolist()}")
        
        self.results['classification'] = results
        return results
    
    def analyze_group_differences(
        self,
        healthy_features: List[Dict],
        disorder_features: List[Dict]
    ) -> Dict:
        """
        Perform complete comparative analysis between groups.
        
        Args:
            healthy_features: List of feature dictionaries for healthy group
            disorder_features: List of feature dictionaries for disorder group
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting comparative analysis")
        
        # Create combined DataFrame
        df = self.create_dataframe(healthy_features, disorder_features)
        
        # Descriptive statistics
        descriptive_stats = self.compute_descriptive_statistics(df)
        
        # Statistical tests
        t_test_results = self.perform_t_tests(df)
        mann_whitney_results = self.perform_mann_whitney_tests(df)
        
        # Classification analysis
        classification_results = self.classify_groups(df)
        
        # Compile results
        analysis_results = {
            'data': df,
            'descriptive_stats': descriptive_stats,
            't_tests': t_test_results,
            'mann_whitney': mann_whitney_results,
            'classification': classification_results,
            'n_healthy': len(healthy_features),
            'n_disorder': len(disorder_features)
        }
        
        logger.info("Comparative analysis completed")
        return analysis_results
    
    def get_top_discriminative_features(
        self,
        n_top: int = 10,
        method: str = 'both'
    ) -> pd.DataFrame:
        """
        Get top features that discriminate between groups.
        
        Args:
            n_top: Number of top features to return
            method: Method to use ('t_test', 'importance', or 'both')
            
        Returns:
            DataFrame with top discriminative features
        """
        if method in ['t_test', 'both'] and 't_tests' in self.results:
            t_test_top = self.results['t_tests'].nsmallest(n_top, 'p_value')[
                ['feature', 'p_value', 'cohens_d', 'mean_difference']
            ]
        else:
            t_test_top = pd.DataFrame()
        
        if method in ['importance', 'both'] and 'classification' in self.results:
            importance_top = self.results['classification']['feature_importances'].head(n_top)
        else:
            importance_top = pd.DataFrame()
        
        if method == 'both':
            # Merge both methods
            combined = pd.merge(
                t_test_top,
                importance_top,
                on='feature',
                how='outer'
            )
            return combined
        elif method == 't_test':
            return t_test_top
        else:
            return importance_top
