"""
Comparative Analysis
Compare different approaches to difficulty prediction
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Tuple


class ComparativeAnalyzer:
    """Compare different difficulty prediction approaches."""
    
    def __init__(self):
        """Initialize comparative analyzer."""
        self.comparison_results = {}
    
    def compare_predictions(self,
                          y_true: pd.Series,
                          predictions: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compare multiple prediction approaches.
        
        Args:
            y_true: True success rates
            predictions: Dictionary mapping approach names to predicted values
            
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for name, y_pred in predictions.items():
            # Align indices
            common_idx = y_true.index.intersection(y_pred.index)
            y_true_aligned = y_true.loc[common_idx]
            y_pred_aligned = y_pred.loc[common_idx]
            
            # Compute metrics
            spearman_corr, spearman_p = spearmanr(y_true_aligned, y_pred_aligned)
            kendall_corr, kendall_p = kendalltau(y_true_aligned, y_pred_aligned)
            
            # R² and errors
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            r2 = r2_score(y_true_aligned, y_pred_aligned)
            mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
            rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
            
            results.append({
                'approach': name,
                'spearman': spearman_corr,
                'kendall': kendall_corr,
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'n_samples': len(common_idx)
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('spearman', ascending=False)
        
        self.comparison_results = df
        return df
    
    def compare_rankings(self,
                        y_true: pd.Series,
                        predictions: Dict[str, pd.Series],
                        top_k: int = 10) -> Dict[str, Dict]:
        """
        Compare top-k rankings between approaches.
        
        Args:
            y_true: True success rates
            predictions: Dictionary mapping approach names to predicted values
            top_k: Number of top bugs to compare
            
        Returns:
            Dictionary with ranking comparison results
        """
        results = {}
        
        # Get true top-k (hardest bugs = lowest success rates)
        true_top_k = set(y_true.nsmallest(top_k).index)
        
        for name, y_pred in predictions.items():
            # Get predicted top-k
            common_idx = y_true.index.intersection(y_pred.index)
            pred_top_k = set(y_pred.loc[common_idx].nsmallest(top_k).index)
            
            # Compute overlap
            overlap = len(true_top_k & pred_top_k)
            precision = overlap / top_k if top_k > 0 else 0
            
            results[name] = {
                'overlap': overlap,
                'precision': precision,
                'true_top_k': list(true_top_k),
                'pred_top_k': list(pred_top_k),
                'intersection': list(true_top_k & pred_top_k)
            }
        
        return results
    
    def analyze_prediction_errors(self,
                                 y_true: pd.Series,
                                 y_pred: pd.Series) -> pd.DataFrame:
        """
        Analyze prediction errors per bug.
        
        Args:
            y_true: True success rates
            y_pred: Predicted success rates
            
        Returns:
            DataFrame with error analysis
        """
        # Align indices
        common_idx = y_true.index.intersection(y_pred.index)
        y_true = y_true.loc[common_idx]
        y_pred = y_pred.loc[common_idx]
        
        # Compute errors
        errors = pd.DataFrame({
            'bug_id': common_idx,
            'true_success_rate': y_true.values,
            'pred_success_rate': y_pred.values,
            'error': y_pred.values - y_true.values,
            'abs_error': np.abs(y_pred.values - y_true.values),
            'squared_error': (y_pred.values - y_true.values) ** 2
        })
        
        errors = errors.sort_values('abs_error', ascending=False)
        
        return errors
    
    def compute_difficulty_agreement(self,
                                    y_true: pd.Series,
                                    predictions: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compute agreement on difficulty categories.
        
        Args:
            y_true: True success rates
            predictions: Dictionary mapping approach names to predicted values
            
        Returns:
            DataFrame with category agreement metrics
        """
        def categorize(rates):
            """Categorize success rates into difficulty levels."""
            categories = []
            for rate in rates:
                if rate >= 0.75:
                    categories.append('Easy')
                elif rate >= 0.5:
                    categories.append('Medium')
                elif rate >= 0.25:
                    categories.append('Hard')
                else:
                    categories.append('Very Hard')
            return categories
        
        results = []
        
        true_categories = categorize(y_true.values)
        
        for name, y_pred in predictions.items():
            common_idx = y_true.index.intersection(y_pred.index)
            y_true_aligned = y_true.loc[common_idx]
            y_pred_aligned = y_pred.loc[common_idx]
            
            true_cat = categorize(y_true_aligned.values)
            pred_cat = categorize(y_pred_aligned.values)
            
            # Compute agreement
            from sklearn.metrics import accuracy_score, cohen_kappa_score
            accuracy = accuracy_score(true_cat, pred_cat)
            kappa = cohen_kappa_score(true_cat, pred_cat)
            
            results.append({
                'approach': name,
                'category_accuracy': accuracy,
                'cohen_kappa': kappa
            })
        
        return pd.DataFrame(results)
