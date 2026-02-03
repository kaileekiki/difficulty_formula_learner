"""
Correlation Metrics
Spearman's ρ, Kendall's τ, Pearson's r
"""
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from typing import Dict, Tuple


class CorrelationMetrics:
    """Compute correlation metrics for ranking evaluation."""
    
    @staticmethod
    def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Compute Spearman's rank correlation coefficient.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            (correlation, p-value)
        """
        return spearmanr(y_true, y_pred)
    
    @staticmethod
    def kendall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Compute Kendall's tau rank correlation coefficient.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            (correlation, p-value)
        """
        return kendalltau(y_true, y_pred)
    
    @staticmethod
    def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Compute Pearson correlation coefficient.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            (correlation, p-value)
        """
        return pearsonr(y_true, y_pred)
    
    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute all correlation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with all correlation metrics
        """
        spearman_corr, spearman_p = CorrelationMetrics.spearman(y_true, y_pred)
        kendall_corr, kendall_p = CorrelationMetrics.kendall(y_true, y_pred)
        pearson_corr, pearson_p = CorrelationMetrics.pearson(y_true, y_pred)
        
        return {
            'spearman': spearman_corr,
            'spearman_pvalue': spearman_p,
            'kendall': kendall_corr,
            'kendall_pvalue': kendall_p,
            'pearson': pearson_corr,
            'pearson_pvalue': pearson_p
        }
