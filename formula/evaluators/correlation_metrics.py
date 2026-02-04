"""
Correlation Metrics
Compute correlation-based evaluation metrics
"""
import numpy as np
from scipy import stats
from typing import Dict


class CorrelationMetrics:
    """Compute correlation metrics between actual and predicted values."""
    
    @staticmethod
    def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Spearman rank correlation coefficient."""
        try:
            if len(y_true) < 2:
                return 0.0
            # Handle constant arrays
            if np.std(y_true) == 0 or np.std(y_pred) == 0:
                return 0.0
            corr, _ = stats.spearmanr(y_true, y_pred)
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def compute_kendall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Kendall tau correlation coefficient."""
        try:
            if len(y_true) < 2:
                return 0.0
            if np.std(y_true) == 0 or np.std(y_pred) == 0:
                return 0.0
            corr, _ = stats.kendalltau(y_true, y_pred)
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        try:
            if len(y_true) < 2:
                return 0.0
            if np.std(y_true) == 0 or np.std(y_pred) == 0:
                return 0.0
            corr, _ = stats.pearsonr(y_true, y_pred)
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all correlation metrics."""
        # Ensure numpy arrays
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        return {
            'spearman': CorrelationMetrics.compute_spearman(y_true, y_pred),
            'kendall': CorrelationMetrics.compute_kendall(y_true, y_pred),
            'pearson': CorrelationMetrics.compute_pearson(y_true, y_pred)
        }
    
    # Keep old methods for backward compatibility
    @staticmethod
    def spearman(y_true: np.ndarray, y_pred: np.ndarray):
        """Compute Spearman's rank correlation coefficient (legacy method)."""
        try:
            return stats.spearmanr(y_true, y_pred)
        except Exception:
            return (0.0, 1.0)
    
    @staticmethod
    def kendall(y_true: np.ndarray, y_pred: np.ndarray):
        """Compute Kendall's tau rank correlation coefficient (legacy method)."""
        try:
            return stats.kendalltau(y_true, y_pred)
        except Exception:
            return (0.0, 1.0)
    
    @staticmethod
    def pearson(y_true: np.ndarray, y_pred: np.ndarray):
        """Compute Pearson correlation coefficient (legacy method)."""
        try:
            return stats.pearsonr(y_true, y_pred)
        except Exception:
            return (0.0, 1.0)
