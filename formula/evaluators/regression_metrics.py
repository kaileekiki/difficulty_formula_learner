"""
Regression Metrics
Compute regression-based evaluation metrics
"""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict


class RegressionMetrics:
    """Compute regression metrics between actual and predicted values."""
    
    @staticmethod
    def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R-squared (coefficient of determination)."""
        try:
            if len(y_true) < 2:
                return 0.0
            score = r2_score(y_true, y_pred)
            return float(score) if not np.isnan(score) else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        try:
            if len(y_true) == 0:
                return 0.0
            mse = mean_squared_error(y_true, y_pred)
            return float(np.sqrt(mse))
        except Exception:
            return 0.0
    
    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        try:
            if len(y_true) == 0:
                return 0.0
            return float(mean_absolute_error(y_true, y_pred))
        except Exception:
            return 0.0
    
    @staticmethod
    def compute_adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
        """Compute Adjusted R-squared."""
        try:
            n = len(y_true)
            if n <= n_features + 1:
                return 0.0
            r2 = RegressionMetrics.compute_r2(y_true, y_pred)
            adjusted = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
            return float(adjusted) if not np.isnan(adjusted) else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 0) -> Dict[str, float]:
        """Compute all regression metrics."""
        # Ensure numpy arrays
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        return {
            'r2': RegressionMetrics.compute_r2(y_true, y_pred),
            'adjusted_r2': RegressionMetrics.compute_adjusted_r2(y_true, y_pred, n_features),
            'rmse': RegressionMetrics.compute_rmse(y_true, y_pred),
            'mae': RegressionMetrics.compute_mae(y_true, y_pred)
        }
    
    # Keep old methods for backward compatibility
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² (coefficient of determination) (legacy method)."""
        return RegressionMetrics.compute_r2(y_true, y_pred)
    
    @staticmethod
    def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
        """Compute adjusted R² (legacy method)."""
        return RegressionMetrics.compute_adjusted_r2(y_true, y_pred, n_features)
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error (legacy method)."""
        try:
            return float(mean_squared_error(y_true, y_pred))
        except Exception:
            return 0.0
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error (legacy method)."""
        return RegressionMetrics.compute_rmse(y_true, y_pred)
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error (legacy method)."""
        return RegressionMetrics.compute_mae(y_true, y_pred)
