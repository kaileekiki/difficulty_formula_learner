"""
Regression Metrics
R², MSE, RMSE, MAE, Adjusted R²
"""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict


class RegressionMetrics:
    """Compute regression metrics for prediction evaluation."""
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute R² (coefficient of determination).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            R² score
        """
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
        """
        Compute adjusted R².
        
        Args:
            y_true: True values
            y_pred: Predicted values
            n_features: Number of features
            
        Returns:
            Adjusted R² score
        """
        n = len(y_true)
        r2 = RegressionMetrics.r2(y_true, y_pred)
        
        if n <= n_features + 1:
            return r2
        
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adj_r2
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MSE
        """
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Root Mean Squared Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Absolute Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE
        """
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 0) -> Dict[str, float]:
        """
        Compute all regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            n_features: Number of features (for adjusted R²)
            
        Returns:
            Dictionary with all regression metrics
        """
        return {
            'r2': RegressionMetrics.r2(y_true, y_pred),
            'adjusted_r2': RegressionMetrics.adjusted_r2(y_true, y_pred, n_features),
            'mse': RegressionMetrics.mse(y_true, y_pred),
            'rmse': RegressionMetrics.rmse(y_true, y_pred),
            'mae': RegressionMetrics.mae(y_true, y_pred)
        }
