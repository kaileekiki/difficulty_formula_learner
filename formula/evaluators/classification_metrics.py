"""
Classification Metrics
For categorizing bugs into difficulty levels
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Dict, List


class ClassificationMetrics:
    """Compute classification metrics for difficulty categories."""
    
    DIFFICULTY_THRESHOLDS = {
        'Easy': 0.75,
        'Medium': 0.5,
        'Hard': 0.25,
        'Very Hard': 0.0
    }
    
    @staticmethod
    def categorize(success_rates: np.ndarray) -> np.ndarray:
        """
        Categorize success rates into difficulty levels.
        
        Args:
            success_rates: Array of success rates (0-1)
            
        Returns:
            Array of category labels
        """
        categories = np.empty(len(success_rates), dtype=object)
        
        for i, rate in enumerate(success_rates):
            if rate >= 0.75:
                categories[i] = 'Easy'
            elif rate >= 0.5:
                categories[i] = 'Medium'
            elif rate >= 0.25:
                categories[i] = 'Hard'
            else:
                categories[i] = 'Very Hard'
        
        return categories
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute accuracy for categorical predictions.
        
        Args:
            y_true: True categories
            y_pred: Predicted categories
            
        Returns:
            Accuracy score
        """
        y_true_cat = ClassificationMetrics.categorize(y_true)
        y_pred_cat = ClassificationMetrics.categorize(y_pred)
        
        return accuracy_score(y_true_cat, y_pred_cat)
    
    @staticmethod
    def f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """
        Compute F1 score for categorical predictions.
        
        Args:
            y_true: True categories
            y_pred: Predicted categories
            average: Averaging method ('weighted', 'macro', 'micro')
            
        Returns:
            F1 score
        """
        y_true_cat = ClassificationMetrics.categorize(y_true)
        y_pred_cat = ClassificationMetrics.categorize(y_pred)
        
        return f1_score(y_true_cat, y_pred_cat, average=average, zero_division=0)
    
    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute all classification metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with all classification metrics
        """
        y_true_cat = ClassificationMetrics.categorize(y_true)
        y_pred_cat = ClassificationMetrics.categorize(y_pred)
        
        return {
            'accuracy': accuracy_score(y_true_cat, y_pred_cat),
            'f1_weighted': f1_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true_cat, y_pred_cat, average='macro', zero_division=0)
        }
    
    @staticmethod
    def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Get detailed classification report.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Classification report string
        """
        y_true_cat = ClassificationMetrics.categorize(y_true)
        y_pred_cat = ClassificationMetrics.categorize(y_pred)
        
        return classification_report(y_true_cat, y_pred_cat, zero_division=0)
