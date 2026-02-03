"""
Base Formula class
Abstract base class for all formula generators
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class BaseFormula(ABC):
    """Abstract base class for difficulty formulas."""
    
    def __init__(self, name: str = "BaseFormula"):
        """
        Initialize base formula.
        
        Args:
            name: Name of the formula
        """
        self.name = name
        self.model = None
        self.coefficients = None
        self.feature_names = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseFormula':
        """
        Fit the formula to the data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict target values.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted values (n_samples,)
        """
        pass
    
    @abstractmethod
    def get_formula_string(self) -> str:
        """
        Get human-readable formula string.
        
        Returns:
            Formula as string
        """
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return None
    
    def get_complexity(self) -> int:
        """
        Get formula complexity score.
        
        Returns:
            Complexity score (higher = more complex)
        """
        return 0
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the formula.
        
        Returns:
            Dictionary with formula metadata
        """
        return {
            'name': self.name,
            'formula': self.get_formula_string(),
            'complexity': self.get_complexity(),
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else 0
        }
