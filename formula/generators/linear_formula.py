"""
Linear Formula
Linear regression with various regularization methods
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from typing import Dict, Optional

from .base_formula import BaseFormula


class LinearFormula(BaseFormula):
    """Linear regression formula with regularization."""
    
    def __init__(self, regularization: str = 'ridge', alpha: float = 1.0):
        """
        Initialize linear formula.
        
        Args:
            regularization: Type of regularization ('none', 'ridge', 'lasso', 'elasticnet')
            alpha: Regularization strength
        """
        super().__init__(name=f"Linear_{regularization}")
        self.regularization = regularization
        self.alpha = alpha
        
        # Create model based on regularization type
        if regularization == 'none':
            self.model = LinearRegression()
        elif regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha, max_iter=10000)
        elif regularization == 'elasticnet':
            self.model = ElasticNet(alpha=alpha, max_iter=10000)
        else:
            raise ValueError(f"Unknown regularization: {regularization}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearFormula':
        """Fit the linear model."""
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.coefficients = self.model.coef_
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the linear model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_formula_string(self) -> str:
        """Get linear formula as string."""
        if not self.is_fitted:
            return "Not fitted"
        
        terms = []
        intercept = self.model.intercept_
        
        # Add feature terms
        for name, coef in zip(self.feature_names, self.coefficients):
            if abs(coef) > 1e-6:  # Skip near-zero coefficients
                sign = "+" if coef >= 0 else "-"
                terms.append(f"{sign} {abs(coef):.4f}*{name}")
        
        # Add intercept
        formula = f"success_rate = {intercept:.4f}"
        if terms:
            formula += " " + " ".join(terms)
        
        return formula
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on absolute coefficients."""
        if not self.is_fitted:
            return {}
        
        return {
            name: abs(coef) 
            for name, coef in zip(self.feature_names, self.coefficients)
        }
    
    def get_complexity(self) -> int:
        """Get complexity as number of non-zero coefficients."""
        if not self.is_fitted:
            return 0
        return int(np.sum(np.abs(self.coefficients) > 1e-6))
