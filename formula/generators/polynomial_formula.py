"""
Polynomial Formula
Polynomial regression with interaction terms
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from typing import Dict

from .base_formula import BaseFormula


class PolynomialFormula(BaseFormula):
    """Polynomial regression formula."""
    
    def __init__(self, degree: int = 2, interaction_only: bool = False, alpha: float = 1.0):
        """
        Initialize polynomial formula.
        
        Args:
            degree: Polynomial degree
            interaction_only: If True, only interaction features are produced
            alpha: Regularization strength for Ridge regression
        """
        super().__init__(name=f"Polynomial_degree{degree}")
        self.degree = degree
        self.interaction_only = interaction_only
        self.alpha = alpha
        
        self.poly_features = PolynomialFeatures(
            degree=degree, 
            interaction_only=interaction_only,
            include_bias=False
        )
        self.model = Ridge(alpha=alpha)
        self.poly_feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PolynomialFormula':
        """Fit the polynomial model."""
        self.feature_names = X.columns.tolist()
        
        # Transform features to polynomial
        X_poly = self.poly_features.fit_transform(X)
        self.poly_feature_names = self.poly_features.get_feature_names_out(self.feature_names)
        
        # Fit model
        self.model.fit(X_poly, y)
        self.coefficients = self.model.coef_
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the polynomial model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_poly = self.poly_features.transform(X)
        return self.model.predict(X_poly)
    
    def get_formula_string(self) -> str:
        """Get polynomial formula as string (truncated for readability)."""
        if not self.is_fitted:
            return "Not fitted"
        
        terms = []
        intercept = self.model.intercept_
        
        # Add top 10 terms by absolute coefficient value
        coef_importance = sorted(
            zip(self.poly_feature_names, self.coefficients),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        for name, coef in coef_importance:
            if abs(coef) > 1e-6:
                sign = "+" if coef >= 0 else "-"
                terms.append(f"{sign} {abs(coef):.4f}*{name}")
        
        formula = f"success_rate = {intercept:.4f}"
        if terms:
            formula += " " + " ".join(terms) + " + ..."
        
        return formula
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on original features."""
        if not self.is_fitted:
            return {}
        
        # Map polynomial features back to original features
        importance = {name: 0.0 for name in self.feature_names}
        
        for poly_name, coef in zip(self.poly_feature_names, self.coefficients):
            # Find which original features contribute to this polynomial term
            for orig_name in self.feature_names:
                if orig_name in poly_name:
                    importance[orig_name] += abs(coef)
        
        return importance
    
    def get_complexity(self) -> int:
        """Get complexity as number of polynomial features."""
        if not self.is_fitted:
            return 0
        return len(self.poly_feature_names)
