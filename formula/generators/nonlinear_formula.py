"""
Nonlinear Formula
Nonlinear transformations (log, sqrt, sigmoid)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from typing import Dict, List

from .base_formula import BaseFormula


class NonlinearFormula(BaseFormula):
    """Nonlinear formula with feature transformations."""
    
    def __init__(self, transformations: List[str] = ['log', 'sqrt'], alpha: float = 1.0):
        """
        Initialize nonlinear formula.
        
        Args:
            transformations: List of transformations to apply ('log', 'sqrt', 'square')
            alpha: Regularization strength
        """
        super().__init__(name=f"Nonlinear_{'_'.join(transformations)}")
        self.transformations = transformations
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.transformed_feature_names = None
    
    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply nonlinear transformations to features."""
        X_transformed = X.copy()
        transformed_cols = []
        
        # Add original features
        for col in X.columns:
            transformed_cols.append(col)
        
        # Add transformed features
        if 'log' in self.transformations:
            for col in X.columns:
                # log(1 + x) to handle zeros
                X_transformed[f'log_{col}'] = np.log1p(np.maximum(X[col], 0))
                transformed_cols.append(f'log_{col}')
        
        if 'sqrt' in self.transformations:
            for col in X.columns:
                # sqrt of absolute value
                X_transformed[f'sqrt_{col}'] = np.sqrt(np.abs(X[col]))
                transformed_cols.append(f'sqrt_{col}')
        
        if 'square' in self.transformations:
            for col in X.columns:
                X_transformed[f'square_{col}'] = X[col] ** 2
                transformed_cols.append(f'square_{col}')
        
        return X_transformed[transformed_cols]
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NonlinearFormula':
        """Fit the nonlinear model."""
        self.feature_names = X.columns.tolist()
        
        # Transform features
        X_transformed = self._transform_features(X)
        self.transformed_feature_names = X_transformed.columns.tolist()
        
        # Fit model
        self.model.fit(X_transformed, y)
        self.coefficients = self.model.coef_
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the nonlinear model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_transformed = self._transform_features(X)
        return self.model.predict(X_transformed)
    
    def get_formula_string(self) -> str:
        """Get nonlinear formula as string."""
        if not self.is_fitted:
            return "Not fitted"
        
        terms = []
        intercept = self.model.intercept_
        
        # Add top terms by absolute coefficient
        coef_importance = sorted(
            zip(self.transformed_feature_names, self.coefficients),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:15]
        
        for name, coef in coef_importance:
            if abs(coef) > 1e-6:
                sign = "+" if coef >= 0 else "-"
                terms.append(f"{sign} {abs(coef):.4f}*{name}")
        
        formula = f"success_rate = {intercept:.4f}"
        if terms:
            formula += " " + " ".join(terms)
        
        return formula
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for original features."""
        if not self.is_fitted:
            return {}
        
        importance = {name: 0.0 for name in self.feature_names}
        
        for trans_name, coef in zip(self.transformed_feature_names, self.coefficients):
            # Map transformed feature back to original
            for orig_name in self.feature_names:
                if orig_name in trans_name:
                    importance[orig_name] += abs(coef)
        
        return importance
    
    def get_complexity(self) -> int:
        """Get complexity as number of transformed features."""
        if not self.is_fitted:
            return 0
        return len(self.transformed_feature_names)
