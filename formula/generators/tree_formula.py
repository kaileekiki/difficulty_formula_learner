"""
Tree-based Formula
Random Forest and XGBoost models with feature importance
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from typing import Dict

from .base_formula import BaseFormula


class TreeFormula(BaseFormula):
    """Tree-based formula using Random Forest or XGBoost."""
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 n_estimators: int = 100, 
                 max_depth: int = 5,
                 random_state: int = 42):
        """
        Initialize tree-based formula.
        
        Args:
            model_type: 'random_forest' or 'xgboost'
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        super().__init__(name=f"Tree_{model_type}")
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            if not HAS_XGBOOST:
                raise ImportError("XGBoost is not installed")
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TreeFormula':
        """Fit the tree model."""
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the tree model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_formula_string(self) -> str:
        """Get tree formula description."""
        if not self.is_fitted:
            return "Not fitted"
        
        # Get top features by importance
        importance = self.get_feature_importance()
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        formula = f"{self.model_type} ensemble with {self.n_estimators} trees (max_depth={self.max_depth})\n"
        formula += "Top features: "
        formula += ", ".join([f"{name}({imp:.3f})" for name, imp in top_features])
        
        return formula
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the tree model."""
        if not self.is_fitted:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            return {}
        
        return {
            name: float(imp) 
            for name, imp in zip(self.feature_names, importances)
        }
    
    def get_complexity(self) -> int:
        """Get complexity as number of trees * max depth."""
        return self.n_estimators * self.max_depth
