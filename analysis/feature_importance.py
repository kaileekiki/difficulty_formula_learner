"""
Feature Importance Analysis
Permutation importance and SHAP values
"""
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
from typing import Dict, Optional

from formula.generators.base_formula import BaseFormula


class FeatureImportanceAnalyzer:
    """Analyze feature importance using multiple methods."""
    
    def __init__(self):
        """Initialize feature importance analyzer."""
        self.importance_scores = {}
    
    def compute_permutation_importance(self, 
                                      formula: BaseFormula,
                                      X: pd.DataFrame,
                                      y: pd.Series,
                                      n_repeats: int = 10,
                                      random_state: int = 42) -> Dict[str, float]:
        """
        Compute permutation importance.
        
        Args:
            formula: Fitted formula
            X: Feature matrix
            y: Target values
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Create a wrapper for sklearn's permutation_importance
        def predict_wrapper(X_array):
            X_df = pd.DataFrame(X_array, columns=X.columns)
            return formula.predict(X_df)
        
        # Compute permutation importance
        result = permutation_importance(
            estimator=type('obj', (object,), {'predict': predict_wrapper})(),
            X=X.values,
            y=y.values,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='neg_mean_squared_error'
        )
        
        importance = {
            name: float(score)
            for name, score in zip(X.columns, result.importances_mean)
        }
        
        self.importance_scores['permutation'] = importance
        return importance
    
    def compute_shap_importance(self,
                               formula: BaseFormula,
                               X: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Compute SHAP values for feature importance.
        
        Args:
            formula: Fitted formula
            X: Feature matrix
            
        Returns:
            Dictionary mapping feature names to importance scores, or None if SHAP not available
        """
        if not HAS_SHAP:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(formula.predict, X)
            shap_values = explainer(X)
            
            # Compute mean absolute SHAP values
            importance = {
                name: float(np.mean(np.abs(shap_values.values[:, i])))
                for i, name in enumerate(X.columns)
            }
            
            self.importance_scores['shap'] = importance
            return importance
        except Exception as e:
            print(f"SHAP computation failed: {e}")
            return None
    
    def compute_coefficient_importance(self, formula: BaseFormula) -> Optional[Dict[str, float]]:
        """
        Get feature importance from formula coefficients (if available).
        
        Args:
            formula: Fitted formula
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance = formula.get_feature_importance()
        
        if importance:
            self.importance_scores['coefficient'] = importance
        
        return importance
    
    def compute_all(self,
                   formula: BaseFormula,
                   X: pd.DataFrame,
                   y: pd.Series,
                   methods: list = ['coefficient', 'permutation']) -> Dict[str, Dict[str, float]]:
        """
        Compute all available importance methods.
        
        Args:
            formula: Fitted formula
            X: Feature matrix
            y: Target values
            methods: List of methods to use
            
        Returns:
            Dictionary mapping method names to importance dictionaries
        """
        results = {}
        
        if 'coefficient' in methods:
            coef_imp = self.compute_coefficient_importance(formula)
            if coef_imp:
                results['coefficient'] = coef_imp
        
        if 'permutation' in methods:
            perm_imp = self.compute_permutation_importance(formula, X, y)
            if perm_imp:
                results['permutation'] = perm_imp
        
        if 'shap' in methods and HAS_SHAP:
            shap_imp = self.compute_shap_importance(formula, X)
            if shap_imp:
                results['shap'] = shap_imp
        
        self.importance_scores = results
        return results
    
    def get_averaged_importance(self) -> Dict[str, float]:
        """
        Get averaged importance across all methods.
        
        Returns:
            Dictionary with averaged importance scores
        """
        if not self.importance_scores:
            return {}
        
        # Get all feature names
        all_features = set()
        for method_scores in self.importance_scores.values():
            all_features.update(method_scores.keys())
        
        # Average scores across methods
        averaged = {}
        for feature in all_features:
            scores = []
            for method_scores in self.importance_scores.values():
                if feature in method_scores:
                    scores.append(method_scores[feature])
            averaged[feature] = np.mean(scores) if scores else 0.0
        
        return averaged
    
    def get_importance_dataframe(self) -> pd.DataFrame:
        """
        Get importance scores as a DataFrame.
        
        Returns:
            DataFrame with features as rows and methods as columns
        """
        if not self.importance_scores:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.importance_scores)
        df['averaged'] = self.get_averaged_importance().values()
        return df.sort_values('averaged', ascending=False)
