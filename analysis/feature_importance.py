"""
Feature Importance Analysis
Permutation importance and SHAP values
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.base import BaseEstimator, RegressorMixin

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from formula.generators.base_formula import BaseFormula


class SklearnWrapper(BaseEstimator, RegressorMixin):
    """Wrapper to make our formula compatible with sklearn's permutation_importance."""
    
    def __init__(self, formula: BaseFormula, feature_names: list):
        self.formula = formula
        self.feature_names = feature_names
        self._is_fitted = True  # Already fitted
    
    def fit(self, X, y):
        """No-op fit since the formula is already fitted."""
        return self
    
    def predict(self, X):
        """Predict using the wrapped formula."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.formula.predict(X)
    
    def __sklearn_is_fitted__(self):
        """Required for sklearn compatibility."""
        return True


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
        from sklearn.inspection import permutation_importance
        
        # Create sklearn-compatible wrapper
        wrapper = SklearnWrapper(formula, X.columns.tolist())
        
        # Reset index to ensure compatibility
        X_reset = X.reset_index(drop=True)
        y_reset = y.reset_index(drop=True) if hasattr(y, 'reset_index') else y
        
        try:
            # Compute permutation importance
            result = permutation_importance(
                estimator=wrapper,
                X=X_reset.values,
                y=y_reset.values if hasattr(y_reset, 'values') else np.array(y_reset),
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
            
        except Exception as e:
            print(f"  ⚠️ Permutation importance failed: {e}")
            # Fallback: return coefficient-based importance if available
            return self._fallback_importance(formula, X)
    
    def _fallback_importance(self, formula: BaseFormula, X: pd.DataFrame) -> Dict[str, float]:
        """Fallback importance calculation using formula's built-in method or variance."""
        # Try formula's own importance method
        importance = formula.get_feature_importance()
        if importance:
            return importance
        
        # Fallback: use variance-based importance (less meaningful but safe)
        print("  Using variance-based fallback importance")
        return {col: float(X[col].var()) for col in X.columns}
    
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
            print("  ⚠️ SHAP not available. Install with: pip install shap")
            return None
        
        try:
            # Sample data if too large
            X_sample = X if len(X) <= 100 else X.sample(n=100, random_state=42)
            X_sample = X_sample.reset_index(drop=True)
            
            # Create SHAP explainer with background data
            def predict_func(X_arr):
                X_df = pd.DataFrame(X_arr, columns=X.columns)
                return formula.predict(X_df)
            
            explainer = shap.Explainer(predict_func, X_sample)
            shap_values = explainer(X_sample)
            
            # Compute mean absolute SHAP values
            importance = {
                name: float(np.mean(np.abs(shap_values.values[:, i])))
                for i, name in enumerate(X.columns)
            }
            
            self.importance_scores['shap'] = importance
            return importance
            
        except Exception as e:
            print(f"  ⚠️ SHAP computation failed: {e}")
            return None
    
    def compute_coefficient_importance(self, formula: BaseFormula) -> Optional[Dict[str, float]]:
        """
        Get feature importance from formula coefficients (if available).
        
        Args:
            formula: Fitted formula
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            importance = formula.get_feature_importance()
            
            if importance:
                self.importance_scores['coefficient'] = importance
            
            return importance
        except Exception as e:
            print(f"  ⚠️ Coefficient importance failed: {e}")
            return None
    
    def compute_correlation_importance(self, 
                                       X: pd.DataFrame, 
                                       y: pd.Series) -> Dict[str, float]:
        """
        Compute importance based on correlation with target.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary mapping feature names to absolute correlation scores
        """
        from scipy import stats
        
        importance = {}
        y_arr = y.values if hasattr(y, 'values') else np.array(y)
        
        for col in X.columns:
            try:
                corr, _ = stats.spearmanr(X[col].values, y_arr)
                importance[col] = abs(float(corr)) if not np.isnan(corr) else 0.0
            except Exception:
                importance[col] = 0.0
        
        self.importance_scores['correlation'] = importance
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
        
        # Reset indices for safety
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True) if hasattr(y, 'reset_index') else pd.Series(y)
        
        if 'coefficient' in methods:
            print("  Computing coefficient importance...")
            coef_imp = self.compute_coefficient_importance(formula)
            if coef_imp:
                results['coefficient'] = coef_imp
                print(f"    ✓ Done")
        
        if 'correlation' in methods:
            print("  Computing correlation importance...")
            corr_imp = self.compute_correlation_importance(X, y)
            if corr_imp:
                results['correlation'] = corr_imp
                print(f"    ✓ Done")
        
        if 'permutation' in methods:
            print("  Computing permutation importance...")
            perm_imp = self.compute_permutation_importance(formula, X, y)
            if perm_imp:
                results['permutation'] = perm_imp
                print(f"    ✓ Done")
        
        if 'shap' in methods:
            if HAS_SHAP:
                print("  Computing SHAP importance...")
                shap_imp = self.compute_shap_importance(formula, X)
                if shap_imp:
                    results['shap'] = shap_imp
                    print(f"    ✓ Done")
            else:
                print("  ⚠️ SHAP not available, skipping")
        
        # Ensure we have at least one result
        if not results:
            print("  ⚠️ All methods failed, using correlation fallback")
            results['correlation'] = self.compute_correlation_importance(X, y)
        
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
            if method_scores:
                all_features.update(method_scores.keys())
        
        if not all_features:
            return {}
        
        # Average scores across methods
        averaged = {}
        for feature in all_features:
            scores = []
            for method_scores in self.importance_scores.values():
                if method_scores and feature in method_scores:
                    score = method_scores[feature]
                    if not np.isnan(score):
                        scores.append(abs(score))  # Use absolute values
            averaged[feature] = float(np.mean(scores)) if scores else 0.0
        
        # Normalize to sum to 1
        total = sum(averaged.values())
        if total > 0:
            averaged = {k: v / total for k, v in averaged.items()}
        
        return averaged
    
    def get_importance_dataframe(self) -> pd.DataFrame:
        """
        Get importance scores as a DataFrame.
        
        Returns:
            DataFrame with features as rows and methods as columns
        """
        if not self.importance_scores:
            return pd.DataFrame()
        
        # Filter out None values
        valid_scores = {k: v for k, v in self.importance_scores.items() if v}
        
        if not valid_scores:
            return pd.DataFrame()
        
        df = pd.DataFrame(valid_scores)
        averaged = self.get_averaged_importance()
        
        if averaged:
            df['averaged'] = [averaged.get(idx, 0.0) for idx in df.index]
            return df.sort_values('averaged', ascending=False)
        
        return df
