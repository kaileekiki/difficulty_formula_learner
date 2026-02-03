"""
GED Design Analysis
Determine optimal GED calculation and aggregation methods
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from typing import Dict, Optional


class GEDDesigner:
    """Design optimal GED calculation and combination strategy."""
    
    GED_METRICS = ['DFG_GED', 'PDG_GED', 'CFG_GED', 'AST_GED', 'CPG_GED']
    
    def __init__(self):
        """Initialize GED designer."""
        self.optimal_weights = None
        self.combined_ged_scores = None
    
    def learn_ged_weights(self, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         alpha: float = 0.1) -> Dict[str, float]:
        """
        Learn optimal weights for combining GED metrics.
        
        Args:
            X: Feature matrix (must include GED metrics)
            y: Target values (success rates)
            alpha: Regularization strength
            
        Returns:
            Dictionary mapping GED metric names to weights
        """
        # Extract GED features
        ged_features = [col for col in X.columns if col in self.GED_METRICS]
        
        if not ged_features:
            print("Warning: No GED metrics found in features")
            return {}
        
        X_ged = X[ged_features]
        
        # Fit Ridge regression to learn weights
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X_ged, y)
        
        # Get weights
        weights = {
            name: float(coef)
            for name, coef in zip(ged_features, model.coef_)
        }
        
        self.optimal_weights = weights
        return weights
    
    def compute_combined_ged(self, X: pd.DataFrame) -> pd.Series:
        """
        Compute combined GED score using learned weights.
        
        Args:
            X: Feature matrix
            
        Returns:
            Series with combined GED scores
        """
        if self.optimal_weights is None:
            raise ValueError("Must call learn_ged_weights first")
        
        combined = pd.Series(0.0, index=X.index)
        
        for metric, weight in self.optimal_weights.items():
            if metric in X.columns:
                combined += weight * X[metric]
        
        self.combined_ged_scores = combined
        return combined
    
    def analyze_normalization_methods(self,
                                     X: pd.DataFrame,
                                     y: pd.Series) -> Dict[str, float]:
        """
        Analyze different GED normalization methods.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary mapping normalization methods to correlation scores
        """
        from scipy.stats import spearmanr
        
        results = {}
        
        # Extract GED features
        ged_features = [col for col in X.columns if col in self.GED_METRICS]
        
        for feature in ged_features:
            if feature not in X.columns:
                continue
            
            # Compute correlation with raw values
            corr_raw, _ = spearmanr(X[feature], y)
            results[f'{feature}_raw'] = corr_raw
            
            # Compute correlation with log-transformed values
            X_log = np.log1p(X[feature])
            corr_log, _ = spearmanr(X_log, y)
            results[f'{feature}_log'] = corr_log
            
            # Compute correlation with sqrt-transformed values
            X_sqrt = np.sqrt(X[feature])
            corr_sqrt, _ = spearmanr(X_sqrt, y)
            results[f'{feature}_sqrt'] = corr_sqrt
        
        return results
    
    def recommend_aggregation_method(self,
                                    metrics_by_aggregation: Dict[str, pd.DataFrame],
                                    y: pd.Series) -> str:
        """
        Recommend best aggregation method (sum, avg, max) for GED metrics.
        
        Args:
            metrics_by_aggregation: Dictionary mapping aggregation method to feature matrix
            y: Target values
            
        Returns:
            Recommended aggregation method
        """
        from scipy.stats import spearmanr
        
        results = {}
        
        for agg_method, X in metrics_by_aggregation.items():
            # Compute average correlation across GED metrics
            correlations = []
            
            for metric in self.GED_METRICS:
                if metric in X.columns:
                    corr, _ = spearmanr(X[metric], y)
                    correlations.append(abs(corr))
            
            if correlations:
                results[agg_method] = np.mean(correlations)
        
        if not results:
            return 'sum'  # Default
        
        best_method = max(results, key=results.get)
        return best_method
    
    def get_ged_summary(self) -> pd.DataFrame:
        """
        Get summary of GED analysis.
        
        Returns:
            DataFrame with GED analysis results
        """
        if self.optimal_weights is None:
            return pd.DataFrame()
        
        summary = pd.DataFrame({
            'metric': list(self.optimal_weights.keys()),
            'weight': list(self.optimal_weights.values())
        })
        
        summary['abs_weight'] = summary['weight'].abs()
        summary = summary.sort_values('abs_weight', ascending=False)
        
        return summary
