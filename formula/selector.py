"""
Formula Selector
Select the best formula from multiple candidates
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from typing import List, Dict, Any, Optional

from .generators.base_formula import BaseFormula
from .evaluators.correlation_metrics import CorrelationMetrics
from .evaluators.regression_metrics import RegressionMetrics


class FormulaSelector:
    """Select the best formula from multiple candidates."""
    
    def __init__(self, 
                 primary_metric: str = 'spearman',
                 complexity_penalty: float = 0.01,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize formula selector.
        
        Args:
            primary_metric: Primary metric for selection ('spearman', 'r2', etc.)
            complexity_penalty: Weight for complexity penalty
            cv_folds: Number of cross-validation folds
            random_state: Random seed
        """
        self.primary_metric = primary_metric
        self.complexity_penalty = complexity_penalty
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.evaluation_results = []
    
    def evaluate_formula(self, 
                        formula: BaseFormula, 
                        X: pd.DataFrame, 
                        y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate a single formula using cross-validation.
        
        Args:
            formula: Formula to evaluate
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        n_samples = len(X)
        
        # Reset index to ensure positional indexing works
        X_reset = X.reset_index(drop=True)
        y_reset = y.reset_index(drop=True)
        
        # Adjust CV folds if not enough data
        actual_cv_folds = min(self.cv_folds, n_samples)
        
        if actual_cv_folds < 2:
            # Not enough data for CV, use train=test evaluation
            print(f"      ⚠️ Only {n_samples} samples, using full-data evaluation")
            formula.fit(X_reset, y_reset)
            y_pred = formula.predict(X_reset)
            
            corr_metrics = CorrelationMetrics.compute_all(y_reset.values, y_pred)
            reg_metrics = RegressionMetrics.compute_all(y_reset.values, y_pred, n_features=len(X.columns))
            
            return {
                'formula_name': formula.name,
                'cv_spearman_mean': corr_metrics.get('spearman', 0),
                'cv_spearman_std': 0.0,
                'cv_r2_mean': reg_metrics.get('r2', 0),
                'cv_r2_std': 0.0,
                'cv_rmse_mean': reg_metrics.get('rmse', 0),
                'cv_rmse_std': 0.0,
                'cv_mae_mean': reg_metrics.get('mae', 0),
                'cv_mae_std': 0.0,
                'complexity': formula.get_complexity(),
                'selection_score': corr_metrics.get('spearman', 0) - self.complexity_penalty * formula.get_complexity(),
                'n_samples': n_samples,
                'cv_folds_used': 1
            }
        
        if actual_cv_folds != self.cv_folds:
            print(f"      ⚠️ Adjusted CV folds: {self.cv_folds} → {actual_cv_folds} (only {n_samples} samples)")
        
        kfold = KFold(n_splits=actual_cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            'spearman': [],
            'r2': [],
            'rmse': [],
            'mae': []
        }
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_reset)):
            try:
                # Use numpy array indexing
                X_train = X_reset.iloc[train_idx].copy()
                X_val = X_reset.iloc[val_idx].copy()
                y_train = y_reset.iloc[train_idx].copy()
                y_val = y_reset.iloc[val_idx].copy()
                
                # Fit and predict
                formula.fit(X_train, y_train)
                y_pred = formula.predict(X_val)
                
                # Ensure y_val is numpy array for metrics computation
                y_val_arr = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
                y_pred_arr = np.array(y_pred)
                
                # Compute metrics
                corr_metrics = CorrelationMetrics.compute_all(y_val_arr, y_pred_arr)
                reg_metrics = RegressionMetrics.compute_all(y_val_arr, y_pred_arr, n_features=len(X.columns))
                
                cv_scores['spearman'].append(corr_metrics.get('spearman', 0))
                cv_scores['r2'].append(reg_metrics.get('r2', 0))
                cv_scores['rmse'].append(reg_metrics.get('rmse', 0))
                cv_scores['mae'].append(reg_metrics.get('mae', 0))
                
            except Exception as e:
                print(f"      ⚠️ Fold {fold_idx + 1} failed: {e}")
                continue
        
        if not cv_scores['spearman']:
            raise ValueError("All CV folds failed")
        
        # Compute mean and std of CV scores
        results = {
            'formula_name': formula.name,
            'cv_spearman_mean': np.mean(cv_scores['spearman']),
            'cv_spearman_std': np.std(cv_scores['spearman']),
            'cv_r2_mean': np.mean(cv_scores['r2']),
            'cv_r2_std': np.std(cv_scores['r2']),
            'cv_rmse_mean': np.mean(cv_scores['rmse']),
            'cv_rmse_std': np.std(cv_scores['rmse']),
            'cv_mae_mean': np.mean(cv_scores['mae']),
            'cv_mae_std': np.std(cv_scores['mae']),
            'complexity': formula.get_complexity(),
            'n_samples': n_samples,
            'cv_folds_used': len(cv_scores['spearman'])
        }
        
        # Compute selection score
        primary_score = results[f'cv_{self.primary_metric}_mean']
        complexity_cost = self.complexity_penalty * results['complexity']
        results['selection_score'] = primary_score - complexity_cost
        
        return results
    
    def select_best_formula(self, 
                           formulas: List[BaseFormula], 
                           X: pd.DataFrame, 
                           y: pd.Series) -> BaseFormula:
        """
        Select the best formula from candidates.
        
        Args:
            formulas: List of formula candidates
            X: Feature matrix
            y: Target values
            
        Returns:
            Best formula
        """
        self.evaluation_results = []
        
        n_samples = len(X)
        print(f"\nEvaluating {len(formulas)} formula candidates on {n_samples} samples...")
        
        if n_samples < 5:
            print(f"⚠️ Warning: Only {n_samples} samples available. Results may not be reliable.")
        if n_samples == 0:
            raise ValueError("No data available for evaluation")
        
        for i, formula in enumerate(formulas):
            print(f"  [{i+1}/{len(formulas)}] Evaluating {formula.name}...")
            try:
                results = self.evaluate_formula(formula, X, y)
                self.evaluation_results.append(results)
                print(f"      {self.primary_metric}: {results[f'cv_{self.primary_metric}_mean']:.4f} "
                      f"(±{results[f'cv_{self.primary_metric}_std']:.4f}), "
                      f"complexity: {results['complexity']}")
            except Exception as e:
                print(f"      Error: {e}")
                continue
        
        if not self.evaluation_results:
            raise ValueError("No formulas could be evaluated successfully. "
                           "Check that your data has enough samples and valid values.")
        
        # Select best by selection score
        best_result = max(self.evaluation_results, key=lambda x: x['selection_score'])
        best_idx = [r['formula_name'] for r in self.evaluation_results].index(best_result['formula_name'])
        best_formula = formulas[best_idx]
        
        print(f"\n✓ Best formula: {best_formula.name}")
        print(f"  Selection score: {best_result['selection_score']:.4f}")
        print(f"  {self.primary_metric}: {best_result[f'cv_{self.primary_metric}_mean']:.4f}")
        
        # Fit best formula on full data
        X_reset = X.reset_index(drop=True)
        y_reset = y.reset_index(drop=True)
        best_formula.fit(X_reset, y_reset)
        
        return best_formula
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        Get summary of all evaluations.
        
        Returns:
            DataFrame with evaluation results
        """
        if not self.evaluation_results:
            return pd.DataFrame()
        
        return pd.DataFrame(self.evaluation_results).sort_values('selection_score', ascending=False)
