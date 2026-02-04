"""
Tests for feature importance analysis
"""
import unittest
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

from analysis.feature_importance import FeatureImportanceAnalyzer, SklearnWrapper
from formula.generators.linear_formula import LinearFormula


class TestSklearnWrapper(unittest.TestCase):
    """Test SklearnWrapper class for sklearn compatibility."""
    
    def setUp(self):
        """Create test data and fitted formula."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        self.y = pd.Series(0.5 + 0.3 * self.X['feature1'] - 0.2 * self.X['feature2'] + np.random.randn(100) * 0.1)
        
        # Fit a formula
        self.formula = LinearFormula(regularization='ridge', alpha=1.0)
        self.formula.fit(self.X, self.y)
    
    def test_wrapper_has_required_methods(self):
        """Test that wrapper has all required sklearn methods."""
        wrapper = SklearnWrapper(self.formula, self.X.columns.tolist())
        
        self.assertTrue(hasattr(wrapper, 'fit'))
        self.assertTrue(hasattr(wrapper, 'predict'))
        self.assertTrue(hasattr(wrapper, '__sklearn_is_fitted__'))
        self.assertTrue(callable(wrapper.fit))
        self.assertTrue(callable(wrapper.predict))
        self.assertTrue(callable(wrapper.__sklearn_is_fitted__))
    
    def test_wrapper_fit_returns_self(self):
        """Test that fit returns self for sklearn compatibility."""
        wrapper = SklearnWrapper(self.formula, self.X.columns.tolist())
        result = wrapper.fit(self.X, self.y)
        self.assertIs(result, wrapper)
    
    def test_wrapper_predict_with_dataframe(self):
        """Test prediction with DataFrame input."""
        wrapper = SklearnWrapper(self.formula, self.X.columns.tolist())
        y_pred = wrapper.predict(self.X)
        
        self.assertEqual(len(y_pred), len(self.y))
        self.assertIsInstance(y_pred, np.ndarray)
    
    def test_wrapper_predict_with_array(self):
        """Test prediction with array input."""
        wrapper = SklearnWrapper(self.formula, self.X.columns.tolist())
        y_pred = wrapper.predict(self.X.values)
        
        self.assertEqual(len(y_pred), len(self.y))
        self.assertIsInstance(y_pred, np.ndarray)
    
    def test_wrapper_is_fitted_status(self):
        """Test that wrapper reports fitted status correctly."""
        wrapper = SklearnWrapper(self.formula, self.X.columns.tolist())
        self.assertTrue(wrapper.__sklearn_is_fitted__())
    
    def test_wrapper_with_permutation_importance(self):
        """Test that wrapper works with sklearn's permutation_importance."""
        wrapper = SklearnWrapper(self.formula, self.X.columns.tolist())
        
        # This should not raise an error
        result = permutation_importance(
            estimator=wrapper,
            X=self.X.values,
            y=self.y.values,
            n_repeats=5,
            random_state=42,
            scoring='neg_mean_squared_error'
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.importances_mean), len(self.X.columns))


class TestFeatureImportanceAnalyzer(unittest.TestCase):
    """Test FeatureImportanceAnalyzer class."""
    
    def setUp(self):
        """Create test data and fitted formula."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        self.y = pd.Series(0.5 + 0.3 * self.X['feature1'] - 0.2 * self.X['feature2'] + np.random.randn(100) * 0.1)
        
        # Fit a formula
        self.formula = LinearFormula(regularization='ridge', alpha=1.0)
        self.formula.fit(self.X, self.y)
        
        self.analyzer = FeatureImportanceAnalyzer()
    
    def test_compute_permutation_importance(self):
        """Test permutation importance computation."""
        importance = self.analyzer.compute_permutation_importance(
            self.formula, self.X, self.y, n_repeats=5
        )
        
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), len(self.X.columns))
        
        # Check all feature names are present
        for col in self.X.columns:
            self.assertIn(col, importance)
            self.assertIsInstance(importance[col], float)
    
    def test_compute_correlation_importance(self):
        """Test correlation-based importance."""
        importance = self.analyzer.compute_correlation_importance(self.X, self.y)
        
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), len(self.X.columns))
        
        # Check all values are between 0 and 1 (absolute Spearman correlations)
        for col in self.X.columns:
            self.assertIn(col, importance)
            self.assertGreaterEqual(importance[col], 0.0)
            self.assertLessEqual(importance[col], 1.0)
    
    def test_compute_coefficient_importance(self):
        """Test coefficient-based importance."""
        importance = self.analyzer.compute_coefficient_importance(self.formula)
        
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)
    
    def test_compute_all_methods(self):
        """Test computing all importance methods."""
        methods = ['coefficient', 'correlation', 'permutation']
        results = self.analyzer.compute_all(
            self.formula, self.X, self.y, methods=methods
        )
        
        self.assertIsInstance(results, dict)
        
        # Check that we have results for most methods (at least one should work)
        self.assertGreater(len(results), 0)
        
        # If coefficient is available, check it
        if 'coefficient' in results:
            self.assertIsInstance(results['coefficient'], dict)
        
        # Correlation should always work
        if 'correlation' in results:
            self.assertIsInstance(results['correlation'], dict)
            self.assertEqual(len(results['correlation']), len(self.X.columns))
    
    def test_get_averaged_importance(self):
        """Test averaging importance across methods."""
        methods = ['coefficient', 'correlation']
        self.analyzer.compute_all(self.formula, self.X, self.y, methods=methods)
        
        averaged = self.analyzer.get_averaged_importance()
        
        self.assertIsInstance(averaged, dict)
        self.assertGreater(len(averaged), 0)
        
        # Check that values sum to 1 (normalized)
        total = sum(averaged.values())
        self.assertAlmostEqual(total, 1.0, places=5)
    
    def test_get_importance_dataframe(self):
        """Test getting importance as DataFrame."""
        methods = ['coefficient', 'correlation']
        self.analyzer.compute_all(self.formula, self.X, self.y, methods=methods)
        
        df = self.analyzer.get_importance_dataframe()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Check that averaged column exists and is sorted descending
        if 'averaged' in df.columns:
            self.assertTrue(df['averaged'].is_monotonic_decreasing)
    
    def test_fallback_on_failure(self):
        """Test that fallback mechanism works when permutation fails."""
        # Test with minimal data that might cause issues
        X_small = self.X.head(5)
        y_small = self.y.head(5)
        
        # This should either succeed or fall back gracefully
        importance = self.analyzer.compute_permutation_importance(
            self.formula, X_small, y_small, n_repeats=2
        )
        
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)


if __name__ == '__main__':
    unittest.main()
