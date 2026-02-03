"""
Tests for formula generators
"""
import unittest
import pandas as pd
import numpy as np

from formula.generators.linear_formula import LinearFormula
from formula.generators.polynomial_formula import PolynomialFormula
from formula.generators.nonlinear_formula import NonlinearFormula


class TestLinearFormula(unittest.TestCase):
    """Test LinearFormula class."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        self.y = pd.Series(0.5 + 0.3 * self.X['feature1'] - 0.2 * self.X['feature2'] + np.random.randn(100) * 0.1)
    
    def test_fit_predict(self):
        """Test fitting and prediction."""
        formula = LinearFormula(regularization='ridge', alpha=1.0)
        formula.fit(self.X, self.y)
        
        self.assertTrue(formula.is_fitted)
        
        y_pred = formula.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))
    
    def test_formula_string(self):
        """Test formula string generation."""
        formula = LinearFormula(regularization='ridge', alpha=1.0)
        formula.fit(self.X, self.y)
        
        formula_str = formula.get_formula_string()
        self.assertIsInstance(formula_str, str)
        self.assertIn('success_rate', formula_str)
    
    def test_feature_importance(self):
        """Test feature importance."""
        formula = LinearFormula(regularization='ridge', alpha=1.0)
        formula.fit(self.X, self.y)
        
        importance = formula.get_feature_importance()
        self.assertEqual(len(importance), 3)
        self.assertIn('feature1', importance)


class TestPolynomialFormula(unittest.TestCase):
    """Test PolynomialFormula class."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        self.y = pd.Series(0.5 + 0.3 * self.X['feature1'] + 0.1 * self.X['feature1']**2 + np.random.randn(100) * 0.1)
    
    def test_fit_predict(self):
        """Test fitting and prediction."""
        formula = PolynomialFormula(degree=2)
        formula.fit(self.X, self.y)
        
        self.assertTrue(formula.is_fitted)
        
        y_pred = formula.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))
    
    def test_complexity(self):
        """Test complexity calculation."""
        formula = PolynomialFormula(degree=2)
        formula.fit(self.X, self.y)
        
        complexity = formula.get_complexity()
        self.assertGreater(complexity, 2)  # More than original features


class TestNonlinearFormula(unittest.TestCase):
    """Test NonlinearFormula class."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.abs(np.random.randn(100)),  # Positive values for log
            'feature2': np.abs(np.random.randn(100))
        })
        self.y = pd.Series(0.5 + 0.3 * np.log1p(self.X['feature1']) + np.random.randn(100) * 0.1)
    
    def test_fit_predict(self):
        """Test fitting and prediction."""
        formula = NonlinearFormula(transformations=['log', 'sqrt'])
        formula.fit(self.X, self.y)
        
        self.assertTrue(formula.is_fitted)
        
        y_pred = formula.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))
    
    def test_transformations(self):
        """Test feature transformations."""
        formula = NonlinearFormula(transformations=['log', 'sqrt'])
        formula.fit(self.X, self.y)
        
        # Check that transformed features are created
        self.assertGreater(len(formula.transformed_feature_names), len(self.X.columns))


if __name__ == '__main__':
    unittest.main()
