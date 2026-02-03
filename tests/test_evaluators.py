"""
Tests for formula evaluators
"""
import unittest
import numpy as np

from formula.evaluators.correlation_metrics import CorrelationMetrics
from formula.evaluators.regression_metrics import RegressionMetrics
from formula.evaluators.classification_metrics import ClassificationMetrics


class TestCorrelationMetrics(unittest.TestCase):
    """Test CorrelationMetrics class."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.y_true = np.random.rand(100)
        self.y_pred = self.y_true + np.random.randn(100) * 0.1
    
    def test_spearman(self):
        """Test Spearman correlation."""
        corr, pval = CorrelationMetrics.spearman(self.y_true, self.y_pred)
        self.assertGreater(corr, 0.8)
        self.assertLess(pval, 0.01)
    
    def test_kendall(self):
        """Test Kendall correlation."""
        corr, pval = CorrelationMetrics.kendall(self.y_true, self.y_pred)
        self.assertGreater(corr, 0.6)
    
    def test_pearson(self):
        """Test Pearson correlation."""
        corr, pval = CorrelationMetrics.pearson(self.y_true, self.y_pred)
        self.assertGreater(corr, 0.8)
    
    def test_compute_all(self):
        """Test computing all metrics."""
        metrics = CorrelationMetrics.compute_all(self.y_true, self.y_pred)
        
        self.assertIn('spearman', metrics)
        self.assertIn('kendall', metrics)
        self.assertIn('pearson', metrics)


class TestRegressionMetrics(unittest.TestCase):
    """Test RegressionMetrics class."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.y_true = np.random.rand(100)
        self.y_pred = self.y_true + np.random.randn(100) * 0.1
    
    def test_r2(self):
        """Test R² score."""
        r2 = RegressionMetrics.r2(self.y_true, self.y_pred)
        self.assertGreater(r2, 0.5)
        self.assertLessEqual(r2, 1.0)
    
    def test_mse(self):
        """Test MSE."""
        mse = RegressionMetrics.mse(self.y_true, self.y_pred)
        self.assertGreater(mse, 0)
    
    def test_rmse(self):
        """Test RMSE."""
        rmse = RegressionMetrics.rmse(self.y_true, self.y_pred)
        self.assertGreater(rmse, 0)
    
    def test_mae(self):
        """Test MAE."""
        mae = RegressionMetrics.mae(self.y_true, self.y_pred)
        self.assertGreater(mae, 0)
    
    def test_compute_all(self):
        """Test computing all metrics."""
        metrics = RegressionMetrics.compute_all(self.y_true, self.y_pred, n_features=5)
        
        self.assertIn('r2', metrics)
        self.assertIn('adjusted_r2', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)


class TestClassificationMetrics(unittest.TestCase):
    """Test ClassificationMetrics class."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.y_true = np.random.rand(100)
        self.y_pred = self.y_true + np.random.randn(100) * 0.1
    
    def test_categorize(self):
        """Test categorization."""
        rates = np.array([0.9, 0.6, 0.4, 0.1])
        categories = ClassificationMetrics.categorize(rates)
        
        self.assertEqual(categories[0], 'Easy')
        self.assertEqual(categories[1], 'Medium')
        self.assertEqual(categories[2], 'Hard')
        self.assertEqual(categories[3], 'Very Hard')
    
    def test_accuracy(self):
        """Test accuracy."""
        accuracy = ClassificationMetrics.accuracy(self.y_true, self.y_pred)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_compute_all(self):
        """Test computing all metrics."""
        metrics = ClassificationMetrics.compute_all(self.y_true, self.y_pred)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_weighted', metrics)


if __name__ == '__main__':
    unittest.main()
