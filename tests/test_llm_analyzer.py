"""
Tests for LLMAnalyzer class
"""
import unittest
from unittest.mock import Mock, patch
import os
import pandas as pd

from analysis.llm_analyzer import LLMAnalyzer


class TestLLMAnalyzer(unittest.TestCase):
    """Test LLMAnalyzer class."""
    
    def test_initialization_without_api_key(self):
        """Test initialization without API key."""
        analyzer = LLMAnalyzer()
        self.assertFalse(analyzer.is_available)
    
    def test_fallback_interpretation(self):
        """Test fallback interpretation without API."""
        analyzer = LLMAnalyzer()
        
        formula = "y = 0.5 * x1 + 0.3 * x2"
        importance = {
            "DFG_GED": 0.8,
            "LOC": 0.5,
            "CFG_GED": 0.3
        }
        
        result = analyzer.interpret_formula(formula, importance)
        self.assertIn("수식", result)
        self.assertIn("DFG_GED", result)
        self.assertIn("0.8", result)
    
    def test_fallback_recommendations(self):
        """Test fallback recommendations without API."""
        analyzer = LLMAnalyzer()
        
        metrics = {
            "spearman": 0.6,
            "r2": 0.4,
            "rmse": 0.15
        }
        
        # Test with empty correlation data
        result = analyzer.recommend_formula_improvements("y = x", metrics, {})
        self.assertIn("개선 제안", result)
        
        # Test with populated correlation data
        correlation_data = {"DFG_GED": 0.8, "LOC": 0.5}
        result2 = analyzer.recommend_formula_improvements("y = x", metrics, correlation_data)
        self.assertIn("개선 제안", result2)
    
    def test_fallback_selection_explanation(self):
        """Test fallback selection explanation without API."""
        analyzer = LLMAnalyzer()
        
        candidates = [
            {"formula_name": "Linear", "cv_spearman_mean": 0.7, "cv_r2_mean": 0.5, "complexity": 1},
            {"formula_name": "Polynomial", "cv_spearman_mean": 0.6, "cv_r2_mean": 0.4, "complexity": 3}
        ]
        best = candidates[0]
        
        result = analyzer.explain_formula_selection(candidates, best)
        self.assertIn("선택된 수식", result)
        self.assertIn("Linear", result)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('builtins.__import__', side_effect=ImportError)
    def test_initialization_with_openai_key_no_module(self, mock_import):
        """Test initialization with OpenAI API key but no module."""
        analyzer = LLMAnalyzer(provider="openai")
        # Should have key but not be available due to import error
        self.assertEqual(analyzer.api_key, "test_key")
        self.assertFalse(analyzer.is_available)
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch('builtins.__import__', side_effect=ImportError)
    def test_initialization_with_anthropic_key_no_module(self, mock_import):
        """Test initialization with Anthropic API key but no module."""
        analyzer = LLMAnalyzer(provider="anthropic")
        self.assertEqual(analyzer.api_key, "test_key")
        self.assertFalse(analyzer.is_available)
    
    def test_explain_formula_selection_with_dataframe(self):
        """Test explain_formula_selection with DataFrame input."""
        analyzer = LLMAnalyzer()
        
        # Create a DataFrame as would be returned by selector.get_evaluation_summary()
        candidates_df = pd.DataFrame([
            {"formula_name": "Linear", "cv_spearman_mean": 0.7, "cv_r2_mean": 0.5, "complexity": 1},
            {"formula_name": "Polynomial", "cv_spearman_mean": 0.6, "cv_r2_mean": 0.4, "complexity": 3}
        ])
        
        best = {"formula_name": "Linear", "cv_spearman_mean": 0.7, "cv_r2_mean": 0.5, "complexity": 1}
        
        # Should not raise an error even with DataFrame input
        result = analyzer.explain_formula_selection(candidates_df, best)
        self.assertIn("선택된 수식", result)
        self.assertIn("Linear", result)
    
    def test_explain_formula_selection_with_empty_dataframe(self):
        """Test explain_formula_selection with empty DataFrame."""
        analyzer = LLMAnalyzer()
        
        candidates_df = pd.DataFrame()
        best = {"formula_name": "Linear", "cv_spearman_mean": 0.7, "cv_r2_mean": 0.5, "complexity": 1}
        
        result = analyzer.explain_formula_selection(candidates_df, best)
        self.assertIsInstance(result, str)
        self.assertTrue("no" in result.lower() or "formula" in result.lower())


if __name__ == '__main__':
    unittest.main()
