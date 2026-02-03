"""
Tests for data loaders
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import json

from data.loaders.metrics_loader import MetricsLoader
from data.loaders.model_results_loader import ModelResultsLoader


class TestMetricsLoader(unittest.TestCase):
    """Test MetricsLoader class."""
    
    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test JSON file
        test_data = {
            "count": 2,
            "results": [
                {
                    "instance_id": "test_bug_1",
                    "metrics": {
                        "changed_files_metrics": {
                            "summary": {
                                "basic": {
                                    "LOC": {"sum": 10, "avg": 5.0, "max": 10},
                                    "Token_Edit_Distance": {"sum": 20, "avg": 10.0, "max": 20},
                                    "Cyclomatic_Complexity": {"sum": 5, "avg": 2.5, "max": 5},
                                    "Halstead_Difficulty": {"sum": 15.0, "avg": 7.5, "max": 15.0},
                                    "Variable_Scope": {"sum": 3, "avg": 1.5, "max": 3}
                                },
                                "ast": {
                                    "AST_GED": {"sum": 8, "avg": 4.0, "max": 8},
                                    "Exception_Handling": {"sum": 2, "avg": 1.0, "max": 2},
                                    "Type_Changes": {"sum": 1, "avg": 0.5, "max": 1}
                                },
                                "graph": {
                                    "CFG_GED": {"sum": 5.0, "avg": 2.5, "max": 5.0},
                                    "DFG_GED": {"sum": 6.0, "avg": 3.0, "max": 6.0},
                                    "Call_Graph_GED": {"sum": 2.0, "avg": 1.0, "max": 2.0},
                                    "PDG_GED": {"sum": 7.0, "avg": 3.5, "max": 7.0},
                                    "CPG_GED": {"sum": 8.0, "avg": 4.0, "max": 8.0}
                                }
                            }
                        }
                    }
                },
                {
                    "instance_id": "test_bug_2",
                    "metrics": {
                        "changed_files_metrics": {
                            "summary": {
                                "basic": {
                                    "LOC": {"sum": 20, "avg": 10.0, "max": 20},
                                    "Token_Edit_Distance": {"sum": 30, "avg": 15.0, "max": 30},
                                    "Cyclomatic_Complexity": {"sum": 8, "avg": 4.0, "max": 8},
                                    "Halstead_Difficulty": {"sum": 20.0, "avg": 10.0, "max": 20.0},
                                    "Variable_Scope": {"sum": 5, "avg": 2.5, "max": 5}
                                },
                                "ast": {
                                    "AST_GED": {"sum": 12, "avg": 6.0, "max": 12},
                                    "Exception_Handling": {"sum": 3, "avg": 1.5, "max": 3},
                                    "Type_Changes": {"sum": 2, "avg": 1.0, "max": 2}
                                },
                                "graph": {
                                    "CFG_GED": {"sum": 8.0, "avg": 4.0, "max": 8.0},
                                    "DFG_GED": {"sum": 9.0, "avg": 4.5, "max": 9.0},
                                    "Call_Graph_GED": {"sum": 3.0, "avg": 1.5, "max": 3.0},
                                    "PDG_GED": {"sum": 10.0, "avg": 5.0, "max": 10.0},
                                    "CPG_GED": {"sum": 11.0, "avg": 5.5, "max": 11.0}
                                }
                            }
                        }
                    }
                }
            ]
        }
        
        json_path = os.path.join(self.temp_dir, 'v3_progress_test.json')
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
    
    def test_load_metrics(self):
        """Test loading metrics from JSON."""
        loader = MetricsLoader(self.temp_dir, aggregation='sum')
        df = loader.load()
        
        self.assertEqual(len(df), 2)
        self.assertEqual(len(df.columns), 13)
        self.assertIn('test_bug_1', df.index)
        self.assertIn('test_bug_2', df.index)
    
    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        # Test sum
        loader_sum = MetricsLoader(self.temp_dir, aggregation='sum')
        df_sum = loader_sum.load()
        self.assertEqual(df_sum.loc['test_bug_1', 'LOC'], 10)
        
        # Test avg
        loader_avg = MetricsLoader(self.temp_dir, aggregation='avg')
        df_avg = loader_avg.load()
        self.assertEqual(df_avg.loc['test_bug_1', 'LOC'], 5.0)
        
        # Test max
        loader_max = MetricsLoader(self.temp_dir, aggregation='max')
        df_max = loader_max.load()
        self.assertEqual(df_max.loc['test_bug_1', 'LOC'], 10)
    
    def test_feature_names(self):
        """Test getting feature names."""
        loader = MetricsLoader(self.temp_dir)
        features = loader.get_feature_names()
        self.assertEqual(len(features), 13)
        self.assertIn('LOC', features)
        self.assertIn('DFG_GED', features)


class TestModelResultsLoader(unittest.TestCase):
    """Test ModelResultsLoader class."""
    
    def setUp(self):
        """Create temporary test data."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Create test CSV
        csv_content = """model_name,bug_1,bug_2,bug_3
model_a,1,0,1
model_b,1,1,0
model_c,0,1,1
"""
        self.temp_file.write(csv_content)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary file."""
        os.unlink(self.temp_file.name)
    
    def test_load_matrix(self):
        """Test loading model-bug matrix."""
        loader = ModelResultsLoader(self.temp_file.name)
        success_rates = loader.load()
        
        self.assertEqual(len(success_rates), 3)
        self.assertAlmostEqual(success_rates['bug_1'], 2/3)
        self.assertAlmostEqual(success_rates['bug_2'], 2/3)
        self.assertAlmostEqual(success_rates['bug_3'], 2/3)
    
    def test_bug_stats(self):
        """Test getting bug statistics."""
        loader = ModelResultsLoader(self.temp_file.name)
        loader.load()
        stats = loader.get_bug_stats()
        
        self.assertIn('success_count', stats.columns)
        self.assertIn('success_rate', stats.columns)
        self.assertEqual(stats.loc['bug_1', 'success_count'], 2)
    
    def test_difficulty_categories(self):
        """Test difficulty categorization."""
        loader = ModelResultsLoader(self.temp_file.name)
        self.assertEqual(loader.get_difficulty_category(0.8), 'Easy')
        self.assertEqual(loader.get_difficulty_category(0.6), 'Medium')
        self.assertEqual(loader.get_difficulty_category(0.4), 'Hard')
        self.assertEqual(loader.get_difficulty_category(0.2), 'Very Hard')


if __name__ == '__main__':
    unittest.main()
