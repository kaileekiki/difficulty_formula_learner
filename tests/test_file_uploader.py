"""
Tests for FileUploader class
"""
import unittest
import tempfile
import os
import json
import shutil
from pathlib import Path

from data.loaders.file_uploader import FileUploader


class TestFileUploader(unittest.TestCase):
    """Test FileUploader class."""
    
    def setUp(self):
        """Create temporary test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.source_dir = os.path.join(self.test_dir, "source")
        self.data_dir = os.path.join(self.test_dir, "data_sample")
        os.makedirs(self.source_dir)
        
        # Create test metrics file
        self.test_metrics = {
            "count": 1,
            "results": [
                {
                    "instance_id": "test_bug_1",
                    "metrics": {
                        "changed_files_metrics": {
                            "summary": {
                                "basic": {"LOC": {"sum": 10}},
                                "ast": {"AST_GED": {"sum": 5}},
                                "graph": {"DFG_GED": {"sum": 3}}
                            }
                        }
                    }
                }
            ]
        }
        
        self.metrics_file = os.path.join(self.source_dir, "v3_progress_test.json")
        with open(self.metrics_file, 'w') as f:
            json.dump(self.test_metrics, f)
        
        # Create test matrix file
        self.matrix_file = os.path.join(self.source_dir, "model_bug_matrix.csv")
        with open(self.matrix_file, 'w') as f:
            f.write("model_name,test_bug_1\n")
            f.write("Model_A,1\n")
            f.write("Model_B,0\n")
        
        self.uploader = FileUploader(data_dir=self.data_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test FileUploader initialization."""
        self.assertTrue(os.path.exists(self.uploader.metrics_dir))
        self.assertTrue(os.path.exists(self.uploader.results_dir))
    
    def test_upload_metrics_file(self):
        """Test uploading a metrics file."""
        dest = self.uploader.upload_metrics_file(self.metrics_file)
        self.assertTrue(os.path.exists(dest))
        self.assertIn("v3_progress_test.json", dest)
    
    def test_upload_invalid_metrics_file(self):
        """Test uploading an invalid metrics file."""
        invalid_file = os.path.join(self.source_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("{}")
        
        with self.assertRaises(ValueError):
            self.uploader.upload_metrics_file(invalid_file)
    
    def test_upload_nonexistent_file(self):
        """Test uploading a nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            self.uploader.upload_metrics_file("nonexistent.json")
    
    def test_upload_matrix_file(self):
        """Test uploading a matrix file."""
        dest = self.uploader.upload_matrix_file(self.matrix_file)
        self.assertTrue(os.path.exists(dest))
        self.assertIn("model_bug_matrix.csv", dest)
    
    def test_validate_uploaded_data(self):
        """Test data validation."""
        # Upload files
        self.uploader.upload_metrics_file(self.metrics_file)
        self.uploader.upload_matrix_file(self.matrix_file)
        
        # Validate
        validation = self.uploader.validate_uploaded_data()
        self.assertEqual(validation['metrics_files_count'], 1)
        self.assertTrue(validation['matrix_file_exists'])
        self.assertTrue(validation['is_valid'])
        self.assertEqual(validation['total_bugs_in_metrics'], 1)
    
    def test_get_paths(self):
        """Test getting paths to uploaded data."""
        self.uploader.upload_matrix_file(self.matrix_file)
        
        metrics_dir = self.uploader.get_metrics_dir()
        matrix_file = self.uploader.get_matrix_file()
        
        self.assertIsNotNone(metrics_dir)
        self.assertIsNotNone(matrix_file)
        self.assertTrue(os.path.exists(matrix_file))


if __name__ == '__main__':
    unittest.main()
