"""
File Uploader for Manual Data Upload
Supports uploading v3_progress JSON files and model_bug_matrix CSV
"""
import os
import shutil
import json
from pathlib import Path
from typing import List, Optional


class FileUploader:
    """Handle manual file uploads for metrics and model results."""
    
    def __init__(self, data_dir: str = "data/sample"):
        self.data_dir = Path(data_dir)
        self.metrics_dir = self.data_dir / "metrics"
        self.results_dir = self.data_dir / "model_results"
        
        # Create directories if not exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def upload_metrics_file(self, source_path: str) -> str:
        """
        Upload a v3_progress JSON file.
        
        Args:
            source_path: Path to the JSON file
            
        Returns:
            Destination path
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source_path}")
        
        # Validate file naming: must start with "v3_progress_" and end with ".json"
        if not (source.name.startswith("v3_progress_") and source.suffix == ".json"):
            raise ValueError(f"File must be named v3_progress_*.json, got: {source.name}")
        
        dest = self.metrics_dir / source.name
        shutil.copy(source, dest)
        return str(dest)
    
    def upload_metrics_batch(self, source_paths: List[str]) -> List[str]:
        """Upload multiple metrics files."""
        return [self.upload_metrics_file(p) for p in source_paths]
    
    def upload_matrix_file(self, source_path: str) -> str:
        """
        Upload model_bug_matrix.csv file.
        
        Args:
            source_path: Path to the CSV file
            
        Returns:
            Destination path
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source_path}")
        
        dest = self.results_dir / "model_bug_matrix.csv"
        shutil.copy(source, dest)
        return str(dest)
    
    def validate_uploaded_data(self) -> dict:
        """
        Validate uploaded data files.
        
        Returns:
            Dictionary with validation results
        """
        metrics_files = list(self.metrics_dir.glob("v3_progress_*.json"))
        matrix_file = self.results_dir / "model_bug_matrix.csv"
        
        results = {
            "metrics_files_count": len(metrics_files),
            "metrics_files": [f.name for f in metrics_files],
            "matrix_file_exists": matrix_file.exists(),
            "is_valid": len(metrics_files) > 0 and matrix_file.exists()
        }
        
        # Count total bugs in metrics files
        total_bugs = 0
        for mf in metrics_files:
            try:
                with open(mf, 'r') as f:
                    data = json.load(f)
                    total_bugs += len(data.get('results', []))
            except (json.JSONDecodeError, KeyError):
                pass
        results["total_bugs_in_metrics"] = total_bugs
        
        return results
    
    def get_metrics_dir(self) -> str:
        """Get path to metrics directory."""
        return str(self.metrics_dir)
    
    def get_matrix_file(self) -> Optional[str]:
        """Get path to matrix file if exists."""
        matrix_file = self.results_dir / "model_bug_matrix.csv"
        return str(matrix_file) if matrix_file.exists() else None
