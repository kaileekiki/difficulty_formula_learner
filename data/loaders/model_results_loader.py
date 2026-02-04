"""
Model Results Loader for model_bug_matrix.csv
Computes bug success rates from model pass/fail data
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class ModelResultsLoader:
    """Load and process model-bug success matrix."""
    
    def __init__(self, matrix_file: str):
        """
        Initialize model results loader.
        
        Args:
            matrix_file: Path to model_bug_matrix.csv file
        """
        self.matrix_file = matrix_file
        self.matrix = None
        self.success_rates = None
    
    def load(self) -> pd.Series:
        """
        Load model-bug matrix and compute success rates.
        
        Returns:
            Series with bug_id as index and success_rate as value
        """
        # Load the CSV file
        self.matrix = pd.read_csv(self.matrix_file)
        
        # Check if first column is model_name
        first_col = self.matrix.columns[0]
        if first_col in ['model_name', 'model_id', 'Unnamed: 0']:
            self.matrix = self.matrix.set_index(first_col)
        
        # Ensure all values are numeric (0 or 1)
        bug_columns = self.matrix.columns
        for col in bug_columns:
            self.matrix[col] = pd.to_numeric(self.matrix[col], errors='coerce').fillna(0).astype(int)
        
        # Compute success rate for each bug (column)
        # Success rate = (number of models that solved it) / (total models)
        total_models = len(self.matrix)
        
        if total_models == 0:
            raise ValueError("No models found in matrix file")
        
        # Sum successes (1s) for each bug
        success_counts = self.matrix.sum(axis=0)
        
        # Calculate success rates
        self.success_rates = success_counts / total_models
        
        # Ensure index is string type for matching
        self.success_rates.index = self.success_rates.index.astype(str)
        
        print(f"  Loaded {len(self.matrix)} models × {len(self.success_rates)} bugs")
        print(f"  Success rate range: {self.success_rates.min():.2%} - {self.success_rates.max():.2%}")
        
        return self.success_rates
    
    def get_bug_ids(self):
        """Get list of all bug IDs."""
        if self.success_rates is not None:
            return self.success_rates.index.tolist()
        return []
    
    def get_model_names(self):
        """Get list of all model names."""
        if self.matrix is not None:
            return self.matrix.index.tolist()
        return []
    
    def get_bug_stats(self) -> pd.DataFrame:
        """
        Get detailed statistics for each bug.
        
        Returns:
            DataFrame with bug statistics
        """
        if self.matrix is None:
            return None
        
        stats = pd.DataFrame({
            'success_count': self.matrix.sum(axis=0),
            'failure_count': (1 - self.matrix).sum(axis=0),
            'success_rate': self.success_rates,
            'total_models': len(self.matrix)
        })
        
        return stats
    
    def get_difficulty_category(self, success_rate: float) -> str:
        """
        Categorize bug difficulty based on success rate.
        
        Args:
            success_rate: Bug success rate (0-1)
            
        Returns:
            Difficulty category string
        """
        if success_rate >= 0.75:
            return 'Easy'
        elif success_rate >= 0.5:
            return 'Medium'
        elif success_rate >= 0.25:
            return 'Hard'
        else:
            return 'Very Hard'
    
    def get_difficulty_categories(self) -> pd.Series:
        """Get difficulty category for each bug."""
        if self.success_rates is None:
            return None
        
        return self.success_rates.apply(self.get_difficulty_category)
