"""
Metrics Loader for v3_progress_*.json files
Extracts 13 code complexity metrics for each bug instance
"""
import json
import glob
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


class MetricsLoader:
    """Load and process bug metrics from v3_progress JSON files."""
    
    METRIC_NAMES = {
        'basic': ['LOC', 'Token_Edit_Distance', 'Cyclomatic_Complexity', 
                  'Halstead_Difficulty', 'Variable_Scope'],
        'ast': ['AST_GED', 'Exception_Handling', 'Type_Changes'],
        'graph': ['CFG_GED', 'DFG_GED', 'Call_Graph_GED', 'PDG_GED', 'CPG_GED']
    }
    
    def __init__(self, metrics_dir: str, aggregation: str = 'sum'):
        """
        Initialize metrics loader.
        
        Args:
            metrics_dir: Directory containing v3_progress_*.json files
            aggregation: How to aggregate metrics ('sum', 'avg', 'max', 'auto')
        """
        self.metrics_dir = metrics_dir
        self.aggregation = aggregation
        self.data = None
    
    def load(self) -> pd.DataFrame:
        """
        Load all metrics from JSON files.
        
        Returns:
            DataFrame with columns: instance_id + 13 metric columns
        """
        json_files = glob.glob(os.path.join(self.metrics_dir, 'v3_progress_*.json'))
        
        if not json_files:
            raise FileNotFoundError(f"No v3_progress_*.json files found in {self.metrics_dir}")
        
        all_data = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            for result in data.get('results', []):
                instance_id = result['instance_id']
                metrics = self._extract_metrics(result)
                metrics['instance_id'] = instance_id
                all_data.append(metrics)
        
        self.data = pd.DataFrame(all_data)
        
        # Set instance_id as index
        if 'instance_id' in self.data.columns:
            self.data = self.data.set_index('instance_id')
        
        return self.data
    
    def _extract_metrics(self, result: Dict) -> Dict[str, float]:
        """
        Extract 13 metrics from a result entry.
        
        Args:
            result: Single result dictionary from JSON
            
        Returns:
            Dictionary mapping metric name to value
        """
        metrics = {}
        
        try:
            summary = result['metrics']['changed_files_metrics']['summary']
            
            # Extract basic metrics
            for metric in self.METRIC_NAMES['basic']:
                value = self._get_aggregated_value(summary['basic'].get(metric, {}))
                metrics[metric] = value
            
            # Extract AST metrics
            for metric in self.METRIC_NAMES['ast']:
                value = self._get_aggregated_value(summary['ast'].get(metric, {}))
                metrics[metric] = value
            
            # Extract graph metrics
            for metric in self.METRIC_NAMES['graph']:
                value = self._get_aggregated_value(summary['graph'].get(metric, {}))
                metrics[metric] = value
                
        except (KeyError, TypeError) as e:
            # If metrics are missing, fill with zeros
            for tier_metrics in self.METRIC_NAMES.values():
                for metric in tier_metrics:
                    metrics[metric] = 0.0
        
        return metrics
    
    def _get_aggregated_value(self, metric_dict: Dict) -> float:
        """
        Get aggregated value from metric dictionary based on aggregation method.
        
        Args:
            metric_dict: Dictionary with 'sum', 'avg', 'max' keys
            
        Returns:
            Aggregated value
        """
        if not metric_dict:
            return 0.0
        
        if self.aggregation == 'sum':
            return float(metric_dict.get('sum', 0))
        elif self.aggregation == 'avg':
            return float(metric_dict.get('avg', 0))
        elif self.aggregation == 'max':
            return float(metric_dict.get('max', 0))
        elif self.aggregation == 'auto':
            # Use sum by default, but could implement smarter logic
            return float(metric_dict.get('sum', 0))
        else:
            return float(metric_dict.get('sum', 0))
    
    def get_feature_names(self) -> List[str]:
        """Get list of all 13 feature names."""
        features = []
        for tier_metrics in self.METRIC_NAMES.values():
            features.extend(tier_metrics)
        return features
    
    def get_metrics_by_tier(self) -> Dict[str, List[str]]:
        """Get metrics organized by tier."""
        return self.METRIC_NAMES.copy()
