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
        Handles duplicate instance_ids by keeping the latest (most complete) entry.
        
        Returns:
            DataFrame with columns: instance_id + 13 metric columns
        """
        json_files = glob.glob(os.path.join(self.metrics_dir, 'v3_progress_*.json'))
        
        if not json_files:
            raise FileNotFoundError(f"No v3_progress_*.json files found in {self.metrics_dir}")
        
        # Sort files by timestamp (newest first) to prefer newer analyses
        json_files = sorted(json_files, reverse=True)
        
        all_data = {}  # Use dict to handle duplicates (keep first = newest)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  ⚠️ Skipping invalid JSON: {os.path.basename(json_file)}")
                continue
                
            for result in data.get('results', []):
                instance_id = result.get('instance_id')
                if not instance_id:
                    continue
                    
                # Skip if already seen (keep first = newest due to sorting)
                if instance_id in all_data:
                    continue
                
                metrics = self._extract_metrics(result)
                if metrics:  # Only add if we got valid metrics
                    all_data[instance_id] = metrics
        
        if not all_data:
            raise ValueError("No valid metrics found in any JSON file")
        
        # Convert to DataFrame
        df_data = []
        for instance_id, metrics in all_data.items():
            row = {'instance_id': instance_id}
            row.update(metrics)
            df_data.append(row)
        
        self.data = pd.DataFrame(df_data)
        
        # Set instance_id as index
        if 'instance_id' in self.data.columns:
            self.data = self.data.set_index('instance_id')
        
        # Remove any duplicate indices (extra safety)
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        
        print(f"  Loaded {len(self.data)} unique bugs from {len(json_files)} JSON files")
        
        return self.data
    
    def _extract_metrics(self, result: Dict) -> Optional[Dict[str, float]]:
        """
        Extract 13 metrics from a result entry.
        
        Args:
            result: Single result dictionary from JSON
            
        Returns:
            Dictionary mapping metric name to value, or None if extraction fails
        """
        metrics = {}
        
        try:
            # Try different possible paths for metrics
            if 'metrics' in result:
                if 'changed_files_metrics' in result['metrics']:
                    summary = result['metrics']['changed_files_metrics'].get('summary', {})
                elif 'summary' in result['metrics']:
                    summary = result['metrics']['summary']
                else:
                    summary = result['metrics']
            else:
                return None
            
            # Extract basic metrics
            basic = summary.get('basic', {})
            for metric in self.METRIC_NAMES['basic']:
                value = self._get_aggregated_value(basic.get(metric, {}))
                metrics[metric] = value
            
            # Extract AST metrics
            ast_metrics = summary.get('ast', {})
            for metric in self.METRIC_NAMES['ast']:
                value = self._get_aggregated_value(ast_metrics.get(metric, {}))
                metrics[metric] = value
            
            # Extract graph metrics
            graph = summary.get('graph', {})
            for metric in self.METRIC_NAMES['graph']:
                value = self._get_aggregated_value(graph.get(metric, {}))
                metrics[metric] = value
            
            return metrics
                
        except (KeyError, TypeError) as e:
            # Return metrics with zeros if partial failure
            for tier_metrics in self.METRIC_NAMES.values():
                for metric in tier_metrics:
                    if metric not in metrics:
                        metrics[metric] = 0.0
            return metrics
    
    def _get_aggregated_value(self, metric_dict: Dict) -> float:
        """
        Get aggregated value from metric dictionary based on aggregation method.
        
        Args:
            metric_dict: Dictionary with 'sum', 'avg', 'max' keys, or a direct value
            
        Returns:
            Aggregated value
        """
        if not metric_dict:
            return 0.0
        
        # Handle case where metric_dict is already a number
        if isinstance(metric_dict, (int, float)):
            return float(metric_dict)
        
        if not isinstance(metric_dict, dict):
            return 0.0
        
        if self.aggregation == 'sum':
            return float(metric_dict.get('sum', metric_dict.get('avg', 0)))
        elif self.aggregation == 'avg':
            return float(metric_dict.get('avg', metric_dict.get('sum', 0)))
        elif self.aggregation == 'max':
            return float(metric_dict.get('max', metric_dict.get('sum', 0)))
        elif self.aggregation == 'auto':
            # Prefer sum, fall back to avg, then max
            return float(metric_dict.get('sum', metric_dict.get('avg', metric_dict.get('max', 0))))
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
