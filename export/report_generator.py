"""
Report Generator
Generate comprehensive analysis reports
"""
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import json


class ReportGenerator:
    """Generate analysis reports in multiple formats."""
    
    def __init__(self, output_dir: str = './outputs/reports'):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_markdown_report(self,
                                 results: Dict[str, Any],
                                 filepath: str) -> None:
        """
        Generate comprehensive Markdown report.
        
        Args:
            results: Dictionary with all analysis results
            filepath: Path to save report
        """
        lines = []
        
        # Header
        lines.append("# Bug Difficulty Formula Learning Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        if 'best_formula' in results:
            formula_info = results['best_formula']
            lines.append(f"**Best Formula:** {formula_info.get('name', 'N/A')}")
            lines.append("")
            
            if 'evaluation' in results:
                eval_metrics = results['evaluation']
                lines.append("**Performance Metrics:**")
                lines.append("")
                lines.append(f"- Spearman's ρ: {eval_metrics.get('spearman', 0):.4f}")
                lines.append(f"- R²: {eval_metrics.get('r2', 0):.4f}")
                lines.append(f"- RMSE: {eval_metrics.get('rmse', 0):.4f}")
                lines.append(f"- MAE: {eval_metrics.get('mae', 0):.4f}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Formula Details
        lines.append("## Best Formula")
        lines.append("")
        if 'best_formula' in results:
            formula_info = results['best_formula']
            lines.append("### Formula String")
            lines.append("")
            lines.append("```")
            lines.append(formula_info.get('formula_string', 'N/A'))
            lines.append("```")
            lines.append("")
            
            lines.append(f"**Complexity:** {formula_info.get('complexity', 'N/A')}")
            lines.append("")
        
        # Feature Importance
        if 'feature_importance' in results:
            lines.append("## Feature Importance")
            lines.append("")
            
            importance = results['feature_importance']
            if isinstance(importance, dict):
                lines.append("### Top Features")
                lines.append("")
                lines.append("| Rank | Feature | Importance |")
                lines.append("|------|---------|------------|")
                
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for rank, (feature, value) in enumerate(sorted_features[:10], 1):
                    lines.append(f"| {rank} | {feature} | {value:.4f} |")
                
                lines.append("")
        
        # All Formulas Comparison
        if 'all_formulas' in results:
            lines.append("## Formula Comparison")
            lines.append("")
            lines.append("### All Evaluated Formulas")
            lines.append("")
            
            formulas_df = results['all_formulas']
            if isinstance(formulas_df, pd.DataFrame):
                lines.append(formulas_df.to_markdown(index=False, floatfmt=".4f"))
            
            lines.append("")
        
        # GED Analysis
        if 'ged_analysis' in results:
            lines.append("## GED Metric Analysis")
            lines.append("")
            
            ged_info = results['ged_analysis']
            if 'weights' in ged_info:
                lines.append("### Optimal GED Weights")
                lines.append("")
                lines.append("| GED Metric | Weight |")
                lines.append("|------------|--------|")
                
                for metric, weight in ged_info['weights'].items():
                    lines.append(f"| {metric} | {weight:.4f} |")
                
                lines.append("")
        
        # Comparative Analysis
        if 'comparative_analysis' in results:
            lines.append("## Comparative Analysis")
            lines.append("")
            
            comp_df = results['comparative_analysis']
            if isinstance(comp_df, pd.DataFrame):
                lines.append(comp_df.to_markdown(index=False, floatfmt=".4f"))
            
            lines.append("")
        
        # Visualizations
        if 'figures' in results:
            lines.append("## Visualizations")
            lines.append("")
            
            for fig_name, fig_path in results['figures'].items():
                lines.append(f"### {fig_name}")
                lines.append("")
                lines.append(f"![{fig_name}]({fig_path})")
                lines.append("")
        
        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        lines.append("Based on the analysis:")
        lines.append("")
        
        if 'evaluation' in results:
            eval_metrics = results['evaluation']
            spearman = eval_metrics.get('spearman', 0)
            r2 = eval_metrics.get('r2', 0)
            
            if spearman >= 0.7 and r2 >= 0.5:
                lines.append("✅ **Excellent Performance:** The formula achieves the target performance goals.")
            elif spearman >= 0.5 or r2 >= 0.3:
                lines.append("⚠️ **Moderate Performance:** The formula shows promise but could be improved.")
                lines.append("")
                lines.append("**Suggestions:**")
                lines.append("- Consider collecting more training data")
                lines.append("- Try different feature engineering approaches")
                lines.append("- Experiment with ensemble methods")
            else:
                lines.append("❌ **Low Performance:** The formula needs significant improvement.")
                lines.append("")
                lines.append("**Suggestions:**")
                lines.append("- Review data quality and feature relevance")
                lines.append("- Consider alternative modeling approaches")
                lines.append("- Investigate potential data issues or outliers")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by Difficulty Formula Learner*")
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"Generated Markdown report: {filepath}")
    
    def generate_html_report(self,
                            results: Dict[str, Any],
                            filepath: str) -> None:
        """
        Generate HTML report.
        
        Args:
            results: Dictionary with all analysis results
            filepath: Path to save report
        """
        # First generate markdown, then convert to HTML
        md_path = filepath.replace('.html', '.md')
        self.generate_markdown_report(results, md_path)
        
        # Simple HTML wrapper
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<title>Bug Difficulty Formula Learning Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }")
        html.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #4CAF50; color: white; }")
        html.append("tr:nth-child(even) { background-color: #f2f2f2; }")
        html.append("pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; }")
        html.append("code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }")
        html.append("img { max-width: 100%; height: auto; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Read markdown and convert to simple HTML
        with open(md_path, 'r') as f:
            md_content = f.read()
        
        # Very simple markdown to HTML conversion
        html_content = md_content.replace('\n## ', '\n<h2>').replace('\n### ', '\n<h3>')
        html_content = html_content.replace('## ', '<h2>').replace('</h2>\n', '</h2>')
        html_content = html_content.replace('### ', '<h3>').replace('</h3>\n', '</h3>')
        html_content = html_content.replace('\n\n', '</p><p>')
        html_content = f"<p>{html_content}</p>"
        
        html.append(html_content)
        html.append("</body>")
        html.append("</html>")
        
        with open(filepath, 'w') as f:
            f.write("\n".join(html))
        
        print(f"Generated HTML report: {filepath}")
    
    def generate_summary_json(self,
                             results: Dict[str, Any],
                             filepath: str) -> None:
        """
        Generate JSON summary of results.
        
        Args:
            results: Dictionary with all analysis results
            filepath: Path to save JSON
        """
        # Convert pandas objects to serializable format
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Generated JSON summary: {filepath}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
