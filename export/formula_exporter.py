"""
Formula Exporter
Export learned formulas in various formats
"""
import json
import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from formula.generators.base_formula import BaseFormula


class FormulaExporter:
    """Export formulas in multiple formats."""
    
    def __init__(self, output_dir: str = './outputs'):
        """
        Initialize formula exporter.
        
        Args:
            output_dir: Directory to save exports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_python(self, formula: BaseFormula, filepath: str) -> None:
        """
        Export formula as Python function.
        
        Args:
            formula: Fitted formula
            filepath: Path to save Python file
        """
        code = self._generate_python_code(formula)
        
        with open(filepath, 'w') as f:
            f.write(code)
        
        print(f"Exported Python function to {filepath}")
    
    def _generate_python_code(self, formula: BaseFormula) -> str:
        """Generate Python code for the formula."""
        code = []
        code.append("\"\"\"")
        code.append("Auto-generated difficulty prediction function")
        code.append(f"Formula: {formula.name}")
        code.append("\"\"\"")
        code.append("import numpy as np")
        code.append("import pandas as pd")
        code.append("")
        code.append("")
        code.append("def predict_difficulty(features):")
        code.append("    \"\"\"")
        code.append("    Predict bug difficulty (success rate).")
        code.append("    ")
        code.append("    Args:")
        code.append("        features: Dictionary or DataFrame with the following keys:")
        
        if formula.feature_names:
            for feature in formula.feature_names:
                code.append(f"            - {feature}")
        
        code.append("    ")
        code.append("    Returns:")
        code.append("        Predicted success rate (0-1)")
        code.append("    \"\"\"")
        code.append("    # Convert input to numpy array")
        code.append("    if isinstance(features, dict):")
        code.append(f"        feature_order = {formula.feature_names}")
        code.append("        X = np.array([features[k] for k in feature_order]).reshape(1, -1)")
        code.append("    elif isinstance(features, pd.DataFrame):")
        code.append("        X = features.values")
        code.append("    else:")
        code.append("        X = np.array(features).reshape(1, -1)")
        code.append("    ")
        code.append("    # Note: This is a simplified export. For full functionality,")
        code.append("    # use the original formula object with pickle/joblib.")
        code.append(f"    # Formula: {formula.get_formula_string()}")
        code.append("    ")
        code.append("    # Placeholder - replace with actual prediction logic")
        code.append("    raise NotImplementedError('Use joblib to load the full model')")
        code.append("")
        code.append("")
        code.append("if __name__ == '__main__':")
        code.append("    # Example usage")
        code.append("    example_features = {")
        
        if formula.feature_names:
            for feature in formula.feature_names:
                code.append(f"        '{feature}': 0.0,")
        
        code.append("    }")
        code.append("    ")
        code.append("    # prediction = predict_difficulty(example_features)")
        code.append("    # print(f'Predicted success rate: {prediction}')")
        code.append("    print('Load the model with joblib for full functionality')")
        
        return "\n".join(code)
    
    def export_json(self, formula: BaseFormula, evaluation_results: Dict[str, Any], filepath: str) -> None:
        """
        Export formula metadata and coefficients as JSON.
        
        Args:
            formula: Fitted formula
            evaluation_results: Evaluation metrics
            filepath: Path to save JSON file
        """
        data = {
            'formula': {
                'name': formula.name,
                'formula_string': formula.get_formula_string(),
                'complexity': formula.get_complexity(),
                'feature_names': formula.feature_names if formula.feature_names else [],
                'feature_importance': formula.get_feature_importance()
            },
            'evaluation': evaluation_results,
            'metadata': formula.get_metadata()
        }
        
        # Convert numpy types to Python types for JSON serialization
        data = self._convert_to_json_serializable(data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported JSON to {filepath}")
    
    def export_latex(self, formula: BaseFormula, filepath: str) -> None:
        """
        Export formula as LaTeX equation.
        
        Args:
            formula: Fitted formula
            filepath: Path to save LaTeX file
        """
        latex = self._generate_latex(formula)
        
        with open(filepath, 'w') as f:
            f.write(latex)
        
        print(f"Exported LaTeX to {filepath}")
    
    def _generate_latex(self, formula: BaseFormula) -> str:
        """Generate LaTeX code for the formula."""
        lines = []
        lines.append("\\documentclass{article}")
        lines.append("\\usepackage{amsmath}")
        lines.append("\\begin{document}")
        lines.append("")
        lines.append("\\section*{Difficulty Prediction Formula}")
        lines.append("")
        lines.append(f"\\textbf{{Formula Name:}} {formula.name}")
        lines.append("")
        lines.append("\\subsection*{Formula}")
        lines.append("")
        lines.append("\\begin{equation}")
        
        # Simplified LaTeX representation
        formula_str = formula.get_formula_string()
        # Basic cleanup for LaTeX
        latex_formula = formula_str.replace('_', '\\_')
        lines.append(f"    {latex_formula}")
        
        lines.append("\\end{equation}")
        lines.append("")
        
        # Add feature importance if available
        importance = formula.get_feature_importance()
        if importance:
            lines.append("\\subsection*{Feature Importance}")
            lines.append("\\begin{itemize}")
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feature, value in sorted_importance[:10]:  # Top 10
                feature_tex = feature.replace('_', '\\_')
                lines.append(f"    \\item {feature_tex}: {value:.4f}")
            lines.append("\\end{itemize}")
        
        lines.append("")
        lines.append("\\end{document}")
        
        return "\n".join(lines)
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
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
    
    def export_model(self, formula: BaseFormula, filepath: str) -> None:
        """
        Export the full model using joblib.
        
        Args:
            formula: Fitted formula
            filepath: Path to save model
        """
        import joblib
        
        joblib.dump(formula, filepath)
        print(f"Exported model to {filepath}")
    
    def load_model(self, filepath: str) -> BaseFormula:
        """
        Load a saved model.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded formula
        """
        import joblib
        
        formula = joblib.load(filepath)
        print(f"Loaded model from {filepath}")
        return formula
