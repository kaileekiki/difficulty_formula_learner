#!/usr/bin/env python3
"""
Main entry point for Difficulty Formula Learner
Learns and evaluates difficulty prediction formulas from bug metrics and model success data
"""
import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.loaders.metrics_loader import MetricsLoader
from data.loaders.model_results_loader import ModelResultsLoader
from data.preprocessors.feature_normalizer import FeatureNormalizer

from formula.generators.linear_formula import LinearFormula
from formula.generators.polynomial_formula import PolynomialFormula
from formula.generators.nonlinear_formula import NonlinearFormula
from formula.generators.tree_formula import TreeFormula
from formula.generators.symbolic_regression import SymbolicRegressionFormula
from formula.selector import FormulaSelector
from formula.evaluators.correlation_metrics import CorrelationMetrics
from formula.evaluators.regression_metrics import RegressionMetrics
from formula.evaluators.classification_metrics import ClassificationMetrics

from analysis.feature_importance import FeatureImportanceAnalyzer
from analysis.ged_design import GEDDesigner
from analysis.comparative_analysis import ComparativeAnalyzer

from visualization.importance_plots import ImportancePlotter
from visualization.correlation_heatmap import CorrelationHeatmapPlotter
from visualization.prediction_plots import PredictionPlotter

from export.formula_exporter import FormulaExporter
from export.report_generator import ReportGenerator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(config: dict):
    """Load metrics and model results data."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load metrics
    print(f"\nLoading metrics from: {config['data']['metrics_dir']}")
    metrics_loader = MetricsLoader(
        metrics_dir=config['data']['metrics_dir'],
        aggregation=config['data']['aggregation']
    )
    X = metrics_loader.load()
    print(f"✓ Loaded {len(X)} bug instances with {len(X.columns)} features")
    
    # Load model results
    print(f"\nLoading model results from: {config['data']['matrix_file']}")
    results_loader = ModelResultsLoader(matrix_file=config['data']['matrix_file'])
    y = results_loader.load()
    print(f"✓ Loaded success rates for {len(y)} bugs")
    
    # Align data
    common_bugs = X.index.intersection(y.index)
    X = X.loc[common_bugs]
    y = y.loc[common_bugs]
    print(f"✓ {len(common_bugs)} bugs have both metrics and success rates")
    
    return X, y, metrics_loader, results_loader


def preprocess_data(X: pd.DataFrame, config: dict):
    """Preprocess features."""
    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)
    
    normalization = config['data'].get('normalization', 'standard')
    print(f"\nNormalization method: {normalization}")
    
    if normalization and normalization != 'none':
        normalizer = FeatureNormalizer(method=normalization)
        X_normalized = normalizer.fit_transform(X)
        print(f"✓ Features normalized using {normalization}")
        return X_normalized, normalizer
    
    return X, None


def generate_formulas(config: dict):
    """Generate formula candidates."""
    print("\n" + "="*80)
    print("GENERATING FORMULA CANDIDATES")
    print("="*80)
    
    formulas = []
    formula_types = config['formula']['types']
    
    # Linear formulas
    if 'linear' in formula_types:
        print("\n→ Linear formulas:")
        for reg in config['formula']['linear']['regularization']:
            for alpha in config['formula']['linear']['alpha']:
                formula = LinearFormula(regularization=reg, alpha=alpha)
                formulas.append(formula)
                print(f"  - {formula.name} (alpha={alpha})")
    
    # Polynomial formulas
    if 'polynomial' in formula_types:
        print("\n→ Polynomial formulas:")
        for degree in config['formula']['polynomial']['degree']:
            formula = PolynomialFormula(
                degree=degree,
                interaction_only=config['formula']['polynomial']['interaction_only']
            )
            formulas.append(formula)
            print(f"  - {formula.name}")
    
    # Nonlinear formulas
    if 'nonlinear' in formula_types:
        print("\n→ Nonlinear formulas:")
        formula = NonlinearFormula(transformations=['log', 'sqrt'])
        formulas.append(formula)
        print(f"  - {formula.name}")
    
    # Tree-based formulas
    if 'tree' in formula_types:
        print("\n→ Tree-based formulas:")
        formula = TreeFormula(
            model_type='random_forest',
            n_estimators=config['formula']['tree']['n_estimators'],
            max_depth=config['formula']['tree']['max_depth'],
            random_state=config['formula']['tree']['random_state']
        )
        formulas.append(formula)
        print(f"  - {formula.name}")
        
        try:
            formula = TreeFormula(
                model_type='xgboost',
                n_estimators=config['formula']['tree']['n_estimators'],
                max_depth=config['formula']['tree']['max_depth'],
                random_state=config['formula']['tree']['random_state']
            )
            formulas.append(formula)
            print(f"  - {formula.name}")
        except ImportError:
            print("  - XGBoost not available, skipping")
    
    # Symbolic regression
    if 'symbolic' in formula_types:
        print("\n→ Symbolic regression:")
        try:
            formula = SymbolicRegressionFormula(
                population_size=config['formula']['symbolic']['population_size'],
                generations=config['formula']['symbolic']['generations'],
                tournament_size=config['formula']['symbolic']['tournament_size'],
                random_state=config['formula']['symbolic']['random_state']
            )
            formulas.append(formula)
            print(f"  - {formula.name}")
        except ImportError:
            print("  - gplearn not available, skipping")
    
    print(f"\n✓ Generated {len(formulas)} formula candidates")
    return formulas


def select_best_formula(formulas, X, y, config):
    """Select the best formula using cross-validation."""
    print("\n" + "="*80)
    print("SELECTING BEST FORMULA")
    print("="*80)
    
    selector = FormulaSelector(
        primary_metric=config['selection']['primary_metric'],
        complexity_penalty=config['selection']['complexity_penalty'],
        cv_folds=config['evaluation']['cv_folds'],
        random_state=config['evaluation']['random_state']
    )
    
    best_formula = selector.select_best_formula(formulas, X, y)
    evaluation_summary = selector.get_evaluation_summary()
    
    return best_formula, evaluation_summary


def analyze_formula(best_formula, X, y, config):
    """Analyze the selected formula."""
    print("\n" + "="*80)
    print("ANALYZING FORMULA")
    print("="*80)
    
    results = {}
    
    # Evaluate on full data
    print("\n→ Computing final metrics...")
    y_pred = best_formula.predict(X)
    
    corr_metrics = CorrelationMetrics.compute_all(y.values, y_pred)
    reg_metrics = RegressionMetrics.compute_all(y.values, y_pred, n_features=len(X.columns))
    class_metrics = ClassificationMetrics.compute_all(y.values, y_pred)
    
    results['evaluation'] = {**corr_metrics, **reg_metrics, **class_metrics}
    
    print(f"  Spearman ρ: {corr_metrics['spearman']:.4f}")
    print(f"  R²: {reg_metrics['r2']:.4f}")
    print(f"  RMSE: {reg_metrics['rmse']:.4f}")
    
    # Feature importance
    print("\n→ Computing feature importance...")
    importance_analyzer = FeatureImportanceAnalyzer()
    importance_methods = config['analysis']['importance_methods']
    importance_results = importance_analyzer.compute_all(best_formula, X, y, methods=importance_methods)
    results['feature_importance'] = importance_analyzer.get_averaged_importance()
    results['feature_importance_by_method'] = importance_results
    
    # GED analysis
    print("\n→ Analyzing GED metrics...")
    ged_designer = GEDDesigner()
    ged_weights = ged_designer.learn_ged_weights(X, y)
    results['ged_analysis'] = {
        'weights': ged_weights
    }
    
    return results, y_pred


def create_visualizations(best_formula, X, y, y_pred, results, config):
    """Create all visualizations."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    output_dir = os.path.join(config['output']['output_dir'], 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    figures = {}
    
    # Importance plots
    print("\n→ Creating importance plots...")
    plotter = ImportancePlotter(dpi=config['visualization']['figure_dpi'])
    
    importance_path = os.path.join(output_dir, 'feature_importance.png')
    plotter.plot_importance_bar(
        results['feature_importance'],
        title="Feature Importance (Averaged)",
        output_path=importance_path
    )
    figures['Feature Importance'] = importance_path
    
    if 'feature_importance_by_method' in results:
        comparison_path = os.path.join(output_dir, 'feature_importance_comparison.png')
        plotter.plot_importance_comparison(
            results['feature_importance_by_method'],
            title="Feature Importance by Method",
            output_path=comparison_path
        )
        figures['Feature Importance Comparison'] = comparison_path
    
    # Correlation heatmaps
    print("\n→ Creating correlation heatmaps...")
    heatmap_plotter = CorrelationHeatmapPlotter(dpi=config['visualization']['figure_dpi'])
    
    feature_corr_path = os.path.join(output_dir, 'feature_correlation.png')
    heatmap_plotter.plot_feature_correlation(
        X,
        method='spearman',
        title="Feature Correlation Matrix",
        output_path=feature_corr_path
    )
    figures['Feature Correlation'] = feature_corr_path
    
    target_corr_path = os.path.join(output_dir, 'target_correlation.png')
    heatmap_plotter.plot_target_correlation(
        X, y,
        method='spearman',
        title="Feature-Target Correlation",
        output_path=target_corr_path
    )
    figures['Target Correlation'] = target_corr_path
    
    # Prediction plots
    print("\n→ Creating prediction plots...")
    pred_plotter = PredictionPlotter(dpi=config['visualization']['figure_dpi'])
    
    actual_vs_pred_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    pred_plotter.plot_actual_vs_predicted(
        y.values, y_pred,
        title="Actual vs Predicted Success Rate",
        output_path=actual_vs_pred_path
    )
    figures['Actual vs Predicted'] = actual_vs_pred_path
    
    residuals_path = os.path.join(output_dir, 'residuals.png')
    pred_plotter.plot_residuals(
        y.values, y_pred,
        title="Residual Distribution",
        output_path=residuals_path
    )
    figures['Residuals'] = residuals_path
    
    print(f"\n✓ Created {len(figures)} visualizations")
    return figures


def export_results(best_formula, evaluation_summary, results, figures, config):
    """Export results in various formats."""
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    output_dir = config['output']['output_dir']
    export_formats = config['output']['export_formats']
    
    # Export formula
    print("\n→ Exporting formula...")
    exporter = FormulaExporter(output_dir=os.path.join(output_dir, 'models'))
    
    if 'python' in export_formats:
        python_path = os.path.join(output_dir, 'models', 'formula.py')
        exporter.export_python(best_formula, python_path)
    
    if 'json' in export_formats:
        json_path = os.path.join(output_dir, 'models', 'formula.json')
        exporter.export_json(best_formula, results['evaluation'], json_path)
    
    if 'latex' in export_formats:
        latex_path = os.path.join(output_dir, 'models', 'formula.tex')
        exporter.export_latex(best_formula, latex_path)
    
    # Export model
    model_path = os.path.join(output_dir, 'models', 'formula_model.pkl')
    exporter.export_model(best_formula, model_path)
    
    # Generate report
    if config['output']['generate_report']:
        print("\n→ Generating report...")
        report_gen = ReportGenerator(output_dir=os.path.join(output_dir, 'reports'))
        
        report_data = {
            'best_formula': best_formula.get_metadata(),
            'evaluation': results['evaluation'],
            'feature_importance': results['feature_importance'],
            'all_formulas': evaluation_summary,
            'ged_analysis': results.get('ged_analysis', {}),
            'figures': figures
        }
        
        md_path = os.path.join(output_dir, 'reports', 'analysis_report.md')
        report_gen.generate_markdown_report(report_data, md_path)
        
        json_path = os.path.join(output_dir, 'reports', 'results_summary.json')
        report_gen.generate_summary_json(report_data, json_path)
    
    print("\n✓ Results exported successfully")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Learn difficulty prediction formulas from bug metrics'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--metrics-dir',
        type=str,
        help='Override metrics directory from config'
    )
    parser.add_argument(
        '--matrix-file',
        type=str,
        help='Override matrix file from config'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--formula-types',
        type=str,
        help='Comma-separated list of formula types to try'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.metrics_dir:
        config['data']['metrics_dir'] = args.metrics_dir
    if args.matrix_file:
        config['data']['matrix_file'] = args.matrix_file
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    if args.formula_types:
        config['formula']['types'] = args.formula_types.split(',')
    
    # Validate required paths
    if not config['data']['metrics_dir']:
        print("Error: --metrics-dir is required")
        sys.exit(1)
    if not config['data']['matrix_file']:
        print("Error: --matrix-file is required")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("DIFFICULTY FORMULA LEARNER")
    print("="*80)
    print(f"\nMetrics directory: {config['data']['metrics_dir']}")
    print(f"Matrix file: {config['data']['matrix_file']}")
    print(f"Output directory: {config['output']['output_dir']}")
    
    # Load data
    X, y, metrics_loader, results_loader = load_data(config)
    
    # Preprocess
    X_processed, normalizer = preprocess_data(X, config)
    
    # Generate formulas
    formulas = generate_formulas(config)
    
    # Select best formula
    best_formula, evaluation_summary = select_best_formula(formulas, X_processed, y, config)
    
    # Analyze formula
    results, y_pred = analyze_formula(best_formula, X_processed, y, config)
    
    # Create visualizations
    figures = create_visualizations(best_formula, X_processed, y, y_pred, results, config)
    
    # Export results
    export_results(best_formula, evaluation_summary, results, figures, config)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\n✓ Best formula: {best_formula.name}")
    print(f"✓ Spearman ρ: {results['evaluation']['spearman']:.4f}")
    print(f"✓ R²: {results['evaluation']['r2']:.4f}")
    print(f"\n✓ All results saved to: {config['output']['output_dir']}")
    print("")


if __name__ == '__main__':
    main()
