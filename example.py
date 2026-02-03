#!/usr/bin/env python3
"""
Example script demonstrating how to use the Difficulty Formula Learner
This creates synthetic data for demonstration purposes
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def create_example_data():
    """Create example data files for demonstration."""
    print("Creating example data...")
    
    # Create example directories
    os.makedirs('example_data/metrics', exist_ok=True)
    os.makedirs('example_data', exist_ok=True)
    
    # Generate synthetic bug metrics
    np.random.seed(42)
    n_bugs = 50
    
    bugs_data = []
    for i in range(n_bugs):
        bug_id = f"example_bug_{i+1:03d}"
        
        # Generate random metrics with some correlation to difficulty
        metrics = {
            "instance_id": bug_id,
            "metrics": {
                "changed_files_metrics": {
                    "summary": {
                        "basic": {
                            "LOC": {"sum": int(np.random.exponential(50)), "avg": 0, "max": 0},
                            "Token_Edit_Distance": {"sum": int(np.random.exponential(30)), "avg": 0, "max": 0},
                            "Cyclomatic_Complexity": {"sum": int(np.random.exponential(10)), "avg": 0, "max": 0},
                            "Halstead_Difficulty": {"sum": float(np.random.exponential(20)), "avg": 0, "max": 0},
                            "Variable_Scope": {"sum": int(np.random.exponential(5)), "avg": 0, "max": 0}
                        },
                        "ast": {
                            "AST_GED": {"sum": int(np.random.exponential(15)), "avg": 0, "max": 0},
                            "Exception_Handling": {"sum": int(np.random.poisson(2)), "avg": 0, "max": 0},
                            "Type_Changes": {"sum": int(np.random.poisson(1)), "avg": 0, "max": 0}
                        },
                        "graph": {
                            "CFG_GED": {"sum": float(np.random.exponential(10)), "avg": 0, "max": 0},
                            "DFG_GED": {"sum": float(np.random.exponential(12)), "avg": 0, "max": 0},
                            "Call_Graph_GED": {"sum": float(np.random.exponential(5)), "avg": 0, "max": 0},
                            "PDG_GED": {"sum": float(np.random.exponential(15)), "avg": 0, "max": 0},
                            "CPG_GED": {"sum": float(np.random.exponential(18)), "avg": 0, "max": 0}
                        }
                    }
                }
            }
        }
        bugs_data.append(metrics)
    
    # Save metrics to JSON
    metrics_file = 'example_data/metrics/v3_progress_example.json'
    with open(metrics_file, 'w') as f:
        json.dump({"count": n_bugs, "results": bugs_data}, f, indent=2)
    
    print(f"✓ Created {metrics_file} with {n_bugs} bugs")
    
    # Generate synthetic model results
    # Create correlation between metrics and success rate
    bug_ids = [f"example_bug_{i+1:03d}" for i in range(n_bugs)]
    n_models = 10
    model_names = [f"Model_{chr(65+i)}" for i in range(n_models)]
    
    # Calculate synthetic difficulty based on metrics
    difficulties = []
    for bug in bugs_data:
        metrics = bug["metrics"]["changed_files_metrics"]["summary"]
        # Higher metrics = harder = lower success rate
        difficulty_score = (
            metrics["basic"]["LOC"]["sum"] * 0.01 +
            metrics["basic"]["Token_Edit_Distance"]["sum"] * 0.02 +
            metrics["basic"]["Cyclomatic_Complexity"]["sum"] * 0.03 +
            metrics["graph"]["DFG_GED"]["sum"] * 0.02 +
            metrics["graph"]["PDG_GED"]["sum"] * 0.02
        )
        difficulties.append(difficulty_score)
    
    # Normalize to success rates
    difficulties = np.array(difficulties)
    max_diff = difficulties.max()
    success_rates = 0.9 - (difficulties / max_diff) * 0.7  # Range from 0.2 to 0.9
    
    # Generate model results based on success rates
    matrix_data = {'model_name': model_names}
    for i, bug_id in enumerate(bug_ids):
        # Each model has a probability of solving based on success rate
        prob = success_rates[i]
        # Add some randomness per model
        results = [1 if np.random.random() < prob else 0 for _ in range(n_models)]
        matrix_data[bug_id] = results
    
    # Save to CSV
    matrix_file = 'example_data/model_bug_matrix.csv'
    df = pd.DataFrame(matrix_data)
    df.to_csv(matrix_file, index=False)
    
    print(f"✓ Created {matrix_file} with {n_models} models and {n_bugs} bugs")
    print(f"\nExample data created successfully!")
    print(f"\nTo run the learner on example data:")
    print(f"  python main.py \\")
    print(f"    --metrics-dir example_data/metrics \\")
    print(f"    --matrix-file example_data/model_bug_matrix.csv \\")
    print(f"    --output-dir example_outputs")


def run_example():
    """Run the learner on example data."""
    from data.loaders.metrics_loader import MetricsLoader
    from data.loaders.model_results_loader import ModelResultsLoader
    from formula.generators.linear_formula import LinearFormula
    from formula.evaluators.correlation_metrics import CorrelationMetrics
    from formula.evaluators.regression_metrics import RegressionMetrics
    
    print("\n" + "="*80)
    print("RUNNING SIMPLE EXAMPLE")
    print("="*80)
    
    # Load data
    print("\nLoading example data...")
    metrics_loader = MetricsLoader('example_data/metrics', aggregation='sum')
    X = metrics_loader.load()
    print(f"✓ Loaded {len(X)} bugs with {len(X.columns)} features")
    
    results_loader = ModelResultsLoader('example_data/model_bug_matrix.csv')
    y = results_loader.load()
    print(f"✓ Loaded success rates for {len(y)} bugs")
    
    # Align data
    common_bugs = X.index.intersection(y.index)
    X = X.loc[common_bugs]
    y = y.loc[common_bugs]
    print(f"✓ {len(common_bugs)} bugs have both metrics and results")
    
    # Train simple linear model
    print("\nTraining simple linear regression model...")
    formula = LinearFormula(regularization='ridge', alpha=1.0)
    formula.fit(X, y)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = formula.predict(X)
    
    corr_metrics = CorrelationMetrics.compute_all(y.values, y_pred)
    reg_metrics = RegressionMetrics.compute_all(y.values, y_pred, n_features=len(X.columns))
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nSpearman's ρ: {corr_metrics['spearman']:.4f}")
    print(f"Kendall's τ: {corr_metrics['kendall']:.4f}")
    print(f"Pearson's r: {corr_metrics['pearson']:.4f}")
    print(f"R²: {reg_metrics['r2']:.4f}")
    print(f"RMSE: {reg_metrics['rmse']:.4f}")
    print(f"MAE: {reg_metrics['mae']:.4f}")
    
    print("\n" + "="*80)
    print("FORMULA")
    print("="*80)
    print(f"\n{formula.get_formula_string()}")
    
    print("\n" + "="*80)
    print("TOP FEATURES")
    print("="*80)
    importance = formula.get_feature_importance()
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, value) in enumerate(sorted_features[:10], 1):
        print(f"{i:2d}. {feature:30s} {value:.4f}")
    
    print("\n✓ Example completed successfully!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Example usage of Difficulty Formula Learner')
    parser.add_argument('--create-data', action='store_true', help='Create example data files')
    parser.add_argument('--run', action='store_true', help='Run example analysis')
    
    args = parser.parse_args()
    
    if args.create_data or (not args.create_data and not args.run):
        create_example_data()
    
    if args.run:
        if not os.path.exists('example_data/model_bug_matrix.csv'):
            print("\nExample data not found. Creating it first...")
            create_example_data()
        run_example()
