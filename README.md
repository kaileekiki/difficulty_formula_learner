# Difficulty Formula Learner

Automatically learn and evaluate difficulty prediction formulas from bug metrics and model success data.

## Overview

The Difficulty Formula Learner analyzes bug complexity metrics from the SWE-bench dataset and learns mathematical formulas to predict bug difficulty (model success rates). It supports multiple formula types including linear regression, polynomial models, tree-based methods, and symbolic regression.

## Features

- **13 Code Complexity Metrics**: Analyzes basic, AST, and graph-based metrics
  - **Tier 1 - Basic (5)**: LOC, Token_Edit_Distance, Cyclomatic_Complexity, Halstead_Difficulty, Variable_Scope
  - **Tier 2 - AST (3)**: AST_GED, Exception_Handling, Type_Changes
  - **Tier 3 - Graph (5)**: CFG_GED, DFG_GED, Call_Graph_GED, PDG_GED, CPG_GED

- **Multiple Formula Types**:
  - Linear regression with regularization (Ridge, Lasso, ElasticNet)
  - Polynomial regression with interaction terms
  - Nonlinear transformations (log, sqrt, square)
  - Tree-based models (Random Forest, XGBoost)
  - Symbolic regression (genetic programming)

- **Comprehensive Evaluation**:
  - Correlation metrics (Spearman's ρ, Kendall's τ, Pearson's r)
  - Regression metrics (R², RMSE, MAE, Adjusted R²)
  - Classification metrics (accuracy, F1-score)
  - Cross-validation for model selection

- **Advanced Analysis**:
  - Feature importance (permutation, SHAP, coefficient-based)
  - GED metric weight optimization
  - Comparative analysis of different approaches

- **Rich Visualizations**:
  - Feature importance plots
  - Correlation heatmaps
  - Actual vs predicted scatter plots
  - Residual distributions
  - Error analysis by bug

- **Multiple Export Formats**:
  - Python functions
  - JSON (metadata + coefficients)
  - LaTeX equations
  - Pickled models (joblib)
  - Markdown/HTML reports

## Installation

```bash
# Clone the repository
git clone https://github.com/kaileekiki/difficulty_formula_learner.git
cd difficulty_formula_learner

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- scipy >= 1.11.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- shap >= 0.42.0
- gplearn >= 0.4.2
- pyyaml >= 6.0

## Usage

### Basic Usage

```bash
python main.py \
    --metrics-dir /path/to/bug_difficulty_analyzer/outputs \
    --matrix-file /path/to/model_bug_matrix.csv \
    --output-dir ./outputs
```

### Advanced Usage

```bash
# Use specific formula types
python main.py \
    --metrics-dir /path/to/metrics \
    --matrix-file /path/to/matrix.csv \
    --formula-types linear,polynomial,symbolic

# Use custom configuration
python main.py \
    --config custom_config.yaml \
    --metrics-dir /path/to/metrics \
    --matrix-file /path/to/matrix.csv
```

### Configuration

Edit `config.yaml` to customize:

- Data loading and preprocessing
- Formula types and hyperparameters
- Evaluation metrics and cross-validation settings
- Feature importance methods
- Visualization settings
- Export formats

Example configuration:

```yaml
data:
  aggregation: 'sum'  # 'sum', 'avg', 'max', or 'auto'
  normalization: 'standard'  # 'standard', 'minmax', 'robust', or null

formula:
  types:
    - linear
    - polynomial
    - tree
    - symbolic

evaluation:
  cv_folds: 5
  test_size: 0.2
  
selection:
  primary_metric: 'spearman'
  complexity_penalty: 0.01
```

## Input Data Format

### 1. Bug Metrics (`v3_progress_*.json`)

JSON files from `kaileekiki/bug_difficulty_analyzer` repository:

```json
{
  "count": 5,
  "results": [
    {
      "instance_id": "astropy__astropy-12907",
      "metrics": {
        "changed_files_metrics": {
          "summary": {
            "basic": {...},
            "ast": {...},
            "graph": {...}
          }
        }
      }
    }
  ]
}
```

### 2. Model Results (`model_bug_matrix.csv`)

CSV file from `kaileekiki/benchmark-difficulty-analyzer` repository:

```csv
model_name,bug_id_1,bug_id_2,bug_id_3,...
Model_A,1,0,1,...
Model_B,0,1,1,...
```

Where:
- Rows: Models
- Columns: Bug IDs
- Values: 1 (solved), 0 (failed)

## Output

The tool generates the following outputs in the specified output directory:

```
outputs/
├── models/
│   ├── formula.py           # Python function
│   ├── formula.json          # JSON metadata
│   ├── formula.tex           # LaTeX equation
│   └── formula_model.pkl     # Pickled model
├── reports/
│   ├── analysis_report.md    # Markdown report
│   └── results_summary.json  # JSON summary
└── figures/
    ├── feature_importance.png
    ├── feature_importance_comparison.png
    ├── feature_correlation.png
    ├── target_correlation.png
    ├── actual_vs_predicted.png
    └── residuals.png
```

## Performance Goals

- **Spearman's ρ ≥ 0.7**: High correlation between predicted and actual difficulty rankings
- **R² ≥ 0.5**: Model explains at least 50% of variance in success rates
- **Interpretability**: Generated formulas should be understandable and explainable

## Project Structure

```
difficulty_formula_learner/
├── data/                      # Data loading and preprocessing
│   ├── loaders/              # JSON and CSV loaders
│   └── preprocessors/        # Normalization and scaling
├── formula/                   # Formula generation and evaluation
│   ├── generators/           # Different formula types
│   ├── evaluators/           # Metrics computation
│   └── selector.py           # Formula selection
├── analysis/                  # Advanced analysis
│   ├── feature_importance.py
│   ├── ged_design.py
│   └── comparative_analysis.py
├── visualization/             # Plotting and visualization
├── export/                    # Export and reporting
├── tests/                     # Unit tests
├── config.yaml               # Configuration file
├── main.py                   # Main entry point
└── requirements.txt          # Dependencies
```

## Examples

### Loading and Using a Saved Model

```python
from export.formula_exporter import FormulaExporter
import pandas as pd

# Load saved model
exporter = FormulaExporter()
formula = exporter.load_model('outputs/models/formula_model.pkl')

# Predict difficulty for new bugs
features = pd.DataFrame({
    'LOC': [100],
    'Token_Edit_Distance': [50],
    # ... other features
})

success_rate = formula.predict(features)
print(f"Predicted success rate: {success_rate[0]:.2%}")
```

### Feature Importance Analysis

```python
from analysis.feature_importance import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()
importance = analyzer.compute_all(formula, X, y, methods=['coefficient', 'permutation'])

# Get top features
df = analyzer.get_importance_dataframe()
print(df.head(10))
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- [kaileekiki/bug_difficulty_analyzer](https://github.com/kaileekiki/bug_difficulty_analyzer): Computes the 13 code complexity metrics
- [kaileekiki/benchmark-difficulty-analyzer](https://github.com/kaileekiki/benchmark-difficulty-analyzer): Analyzes model performance on benchmarks

## License

MIT License

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{difficulty_formula_learner,
  title={Difficulty Formula Learner: Automated Learning of Bug Difficulty Prediction Formulas},
  author={kaileekiki},
  year={2024},
  url={https://github.com/kaileekiki/difficulty_formula_learner}
}
```

## Contact

For questions or issues, please open an issue on GitHub.