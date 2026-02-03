# Difficulty Formula Learner - Usage Guide

## Quick Start

### 1. Installation

```bash
git clone https://github.com/kaileekiki/difficulty_formula_learner.git
cd difficulty_formula_learner
pip install -r requirements.txt
```

### 2. Try the Example

```bash
# Create example data
python example.py --create-data

# Run the learner on example data
python example.py --run
```

### 3. Use Your Own Data

```bash
python main.py \
    --metrics-dir /path/to/bug_difficulty_analyzer/outputs \
    --matrix-file /path/to/model_bug_matrix.csv \
    --output-dir ./outputs
```

## Detailed Usage

### Data Preparation

#### Input 1: Bug Metrics (v3_progress_*.json)

Place your JSON files from `bug_difficulty_analyzer` in a directory:

```
metrics_dir/
├── v3_progress_batch1.json
├── v3_progress_batch2.json
└── v3_progress_batch3.json
```

Each JSON should contain:
```json
{
  "count": N,
  "results": [
    {
      "instance_id": "repo__name-issue",
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

#### Input 2: Model Results (model_bug_matrix.csv)

CSV file with model performance:

```csv
model_name,bug_1,bug_2,bug_3,...
GPT-4,1,0,1,...
Claude-3,1,1,0,...
Gemini-Pro,0,1,1,...
```

- Rows: AI models
- Columns: Bug IDs (must match instance_id from JSON)
- Values: 1 (solved), 0 (failed)

### Configuration

Edit `config.yaml` to customize settings:

#### Data Settings

```yaml
data:
  aggregation: 'sum'  # How to aggregate metrics: 'sum', 'avg', 'max'
  normalization: 'standard'  # Feature scaling: 'standard', 'minmax', 'robust', null
```

#### Formula Selection

```yaml
formula:
  types:
    - linear        # Linear regression with regularization
    - polynomial    # Polynomial features
    - nonlinear     # Log/sqrt transformations
    - tree          # Random Forest / XGBoost
    - symbolic      # Genetic programming
```

#### Linear Formula Settings

```yaml
formula:
  linear:
    regularization: ['ridge', 'lasso', 'elasticnet']
    alpha: [0.1, 1.0, 10.0]  # Regularization strengths to try
```

#### Evaluation Settings

```yaml
evaluation:
  cv_folds: 5  # Cross-validation folds
  test_size: 0.2  # Train/test split
  random_state: 42  # For reproducibility
```

#### Selection Criteria

```yaml
selection:
  primary_metric: 'spearman'  # Metric for model selection
  complexity_penalty: 0.01  # Penalty for complex formulas
```

### Command-Line Options

```bash
python main.py \
    --config config.yaml \                    # Config file (default: config.yaml)
    --metrics-dir PATH \                      # Directory with v3_progress_*.json
    --matrix-file PATH \                      # Path to model_bug_matrix.csv
    --output-dir PATH \                       # Output directory (default: ./outputs)
    --formula-types linear,polynomial,tree    # Comma-separated formula types
```

### Output Structure

The tool generates:

```
outputs/
├── models/
│   ├── formula.py              # Python function
│   ├── formula.json            # JSON metadata
│   ├── formula.tex             # LaTeX equation
│   └── formula_model.pkl       # Full model (joblib)
│
├── reports/
│   ├── analysis_report.md      # Comprehensive report
│   └── results_summary.json    # Summary statistics
│
└── figures/
    ├── feature_importance.png
    ├── feature_correlation.png
    ├── target_correlation.png
    ├── actual_vs_predicted.png
    └── residuals.png
```

## Understanding Results

### Performance Metrics

- **Spearman's ρ** (0 to 1): Rank correlation. Higher is better.
  - ≥ 0.7: Excellent
  - 0.5-0.7: Good
  - < 0.5: Needs improvement

- **R²** (0 to 1): Variance explained. Higher is better.
  - ≥ 0.5: Good explanatory power
  - 0.3-0.5: Moderate
  - < 0.3: Poor

- **RMSE/MAE**: Prediction errors. Lower is better.
  - Compare to range of target values

### Reading the Formula

Example output:
```
success_rate = 0.850 
  - 0.015*DFG_GED 
  - 0.012*PDG_GED 
  - 0.008*LOC 
  + 0.003*Variable_Scope
```

Interpretation:
- Base success rate: 85%
- Higher DFG_GED → Lower success rate (harder bug)
- Each unit increase in DFG_GED reduces success by 1.5%

### Feature Importance

The report shows which metrics matter most:

1. **High importance**: Key difficulty indicators
2. **Low importance**: Less relevant or redundant
3. **Negative coefficients**: Higher value = harder bug
4. **Positive coefficients**: Higher value = easier bug

## Advanced Usage

### Using a Saved Model

```python
from export.formula_exporter import FormulaExporter
import pandas as pd

# Load model
exporter = FormulaExporter()
formula = exporter.load_model('outputs/models/formula_model.pkl')

# Predict for new bugs
new_bugs = pd.DataFrame({
    'LOC': [100, 200],
    'Token_Edit_Distance': [50, 100],
    # ... other features
})

predictions = formula.predict(new_bugs)
print(f"Success rates: {predictions}")
```

### Custom Analysis

```python
from analysis.feature_importance import FeatureImportanceAnalyzer
from analysis.ged_design import GEDDesigner

# Analyze feature importance
analyzer = FeatureImportanceAnalyzer()
importance = analyzer.compute_all(
    formula, X, y, 
    methods=['coefficient', 'permutation', 'shap']
)

# Optimize GED weights
ged_designer = GEDDesigner()
weights = ged_designer.learn_ged_weights(X, y)
print(f"Optimal GED weights: {weights}")
```

### Creating Custom Formulas

```python
from formula.generators.base_formula import BaseFormula
import numpy as np

class CustomFormula(BaseFormula):
    def __init__(self):
        super().__init__(name="Custom")
    
    def fit(self, X, y):
        # Your fitting logic
        self.is_fitted = True
        return self
    
    def predict(self, X):
        # Your prediction logic
        return np.zeros(len(X))
    
    def get_formula_string(self):
        return "custom_formula"
```

### Batch Processing

```python
import glob

# Process multiple datasets
for metrics_dir in glob.glob('data/metrics_*'):
    matrix_file = metrics_dir.replace('metrics', 'matrix') + '.csv'
    output_dir = f'outputs/{Path(metrics_dir).name}'
    
    os.system(f"""
        python main.py \
            --metrics-dir {metrics_dir} \
            --matrix-file {matrix_file} \
            --output-dir {output_dir}
    """)
```

## Troubleshooting

### Issue: "No v3_progress_*.json files found"

**Solution**: Check that:
1. Files are named `v3_progress_*.json`
2. Directory path is correct
3. Files contain valid JSON

### Issue: "No common bugs between metrics and results"

**Solution**: Ensure:
1. Bug IDs match between JSON (instance_id) and CSV (column names)
2. IDs use same format (e.g., `repo__name-issue`)

### Issue: Low performance (Spearman ρ < 0.5)

**Solutions**:
1. Check data quality and completeness
2. Try different formula types
3. Adjust normalization method
4. Add more training data
5. Check for data entry errors

### Issue: "gplearn not installed" or "XGBoost not available"

**Solution**: These are optional dependencies. The tool will skip them and use other formula types. To install:
```bash
pip install gplearn xgboost
```

### Issue: Very long training time

**Solutions**:
1. Reduce symbolic regression parameters in config.yaml:
   ```yaml
   symbolic:
     population_size: 1000  # Lower from 5000
     generations: 10  # Lower from 20
   ```
2. Use fewer formula types
3. Reduce cross-validation folds

## Best Practices

1. **Start Simple**: Begin with linear and polynomial formulas before trying symbolic regression

2. **Validate Results**: Always check if predicted rankings make sense given domain knowledge

3. **Use Cross-Validation**: Don't just trust training performance; check CV scores

4. **Interpret Features**: Understanding why certain features are important helps validate the model

5. **Document Assumptions**: Note any data preprocessing or filtering decisions

6. **Version Control**: Save config files with outputs for reproducibility

7. **Compare Approaches**: Try multiple formula types and compare results

## Tips for Better Results

1. **Data Quality**: Ensure metrics are computed correctly and consistently

2. **Feature Engineering**: Consider creating derived features (ratios, differences)

3. **Outlier Handling**: Check for and handle extreme values appropriately

4. **Regularization**: Use Ridge/Lasso to prevent overfitting with many features

5. **Ensemble Methods**: Tree-based models often perform well with diverse features

6. **Domain Knowledge**: Incorporate understanding of bug difficulty into feature selection

## Citation

If you use this tool in research, please cite:

```bibtex
@software{difficulty_formula_learner,
  title={Difficulty Formula Learner: Automated Learning of Bug Difficulty Prediction Formulas},
  author={kaileekiki},
  year={2024},
  url={https://github.com/kaileekiki/difficulty_formula_learner}
}
```
