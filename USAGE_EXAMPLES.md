# Usage Examples

Quick reference guide for the enhanced features.

## 1. Quick Start - View Formula Catalog

```bash
# Display all formula types with explanations
python main.py --explain-formulas

# Output saved to FORMULA_CATALOG.md
```

## 2. Manual Data Upload Mode

### Step 1: Prepare Your Data
```bash
# Copy your metrics files
cp /path/to/your/v3_progress_*.json data/sample/metrics/

# Copy your matrix file
cp /path/to/your/model_bug_matrix.csv data/sample/model_results/
```

### Step 2: Run Analysis
```bash
# Run with uploaded data
python main.py --upload-mode
```

### Step 3: Check Results
```
outputs/
├── figures/          # Visualizations
├── models/           # Exported models
└── reports/          # Analysis reports
```

## 3. Using LLM API for Enhanced Analysis

### With OpenAI
```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Run with API analysis
python main.py --upload-mode --use-api
```

### With Anthropic Claude
```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Configure provider in config.yaml
# api:
#   provider: "anthropic"

# Run analysis
python main.py --upload-mode --use-api
```

### Using Different Models
Edit `config.yaml`:
```yaml
api:
  provider: "openai"
  model: "gpt-3.5-turbo"  # or "gpt-4-turbo"
  use_llm_analysis: true
```

## 4. Combined Usage

```bash
# 1. First, explore formula options
python main.py --explain-formulas

# 2. Then run full analysis with specific formula types
python main.py \
  --upload-mode \
  --use-api \
  --formula-types linear,tree \
  --output-dir ./my_results
```

## 5. Python API Usage

### FileUploader
```python
from data.loaders.file_uploader import FileUploader

# Initialize
uploader = FileUploader(data_dir="data/sample")

# Upload files
uploader.upload_metrics_file("path/to/v3_progress_1.json")
uploader.upload_matrix_file("path/to/model_bug_matrix.csv")

# Validate
validation = uploader.validate_uploaded_data()
if validation['is_valid']:
    print(f"Ready to analyze {validation['total_bugs_in_metrics']} bugs")
```

### FormulaCatalog
```python
from formula.formula_catalog import FormulaCatalog

# Get all formulas
all_formulas = FormulaCatalog.get_all_formulas()

# Get by category
linear_formulas = FormulaCatalog.get_formulas_by_category("linear")

# Get specific formula
ridge = FormulaCatalog.get_formula_by_type("linear_ridge")
print(f"Description: {ridge.description}")
print(f"Best for: {ridge.best_for}")

# Get recommendations
recommended = FormulaCatalog.get_recommended_formulas(
    data_size=100,
    need_interpretability=True,
    has_multicollinearity=True
)
for f in recommended[:3]:
    print(f"- {f.name}")
```

### LLMAnalyzer
```python
from analysis.llm_analyzer import LLMAnalyzer

# Initialize
analyzer = LLMAnalyzer(provider="openai", model="gpt-4")

if analyzer.is_available:
    # Interpret a formula
    interpretation = analyzer.interpret_formula(
        formula_string="y = 0.5 * DFG_GED + 0.3 * LOC",
        feature_importance={"DFG_GED": 0.8, "LOC": 0.5}
    )
    print(interpretation)
    
    # Get improvement recommendations
    recommendations = analyzer.recommend_formula_improvements(
        current_formula="y = 0.5 * x",
        metrics={"spearman": 0.7, "r2": 0.5},
        correlation_data={"DFG_GED": 0.8}
    )
    print(recommendations)
else:
    print("API key not set - using fallback analysis")
```

## 6. Configuration Options

### config.yaml
```yaml
# Data settings
data:
  metrics_dir: null  # Set via --upload-mode or --metrics-dir
  matrix_file: null  # Set via --upload-mode or --matrix-file
  aggregation: 'auto'
  normalization: 'standard'

# Formula types to evaluate
formula:
  types:
    - linear
    - polynomial
    - tree
    - symbolic

# API configuration
api:
  provider: "openai"  # or "anthropic"
  use_llm_analysis: false  # Set to true or use --use-api flag
  model: "gpt-4"  # Optional: customize model

# Output settings
output:
  output_dir: './outputs'
  export_formats:
    - python
    - json
    - latex
  generate_report: true
```

## 7. Common Workflows

### Workflow 1: Quick Exploration
```bash
# Just see what formula types are available
python main.py --explain-formulas
```

### Workflow 2: Basic Analysis (No API)
```bash
# Prepare data
cp my_data/v3_progress_*.json data/sample/metrics/
cp my_data/model_bug_matrix.csv data/sample/model_results/

# Run analysis
python main.py --upload-mode
```

### Workflow 3: Full Analysis with LLM
```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Run complete analysis
python main.py --upload-mode --use-api

# Results include:
# - Best formula selection
# - LLM interpretation
# - Improvement recommendations
# - Detailed reports
```

### Workflow 4: Custom Analysis
```bash
# Test only specific formula types
python main.py \
  --upload-mode \
  --use-api \
  --formula-types linear,polynomial \
  --output-dir ./linear_analysis

# Then try different types
python main.py \
  --upload-mode \
  --use-api \
  --formula-types tree,symbolic \
  --output-dir ./ml_analysis
```

## 8. Troubleshooting

### "Data validation failed"
```bash
# Check files exist
ls -la data/sample/metrics/
ls -la data/sample/model_results/

# Verify file names
# Metrics: v3_progress_*.json
# Matrix: model_bug_matrix.csv
```

### "API key not set"
```bash
# Check environment variable
echo $OPENAI_API_KEY

# Set if needed
export OPENAI_API_KEY="sk-..."
```

### Import errors
```bash
# Install all dependencies
pip install -r requirements.txt
```

## 9. Output Files

After running analysis, you'll find:

```
outputs/
├── figures/
│   ├── feature_importance.png
│   ├── feature_correlation.png
│   ├── actual_vs_predicted.png
│   └── residuals.png
├── models/
│   ├── formula.py          # Python code
│   ├── formula.json        # JSON format
│   ├── formula.tex         # LaTeX format
│   └── formula_model.pkl   # Pickled model
└── reports/
    ├── analysis_report.md  # Markdown report
    └── results_summary.json # JSON summary
```

If `--use-api` was enabled, reports include:
- Formula interpretation in natural language
- Improvement recommendations
- Selection explanation

## 10. Best Practices

1. **Start Simple**: Use `--explain-formulas` first to understand options
2. **API Costs**: Keep `use_llm_analysis: false` by default
3. **Data Quality**: Validate your data before running analysis
4. **Formula Selection**: Start with linear formulas for interpretability
5. **Iterative Analysis**: Try different formula types and compare results

For more details, see [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md)
