# Enhanced Features Guide

이 문서는 새로 추가된 기능들의 사용 방법을 설명합니다.

## 1. 수동 데이터 업로드 (Manual Data Upload)

### 개요
`data/sample/` 디렉토리를 통해 수동으로 데이터를 업로드하여 분석할 수 있습니다.

### 사용 방법

#### 1.1 데이터 준비
```bash
# 메트릭 파일 복사 (v3_progress_*.json 형식)
cp /path/to/your/v3_progress_*.json data/sample/metrics/

# 모델-버그 매트릭스 복사
cp /path/to/your/model_bug_matrix.csv data/sample/model_results/
```

#### 1.2 실행
```bash
# --upload-mode 플래그 사용
python main.py --upload-mode
```

#### 1.3 프로그래밍 방식 사용
```python
from data.loaders.file_uploader import FileUploader

# 초기화
uploader = FileUploader(data_dir="data/sample")

# 메트릭 파일 업로드
uploader.upload_metrics_file("path/to/v3_progress_1.json")
uploader.upload_metrics_file("path/to/v3_progress_2.json")

# 매트릭스 파일 업로드
uploader.upload_matrix_file("path/to/model_bug_matrix.csv")

# 검증
validation = uploader.validate_uploaded_data()
print(f"Valid: {validation['is_valid']}")
print(f"Total bugs: {validation['total_bugs_in_metrics']}")
```

## 2. 수식 카탈로그 (Formula Catalog)

### 개요
11가지 수식 유형에 대한 상세한 설명과 추천 기능을 제공합니다.

### 사용 방법

#### 2.1 명령줄에서 카탈로그 보기
```bash
# 전체 카탈로그 출력 및 FORMULA_CATALOG.md 생성
python main.py --explain-formulas
```

#### 2.2 프로그래밍 방식 사용
```python
from formula.formula_catalog import FormulaCatalog

# 모든 수식 가져오기
all_formulas = FormulaCatalog.get_all_formulas()
print(f"Total formulas: {len(all_formulas)}")

# 카테고리별로 가져오기
linear_formulas = FormulaCatalog.get_formulas_by_category("linear")
for f in linear_formulas:
    print(f"- {f.name}: {f.description}")

# 특정 수식 정보
ridge = FormulaCatalog.get_formula_by_type("linear_ridge")
print(f"Name: {ridge.name}")
print(f"Advantages: {ridge.advantages}")
print(f"Best for: {ridge.best_for}")

# 데이터 특성에 따른 추천
recommended = FormulaCatalog.get_recommended_formulas(
    data_size=100,
    need_interpretability=True,
    has_multicollinearity=True
)
print("\nRecommended formulas:")
for f in recommended[:3]:
    print(f"- {f.name}")

# 마크다운 보고서 생성
report = FormulaCatalog.generate_catalog_report()
with open("my_catalog.md", "w") as f:
    f.write(report)
```

### 수식 카테고리

1. **선형 (Linear)**: 4가지
   - Linear Regression (No Regularization)
   - Ridge Regression (L2)
   - Lasso Regression (L1)
   - ElasticNet (L1 + L2)

2. **다항식 (Polynomial)**: 2가지
   - Polynomial Degree 2
   - Polynomial Degree 3

3. **비선형 변환 (Nonlinear)**: 2가지
   - Log-Transformed Linear
   - Sqrt-Transformed Linear

4. **트리 기반 (Tree-based)**: 2가지
   - Random Forest
   - XGBoost

5. **기호적 회귀 (Symbolic)**: 1가지
   - Symbolic Regression (gplearn)

## 3. LLM 기반 분석 (API Integration)

### 개요
OpenAI 또는 Anthropic API를 사용하여 수식 해석 및 개선 제안을 받을 수 있습니다.

### 사용 방법

#### 3.1 API 키 설정
```bash
# OpenAI 사용
export OPENAI_API_KEY="your-api-key-here"

# 또는 Anthropic 사용
export ANTHROPIC_API_KEY="your-api-key-here"
```

#### 3.2 실행
```bash
# --use-api 플래그 사용
python main.py --use-api --metrics-dir /path/to/metrics --matrix-file /path/to/matrix.csv
```

#### 3.3 Config 설정
`config.yaml`에서 API 설정:
```yaml
api:
  provider: "openai"  # 또는 "anthropic"
  use_llm_analysis: true
```

#### 3.4 프로그래밍 방식 사용
```python
from analysis.llm_analyzer import LLMAnalyzer

# 초기화
analyzer = LLMAnalyzer(provider="openai")

if analyzer.is_available:
    # 수식 해석
    interpretation = analyzer.interpret_formula(
        formula_string="y = 0.5 * DFG_GED + 0.3 * LOC",
        feature_importance={"DFG_GED": 0.8, "LOC": 0.5}
    )
    print(interpretation)
    
    # 개선 제안
    recommendations = analyzer.recommend_formula_improvements(
        current_formula="y = 0.5 * x",
        metrics={"spearman": 0.7, "r2": 0.5},
        correlation_data={"DFG_GED": 0.8}
    )
    print(recommendations)
    
    # 수식 선택 설명
    explanation = analyzer.explain_formula_selection(
        formula_candidates=[...],
        best_formula={...}
    )
    print(explanation)
else:
    print("API key not set, using fallback analysis")
```

### Fallback 기능
API 키가 설정되지 않았거나 API 호출이 실패하면 자동으로 기본 분석으로 대체됩니다.

## 4. 통합 사용 예제

### 전체 워크플로우
```bash
# 1. 데이터 업로드 및 수식 카탈로그 확인
python main.py --explain-formulas
python main.py --upload-mode

# 2. LLM 분석과 함께 실행
python main.py --upload-mode --use-api

# 3. 특정 수식 타입만 테스트
python main.py --upload-mode --use-api --formula-types linear,tree
```

### Python 스크립트 예제
```python
from data.loaders.file_uploader import FileUploader
from formula.formula_catalog import FormulaCatalog
from analysis.llm_analyzer import LLMAnalyzer

# 1. 데이터 업로드
uploader = FileUploader()
uploader.upload_metrics_file("v3_progress_1.json")
uploader.upload_matrix_file("model_bug_matrix.csv")

validation = uploader.validate_uploaded_data()
if validation['is_valid']:
    print("✓ Data uploaded successfully")
    
    # 2. 수식 추천
    recommendations = FormulaCatalog.get_recommended_formulas(
        data_size=validation['total_bugs_in_metrics'],
        need_interpretability=True,
        has_multicollinearity=True
    )
    
    print("\nRecommended formulas:")
    for f in recommendations[:3]:
        print(f"- {f.name}: {f.best_for}")
    
    # 3. LLM 분석 준비
    analyzer = LLMAnalyzer()
    if analyzer.is_available:
        print("\n✓ LLM analysis available")
    else:
        print("\n⚠️  Set API key for enhanced analysis")
```

## 5. 주의사항

### 데이터 업로드
- 메트릭 파일은 반드시 `v3_progress_*.json` 형식이어야 합니다
- 매트릭스 파일은 `model_bug_matrix.csv` 이름으로 저장됩니다
- 최대 100개의 메트릭 파일을 지원합니다

### API 사용
- API 키는 환경변수로 설정하는 것을 권장합니다
- API 호출에는 비용이 발생할 수 있습니다
- API 키 없이도 fallback 분석이 제공됩니다

### 수식 선택
- 데이터 크기에 따라 적절한 복잡도의 수식을 선택하세요
- 해석 가능성이 중요한 경우 선형 수식을 고려하세요
- 다중공선성이 있는 경우 Ridge나 ElasticNet을 사용하세요

## 6. 문제 해결

### "Data validation failed"
- `data/sample/metrics/`에 v3_progress_*.json 파일이 있는지 확인
- `data/sample/model_results/model_bug_matrix.csv`가 있는지 확인

### "API key not set"
- 환경변수 `OPENAI_API_KEY` 또는 `ANTHROPIC_API_KEY`를 설정했는지 확인
- `openai` 또는 `anthropic` 패키지가 설치되었는지 확인

### Import 오류
- `pip install -r requirements.txt`로 모든 의존성을 설치하세요

## 7. 참고 자료

- [README.md](README.md): 기본 사용법
- [USAGE_GUIDE.md](USAGE_GUIDE.md): 상세 가이드
- [data/sample/README.md](data/sample/README.md): 데이터 형식 설명
- [config.yaml](config.yaml): 설정 파일 예제
