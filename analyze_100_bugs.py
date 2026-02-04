"""
전체 분석 파이프라인을 100개 버그로 실행하는 스크립트
- 데이터 로드 및 100개 선택
- Feature-Target 상관관계 분석 (Spearman)
- 여러 모델 테스트 (Ridge, Lasso, ElasticNet)
- 5-Fold Cross Validation
- Feature 중요도 출력
- 최종 결과 (Spearman ρ, R², RMSE)
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error

from data.loaders.metrics_loader import MetricsLoader
from data.loaders.model_results_loader import ModelResultsLoader
from data.preprocessors.feature_normalizer import FeatureNormalizer

print("="*80)
print("100개 버그 분석")
print("="*80)

# 1. 데이터 로드
print("\n→ 데이터 로딩...")
metrics_loader = MetricsLoader(
    metrics_dir='data/sample/metrics',
    aggregation='sum'
)
metrics_df = metrics_loader.load()

results_loader = ModelResultsLoader('data/sample/model_results/model_bug_matrix.csv')
success_rates = results_loader.load()

# 2. 100개만 선택
common_bugs = metrics_df.index.intersection(success_rates.index)
selected_bugs = sorted(list(common_bugs))[:100]  # 앞에서 100개만

X = metrics_df.loc[selected_bugs]
y = success_rates.loc[selected_bugs]

print(f"✓ 전체 버그: {len(common_bugs)}")
print(f"✓ 선택된 버그: {len(selected_bugs)}")
print(f"✓ Features: {X.shape[1]}")

# 3. 정규화
print("\n→ Feature 정규화...")
normalizer = FeatureNormalizer(method='standard')
X_normalized = normalizer.fit_transform(X)
print("✓ Standard normalization 완료")

# 4. Feature-Target 상관관계 분석
print("\n→ Feature-Target 상관관계 (Spearman)...")
correlations = {}
for feature in X.columns:
    corr, p_value = spearmanr(X[feature], y)
    correlations[feature] = {'correlation': corr, 'p_value': p_value}

# 상관관계 높은 순으로 정렬
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
print("\nTop 5 Features:")
for i, (feature, stats) in enumerate(sorted_corr[:5], 1):
    print(f"  {i}. {feature}: ρ={stats['correlation']:.4f} (p={stats['p_value']:.4e})")

# 5. 모델 학습 및 평가
print("\n→ 모델 학습 (5-Fold CV)...")
models = {
    'Ridge (α=1.0)': Ridge(alpha=1.0, random_state=42),
    'Ridge (α=0.1)': Ridge(alpha=0.1, random_state=42),
    'Lasso (α=0.01)': Lasso(alpha=0.01, random_state=42),
    'ElasticNet (α=0.01)': ElasticNet(alpha=0.01, random_state=42)
}

results = {}
for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_normalized, y, cv=5, scoring='r2')
    
    # 전체 데이터로 학습
    model.fit(X_normalized, y)
    y_pred = model.predict(X_normalized)
    
    # 평가 지표
    spearman_corr, _ = spearmanr(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    results[name] = {
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'spearman': spearman_corr,
        'r2': r2,
        'rmse': rmse
    }
    
    print(f"\n{name}:")
    print(f"  CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Spearman ρ = {spearman_corr:.4f}")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")

# 6. 최적 모델 선택 및 Feature 중요도
print("\n→ 최적 모델 및 Feature 중요도...")
best_model_name = max(results.items(), key=lambda x: x[1]['spearman'])[0]
best_model = models[best_model_name]

print(f"\n최적 모델: {best_model_name}")

# Feature 중요도 (계수 기반)
if hasattr(best_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': best_model.coef_
    })
    feature_importance['abs_coefficient'] = feature_importance['coefficient'].abs()
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 5 중요 Features (계수 기반):")
    for i, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")

# 7. 최종 요약
print("\n" + "="*80)
print("최종 결과")
print("="*80)
print(f"✓ 분석 버그 수: {len(selected_bugs)}")
print(f"✓ 최적 모델: {best_model_name}")
print(f"✓ Spearman ρ: {results[best_model_name]['spearman']:.4f}")
print(f"✓ R²: {results[best_model_name]['r2']:.4f}")
print(f"✓ RMSE: {results[best_model_name]['rmse']:.4f}")
print("="*80)
