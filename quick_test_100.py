"""
100개 버그로 빠른 테스트 실행
"""
import sys
sys.path.insert(0, '.')

from data.loaders.metrics_loader import MetricsLoader
from data.loaders.model_results_loader import ModelResultsLoader
import pandas as pd

# 1. 데이터 로드
metrics_loader = MetricsLoader(
    metrics_dir='data/sample/metrics',
    aggregation='sum'
)
metrics_df = metrics_loader.load()

results_loader = ModelResultsLoader('data/sample/model_results/model_bug_matrix.csv')
success_rates = results_loader.load()

# 2. 100개만 선택
common_bugs = metrics_df.index.intersection(success_rates.index)
selected_bugs = list(common_bugs)[:100]  # 앞에서 100개만

print(f"전체 버그: {len(common_bugs)}")
print(f"선택된 버그: {len(selected_bugs)}")

# 3. 필터링
X = metrics_df.loc[selected_bugs]
y = success_rates.loc[selected_bugs]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\n샘플 버그 IDs: {selected_bugs[:5]}")
