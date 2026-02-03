# 샘플 데이터 디렉토리

이 디렉토리에 분석할 데이터를 업로드하세요.

## 필요한 파일

### 1. 메트릭 파일 (`metrics/` 디렉토리)
- 파일명: `v3_progress_*.json`
- 출처: `kaileekiki/bug_difficulty_analyzer/outputs/`
- 최대 100개 파일 지원

### 2. 모델-버그 매트릭스 (`model_results/` 디렉토리)
- 파일명: `model_bug_matrix.csv`
- 출처: `kaileekiki/benchmark-difficulty-analyzer/data/processed/`

## 사용 방법

```bash
# 1. 파일 복사
cp /path/to/v3_progress_*.json data/sample/metrics/
cp /path/to/model_bug_matrix.csv data/sample/model_results/

# 2. 실행
python main.py --upload-mode
```

## 데이터 형식

### v3_progress_*.json
```json
{
  "count": 5,
  "results": [
    {
      "instance_id": "repo__issue-123",
      "metrics": {
        "changed_files_metrics": {
          "summary": {
            "basic": { ... },
            "ast": { ... },
            "graph": { ... }
          }
        }
      }
    }
  ]
}
```

### model_bug_matrix.csv
```csv
model_name,bug_id_1,bug_id_2,...
Model_A,1,0,...
Model_B,0,1,...
```
