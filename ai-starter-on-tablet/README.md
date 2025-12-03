# 상권 트렌드 분석 대시보드

공개 상권 데이터 API를 활용해 상권별 성장/쇠퇴 업종을 파악하고, 간단한 대시보드를 통해 시각화하는 스타터 프로젝트입니다.

## 폴더 구조
- `data/sample_commercial_data.csv` : 샘플 상권 데이터. 실제 API 응답 형식에 맞춰 컬럼을 유지하면 바로 대시보드에서 활용할 수 있습니다.
- `src/commercial_trends/data_pipeline.py` : API 호출과 수집 데이터를 로컬에 저장하기 위한 유틸리티.
- `src/commercial_trends/analysis.py` : 성장/쇠퇴 업종 분석과 시각화 함수.
- `src/commercial_trends/dashboard.py` : Gradio 기반 대시보드 진입점.

## 설치
```bash
pip install -r requirements.txt
```

## 바로 써보기 (로컬 대시보드)
```bash
cd ai-starter-on-tablet
python -m src.commercial_trends.dashboard --data data/sample_commercial_data.csv --top-n 5
```
- 다른 파일을 쓰고 싶다면 `--data`에 CSV 경로를 지정합니다.
- 서버 노출이 필요하면 `--listen 0.0.0.0 --port 7860`처럼 옵션을 추가합니다.

## 데이터 수집 예시
```python
from datetime import date
from pathlib import Path
from src.commercial_trends.data_pipeline import APIConfig, collect_and_save

config = APIConfig(
    base_url="https://api.example.com",  # 실제 공개 상권 API URL
    api_key="YOUR_KEY",
    service="store-growth",
    page_size=1000,
)
collect_and_save(
    config=config,
    region_code="서울시",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 5, 1),
    output_path=Path("data/raw/records.ndjson"),
)
```

수집 후 CSV로 변환해 대시보드에서 곧바로 활용할 수 있습니다.
```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

raw = Path("data/raw/records.ndjson")
df = pd.read_json(raw, lines=True)
# 필요한 컬럼(date, district, commercial_area, business_type, sales, transactions, customer_count)만 남깁니다.
df.to_csv("data/my_commercial_data.csv", index=False)
print("Saved -> data/my_commercial_data.csv")
PY
```

## 분석 및 대시보드 실행
1. `data/sample_commercial_data.csv`를 자신의 데이터로 교체하거나, 수집한 NDJSON을 CSV로 변환하여 동일한 스키마(`date, district, commercial_area, business_type, sales, transactions, customer_count`)를 맞춥니다.
2. 대시보드 실행:
   ```bash
   python -m src.commercial_trends.dashboard
   ```
3. 슬라이더와 드롭다운을 통해 성장/쇠퇴 업종 순위를 확인하고, 선택 업종의 시계열 추이를 확인할 수 있습니다.

## 참고
- 기본 시각화는 Matplotlib/Seaborn을 활용하며, 그리디오(Gradio) 컴포넌트를 통해 웹 UI로 노출됩니다.
- 분석 지표는 업종별 매출 시계열의 기울기 대비 평균 매출 비율을 성장 점수(growth score)로 사용합니다.
