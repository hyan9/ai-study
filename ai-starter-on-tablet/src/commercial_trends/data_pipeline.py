"""
Data collection utilities for commercial area trend analysis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests


@dataclass
class APIConfig:
    """Configuration for the open commercial data API."""

    base_url: str
    api_key: str
    service: str
    page_size: int = 1000


def build_query_params(
    region_code: str,
    start_date: date,
    end_date: date,
    page: int = 1,
    page_size: int = 1000,
) -> Dict[str, str]:
    """Build standard query params for the API."""

    return {
        "divId": region_code,
        "startDate": start_date.strftime("%Y%m%d"),
        "endDate": end_date.strftime("%Y%m%d"),
        "page": str(page),
        "perPage": str(page_size),
    }


def fetch_page(config: APIConfig, params: Dict[str, str]) -> Dict:
    """Fetch a single page from the API.

    The function keeps responsibilities narrow so it can be mocked easily in tests.
    """

    url = f"{config.base_url}/{config.service}/{config.api_key}/json"
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def paginate_results(
    config: APIConfig,
    region_code: str,
    start_date: date,
    end_date: date,
    max_pages: Optional[int] = None,
) -> Iterable[Dict]:
    """Yield rows from all pages in the requested date range.

    The API the code targets typically returns a "rows" list and a "totalCount".
    Pagination stops when no more rows are returned or the optional `max_pages`
    limit is reached.
    """

    page = 1
    while True:
        params = build_query_params(region_code, start_date, end_date, page=page, page_size=config.page_size)
        payload = fetch_page(config, params)
        rows: List[Dict] = payload.get("rows", [])
        if not rows:
            break

        for row in rows:
            yield row

        if max_pages is not None and page >= max_pages:
            break
        page += 1


def collect_and_save(
    config: APIConfig,
    region_code: str,
    start_date: date,
    end_date: date,
    output_path: Path,
    max_pages: Optional[int] = None,
) -> Path:
    """Collect rows from the API and store them as a newline-delimited JSON file.

    Using a line-delimited format allows easy incremental processing and appends
    without loading the entire file in memory.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in paginate_results(config, region_code, start_date, end_date, max_pages=max_pages):
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return output_path


def load_records(raw_path: Path) -> pd.DataFrame:
    """Load newline-delimited JSON or CSV into a tidy DataFrame."""

    if raw_path.suffix.lower() == ".csv":
        df = pd.read_csv(raw_path)
    else:
        df = pd.read_json(raw_path, lines=True)
    return df
