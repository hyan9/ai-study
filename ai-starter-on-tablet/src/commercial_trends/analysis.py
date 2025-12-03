"""
Analysis helpers for spotting growing and declining businesses in a commercial area.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(style="whitegrid")


@dataclass
class TrendResult:
    business_type: str
    growth_score: float
    change_pct: float
    start_sales: float
    end_sales: float
    observations: int


EXPECTED_COLUMNS = {
    "date": "date",
    "district": "district",
    "commercial_area": "commercial_area",
    "business_type": "business_type",
    "sales": "sales",
    "transactions": "transactions",
    "customer_count": "customer_count",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to the expected schema if possible."""

    normalized = df.rename(columns={k: v for k, v in EXPECTED_COLUMNS.items() if k in df.columns})
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"])
    return normalized


def compute_growth(df: pd.DataFrame, min_points: int = 3) -> List[TrendResult]:
    """Compute growth metrics for each business type within each commercial area."""

    required_cols = {"date", "business_type", "sales"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Input data must include {required_cols}")

    df_sorted = df.sort_values("date")
    trend_results: List[TrendResult] = []

    for business_type, group in df_sorted.groupby("business_type"):
        if len(group) < min_points:
            continue
        sales = group["sales"].astype(float).to_numpy()
        x = np.arange(len(group))
        slope, intercept = np.polyfit(x, sales, 1)
        growth_score = float(slope / sales.mean())
        start_sales = float(sales[0])
        end_sales = float(sales[-1])
        change_pct = float((end_sales - start_sales) / start_sales * 100)
        trend_results.append(
            TrendResult(
                business_type=business_type,
                growth_score=growth_score,
                change_pct=change_pct,
                start_sales=start_sales,
                end_sales=end_sales,
                observations=len(group),
            )
        )
    return trend_results


def top_growth_and_decline(trends: List[TrendResult], top_n: int = 3) -> Tuple[List[TrendResult], List[TrendResult]]:
    """Return lists of growing and declining industries ranked by growth score."""

    sorted_trends = sorted(trends, key=lambda t: t.growth_score, reverse=True)
    return sorted_trends[:top_n], sorted_trends[-top_n:][::-1]


def summarize_from_file(raw_path: Path, top_n: int = 3):
    """Convenience wrapper that reads a file, computes metrics, and returns summaries."""

    df = pd.read_csv(raw_path)
    normalized = normalize_columns(df)
    trends = compute_growth(normalized)
    return top_growth_and_decline(trends, top_n=top_n)


def plot_time_series(df: pd.DataFrame, business_type: str, metric: str = "sales"):
    """Render a line plot for a given business type."""

    filtered = df[df["business_type"] == business_type].sort_values("date")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=filtered, x="date", y=metric, marker="o", ax=ax)
    ax.set_title(f"{business_type} {metric} trend")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric.capitalize())
    fig.autofmt_xdate()
    return fig


def plot_growth_bar(trends: List[TrendResult], title: str):
    """Render a bar chart visualizing growth scores."""

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x=[t.business_type for t in trends],
        y=[t.growth_score for t in trends],
        palette="crest",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Business type")
    ax.set_ylabel("Growth score (slope / mean sales)")
    ax.tick_params(axis="x", rotation=20)
    return fig
