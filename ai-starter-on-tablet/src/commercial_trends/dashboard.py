"""
Lightweight Gradio dashboard to explore commercial area growth and decline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import argparse
import pandas as pd

from .analysis import TrendResult, compute_growth, normalize_columns, plot_growth_bar, plot_time_series, top_growth_and_decline


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_commercial_data.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_columns(df)


def analyze(path: Path = DATA_PATH, top_n: int = 3) -> Tuple[List[TrendResult], List[TrendResult], pd.DataFrame]:
    df = load_data(path)
    trends = compute_growth(df)
    growth, decline = top_growth_and_decline(trends, top_n=top_n)
    return growth, decline, df


def render_dashboard(path: Path = DATA_PATH, top_n: int = 3):
    import gradio as gr

    growth, decline, df = analyze(path, top_n=top_n)
    business_choices = sorted(df["business_type"].unique())

    def refresh_results(top_n: int):
        growth_trends, decline_trends, updated_df = analyze(path, top_n=top_n)
        growth_fig = plot_growth_bar(growth_trends, "성장 업종 TOP")
        decline_fig = plot_growth_bar(decline_trends, "쇠퇴 업종 TOP")
        return growth_trends, decline_trends, growth_fig, decline_fig

    def plot_for_type(business_type: str, metric: str):
        _, _, local_df = analyze(path)
        fig = plot_time_series(local_df, business_type, metric=metric)
        return fig

    growth_table = gr.DataFrame(headers=list(TrendResult.__annotations__.keys()), interactive=False)
    decline_table = gr.DataFrame(headers=list(TrendResult.__annotations__.keys()), interactive=False)

    with gr.Blocks(title="상권 트렌드 대시보드") as demo:
        gr.Markdown("""
        # 상권 트렌드 대시보드
        공개 상권 데이터에서 성장/쇠퇴 업종을 빠르게 확인하고, 시간대별 추이를 시각화합니다.
        데이터 파일 경로를 바꾸면 바로 새로운 상권 데이터를 탐색할 수 있습니다.
        """)

        with gr.Row():
            top_n_slider = gr.Slider(3, 10, value=3, step=1, label="순위 표시 개수")
            metric_dropdown = gr.Dropdown(["sales", "transactions", "customer_count"], value="sales", label="시계열 지표")
            business_dropdown = gr.Dropdown(business_choices, value=business_choices[0], label="업종 선택")

        with gr.Row():
            growth_plot = gr.Plot(label="성장 업종 TOP")
            decline_plot = gr.Plot(label="쇠퇴 업종 TOP")

        with gr.Row():
            growth_table_comp = growth_table
            decline_table_comp = decline_table

        time_series_plot = gr.Plot(label="업종별 시계열")

        top_n_slider.change(fn=refresh_results, inputs=top_n_slider, outputs=[growth_table_comp, decline_table_comp, growth_plot, decline_plot])
        metric_dropdown.change(fn=plot_for_type, inputs=[business_dropdown, metric_dropdown], outputs=time_series_plot)
        business_dropdown.change(fn=plot_for_type, inputs=[business_dropdown, metric_dropdown], outputs=time_series_plot)

        # Initial render
        initial_growth_fig = plot_growth_bar(growth, "성장 업종 TOP")
        initial_decline_fig = plot_growth_bar(decline, "쇠퇴 업종 TOP")
        initial_ts_fig = plot_time_series(df, business_choices[0], metric="sales")

        demo.load(lambda: (growth, decline, initial_growth_fig, initial_decline_fig, initial_ts_fig), None, [growth_table_comp, decline_table_comp, growth_plot, decline_plot, time_series_plot])

    return demo


def main():
    parser = argparse.ArgumentParser(description="상권 트렌드 대시보드 실행기")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="CSV 데이터 경로 (샘플 스키마 유지)")
    parser.add_argument("--top-n", type=int, default=3, help="초기 TOP 순위 개수")
    parser.add_argument("--listen", type=str, default=None, help="Gradio listen 호스트 (예: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Gradio 포트")
    args = parser.parse_args()

    app = render_dashboard(args.data, top_n=args.top_n)
    app.launch(server_name=args.listen, server_port=args.port)


if __name__ == "__main__":
    main()
