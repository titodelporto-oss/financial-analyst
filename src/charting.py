"""Build the per-signal price+indicator chart (Plotly) from the cached chart payload."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_PRICE = "#2a78d6"
COLOR_BAND = "rgba(42, 120, 214, 0.10)"
COLOR_BAND_LINE = "#898781"
COLOR_STOCH_K = "#2a78d6"
COLOR_STOCH_D = "#eb6834"
COLOR_RSI = "#4a3aa7"
COLOR_GOOD = "#0ca30c"
COLOR_CRITICAL = "#d03b3b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"


def build_signal_chart(ticker: str, chart: dict, signal_kind: str, entry_price, stop_loss, take_profit) -> go.Figure:
    x = chart["dates"]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.17, 0.17, 0.16],
        vertical_spacing=0.03,
        subplot_titles=("Prezzo e Bande di Bollinger", "RSI", "Stocastico", "Istogramma MACD"),
    )

    # Row 1: price + Bollinger band
    fig.add_trace(go.Scatter(x=x, y=chart["bb_upper"], line=dict(width=1, color=COLOR_BAND_LINE, dash="dot"),
                              name="Banda superiore", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=chart["bb_lower"], line=dict(width=1, color=COLOR_BAND_LINE, dash="dot"),
                              fill="tonexty", fillcolor=COLOR_BAND, name="Banda inferiore", showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=chart["close"], line=dict(width=2, color=COLOR_PRICE), name="Prezzo"),
                  row=1, col=1)

    marker_color = COLOR_GOOD if signal_kind == "BUY" else COLOR_CRITICAL
    marker_symbol = "triangle-up" if signal_kind == "BUY" else "triangle-down"
    fig.add_trace(
        go.Scatter(
            x=[x[-1]], y=[entry_price], mode="markers", marker=dict(size=14, color=marker_color, symbol=marker_symbol),
            name=f"Segnale {signal_kind}",
        ),
        row=1, col=1,
    )
    if signal_kind == "BUY" and stop_loss:
        fig.add_hline(y=stop_loss, line=dict(color=COLOR_CRITICAL, width=1, dash="dash"),
                      annotation_text="Stop loss", annotation_position="bottom right", row=1, col=1)
    if signal_kind == "BUY" and take_profit:
        fig.add_hline(y=take_profit, line=dict(color=COLOR_GOOD, width=1, dash="dash"),
                      annotation_text="Take profit", annotation_position="top right", row=1, col=1)

    # Row 2: RSI
    fig.add_trace(go.Scatter(x=x, y=chart["rsi"], line=dict(width=2, color=COLOR_RSI), name="RSI", showlegend=False),
                  row=2, col=1)
    fig.add_hline(y=70, line=dict(color=COLOR_MUTED, width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color=COLOR_MUTED, width=1, dash="dot"), row=2, col=1)

    # Row 3: Stochastic
    fig.add_trace(go.Scatter(x=x, y=chart["stoch_k"], line=dict(width=2, color=COLOR_STOCH_K), name="%K"),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=chart["stoch_d"], line=dict(width=2, color=COLOR_STOCH_D), name="%D"),
                  row=3, col=1)
    fig.add_hline(y=80, line=dict(color=COLOR_MUTED, width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=20, line=dict(color=COLOR_MUTED, width=1, dash="dot"), row=3, col=1)

    # Row 4: MACD histogram
    hist = chart["macd_hist"]
    bar_colors = [COLOR_GOOD if (v or 0) >= 0 else COLOR_CRITICAL for v in hist]
    fig.add_trace(go.Bar(x=x, y=hist, marker_color=bar_colors, name="Istogramma MACD", showlegend=False),
                  row=4, col=1)

    fig.update_layout(
        height=760,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=f"{ticker}",
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, showgrid=True)
    fig.update_yaxes(gridcolor=COLOR_GRID, showgrid=True)

    return fig
