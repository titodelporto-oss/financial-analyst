"""Build the per-signal price+indicator chart (Plotly), styled like a TradingView
panel: dark surface, candlesticks, volume, and oscillator rows sharing one x-axis."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG = "#131722"
GRID = "#2a2e39"
TEXT = "#d1d4dc"
TEXT_MUTED = "#787b86"
UP = "#089981"
DOWN = "#f23645"
BAND_LINE = "#5d606b"
BAND_FILL = "rgba(93, 96, 107, 0.12)"
STOCH_K = "#2962ff"
STOCH_D = "#ff6d00"
RSI_LINE = "#b388ff"


def build_signal_chart(ticker: str, chart: dict, signal_kind: str, entry_price, stop_loss, take_profit) -> go.Figure:
    x = chart["dates"]

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.40, 0.12, 0.16, 0.16, 0.16],
        vertical_spacing=0.02,
        subplot_titles=(None, "Volume", "RSI", "Stocastico", "MACD"),
    )

    # Row 1: candlestick + Bollinger bands
    fig.add_trace(
        go.Scatter(x=x, y=chart["bb_upper"], line=dict(width=1, color=BAND_LINE), name="Banda sup.", showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=chart["bb_lower"], line=dict(width=1, color=BAND_LINE), fill="tonexty",
                    fillcolor=BAND_FILL, name="Banda inf.", showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Candlestick(
            x=x, open=chart["open"], high=chart["high"], low=chart["low"], close=chart["close"],
            increasing_line_color=UP, decreasing_line_color=DOWN,
            increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
            name="Prezzo", showlegend=False,
        ),
        row=1, col=1,
    )

    marker_color = UP if signal_kind == "BUY" else DOWN
    marker_symbol = "triangle-up" if signal_kind == "BUY" else "triangle-down"
    fig.add_trace(
        go.Scatter(x=[x[-1]], y=[entry_price], mode="markers",
                   marker=dict(size=16, color=marker_color, symbol=marker_symbol, line=dict(width=1, color=TEXT)),
                   name=f"Segnale {signal_kind}", showlegend=False),
        row=1, col=1,
    )
    if signal_kind == "BUY" and stop_loss:
        fig.add_hline(y=stop_loss, line=dict(color=DOWN, width=1, dash="dash"),
                      annotation_text="Stop loss", annotation_font_color=TEXT,
                      annotation_position="bottom right", row=1, col=1)
    if signal_kind == "BUY" and take_profit:
        fig.add_hline(y=take_profit, line=dict(color=UP, width=1, dash="dash"),
                      annotation_text="Take profit", annotation_font_color=TEXT,
                      annotation_position="top right", row=1, col=1)

    # Row 2: volume, colored by candle direction
    vol_colors = [UP if (c or 0) >= (o or 0) else DOWN for o, c in zip(chart["open"], chart["close"])]
    fig.add_trace(go.Bar(x=x, y=chart["volume"], marker_color=vol_colors, name="Volume", showlegend=False),
                  row=2, col=1)

    # Row 3: RSI
    fig.add_trace(go.Scatter(x=x, y=chart["rsi"], line=dict(width=1.5, color=RSI_LINE), name="RSI", showlegend=False),
                  row=3, col=1)
    fig.add_hline(y=70, line=dict(color=TEXT_MUTED, width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color=TEXT_MUTED, width=1, dash="dot"), row=3, col=1)

    # Row 4: Stochastic
    fig.add_trace(go.Scatter(x=x, y=chart["stoch_k"], line=dict(width=1.5, color=STOCH_K), name="%K"),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=chart["stoch_d"], line=dict(width=1.5, color=STOCH_D), name="%D"),
                  row=4, col=1)
    fig.add_hline(y=80, line=dict(color=TEXT_MUTED, width=1, dash="dot"), row=4, col=1)
    fig.add_hline(y=20, line=dict(color=TEXT_MUTED, width=1, dash="dot"), row=4, col=1)

    # Row 5: MACD histogram
    hist = chart["macd_hist"]
    bar_colors = [UP if (v or 0) >= 0 else DOWN for v in hist]
    fig.add_trace(go.Bar(x=x, y=hist, marker_color=bar_colors, name="MACD", showlegend=False),
                  row=5, col=1)

    fig.update_layout(
        height=820,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Helvetica, Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        title=dict(text=ticker, font=dict(color=TEXT, size=15)),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    for r in range(1, 6):
        fig.update_xaxes(gridcolor=GRID, showgrid=True, zeroline=False, showspikes=True,
                          spikemode="across", spikecolor=TEXT_MUTED, spikethickness=1, row=r, col=1)
        fig.update_yaxes(gridcolor=GRID, showgrid=True, zeroline=False, row=r, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    for annotation in fig["layout"]["annotations"]:
        if annotation["text"] in ("Volume", "RSI", "Stocastico", "MACD"):
            annotation["font"] = dict(color=TEXT_MUTED, size=11)

    return fig
