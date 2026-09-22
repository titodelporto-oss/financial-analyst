"""Build the per-signal price+indicator chart (Plotly), styled to match TradingView's
own chart panels: dark surface, right-side price axis, per-pane live-value labels
instead of a legend, a faint ticker watermark, and a dashed last-price line."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG = "#131722"
GRID = "#1e222d"
TEXT = "#d1d4dc"
TEXT_MUTED = "#787b86"
UP = "#089981"
DOWN = "#f23645"
BAND_LINE = "#5d606b"
BAND_FILL = "rgba(93, 96, 107, 0.10)"
STOCH_K = "#2962ff"
STOCH_D = "#ff6d00"
RSI_LINE = "#b388ff"
LAST_PRICE = "#787b86"

ROW_RSI, ROW_STOCH, ROW_MACD, ROW_VOL, ROW_PRICE = 3, 4, 5, 2, 1


def _pane_label(fig, text: str, row: int, color: str = TEXT) -> None:
    fig.add_annotation(
        text=text, xref=f"x{row} domain" if row > 1 else "x domain",
        yref=f"y{row} domain" if row > 1 else "y domain",
        x=0.005, y=0.97, showarrow=False, xanchor="left", yanchor="top",
        font=dict(color=color, size=12, family="Helvetica, Arial, sans-serif"),
        bgcolor="rgba(19,23,34,0.72)", borderpad=3,
        row=row, col=1,
    )


def build_signal_chart(ticker: str, chart: dict, signal_kind: str, entry_price, stop_loss, take_profit) -> go.Figure:
    x = chart["dates"]
    last_close = chart["close"][-1]

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.42, 0.10, 0.16, 0.16, 0.16],
        vertical_spacing=0.015,
    )

    # Row 1: candlestick + Bollinger bands + watermark + last-price line
    fig.add_annotation(
        text=ticker, xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False,
        font=dict(color="rgba(209, 212, 220, 0.07)", size=90, family="Helvetica, Arial, sans-serif"),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=x, y=chart["bb_upper"], line=dict(width=1, color=BAND_LINE),
                              name="Banda sup.", showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=chart["bb_lower"], line=dict(width=1, color=BAND_LINE), fill="tonexty",
                              fillcolor=BAND_FILL, name="Banda inf.", showlegend=False, hoverinfo="skip"),
                  row=1, col=1)
    fig.add_trace(
        go.Candlestick(
            x=x, open=chart["open"], high=chart["high"], low=chart["low"], close=chart["close"],
            increasing_line_color=UP, decreasing_line_color=DOWN,
            increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
            increasing_line_width=1, decreasing_line_width=1,
            name="Prezzo", showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_hline(y=last_close, line=dict(color=LAST_PRICE, width=1, dash="dot"),
                  annotation_text=f"{last_close:.2f}", annotation_font_color=BG,
                  annotation_bgcolor=LAST_PRICE, annotation_font_size=11,
                  annotation_position="right", row=1, col=1)

    marker_color = UP if signal_kind == "BUY" else DOWN
    marker_symbol = "triangle-up" if signal_kind == "BUY" else "triangle-down"
    fig.add_trace(
        go.Scatter(x=[x[-1]], y=[entry_price], mode="markers",
                   marker=dict(size=15, color=marker_color, symbol=marker_symbol, line=dict(width=1, color=TEXT)),
                   name=f"Segnale {signal_kind}", showlegend=False),
        row=1, col=1,
    )
    if signal_kind == "BUY" and stop_loss:
        fig.add_hline(y=stop_loss, line=dict(color=DOWN, width=1, dash="dash"),
                      annotation_text=f"Stop loss {stop_loss:.2f}", annotation_font_color=DOWN,
                      annotation_bgcolor="rgba(19,23,34,0.85)", annotation_font_size=11,
                      annotation_position="top right", annotation_xanchor="right",
                      annotation_xshift=-58, row=1, col=1)
    if signal_kind == "BUY" and take_profit:
        fig.add_hline(y=take_profit, line=dict(color=UP, width=1, dash="dash"),
                      annotation_text=f"Take profit {take_profit:.2f}", annotation_font_color=UP,
                      annotation_bgcolor="rgba(19,23,34,0.85)", annotation_font_size=11,
                      annotation_position="top right", annotation_xanchor="right",
                      annotation_xshift=-58, row=1, col=1)
    _pane_label(fig, f"<b>{ticker}</b>  O {chart['open'][-1]:.2f}  H {chart['high'][-1]:.2f}  "
                      f"L {chart['low'][-1]:.2f}  C {chart['close'][-1]:.2f}", ROW_PRICE)

    # Row 2: volume
    vol_colors = [UP if (c or 0) >= (o or 0) else DOWN for o, c in zip(chart["open"], chart["close"])]
    fig.add_trace(go.Bar(x=x, y=chart["volume"], marker_color=vol_colors, marker_opacity=0.7,
                          name="Volume", showlegend=False), row=ROW_VOL, col=1)
    _pane_label(fig, f"Volume  {chart['volume'][-1]:,.0f}", ROW_VOL, TEXT_MUTED)

    # Row 3: RSI
    fig.add_trace(go.Scatter(x=x, y=chart["rsi"], line=dict(width=1.5, color=RSI_LINE), name="RSI", showlegend=False),
                  row=ROW_RSI, col=1)
    fig.add_hline(y=70, line=dict(color=TEXT_MUTED, width=1, dash="dot"), row=ROW_RSI, col=1)
    fig.add_hline(y=30, line=dict(color=TEXT_MUTED, width=1, dash="dot"), row=ROW_RSI, col=1)
    _pane_label(fig, f"RSI (14)  {chart['rsi'][-1]:.1f}", ROW_RSI, RSI_LINE)

    # Row 4: Stochastic
    fig.add_trace(go.Scatter(x=x, y=chart["stoch_k"], line=dict(width=1.5, color=STOCH_K), name="%K", showlegend=False),
                  row=ROW_STOCH, col=1)
    fig.add_trace(go.Scatter(x=x, y=chart["stoch_d"], line=dict(width=1.5, color=STOCH_D), name="%D", showlegend=False),
                  row=ROW_STOCH, col=1)
    fig.add_hline(y=80, line=dict(color=TEXT_MUTED, width=1, dash="dot"), row=ROW_STOCH, col=1)
    fig.add_hline(y=20, line=dict(color=TEXT_MUTED, width=1, dash="dot"), row=ROW_STOCH, col=1)
    _pane_label(fig, f"Stoc %K {chart['stoch_k'][-1]:.1f}  %D {chart['stoch_d'][-1]:.1f}", ROW_STOCH, STOCH_K)

    # Row 5: MACD histogram
    hist = chart["macd_hist"]
    bar_colors = [UP if (v or 0) >= 0 else DOWN for v in hist]
    fig.add_trace(go.Bar(x=x, y=hist, marker_color=bar_colors, name="MACD", showlegend=False), row=ROW_MACD, col=1)
    _pane_label(fig, f"MACD hist  {hist[-1]:.2f}", ROW_MACD, TEXT_MUTED)

    fig.update_layout(
        height=760,
        margin=dict(l=8, r=55, t=10, b=10),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Helvetica, Arial, sans-serif", size=12),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e222d", font=dict(color=TEXT)),
        xaxis_rangeslider_visible=False,
    )
    for r in range(1, 6):
        fig.update_xaxes(gridcolor=GRID, showgrid=True, zeroline=False, showspikes=True,
                          spikemode="across", spikecolor=TEXT_MUTED, spikethickness=1,
                          spikedash="dot", row=r, col=1)
        fig.update_yaxes(gridcolor=GRID, showgrid=True, zeroline=False, side="right",
                          tickfont=dict(color=TEXT_MUTED), row=r, col=1)
    fig.update_xaxes(tickfont=dict(color=TEXT_MUTED), row=5, col=1)

    return fig
