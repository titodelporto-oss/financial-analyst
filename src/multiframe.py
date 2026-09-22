"""Build chart payloads (OHLC + indicators, JSON-safe) across multiple timeframes
for a single ticker, so the dashboard can let the viewer switch between them."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from .indicators import calculate_bollinger_bands, calculate_macd, calculate_rsi, calculate_stochastic

TIMEFRAMES = ["1S", "1G", "4H", "30M"]  # Settimana, Giorno, 4 Ore, 30 Minuti
TIMEFRAME_LABELS = {"1S": "1 Settimana", "1G": "1 Giorno", "4H": "4 Ore", "30M": "30 Minuti"}


def _safe_list(series, digits: int = 4) -> list:
    return [None if pd.isna(v) else round(float(v), digits) for v in series]


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return out.dropna(subset=["Close"])


def _chart_payload_from_ohlc(df: pd.DataFrame, window: int) -> dict | None:
    if df.empty or len(df) < 20:
        return None
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    rsi = calculate_rsi(close)
    macd_df = calculate_macd(close)
    bb = calculate_bollinger_bands(close)
    stoch = calculate_stochastic(high, low, close)

    idx = df.index[-window:]
    return {
        "dates": [d.strftime("%Y-%m-%d %H:%M") for d in idx],
        "open": _safe_list(df["Open"].tail(window), 2),
        "high": _safe_list(high.tail(window), 2),
        "low": _safe_list(low.tail(window), 2),
        "close": _safe_list(close.tail(window), 2),
        "volume": _safe_list(volume.tail(window), 0),
        "bb_upper": _safe_list(bb["upper"].tail(window), 2),
        "bb_mid": _safe_list(bb["mid"].tail(window), 2),
        "bb_lower": _safe_list(bb["lower"].tail(window), 2),
        "rsi": _safe_list(rsi.tail(window)),
        "stoch_k": _safe_list(stoch["k"].tail(window)),
        "stoch_d": _safe_list(stoch["d"].tail(window)),
        "macd_hist": _safe_list(macd_df["histogram"].tail(window)),
    }


def build_multi_timeframe_charts(ticker_obj: yf.Ticker, daily_history: pd.DataFrame) -> dict:
    """daily_history: the already-fetched ~2y daily OHLC (avoids re-fetching it)."""
    charts = {}

    charts["1G"] = _chart_payload_from_ohlc(daily_history, window=180)

    weekly = _resample_ohlc(daily_history, "W")
    charts["1S"] = _chart_payload_from_ohlc(weekly, window=104)

    try:
        hourly = ticker_obj.history(period="60d", interval="1h")
        if isinstance(hourly.columns, pd.MultiIndex):
            hourly.columns = hourly.columns.get_level_values(0)
        if not hourly.empty:
            four_hour = _resample_ohlc(hourly, "4h")
            charts["4H"] = _chart_payload_from_ohlc(four_hour, window=120)
    except Exception:
        charts["4H"] = None

    try:
        thirty_min = ticker_obj.history(period="60d", interval="30m")
        if isinstance(thirty_min.columns, pd.MultiIndex):
            thirty_min.columns = thirty_min.columns.get_level_values(0)
        if not thirty_min.empty:
            charts["30M"] = _chart_payload_from_ohlc(thirty_min, window=180)
    except Exception:
        charts["30M"] = None

    return {k: v for k, v in charts.items() if v is not None}
