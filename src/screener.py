"""Combine price history + fundamentals into a growth-potential score per ticker.

This is a heuristic screening tool, not investment advice: RSI/MACD are lagging
technical signals, the "fair value" figure is just the analysts' average target
price reported by Yahoo Finance (not an independent valuation model), and Altman
Z / Piotroski F / Beneish M are standard academic formulas with known blind spots
(they were designed mainly for industrial/manufacturing companies and can misfire
on banks, REITs, or young/loss-making companies).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import yfinance as yf

from .data import fetch_fundamentals, fetch_price_history
from .fundamentals import (
    FinancialStatements,
    altman_zone,
    beneish_flag,
    calculate_altman_z,
    calculate_beneish_m,
    calculate_piotroski_f,
)
from .indicators import (
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    calculate_stochastic,
    volume_spike_ratio,
)
from .signals import classify_signal
from .narrative import build_narrative
from .multiframe import build_multi_timeframe_charts


@dataclass
class TickerAnalysis:
    ticker: str
    name: str | None
    price: float | None
    market_cap: float | None
    rsi: float | None
    rsi_zone: str
    macd_signal: str
    volume_ratio: float | None
    target_upside_pct: float | None
    altman_z: float | None
    altman_zone: str
    piotroski_f: int | None
    beneish_m: float | None
    beneish_flag: str
    score: float
    error: str | None = None
    signal: str | None = None
    signal_reasons: list = field(default_factory=list)
    narrative: list = field(default_factory=list)
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    charts: dict | None = None


def _empty(ticker: str, error: str) -> TickerAnalysis:
    return TickerAnalysis(
        ticker, None, None, None, None, "N/D", "N/D", None, None,
        None, "N/D", None, None, "N/D", 0, error=error,
    )


def _rsi_zone_and_score(rsi: float | None) -> tuple[str, float]:
    if rsi is None:
        return "N/D", 0
    if rsi < 30:
        return "Ipervenduto", 2
    if rsi <= 60:
        return "Neutrale/Sano", 1
    if rsi <= 70:
        return "Forte (attenzione)", 0.5
    return "Ipercomprato", -1


def _macd_signal_and_score(histogram) -> tuple[str, float]:
    if len(histogram.dropna()) < 2:
        return "N/D", 0
    latest_hist = histogram.iloc[-1]
    prev_hist = histogram.iloc[-2]

    if prev_hist <= 0 < latest_hist:
        return "Incrocio rialzista recente", 2
    if latest_hist > 0:
        return "Rialzista", 1
    if latest_hist < 0 <= prev_hist:
        return "Incrocio ribassista recente", -2
    return "Ribassista", -1


def _volume_score(volume_ratio: float | None) -> float:
    if volume_ratio is None or volume_ratio != volume_ratio:  # NaN check
        return 0
    return 1 if volume_ratio > 1.5 else 0


def _fair_value_upside_and_score(current_price, target_mean_price) -> tuple[float | None, float]:
    if not current_price or not target_mean_price:
        return None, 0
    upside_pct = (target_mean_price - current_price) / current_price * 100
    if upside_pct > 15:
        return upside_pct, 2
    if upside_pct > 5:
        return upside_pct, 1
    if upside_pct < -5:
        return upside_pct, -1
    return upside_pct, 0


def _altman_score(z: float | None) -> float:
    if z is None:
        return 0
    if z > 2.99:
        return 1
    if z >= 1.81:
        return 0
    return -2


def _piotroski_score(f: int | None) -> float:
    if f is None:
        return 0
    if f >= 7:
        return 1
    if f <= 2:
        return -1
    return 0


def _beneish_score(m: float | None) -> float:
    if m is None:
        return 0
    return -2 if m > -1.78 else 0


RETRYABLE_MARKERS = ("Invalid Crumb", "Too Many Requests", "429", "401")


def analyze_ticker(ticker: str, retries: int = 3) -> TickerAnalysis:
    last_error = None
    for attempt in range(retries):
        result = _analyze_ticker_once(ticker)
        if result.error is None or not any(m in result.error for m in RETRYABLE_MARKERS):
            return result
        last_error = result.error
        time.sleep(1.5 * (attempt + 1))  # backoff before retrying a rate-limited call
    return _empty(ticker, last_error)


def _analyze_ticker_once(ticker: str) -> TickerAnalysis:
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        history = fetch_price_history(ticker_obj)
        if history.empty:
            return _empty(ticker, "Nessun dato storico trovato")

        close = history["Close"]
        rsi_series = calculate_rsi(close)
        macd_df = calculate_macd(close)
        vol_ratio_series = volume_spike_ratio(history["Volume"])
        bb = calculate_bollinger_bands(close)
        stoch = calculate_stochastic(history["High"], history["Low"], close)

        latest_rsi = rsi_series.iloc[-1] if not rsi_series.dropna().empty else None
        latest_vol_ratio = vol_ratio_series.iloc[-1] if not vol_ratio_series.dropna().empty else None
        current_price = float(close.iloc[-1])

        fundamentals = fetch_fundamentals(ticker_obj, info)
        target_mean_price = fundamentals.get("targetMeanPrice")

        fs = FinancialStatements(ticker_obj, info)
        z = calculate_altman_z(fs)
        f_score = calculate_piotroski_f(fs)
        m_score = calculate_beneish_m(fs)

        rsi_zone, rsi_score = _rsi_zone_and_score(latest_rsi)
        macd_signal, macd_score = _macd_signal_and_score(macd_df["histogram"])
        volume_score = _volume_score(latest_vol_ratio)
        upside_pct, fair_value_score = _fair_value_upside_and_score(current_price, target_mean_price)

        total_score = (
            rsi_score
            + macd_score
            + volume_score
            + fair_value_score
            + _altman_score(z)
            + _piotroski_score(f_score)
            + _beneish_score(m_score)
        )

        ta = TickerAnalysis(
            ticker=ticker,
            name=fundamentals.get("shortName"),
            price=current_price,
            market_cap=fs.info.get("marketCap"),
            rsi=float(latest_rsi) if latest_rsi is not None else None,
            rsi_zone=rsi_zone,
            macd_signal=macd_signal,
            volume_ratio=float(latest_vol_ratio) if latest_vol_ratio is not None else None,
            target_upside_pct=upside_pct,
            altman_z=z,
            altman_zone=altman_zone(z),
            piotroski_f=f_score,
            beneish_m=m_score,
            beneish_flag=beneish_flag(m_score),
            score=total_score,
        )

        signal = classify_signal(ta, history, rsi_series, macd_df, stoch)
        if signal is not None:
            ta.signal = signal.kind
            ta.signal_reasons = signal.reasons
            ta.entry_price = signal.entry_price
            ta.stop_loss = signal.stop_loss
            ta.take_profit = signal.take_profit
            ta.narrative = build_narrative(ta, fundamentals, history, rsi_series, macd_df, stoch, signal.kind)
            ta.charts = build_multi_timeframe_charts(ticker_obj, history)

        return ta
    except Exception as exc:  # noqa: BLE001 - surface any per-ticker failure in the UI/batch log
        return _empty(ticker, str(exc))


def screen_tickers(tickers: list[str]) -> list[TickerAnalysis]:
    results = [analyze_ticker(t.strip().upper()) for t in tickers if t.strip()]
    return sorted(results, key=lambda r: r.score, reverse=True)
