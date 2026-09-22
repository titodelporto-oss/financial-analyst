"""Turn raw indicators into a small number of high-conviction BUY/SELL signals.

Philosophy: MACD's moving-average crossover confirms a move only after it has
already started. To catch a move earlier we require at least 2 of 3 *leading*
technical conditions (RSI divergence, an early stochastic turn, or MACD momentum
already accelerating before the crossover) - AND a fundamentals quality gate, so a
signal never fires on pure momentum with a rotten balance sheet underneath. Most
tickers will trigger nothing at all: that's intentional, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import detect_rsi_divergence

MIN_TECHNICAL_CONFIRMATIONS = 2


@dataclass
class Signal:
    kind: str  # "BUY" or "SELL"
    reasons: list = field(default_factory=list)
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None


def _stochastic_cross(stoch: pd.DataFrame, lookback: int = 3) -> str | None:
    """'bullish' if %K crossed above %D from below 30 in the last `lookback` bars,
    'bearish' if %K crossed below %D from above 70. None otherwise."""
    tail = stoch.tail(lookback + 1)
    if tail["k"].isna().any() or tail["d"].isna().any() or len(tail) < 2:
        return None
    for i in range(1, len(tail)):
        k_prev, d_prev = tail["k"].iloc[i - 1], tail["d"].iloc[i - 1]
        k_now, d_now = tail["k"].iloc[i], tail["d"].iloc[i]
        if k_prev <= d_prev and k_now > d_now and d_prev < 30:
            return "bullish"
        if k_prev >= d_prev and k_now < d_now and d_prev > 70:
            return "bearish"
    return None


def _macd_momentum_direction(histogram: pd.Series, bars: int = 3) -> str | None:
    """'rising'/'falling' if the histogram has moved consistently for `bars` bars -
    this catches acceleration before the lagging zero-line crossover happens."""
    tail = histogram.tail(bars + 1)
    if tail.isna().any() or len(tail) < bars + 1:
        return None
    diffs = tail.diff().dropna()
    if (diffs > 0).all():
        return "rising"
    if (diffs < 0).all():
        return "falling"
    return None


def _fundamentals_buy_gate(ta) -> bool:
    return (
        ta.piotroski_f is not None
        and ta.piotroski_f >= 6
        and ta.altman_zone != "Distress"
        and ta.beneish_flag != "Possibile manipolazione"
        and ta.target_upside_pct is not None
        and ta.target_upside_pct > 5
    )


def _fundamentals_sell_flag(ta) -> str | None:
    if ta.beneish_flag == "Possibile manipolazione":
        return "Beneish M-Score segnala possibile manipolazione contabile"
    if ta.altman_zone == "Distress":
        return "Altman Z-Score in zona di distress finanziario"
    if ta.piotroski_f is not None and ta.piotroski_f <= 3:
        return f"Piotroski F-Score debole ({ta.piotroski_f}/9)"
    if ta.target_upside_pct is not None and ta.target_upside_pct < -10:
        return f"Prezzo {abs(ta.target_upside_pct):.0f}% sopra il target medio degli analisti"
    return None


def classify_signal(ta, history: pd.DataFrame, rsi: pd.Series, macd_df: pd.DataFrame, stoch: pd.DataFrame) -> Signal | None:
    close = history["Close"]
    divergence = detect_rsi_divergence(close, rsi)
    stoch_cross = _stochastic_cross(stoch)
    macd_dir = _macd_momentum_direction(macd_df["histogram"])

    current_price = float(close.iloc[-1])

    # --- BUY ---
    buy_reasons = []
    if divergence and divergence["type"] == "bullish":
        buy_reasons.append(
            f"Divergenza rialzista sull'RSI: il prezzo ha fatto un minimo più basso "
            f"({divergence['price_low_1']:.2f} -> {divergence['price_low_2']:.2f}) ma l'RSI un minimo più "
            f"alto ({divergence['rsi_low_1']:.1f} -> {divergence['rsi_low_2']:.1f}), segno che la pressione "
            f"di vendita si sta esaurendo prima ancora che il prezzo giri."
        )
    if stoch_cross == "bullish":
        buy_reasons.append(
            "Lo Stocastico è uscito dalla zona di ipervenduto (%K ha incrociato %D dal basso), "
            "un segnale di svolta più rapido del MACD."
        )
    if macd_dir == "rising":
        buy_reasons.append(
            "L'istogramma MACD sta accelerando al rialzo da 3 sedute: il momentum sta migliorando "
            "anche prima dell'incrocio delle medie."
        )

    if len(buy_reasons) >= MIN_TECHNICAL_CONFIRMATIONS and _fundamentals_buy_gate(ta):
        buy_reasons.append(
            f"Bilancio solido: Piotroski F {ta.piotroski_f}/9, Altman Z in zona {ta.altman_zone.lower()}, "
            f"nessun segnale Beneish di manipolazione."
        )
        buy_reasons.append(
            f"Il titolo tratta {ta.target_upside_pct:.0f}% sotto il target medio degli analisti "
            f"({ta.name or ta.ticker})."
        )
        recent_low = float(history["Low"].tail(10).min())
        stop_loss = min(recent_low, current_price * 0.95)
        risk = current_price - stop_loss
        take_profit = None
        if ta.target_upside_pct and ta.target_upside_pct > 0:
            take_profit = current_price * (1 + ta.target_upside_pct / 100)
        if take_profit is None or take_profit <= current_price:
            take_profit = current_price + 2 * risk
        return Signal(kind="BUY", reasons=buy_reasons, entry_price=current_price,
                      stop_loss=stop_loss, take_profit=take_profit)

    # --- SELL ---
    sell_reasons = []
    if divergence and divergence["type"] == "bearish":
        sell_reasons.append(
            f"Divergenza ribassista sull'RSI: il prezzo ha fatto un massimo più alto "
            f"({divergence['price_high_1']:.2f} -> {divergence['price_high_2']:.2f}) ma l'RSI un massimo più "
            f"basso ({divergence['rsi_high_1']:.1f} -> {divergence['rsi_high_2']:.1f}), la spinta rialzista "
            f"si sta esaurendo prima ancora che il prezzo giri."
        )
    if stoch_cross == "bearish":
        sell_reasons.append(
            "Lo Stocastico è uscito dalla zona di ipercomprato (%K ha incrociato %D dall'alto), "
            "un segnale di svolta più rapido del MACD."
        )
    if macd_dir == "falling":
        sell_reasons.append(
            "L'istogramma MACD sta rallentando da 3 sedute: il momentum sta peggiorando "
            "anche prima dell'incrocio delle medie."
        )

    fundamentals_flag = _fundamentals_sell_flag(ta)
    if len(sell_reasons) >= MIN_TECHNICAL_CONFIRMATIONS and fundamentals_flag:
        sell_reasons.append(fundamentals_flag)
        return Signal(kind="SELL", reasons=sell_reasons, entry_price=current_price)

    return None
