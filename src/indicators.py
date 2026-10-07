"""Technical indicators computed with plain pandas (no external TA library)."""

from __future__ import annotations

import pandas as pd


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def volume_spike_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Latest volume divided by its rolling average (>1 means above-average volume)."""
    avg_volume = volume.rolling(window=window).mean()
    return volume / avg_volume


def calculate_bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return pd.DataFrame({"mid": sma, "upper": sma + num_std * std, "lower": sma - num_std * std})


def calculate_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator: a leading momentum indicator - it turns *before* MACD's
    lagging moving-average crossover, because it tracks where price sits within its
    recent range rather than waiting for two averages to cross."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    price_range = (highest_high - lowest_low).replace(0, float("nan"))  # flat range (e.g. halted stock)
    k = 100 * (close - lowest_low) / price_range
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: running total of volume added/subtracted based on price
    direction. A rising OBV while price is flat/falling suggests quiet accumulation
    ahead of a move (an earlier tell than waiting for the price move itself)."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def find_pivots(values, order: int, kind: str) -> list:
    """Indices i that are a strict swing low/high: lower (higher) than the `order`
    bars before it and not exceeded by the `order` bars after it. A pivot at bar i
    is therefore only *confirmed* at bar i + order - the last `order` bars never pivot."""
    pivots = []
    for i in range(order, len(values) - order):
        left, right = values[i - order:i], values[i + 1:i + order + 1]
        if kind == "min" and values[i] < left.min() and values[i] <= right.min():
            pivots.append(i)
        elif kind == "max" and values[i] > left.max() and values[i] >= right.max():
            pivots.append(i)
    return pivots


def detect_rsi_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 60, order: int = 3,
                          max_age: int = 5) -> dict | None:
    """Compare the two most recent swing lows/highs in price vs. RSI to catch a
    reversal *before* it shows up as a moving-average crossover:

    - Bullish divergence: price makes a lower low, but RSI makes a higher low
      (selling pressure is fading even though price hasn't turned yet).
    - Bearish divergence: price makes a higher high, but RSI makes a lower high
      (buying pressure is fading even though price is still rising).

    Only uses the bars it receives (no look-ahead), and only reports a divergence
    whose second pivot was confirmed at most `max_age` bars ago - an old divergence
    is no longer a signal.
    """
    recent_close = close.tail(lookback)
    recent_rsi = rsi.tail(lookback)
    if len(recent_close) < 2 * order + 2 or recent_close.isna().any() or recent_rsi.isna().any():
        return None
    cv = recent_close.to_numpy(dtype=float)
    rv = recent_rsi.to_numpy(dtype=float)
    last = len(cv) - 1

    found = []
    minima = find_pivots(cv, order, "min")
    if len(minima) >= 2:
        i1, i2 = minima[-2], minima[-1]
        if cv[i2] < cv[i1] and rv[i2] > rv[i1] and last - (i2 + order) <= max_age:
            found.append({
                "type": "bullish",
                "price_low_1": float(cv[i1]), "price_low_2": float(cv[i2]),
                "rsi_low_1": float(rv[i1]), "rsi_low_2": float(rv[i2]),
                "bars_ago": last - (i2 + order),
            })
    maxima = find_pivots(cv, order, "max")
    if len(maxima) >= 2:
        i1, i2 = maxima[-2], maxima[-1]
        if cv[i2] > cv[i1] and rv[i2] < rv[i1] and last - (i2 + order) <= max_age:
            found.append({
                "type": "bearish",
                "price_high_1": float(cv[i1]), "price_high_2": float(cv[i2]),
                "rsi_high_1": float(rv[i1]), "rsi_high_2": float(rv[i2]),
                "bars_ago": last - (i2 + order),
            })
    if not found:
        return None
    return min(found, key=lambda d: d["bars_ago"])  # bars_ago = sessions since confirmation
