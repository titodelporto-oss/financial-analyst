"""Fetch market data and fundamentals via yfinance.

All functions take an existing `yf.Ticker` instance (created once per ticker by the
caller) instead of creating their own, to avoid duplicate network round-trips when
running a batch screening over thousands of tickers.
"""

import pandas as pd
import yfinance as yf


def fetch_price_history(ticker_obj: yf.Ticker, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = ticker_obj.history(period=period, interval=interval)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_fundamentals(ticker_obj: yf.Ticker, info: dict) -> dict:
    """Best-effort fundamentals lookup; fields can be missing depending on the ticker."""
    return {
        "shortName": info.get("shortName"),
        "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
        "targetMeanPrice": info.get("targetMeanPrice"),
        "targetLowPrice": info.get("targetLowPrice"),
        "targetHighPrice": info.get("targetHighPrice"),
        "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "pegRatio": info.get("trailingPegRatio"),
        "recommendationKey": info.get("recommendationKey"),
        "marketCap": info.get("marketCap"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "profitMargins": info.get("profitMargins"),
        "revenueGrowth": info.get("revenueGrowth"),
        "earningsGrowth": info.get("earningsQuarterlyGrowth"),
        "returnOnEquity": info.get("returnOnEquity"),
        "debtToEquity": info.get("debtToEquity"),
        "beta": info.get("beta"),
        "dividendYield": info.get("dividendYield"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "averageVolume": info.get("averageVolume"),
    }
