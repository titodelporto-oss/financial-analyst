"""Build the list of tradable US common-stock tickers from NASDAQ's official symbol directory.

Source: nasdaqtrader.com/dynamic/SymDir (public, no API key needed). Cached locally
for a day since the exchange listing barely changes intraday - no need to
re-download it on every screening run.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "us_universe.csv"
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60

EXCLUDE_NAME_KEYWORDS = ("Warrant", "Right", "Unit", "Preferred", "Notes")


def _download(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    lines = [l for l in text.splitlines() if not l.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|")


def _build_universe() -> pd.DataFrame:
    nasdaq = _download(NASDAQ_LISTED_URL)
    nasdaq = nasdaq.rename(columns={"Symbol": "symbol", "Security Name": "name", "ETF": "etf", "Test Issue": "test"})
    nasdaq = nasdaq[["symbol", "name", "etf", "test"]]

    other = _download(OTHER_LISTED_URL)
    other = other.rename(
        columns={"ACT Symbol": "symbol", "Security Name": "name", "ETF": "etf", "Test Issue": "test"}
    )
    other = other[["symbol", "name", "etf", "test"]]

    combined = pd.concat([nasdaq, other], ignore_index=True)
    combined = combined[combined["test"] != "Y"]
    combined = combined[combined["etf"] != "Y"]
    combined = combined[~combined["name"].str.contains("|".join(EXCLUDE_NAME_KEYWORDS), case=False, na=False)]
    combined = combined[combined["symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]
    combined = combined.drop_duplicates(subset="symbol").sort_values("symbol").reset_index(drop=True)
    return combined[["symbol", "name"]]


def get_us_ticker_universe(force_refresh: bool = False) -> list[str]:
    """Every tradable US common stock (~5800 tickers) - too many for free-tier data
    sources to handle reliably in a daily batch. Kept for reference / future use with
    a paid data provider. Use get_core_universe() for the actual daily screening."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh and CACHE_PATH.exists():
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < CACHE_MAX_AGE_SECONDS:
            return pd.read_csv(CACHE_PATH)["symbol"].tolist()

    df = _build_universe()
    df.to_csv(CACHE_PATH, index=False)
    return df["symbol"].tolist()


CORE_CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "core_universe.csv"

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"


def _clean_symbol(symbol: str) -> str:
    return symbol.strip().replace(".", "-")


def _fetch_wiki_tables(url: str) -> list[pd.DataFrame]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; financial-analyst-app/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def _fetch_sp500_symbols() -> list[str]:
    tables = _fetch_wiki_tables(SP500_WIKI_URL)
    df = tables[0]
    return [_clean_symbol(s) for s in df["Symbol"].tolist()]


def _fetch_nasdaq100_symbols() -> list[str]:
    tables = _fetch_wiki_tables(NASDAQ100_WIKI_URL)
    for t in tables:
        cols = [c.lower() for c in t.columns.astype(str)]
        if "ticker" in cols:
            col = t.columns[cols.index("ticker")]
            return [_clean_symbol(s) for s in t[col].tolist()]
    raise ValueError("Tabella Nasdaq-100 non trovata su Wikipedia")


def get_core_universe(force_refresh: bool = False) -> list[str]:
    """S&P 500 + Nasdaq 100 constituents (~550 unique tickers after overlap):
    a free, reliable universe that stays well within Yahoo Finance's rate limits
    while covering the large majority of stocks with real growth-analysis interest."""
    CORE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh and CORE_CACHE_PATH.exists():
        age = time.time() - CORE_CACHE_PATH.stat().st_mtime
        if age < CACHE_MAX_AGE_SECONDS:
            return pd.read_csv(CORE_CACHE_PATH)["symbol"].tolist()

    symbols = sorted(set(_fetch_sp500_symbols()) | set(_fetch_nasdaq100_symbols()))
    pd.DataFrame({"symbol": symbols}).to_csv(CORE_CACHE_PATH, index=False)
    return symbols
