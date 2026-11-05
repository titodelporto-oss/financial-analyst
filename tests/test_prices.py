"""Prezzi da Yahoo: le righe senza chiusura vengono scartate (nessuna richiesta vera)."""

import numpy as np
import pandas as pd

from src.data import fetch_price_history


def test_price_history_drops_rows_without_close():
    """Screening ufficiali di fine ottobre / novembre 2026: ultima chiusura NaN per tutti i titoli."""

    class T:
        def history(self, **kwargs):
            idx = pd.to_datetime(["2026-11-03", "2026-11-04", "2026-11-05"])
            return pd.DataFrame({"Close": [10.0, 11.0, np.nan], "Volume": [1, 2, np.nan]}, index=idx)

    h = fetch_price_history(T())
    assert len(h) == 2 and h["Close"].iloc[-1] == 11.0


def test_price_history_without_problems_is_unchanged():
    class T:
        def history(self, **kwargs):
            idx = pd.to_datetime(["2026-11-03", "2026-11-04"])
            return pd.DataFrame({"Close": [10.0, 11.0], "Volume": [1, 2]}, index=idx)

    assert len(fetch_price_history(T())) == 2
