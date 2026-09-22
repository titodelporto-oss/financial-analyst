import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.screener import screen_tickers

st.set_page_config(page_title="Analista Azionario", layout="wide")

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "latest_screening.json"

st.title("📈 Analista Azionario")
st.caption(
    "Screening automatico di tutto il mercato USA basato su RSI, MACD, volume, "
    "target price medio degli analisti, Altman Z-Score, Piotroski F-Score e Beneish "
    "M-Score. Non è consulenza finanziaria: è un aiuto per restringere la ricerca."
)


def _rows_from_results(results: list) -> list[dict]:
    rows = []
    for r in results:
        rows.append(
            {
                "Ticker": r["ticker"] if isinstance(r, dict) else r.ticker,
                "Nome": (r["name"] if isinstance(r, dict) else r.name) or "-",
                "Prezzo": _round(r, "price"),
                "Market Cap": _fmt_cap(r),
                "RSI": _round(r, "rsi", 1),
                "Zona RSI": _get(r, "rsi_zone"),
                "Segnale MACD": _get(r, "macd_signal"),
                "Volume vs media": _round(r, "volume_ratio"),
                "Upside target analisti %": _round(r, "target_upside_pct", 1),
                "Altman Z": _round(r, "altman_z"),
                "Zona Altman": _get(r, "altman_zone"),
                "Piotroski F": _get(r, "piotroski_f"),
                "Beneish M": _round(r, "beneish_m"),
                "Rischio manipolazione": _get(r, "beneish_flag"),
                "Punteggio": _get(r, "score"),
                "Errore": _get(r, "error") or "",
            }
        )
    return rows


def _get(r, field):
    return r[field] if isinstance(r, dict) else getattr(r, field)


def _round(r, field, digits=2):
    v = _get(r, field)
    return round(v, digits) if v is not None else None


def _fmt_cap(r):
    v = _get(r, "market_cap")
    return f"{v:,.0f}" if v else None


tab_auto, tab_manual = st.tabs(["🌐 Screening automatico (tutto il mercato)", "🔍 Analisi manuale"])

with tab_auto:
    if not CACHE_PATH.exists():
        st.info(
            "Nessuno screening ancora eseguito. Lancia `python3 run_screening.py` "
            "oppure attendi il prossimo giro programmato."
        )
    else:
        data = json.loads(CACHE_PATH.read_text())
        generated_at = datetime.fromisoformat(data["generated_at"])
        st.caption(
            f"Ultimo aggiornamento: {generated_at.strftime('%d/%m/%Y %H:%M UTC')} — "
            f"{data['analyzed_ok']}/{data['universe_size']} titoli analizzati "
            f"({data['errors']} con errori/dati mancanti) in {data['elapsed_seconds']/60:.1f} min."
        )

        min_score = st.slider("Punteggio minimo da mostrare", -10, 15, 3)
        df = pd.DataFrame(_rows_from_results(data["results"]))
        df = df[df["Punteggio"] >= min_score].sort_values("Punteggio", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab_manual:
    default_tickers = "AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA"
    tickers_input = st.text_input(
        "Elenco titoli (separati da virgola, ticker USA es. AAPL)", value=default_tickers
    )

    if st.button("Analizza", type="primary"):
        tickers = tickers_input.split(",")
        with st.spinner("Scarico dati e calcolo indicatori..."):
            results = screen_tickers(tickers)

        df = pd.DataFrame(_rows_from_results(results))
        st.dataframe(df, use_container_width=True, hide_index=True)

st.caption(
    "Punteggio più alto = più segnali positivi allineati (RSI sano/ipervenduto, MACD "
    "rialzista, volume sopra media, target price sopra al prezzo attuale, bilancio "
    "solido secondo Altman/Piotroski, nessun segnale di manipolazione contabile). "
    "Non garantisce risultati futuri."
)
