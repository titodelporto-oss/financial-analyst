import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.charting import build_signal_chart
from src.screener import screen_tickers

st.set_page_config(page_title="Analista Azionario", layout="wide", page_icon="📈")

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "latest_screening.json"

st.markdown(
    """
    <style>
    .signal-card {
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 18px;
        border: 1px solid rgba(11,11,11,0.10);
    }
    .signal-card.buy { border-left: 5px solid #0ca30c; }
    .signal-card.sell { border-left: 5px solid #d03b3b; }
    .signal-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
    }
    .signal-badge.buy { background-color: #0ca30c; }
    .signal-badge.sell { background-color: #d03b3b; }
    .signal-ticker { font-size: 1.4rem; font-weight: 700; margin-left: 10px; }
    .signal-sub { color: #898781; font-size: 0.9rem; }
    .signal-reason { margin: 4px 0; }
    .levels-row { display: flex; gap: 28px; margin-top: 10px; margin-bottom: 6px; }
    .level-box { text-align: left; }
    .level-label { font-size: 0.78rem; color: #898781; text-transform: uppercase; }
    .level-value { font-size: 1.1rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Analista Azionario")
st.caption(
    "Segnali selettivi su S&P 500 + Nasdaq 100: uniscono indicatori tecnici che anticipano "
    "il movimento (divergenze RSI, incroci Stocastico, accelerazione MACD) a un filtro sui "
    "fondamentali (Altman Z-Score, Piotroski F-Score, Beneish M-Score, target price analisti). "
    "Non è consulenza finanziaria — è un supporto alla decisione: verifica sempre autonomamente."
)


def _signal_card(r: dict) -> None:
    kind = r["signal"].lower()
    label = "SEGNALE DI ACQUISTO" if r["signal"] == "BUY" else "SEGNALE DI VENDITA"

    st.markdown(f'<div class="signal-card {kind}">', unsafe_allow_html=True)
    st.markdown(
        f'<span class="signal-badge {kind}">{label}</span>'
        f'<span class="signal-ticker">{r["ticker"]}</span> '
        f'<span class="signal-sub">{r["name"] or ""}</span>',
        unsafe_allow_html=True,
    )

    levels = f"""
    <div class="levels-row">
        <div class="level-box"><div class="level-label">Prezzo attuale</div>
            <div class="level-value">${r['entry_price']:.2f}</div></div>
    """
    if r["signal"] == "BUY":
        if r.get("stop_loss"):
            levels += f"""<div class="level-box"><div class="level-label">Stop loss</div>
                <div class="level-value" style="color:#d03b3b">${r['stop_loss']:.2f}</div></div>"""
        if r.get("take_profit"):
            levels += f"""<div class="level-box"><div class="level-label">Take profit</div>
                <div class="level-value" style="color:#0ca30c">${r['take_profit']:.2f}</div></div>"""
    levels += "</div>"
    st.markdown(levels, unsafe_allow_html=True)

    st.markdown("**Perché:**")
    for reason in r["signal_reasons"]:
        st.markdown(f"- {reason}")

    if r.get("chart"):
        fig = build_signal_chart(r["ticker"], r["chart"], r["signal"], r["entry_price"], r.get("stop_loss"), r.get("take_profit"))
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{r['ticker']}")

    st.markdown("</div>", unsafe_allow_html=True)


tab_signals, tab_manual = st.tabs(["🎯 Segnali di oggi", "🔍 Analisi manuale"])

with tab_signals:
    if not CACHE_PATH.exists():
        st.info("Nessuno screening ancora eseguito.")
    else:
        data = json.loads(CACHE_PATH.read_text())
        generated_at = datetime.fromisoformat(data["generated_at"])
        st.caption(
            f"Ultimo aggiornamento: {generated_at.strftime('%d/%m/%Y %H:%M UTC')} — "
            f"{data['analyzed_ok']}/{data['universe_size']} titoli analizzati."
        )

        signals = [r for r in data["results"] if r.get("signal")]
        buys = [r for r in signals if r["signal"] == "BUY"]
        sells = [r for r in signals if r["signal"] == "SELL"]

        if not signals:
            st.info(
                "Nessun segnale ad alta convinzione oggi. Il sistema è volutamente selettivo: "
                "niente di interessante è meglio di un falso segnale."
            )
        else:
            if buys:
                st.subheader(f"🟢 Acquisto ({len(buys)})")
                for r in sorted(buys, key=lambda r: r["target_upside_pct"] or 0, reverse=True):
                    _signal_card(r)
            if sells:
                st.subheader(f"🔴 Vendita ({len(sells)})")
                for r in sells:
                    _signal_card(r)

with tab_manual:
    st.caption("Analizza un titolo specifico su richiesta (non filtrato per segnale).")
    default_tickers = "AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA"
    tickers_input = st.text_input(
        "Elenco titoli (separati da virgola, ticker USA es. AAPL)", value=default_tickers
    )

    if st.button("Analizza", type="primary"):
        tickers = tickers_input.split(",")
        with st.spinner("Scarico dati e calcolo indicatori..."):
            results = screen_tickers(tickers)

        rows = []
        for r in results:
            rows.append(
                {
                    "Ticker": r.ticker,
                    "Nome": r.name or "-",
                    "Prezzo": round(r.price, 2) if r.price else None,
                    "Segnale": r.signal or "-",
                    "RSI": round(r.rsi, 1) if r.rsi else None,
                    "Zona RSI": r.rsi_zone,
                    "Segnale MACD": r.macd_signal,
                    "Altman Z": round(r.altman_z, 2) if r.altman_z is not None else None,
                    "Piotroski F": r.piotroski_f,
                    "Upside target %": round(r.target_upside_pct, 1) if r.target_upside_pct is not None else None,
                    "Errore": r.error or "",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for r in results:
            if r.signal and r.chart:
                _signal_card(
                    {
                        "ticker": r.ticker, "name": r.name, "signal": r.signal,
                        "signal_reasons": r.signal_reasons, "entry_price": r.entry_price,
                        "stop_loss": r.stop_loss, "take_profit": r.take_profit, "chart": r.chart,
                    }
                )
