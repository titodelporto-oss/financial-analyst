import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.charting import build_signal_chart
from src.screener import screen_tickers

st.set_page_config(page_title="Analista Azionario", layout="wide")

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "latest_screening.json"

INK = "#111111"
INK_SECONDARY = "#4a4a4a"
BRAND = "#0d2f5e"  # dark navy masthead accent - kept separate from the green/red signal colors
GOOD = "#0a6b2d"
CRITICAL = "#a3231f"
RULE = "#111111"

st.markdown(
    f"""
    <style>
    html, body, [class*="st-"], .stMarkdown, .stText, p, span, div, table, th, td {{
        font-family: "Times New Roman", Times, Georgia, serif !important;
    }}
    .stApp {{ background-color: #f7f6f2; }}

    .masthead {{
        border-top: 4px double {RULE};
        border-bottom: 2px solid {RULE};
        padding: 14px 0 10px 0;
        margin-bottom: 6px;
        text-align: center;
    }}
    .masthead h1 {{
        font-family: "Times New Roman", Times, serif !important;
        font-weight: 700;
        letter-spacing: 2px;
        font-size: 2.4rem;
        color: {BRAND};
        margin: 0;
        text-transform: uppercase;
    }}
    .masthead .kicker {{
        font-style: italic;
        color: {INK_SECONDARY};
        font-size: 0.95rem;
        margin-top: 4px;
    }}

    .section-rule {{
        border-bottom: 2px solid {RULE};
        margin: 22px 0 14px 0;
        padding-bottom: 4px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 1.1rem;
    }}
    .section-rule.buy {{ color: {GOOD}; border-color: {GOOD}; }}
    .section-rule.sell {{ color: {CRITICAL}; border-color: {CRITICAL}; }}

    .signal-card {{
        border: 1px solid #cfcabf;
        background-color: #ffffff;
        padding: 20px 24px;
        margin-bottom: 24px;
    }}
    .signal-card.buy {{ border-top: 3px solid {GOOD}; }}
    .signal-card.sell {{ border-top: 3px solid {CRITICAL}; }}
    .signal-badge {{
        display: inline-block;
        padding: 2px 10px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 1px;
        border: 1px solid;
    }}
    .signal-badge.buy {{ color: {GOOD}; border-color: {GOOD}; }}
    .signal-badge.sell {{ color: {CRITICAL}; border-color: {CRITICAL}; }}
    .signal-ticker {{ font-size: 1.5rem; font-weight: 700; margin-left: 12px; color: {INK}; }}
    .signal-sub {{ color: {INK_SECONDARY}; font-size: 0.92rem; font-style: italic; }}
    .levels-row {{ display: flex; gap: 32px; margin-top: 12px; margin-bottom: 8px; }}
    .level-box {{ text-align: left; }}
    .level-label {{ font-size: 0.75rem; color: {INK_SECONDARY}; text-transform: uppercase; letter-spacing: 0.5px; }}
    .level-value {{ font-size: 1.15rem; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="masthead">
        <h1>Analista Azionario</h1>
        <div class="kicker">Screening quantitativo &middot; S&amp;P 500 + Nasdaq 100</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Segnali selettivi: indicatori tecnici che anticipano il movimento (divergenze RSI, "
    "incroci Stocastico, accelerazione MACD) combinati a un filtro sui fondamentali (Altman "
    "Z-Score, Piotroski F-Score, Beneish M-Score, target price degli analisti). "
    "Non è consulenza finanziaria — è un supporto alla decisione: verifica sempre autonomamente."
)


def _signal_card(r: dict) -> None:
    kind = r["signal"].lower()
    label = "ACQUISTO" if r["signal"] == "BUY" else "VENDITA"

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
                <div class="level-value" style="color:{CRITICAL}">${r['stop_loss']:.2f}</div></div>"""
        if r.get("take_profit"):
            levels += f"""<div class="level-box"><div class="level-label">Take profit</div>
                <div class="level-value" style="color:{GOOD}">${r['take_profit']:.2f}</div></div>"""
    levels += "</div>"
    st.markdown(levels, unsafe_allow_html=True)

    st.markdown("**Perché:**")
    for reason in r["signal_reasons"]:
        st.markdown(f"- {reason}")

    if r.get("chart"):
        fig = build_signal_chart(r["ticker"], r["chart"], r["signal"], r["entry_price"], r.get("stop_loss"), r.get("take_profit"))
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{r['ticker']}")

    st.markdown("</div>", unsafe_allow_html=True)


tab_signals, tab_manual = st.tabs(["SEGNALI DI OGGI", "ANALISI MANUALE"])

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
                st.markdown(f'<div class="section-rule buy">Acquisto — {len(buys)}</div>', unsafe_allow_html=True)
                for r in sorted(buys, key=lambda r: r["target_upside_pct"] or 0, reverse=True):
                    _signal_card(r)
            if sells:
                st.markdown(f'<div class="section-rule sell">Vendita — {len(sells)}</div>', unsafe_allow_html=True)
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
