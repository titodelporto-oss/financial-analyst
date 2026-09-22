"""Long-form analyst-style write-up for a triggered signal.

Everything here is derived from data actually fetched (price history + yfinance
`info` fundamentals) - no invented claims, no news. Paragraphs are skipped
gracefully when the underlying field is missing for a given ticker (common for
younger companies, foreign filers, or financials where some ratios don't apply).
"""

from __future__ import annotations

import pandas as pd

from .signals import _macd_momentum_direction, _stochastic_cross
from .indicators import detect_rsi_divergence


def _pct(x, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%" if x is not None else "n/d"


def _pct_already(x, digits: int = 1) -> str:
    return f"{x:.{digits}f}%" if x is not None else "n/d"


def _num(x, digits: int = 2) -> str:
    return f"{x:.{digits}f}" if x is not None else "n/d"


def build_narrative(
    ta, fundamentals: dict, history: pd.DataFrame, rsi: pd.Series,
    macd_df: pd.DataFrame, stoch: pd.DataFrame, signal_kind: str,
) -> list:
    close = history["Close"]
    current_price = float(close.iloc[-1])
    p = []  # paragraphs

    divergence = detect_rsi_divergence(close, rsi)
    stoch_cross = _stochastic_cross(stoch)
    macd_dir = _macd_momentum_direction(macd_df["histogram"])
    verb = "acquisto" if signal_kind == "BUY" else "vendita"
    direction_word = "rialzista" if signal_kind == "BUY" else "ribassista"

    # 1. Opening
    p.append(
        f"Il sistema ha generato un segnale di {verb} su {ta.name or ta.ticker} ({ta.ticker}) perché tre "
        f"indicatori tecnici pensati per anticipare il movimento (non per confermarlo a cose fatte, come fa "
        f"il solo MACD) puntano tutti nella stessa direzione {direction_word}, e il bilancio della società "
        f"supera i filtri minimi di qualità richiesti prima che un segnale venga mostrato."
    )

    # 2. Technical detail (RSI divergence)
    if divergence:
        kind_word = "rialzista" if divergence["type"] == "bullish" else "ribassista"
        if divergence["type"] == "bullish":
            p.append(
                f"Sul piano tecnico, l'RSI mostra una divergenza {kind_word}: negli ultimi giorni il prezzo "
                f"ha segnato un minimo più basso del precedente ({divergence['price_low_1']:.2f} contro "
                f"{divergence['price_low_2']:.2f}), ma l'RSI nello stesso intervallo ha fatto un minimo più "
                f"alto ({divergence['rsi_low_1']:.1f} contro {divergence['rsi_low_2']:.1f}). Questo scollamento "
                f"è tipicamente il primo segno che la pressione di vendita si sta esaurendo, e arriva prima "
                f"che il prezzo stesso lo confermi."
            )
        else:
            p.append(
                f"Sul piano tecnico, l'RSI mostra una divergenza {kind_word}: il prezzo ha segnato un massimo "
                f"più alto del precedente ({divergence['price_high_1']:.2f} contro {divergence['price_high_2']:.2f}), "
                f"ma l'RSI ha fatto un massimo più basso ({divergence['rsi_high_1']:.1f} contro "
                f"{divergence['rsi_high_2']:.1f}), segno che la spinta rialzista si sta esaurendo prima che il "
                f"prezzo lo mostri apertamente."
            )

    # 3. Stochastic
    if stoch_cross:
        zone = "ipervenduto" if stoch_cross == "bullish" else "ipercomprato"
        p.append(
            f"Lo Stocastico conferma: le linee %K e %D si sono appena incrociate uscendo dalla zona di "
            f"{zone}, un segnale che per costruzione reagisce più rapidamente del MACD perché misura dove "
            f"il prezzo si trova rispetto al proprio range recente, invece di aspettare l'incrocio di due "
            f"medie mobili."
        )

    # 4. MACD momentum (not crossover)
    if macd_dir:
        trend_word = "accelerando al rialzo" if macd_dir == "rising" else "rallentando"
        p.append(
            f"Anche il MACD, pur essendo l'indicatore più lento del gruppo, dà un primo indizio: il suo "
            f"istogramma sta {trend_word} da tre sedute consecutive. Non è ancora l'incrocio delle medie "
            f"(quello arriverebbe più avanti, a movimento già iniziato) ma la variazione di pendenza che lo "
            f"precede tipicamente di qualche seduta."
        )

    # 5. Trend positioning: SMA50/200, 52-week range
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    if pd.notna(sma50) and pd.notna(sma200):
        cross_word = "sopra" if sma50 > sma200 else "sotto"
        p.append(
            f"Sul trend di fondo, la media mobile a 50 giorni (${sma50:.2f}) è {cross_word} quella a 200 "
            f"giorni (${sma200:.2f}): un contesto di {'medio termine costruttivo' if sma50 > sma200 else 'medio termine ancora debole, da monitorare'}, "
            f"che va letto insieme al segnale di breve termine descritto sopra, non al posto di esso."
        )
    week_high = fundamentals.get("fiftyTwoWeekHigh")
    week_low = fundamentals.get("fiftyTwoWeekLow")
    if week_high and week_low and week_high > week_low:
        position_pct = (current_price - week_low) / (week_high - week_low) * 100
        p.append(
            f"Il prezzo attuale (${current_price:.2f}) si trova al {position_pct:.0f}% del proprio range a "
            f"52 settimane (minimo ${week_low:.2f}, massimo ${week_high:.2f})."
        )

    # 6. Volume/liquidity
    avg_vol = fundamentals.get("averageVolume")
    if avg_vol and ta.volume_ratio:
        p.append(
            f"Il volume di scambio odierno è {ta.volume_ratio:.2f}x la sua media recente (volume medio "
            f"giornaliero storico: {avg_vol:,.0f} titoli): {'un interesse sopra la norma che accompagna il segnale' if ta.volume_ratio > 1.2 else 'un livello di scambi nella norma, senza particolare accelerazione'}."
        )

    # 7. Business & valuation context
    sector, industry = fundamentals.get("sector"), fundamentals.get("industry")
    if sector or industry:
        p.append(
            f"L'azienda opera nel settore {sector or 'n/d'}" + (f", segmento {industry}" if industry else "") + "."
        )
    trailing_pe, forward_pe = fundamentals.get("trailingPE"), fundamentals.get("forwardPE")
    if trailing_pe or forward_pe:
        pe_trend = ""
        if trailing_pe and forward_pe:
            pe_trend = (" — gli utili attesi in crescita fanno scendere il multiplo" if forward_pe < trailing_pe
                        else " — gli utili attesi in calo fanno salire il multiplo")
        p.append(
            f"Valutazione: P/E storico {_num(trailing_pe)}, P/E atteso {_num(forward_pe)}{pe_trend}."
        )
    rev_growth, earn_growth = fundamentals.get("revenueGrowth"), fundamentals.get("earningsGrowth")
    if rev_growth is not None or earn_growth is not None:
        p.append(
            f"Crescita: ricavi {_pct(rev_growth)} su base annua, utili trimestrali {_pct(earn_growth)}."
        )
    margins, roe = fundamentals.get("profitMargins"), fundamentals.get("returnOnEquity")
    if margins is not None or roe is not None:
        p.append(
            f"Redditività: margine netto {_pct(margins)}, ROE {_pct(roe)}."
        )
    debt_eq, beta = fundamentals.get("debtToEquity"), fundamentals.get("beta")
    if debt_eq is not None or beta is not None:
        beta_word = ""
        if beta is not None:
            beta_word = " (più volatile del mercato)" if beta > 1.2 else (" (meno volatile del mercato)" if beta < 0.8 else " (in linea col mercato)")
        p.append(
            f"Struttura finanziaria e rischio: rapporto debito/patrimonio {_num(debt_eq, 0)}, beta {_num(beta)}{beta_word}."
        )
    div_yield = fundamentals.get("dividendYield")
    if div_yield:
        p.append(f"Il titolo distribuisce un dividendo con rendimento del {_pct_already(div_yield)}.")

    # 8. Quality scores, explained
    p.append(
        f"Punteggio Piotroski F: {ta.piotroski_f}/9 — misura la solidità fondamentale su nove criteri "
        f"(redditività, flusso di cassa, leva finanziaria, liquidità ed efficienza); sopra 7 indica un "
        f"bilancio strutturalmente solido, non solo un buon momento di mercato."
    )
    p.append(
        f"Altman Z-Score: {_num(ta.altman_z)} (zona {ta.altman_zone.lower()}) — stima il rischio di "
        f"insolvenza combinando liquidità, redditività e leva; valori sopra 2,99 indicano un'azienda "
        f"finanziariamente solida nel breve-medio termine."
    )
    p.append(
        f"Beneish M-Score: {_num(ta.beneish_m)} ({ta.beneish_flag.lower()}) — non rileva segnali statistici "
        f"tipici della manipolazione contabile (crescita anomala dei crediti, margini o accantonamenti "
        f"fuori pattern)."
    )

    # 9. Analyst consensus
    n_analysts = fundamentals.get("numberOfAnalystOpinions")
    low, high = fundamentals.get("targetLowPrice"), fundamentals.get("targetHighPrice")
    if n_analysts:
        range_txt = f", range {low:.2f}-{high:.2f}" if low and high else ""
        p.append(
            f"{n_analysts} analisti coprono il titolo con un target price medio di "
            f"${fundamentals.get('targetMeanPrice', 0):.2f}{range_txt}, il {ta.target_upside_pct:.0f}% "
            f"{'sopra' if ta.target_upside_pct and ta.target_upside_pct > 0 else 'sotto'} il prezzo attuale."
        )

    # 10. Risk / invalidation
    if signal_kind == "BUY" and ta.stop_loss:
        p.append(
            f"Livello di invalidazione: una chiusura sotto ${ta.stop_loss:.2f} negherebbe l'impostazione "
            f"rialzista descritta sopra e andrebbe considerata come uscita dalla posizione, "
            f"indipendentemente da quanto sembrava solida la tesi al momento dell'ingresso."
        )
    p.append(
        "Questa è un'analisi quantitativa basata su dati storici e di bilancio: non incorpora notizie "
        "dell'ultima ora, eventi macroeconomici imminenti o annunci societari non ancora riflessi nei dati. "
        "Non è consulenza finanziaria — verifica sempre il contesto attuale prima di decidere."
    )

    return p
