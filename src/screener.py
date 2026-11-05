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

from . import edgar, profiles
from .data import fetch_fundamentals, fetch_price_history
from .models import altman_z, beneish_m, piotroski_f
from .ratelimit import RateLimitGate
from .sector import classify
from .statements import fill_from, from_yahoo, recompute_derived, split_adjust
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
    piotroski_evaluable: int | None = None
    altman_variant: str = ""
    sector_category: str = ""
    # Raw Yahoo fields read by the "Analisi Azionaria" app to build sector peer groups
    peer_info: dict = field(default_factory=dict)
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


def _altman_score(zone: str) -> float:
    """Uses the zone, whose thresholds depend on the Altman variant (original vs Z'')."""
    if zone == "Sicura":
        return 1
    if zone == "Distress":
        return -2
    return 0


def _piotroski_score(f: int | None) -> float:
    if f is None:
        return 0
    if f >= 7:
        return 1
    if f <= 2:
        return -1
    return 0


def _beneish_score(flag: str) -> float:
    return -2 if flag == "Possibile manipolazione" else 0


def _market_value_at_fye(history, annual, shares_on_today_basis: bool = False) -> float | None:
    """Market cap on the last fiscal year end: price then x shares then.

    Yahoo's Close is retroactively divided by later stock splits while the share
    count in the old balance sheet is not, so later splits are multiplied back.
    SEC share counts already went through split_adjust (`shares_on_today_basis`):
    there the split-adjusted Close is used as it is."""
    if annual.empty or history.empty:
        return None
    fye = annual.index[-1]
    shares = annual["shares_outstanding"].iloc[-1]
    if shares != shares:  # NaN
        return None
    idx = history.index
    when = fye.tz_localize(idx.tz) if idx.tz is not None else fye
    before = history.loc[:when, "Close"].dropna()
    if before.empty or (when - before.index[-1]).days > 7:
        return None
    factor = 1.0
    if not shares_on_today_basis and "Stock Splits" in history.columns:
        later = history.loc[history.index > before.index[-1], "Stock Splits"]
        later = later[later > 0]
        if not later.empty:
            factor = float(later.prod())
    return float(before.iloc[-1]) * factor * float(shares)


PEER_INFO_FIELDS = (
    "sector", "industry", "marketCap", "currentPrice", "regularMarketPrice", "trailingPE", "forwardPE",
    "enterpriseToEbitda", "enterpriseToRevenue", "priceToBook", "trailingPegRatio",
    "dividendRate", "trailingAnnualDividendYield", "currency", "financialCurrency",
)

def _peer_info(info: dict, annual) -> dict:
    out = {k: info.get(k) for k in PEER_INFO_FIELDS if info.get(k) is not None}
    # free cash flow defined as operating cash flow - capex of the last annual report,
    # the same definition the analysis app uses (Yahoo's own "freeCashflow" differs)
    if not annual.empty and annual["fcf"].notna().iloc[-1]:
        out["fcfAnnual"] = float(annual["fcf"].iloc[-1])
    return out


def load_statements(ticker: str, ticker_obj, history, info: dict | None = None):
    """Bilanci annuali: dalla SEC (ufficiali, gratuiti, senza blocchi); da Yahoo solo se la SEC
    non li ha (società estere che non depositano il 10-K, ticker non trovato, errore della SEC).
    Le singole voci che mancano alla SEC si prendono da Yahoo solo se servono (fill_missing_from_yahoo).

    Restituisce (bilanci, codice SIC, descrizione SIC, True se dalla SEC)."""
    info = info or {}
    if not edgar.user_agent():
        reason = "SEC_USER_AGENT mancante"
    else:
        try:
            cik = edgar.get_cik(ticker)
            if not cik:
                reason = "ticker non trovato alla SEC"
            else:
                annual = edgar.parse_annual_statements(edgar.fetch_company_facts(cik))
                if annual.empty:
                    reason = "nessun bilancio annuale (10-K) alla SEC"
                else:
                    sic, sic_description = None, None
                    try:
                        sub = edgar.fetch_submissions(cik)
                        sic, sic_description = sub.get("sic") or None, sub.get("sicDescription")
                    except Exception:  # noqa: BLE001 - senza SIC si usa il settore di Yahoo
                        pass
                    # split dallo storico prezzi già scaricato (2 anni: copre i bilanci usati dai modelli)
                    splits = history["Stock Splits"] if "Stock Splits" in history.columns else None
                    annual = drop_share_unit_errors(split_adjust(annual, splits), ticker,
                                                    reference=info.get("sharesOutstanding"))
                    sector = classify(sic, sic_description, info.get("sector"), info.get("industry"))
                    annual = fill_missing_from_yahoo(annual, ticker, ticker_obj, sector, splits)
                    return annual, sic, sic_description, True
        except Exception as exc:  # noqa: BLE001
            reason = f"errore SEC ({' '.join(str(exc).split())[:120]})"
    print(f"  BILANCI DA YAHOO {ticker}: {reason}", flush=True)
    annual = from_yahoo(ticker_obj.balance_sheet, ticker_obj.income_stmt, ticker_obj.cashflow)
    return annual, None, None, False


SHARE_JUMP = 50  # azioni oltre 50 volte (o sotto 1/50) il riferimento: errore di unità


def drop_share_unit_errors(annual, ticker: str = "", reference: float | None = None):
    """Numero di azioni con un errore di unità nei dati SEC (PKG 2025: 89 miliardi invece di 89 milioni;
    RTX 2011-2021: in migliaia; PATTY D36): l'anno sospetto diventa N/D invece di falsare Altman e
    Piotroski. Riferimento: le azioni di oggi secondo Yahoo (`reference`, dalla richiesta già fatta),
    altrimenti la mediana degli ultimi 3 anni. Gli split sono già stati applicati prima."""
    sh = annual["shares_outstanding"]
    typical = reference if reference and reference > 0 else sh[sh > 0].tail(3).median()
    if typical != typical:  # nessun dato
        return annual
    ratio = sh / typical
    bad = (ratio > SHARE_JUMP) | (ratio < 1 / SHARE_JUMP)
    if not bad.any():
        return annual
    annual = annual.copy()
    annual.loc[bad, "shares_outstanding"] = float("nan")
    years = ", ".join(str(d.year) for d in annual.index[bad])
    print(f"  AZIONI SOSPETTE {ticker}: numero di azioni {years} lontano dal valore tipico, considerato N/D", flush=True)
    return annual


# voci usate da ciascun modello, negli ultimi due esercizi (Beneish confronta l'anno t con t-1)
ALTMAN_FIELDS = ("total_assets", "current_assets", "current_liabilities", "retained_earnings", "ebit",
                 "total_liabilities", "revenue", "equity", "shares_outstanding")
BENEISH_FIELDS = ("receivables", "revenue", "gross_profit", "current_assets", "net_ppe", "total_assets",
                  "d_and_a", "sga", "long_term_debt", "current_liabilities", "ocf", "net_income")
# in quale prospetto di Yahoo si trova ogni voce: si scarica solo quello che serve
YAHOO_STATEMENT = {
    "income_stmt": ("revenue", "gross_profit", "ebit", "sga", "net_income"),
    "cashflow": ("d_and_a", "ocf"),
}  # tutte le altre voci: stato patrimoniale ("balance_sheet")


def fill_missing_from_yahoo(annual, ticker: str, ticker_obj, sector, splits=None):
    """Voci che mancano alla SEC negli ultimi due esercizi: prese da Yahoo, solo se servono.

    - solo per i modelli che valgono per questo tipo di azienda (banche, REIT e utility: nessuna);
    - si scarica solo il prospetto Yahoo che contiene la voce (1-3 richieste, di solito 1);
    - per una voce mancante si prendono da Yahoo *entrambi* gli anni, così il confronto t / t-1 di
      Beneish non mescola due fonti; se Yahoo non li ha tutti e due, la voce resta com'era;
    - se Yahoo non risponde (blocco), il titolo non fallisce: la voce resta N/D e lo dice il log."""
    if annual.empty or sector.is_financial:
        return annual
    rules = profiles.get(sector.profile)
    wanted = set()
    if rules["altman"]:
        wanted.update(ALTMAN_FIELDS)
    if rules["beneish"]:
        wanted.update(BENEISH_FIELDS)
    last2 = annual.index[-2:]
    missing = sorted(f for f in wanted if annual.loc[last2, f].isna().any())
    if not missing:
        return annual

    needed_statements = {st for st, fields in YAHOO_STATEMENT.items() if set(fields) & set(missing)}
    other = [f for f in missing if not any(f in v for v in YAHOO_STATEMENT.values())]
    if other:
        needed_statements.add("balance_sheet")
    try:
        frames = {st: getattr(ticker_obj, st) for st in ("balance_sheet", "income_stmt", "cashflow")
                  if st in needed_statements}
        yahoo = from_yahoo(frames.get("balance_sheet"), frames.get("income_stmt"), frames.get("cashflow"),
                           require_core=False)
    except Exception as exc:  # noqa: BLE001 - un blocco di Yahoo non deve far fallire il titolo
        print(f"  VOCI DA YAHOO NON DISPONIBILI {ticker} ({' '.join(str(exc).split())[:80]}): "
              f"restano N/D {', '.join(missing)}", flush=True)
        return annual

    out, filled = annual.copy(), []
    for f in missing:
        if f not in yahoo or yahoo[f].dropna().empty:
            continue
        both = fill_from(out.loc[last2, [f]].assign(**{f: float("nan")}), yahoo, f)[f]
        if both.notna().all():  # Yahoo ha tutti e due gli anni: si usano i suoi, per entrambi
            out.loc[last2, f] = both
            filled.append(f)
            if f == "shares_outstanding":  # come nel percorso Yahoo: split dopo la fine dell'esercizio
                out.loc[last2, "filed_shares_outstanding"] = last2
    if "shares_outstanding" in filled:
        out.loc[last2, "shares_outstanding"] = split_adjust(out.loc[last2], splits)["shares_outstanding"]
    if any(f in filled for f in ("current_assets", "current_liabilities", "ebit", "d_and_a", "ocf")):
        for derived in ("working_capital", "ebitda", "fcf"):
            out.loc[last2, derived] = float("nan")
        out = recompute_derived(out)
    still = [f for f in missing if f not in filled]
    print(f"  VOCI DA YAHOO {ticker}: {', '.join(filled) or 'nessuna'}"
          + (f"; restano N/D {', '.join(still)}" if still else ""), flush=True)
    return out


RETRYABLE_MARKERS = ("Invalid Crumb", "Too Many Requests", "429", "401")

# usato quando il titolo non viene (ri)provato perché il tempo massimo è finito durante un blocco;
# contiene "Too Many Requests", così il secondo giro dello screening lo riconosce come bloccato
STOPPED_BY_DEADLINE = "Too Many Requests: non analizzato, tempo massimo finito durante il blocco di Yahoo"


def is_rate_limited(error: str | None) -> bool:
    """True se l'errore viene dal blocco di Yahoo (troppe richieste) e quindi vale la pena ritentare."""
    return error is not None and any(m in error for m in RETRYABLE_MARKERS)


def analyze_ticker(ticker: str, retries: int = 3, gate: RateLimitGate | None = None) -> TickerAnalysis:
    """Analizza un titolo, ritentando se Yahoo blocca le richieste.

    Senza `gate` (il sito Streamlit) si comporta come sempre: brevi attese di 1,5 / 3 / 4,5 s.
    Con `gate` (lo screening notturno) le attese sono comuni a tutti i titoli e più lunghe."""
    if gate is not None:
        return _analyze_with_gate(ticker, retries, gate)
    last_error = None
    for attempt in range(retries):
        result = _analyze_ticker_once(ticker)
        if not is_rate_limited(result.error):
            return result
        last_error = result.error
        time.sleep(1.5 * (attempt + 1))  # backoff before retrying a rate-limited call
    return _empty(ticker, last_error)


def _analyze_with_gate(ticker: str, retries: int, gate: RateLimitGate) -> TickerAnalysis:
    last_error = None
    for _ in range(retries):
        if not gate.wait():  # il tempo massimo è finito (o finirebbe durante la pausa): niente richieste
            return _empty(ticker, last_error or STOPPED_BY_DEADLINE)
        result = _analyze_ticker_once(ticker)
        if not is_rate_limited(result.error):
            gate.report_ok()
            return result
        last_error = result.error
        gate.report_limit()
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

        annual, sic, sic_description, from_sec = load_statements(ticker, ticker_obj, history, info)
        sector = classify(sic, sic_description, info.get("sector"), info.get("industry"))
        altman = altman_z(annual, sector, _market_value_at_fye(history, annual, shares_on_today_basis=from_sec))
        piotroski = piotroski_f(annual, sector)
        beneish = beneish_m(annual, sector)
        f_score = int(piotroski.value) if piotroski.available else None

        rsi_zone, rsi_score = _rsi_zone_and_score(latest_rsi)
        macd_signal, macd_score = _macd_signal_and_score(macd_df["histogram"])
        volume_score = _volume_score(latest_vol_ratio)
        upside_pct, fair_value_score = _fair_value_upside_and_score(current_price, target_mean_price)

        total_score = (
            rsi_score
            + macd_score
            + volume_score
            + fair_value_score
            + _altman_score(altman.label)
            + _piotroski_score(f_score)
            + _beneish_score(beneish.label)
        )

        ta = TickerAnalysis(
            ticker=ticker,
            name=fundamentals.get("shortName"),
            price=current_price,
            market_cap=info.get("marketCap"),
            rsi=float(latest_rsi) if latest_rsi is not None else None,
            rsi_zone=rsi_zone,
            macd_signal=macd_signal,
            volume_ratio=float(latest_vol_ratio) if latest_vol_ratio is not None else None,
            target_upside_pct=upside_pct,
            altman_z=altman.value,
            altman_zone=altman.label,
            piotroski_f=f_score,
            beneish_m=beneish.value,
            beneish_flag=beneish.label,
            score=total_score,
            piotroski_evaluable=sum(1 for c in piotroski.components if c["Esito"] != "N/D") or None,
            altman_variant=altman.variant,
            sector_category=sector.category,
            peer_info=_peer_info(info, annual),
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
