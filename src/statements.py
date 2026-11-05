"""Canonical annual financial statements, independent of the data source.

Every model (Altman, Piotroski, Beneish, and later DCF/ROIC...) reads from one
DataFrame with:
- index: fiscal year end date (pd.Timestamp), sorted oldest -> newest
- columns: the canonical FIELDS below, plus "filed" (date the figures became public,
  NaT when the source doesn't say)

Rows are matched by *date*, never by column position, so a balance sheet with five
years and an income statement with four can't get silently misaligned. Outflows
(capex, buybacks, dividends) are stored as positive amounts regardless of source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FLOW_FIELDS = [
    "revenue", "cost_of_revenue", "gross_profit", "operating_income", "ebit",
    "pretax_income", "tax_provision", "interest_expense", "net_income",
    "net_income_cont_ops", "sga", "d_and_a", "ocf", "capex", "sbc", "buyback",
    "dividends_paid",
]
INSTANT_FIELDS = [
    "total_assets", "current_assets", "current_liabilities", "total_liabilities",
    "equity", "retained_earnings", "receivables", "net_ppe", "long_term_debt",
    "total_debt", "cash", "short_term_investments", "inventory", "shares_outstanding", "tangible_book",
]
DERIVED_FIELDS = ["working_capital", "ebitda", "fcf"]
FIELDS = FLOW_FIELDS + INSTANT_FIELDS + DERIVED_FIELDS

OUTFLOW_FIELDS = ("capex", "buyback", "dividends_paid")

# figures that a stock split changes; their own filing date is kept (SEC data, see split_adjust)
SPLIT_SENSITIVE = ("shares_outstanding",)

# Plain-language labels for the UI
FIELD_LABELS = {
    "revenue": "Ricavi", "cost_of_revenue": "Costo del venduto", "gross_profit": "Utile lordo",
    "operating_income": "Reddito operativo", "ebit": "EBIT", "pretax_income": "Utile ante imposte",
    "tax_provision": "Imposte", "interest_expense": "Interessi passivi", "net_income": "Utile netto",
    "net_income_cont_ops": "Utile netto attività in funzionamento", "sga": "Spese generali (SG&A)",
    "d_and_a": "Ammortamenti", "ocf": "Flusso di cassa operativo", "capex": "Capex",
    "sbc": "Compensi in azioni (SBC)", "buyback": "Riacquisto azioni", "dividends_paid": "Dividendi pagati",
    "total_assets": "Totale attivo", "current_assets": "Attivo corrente",
    "current_liabilities": "Passivo corrente", "total_liabilities": "Totale passività",
    "equity": "Patrimonio netto", "retained_earnings": "Utili non distribuiti",
    "receivables": "Crediti commerciali", "net_ppe": "Immobilizzazioni materiali nette",
    "long_term_debt": "Debito a lungo termine", "total_debt": "Debito totale", "cash": "Cassa",
    "short_term_investments": "Investimenti a breve termine", "ebitda": "EBITDA",
    "fcf": "Free cash flow (cassa operativa - capex)",
    "inventory": "Magazzino", "shares_outstanding": "Azioni in circolazione",
    "tangible_book": "Patrimonio tangibile", "working_capital": "Capitale circolante",
}


@dataclass
class Statements:
    annual: pd.DataFrame
    source: str
    notes: list = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return self.annual is None or self.annual.empty


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FIELDS + ["filed"], dtype=float)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Shared post-processing: column order, sign conventions, derived fields."""
    for col in FIELDS:
        if col not in df.columns:
            df[col] = np.nan
    if "filed" not in df.columns:
        df["filed"] = pd.NaT
    for col in OUTFLOW_FIELDS:
        df[col] = df[col].abs()
    df["gross_profit"] = df["gross_profit"].fillna(df["revenue"] - df["cost_of_revenue"])
    # operating income first: pretax income + interest also contains gains on investments and other
    # non-operating items (PATTY audit D13, Alphabet 2025 +24%); it is only a fallback. The SEC has no
    # "EBIT" tag, so with SEC data this order decides the figure Altman uses.
    df["ebit"] = df["operating_income"].fillna(df["ebit"]).fillna(df["pretax_income"] + df["interest_expense"])
    df["working_capital"] = df["current_assets"] - df["current_liabilities"]
    df["ebitda"] = df["ebit"] + df["d_and_a"]
    df["fcf"] = df["ocf"] - df["capex"]
    extra = [f"filed_{f}" for f in SPLIT_SENSITIVE if f"filed_{f}" in df.columns]
    df = df[FIELDS + ["filed"] + extra]
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def fill_from(df: pd.DataFrame, other: pd.DataFrame, field: str, tolerance_days: int = 7) -> pd.DataFrame:
    """Fill the gaps of one field with another source's values for the same fiscal
    year ends (dates matched within a few days). Returns the filled frame. (Copy of PATTY's.)"""
    if other is None or other.empty or field not in other or other[field].dropna().empty:
        return df
    src = other[field].dropna().sort_index()
    aligned = src.reindex(df.index, method="nearest", tolerance=pd.Timedelta(days=tolerance_days))
    out = df.copy()
    out[field] = out[field].fillna(aligned)
    return out


def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Derived fields again, after some inputs were filled from another source."""
    df = df.copy()
    df["working_capital"] = df["working_capital"].fillna(df["current_assets"] - df["current_liabilities"])
    df["ebitda"] = df["ebitda"].fillna(df["ebit"] + df["d_and_a"])
    df["fcf"] = df["fcf"].fillna(df["ocf"] - df["capex"])
    return df


def split_adjust(annual: pd.DataFrame, splits: pd.Series) -> pd.DataFrame:
    """Put share counts reported before a stock split on today's basis (copy of PATTY's).

    SEC figures are kept as originally published: a 10:1 split after a 10-K was filed makes its
    share count 10x too small compared with later years (Piotroski would see a fake dilution) and
    with today's split-adjusted prices. Each figure is multiplied by the splits that happened
    *after* its own filing date (a split before filing is already reflected in the 10-K)."""
    df = annual.copy()
    if splits is None or len(splits) == 0 or df.empty:
        return df
    s = splits[splits > 0]
    idx = pd.to_datetime(s.index)
    s.index = idx.tz_localize(None) if idx.tz is not None else idx

    def factor(filed):
        if pd.isna(filed):
            return 1.0  # unknown filing date: nothing to adjust
        later = s[s.index > filed]
        return float(later.prod()) if len(later) else 1.0

    for col in SPLIT_SENSITIVE:
        dates = df[f"filed_{col}"] if f"filed_{col}" in df.columns else df["filed"]
        df[col] = df[col] * dates.map(factor)
    return df


# yfinance row names, in order of preference, for each canonical field
YAHOO_ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cost_of_revenue": ["Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income"],
    "ebit": ["EBIT"],
    "pretax_income": ["Pretax Income"],
    "tax_provision": ["Tax Provision"],
    "interest_expense": ["Interest Expense"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "net_income_cont_ops": ["Net Income Continuous Operations"],
    "sga": ["Selling General And Administration"],
    "d_and_a": ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Reconciled Depreciation"],
    "ocf": ["Operating Cash Flow"],
    "capex": ["Capital Expenditure"],
    "sbc": ["Stock Based Compensation"],
    "buyback": ["Repurchase Of Capital Stock"],
    "dividends_paid": ["Cash Dividends Paid"],
    "total_assets": ["Total Assets"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "equity": ["Stockholders Equity", "Common Stock Equity"],
    "retained_earnings": ["Retained Earnings"],
    "receivables": ["Accounts Receivable", "Receivables"],
    "net_ppe": ["Net PPE"],
    "long_term_debt": ["Long Term Debt"],
    "total_debt": ["Total Debt"],
    "cash": ["Cash And Cash Equivalents"],
    "short_term_investments": ["Other Short Term Investments"],
    "inventory": ["Inventory"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
    "tangible_book": ["Tangible Book Value"],
}


def from_yahoo(balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame, cashflow: pd.DataFrame,
               require_core: bool = True) -> pd.DataFrame:
    """Merge yfinance's three annual statements (rows = line items, columns = dates)
    into the canonical frame, aligned by fiscal year end date.
    require_core=False keeps every year: used when only one statement was downloaded to fill gaps."""
    frames = [f for f in (balance_sheet, income_stmt, cashflow) if f is not None and not f.empty]
    if not frames:
        return empty_frame()

    dates = sorted({pd.Timestamp(c) for f in frames for c in f.columns})
    out = pd.DataFrame(index=pd.DatetimeIndex(dates), dtype=float)
    for field_name, candidates in YAHOO_ROWS.items():
        series = pd.Series(np.nan, index=out.index)
        for name in candidates:
            for f in frames:
                if name in f.index:
                    row = pd.to_numeric(f.loc[name], errors="coerce")
                    row.index = pd.to_datetime(row.index)
                    series = series.fillna(row.reindex(out.index))
        out[field_name] = series

    # A missing "Long Term Debt" row is ambiguous; only treat it as zero when Yahoo
    # explicitly reports zero total debt for that year.
    no_debt = out["long_term_debt"].isna() & (out["total_debt"] == 0)
    out.loc[no_debt, "long_term_debt"] = 0.0

    # yfinance pads the oldest year with an almost-empty column: drop years with no core data
    core = ["revenue", "total_assets", "net_income"]
    if require_core:
        out = out[out[core].notna().any(axis=1)]
    return finalize(out)


def consecutive(prev: pd.Timestamp, cur: pd.Timestamp) -> bool:
    """True when two fiscal year ends are one year apart (52/53-week years allowed)."""
    return 330 <= (cur - prev).days <= 400


def value(row: pd.Series, name: str):
    v = row.get(name)
    if v is None or pd.isna(v):
        return None
    return float(v)
