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
    "total_debt", "cash", "inventory", "shares_outstanding", "tangible_book",
]
DERIVED_FIELDS = ["working_capital"]
FIELDS = FLOW_FIELDS + INSTANT_FIELDS + DERIVED_FIELDS

OUTFLOW_FIELDS = ("capex", "buyback", "dividends_paid")

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
    df["ebit"] = df["ebit"].fillna(df["pretax_income"] + df["interest_expense"]).fillna(df["operating_income"])
    df["working_capital"] = df["current_assets"] - df["current_liabilities"]
    df = df[FIELDS + ["filed"]]
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


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
    "inventory": ["Inventory"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
    "tangible_book": ["Tangible Book Value"],
}


def from_yahoo(balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame, cashflow: pd.DataFrame) -> pd.DataFrame:
    """Merge yfinance's three annual statements (rows = line items, columns = dates)
    into the canonical frame, aligned by fiscal year end date."""
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
