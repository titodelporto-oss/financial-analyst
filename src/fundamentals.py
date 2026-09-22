"""Fundamental quality/risk scores computed from yfinance financial statements.

- Altman Z-Score: bankruptcy risk (higher = safer).
- Piotroski F-Score: fundamental strength, 0-9 (higher = stronger).
- Beneish M-Score: likelihood of earnings manipulation (higher/less negative = more suspicious).

All three need real balance sheet / income statement / cash flow data, which is not
available for every ticker (young companies, some ADRs, etc.) - functions return
None when a required line item is missing rather than guessing.
"""

from __future__ import annotations

import yfinance as yf


def _row(df, name: str):
    """Return a statement row (a pandas Series across fiscal years) or None if absent."""
    if df is None or df.empty or name not in df.index:
        return None
    return df.loc[name]


def _val(row, col_idx: int = 0):
    """Safely get the value at a given year offset (0 = most recent) from a statement row."""
    if row is None or col_idx >= len(row):
        return None
    v = row.iloc[col_idx]
    if v is None or v != v:  # NaN check
        return None
    return float(v)


class FinancialStatements:
    def __init__(self, ticker_obj: yf.Ticker, info: dict):
        self.balance_sheet = ticker_obj.balance_sheet
        self.income_stmt = ticker_obj.income_stmt
        self.cashflow = ticker_obj.cashflow
        self.info = info


def calculate_altman_z(fs: FinancialStatements) -> float | None:
    bs, inc = fs.balance_sheet, fs.income_stmt

    total_assets = _val(_row(bs, "Total Assets"))
    working_capital = _val(_row(bs, "Working Capital"))
    retained_earnings = _val(_row(bs, "Retained Earnings"))
    ebit = _val(_row(inc, "EBIT"))
    total_liabilities = _val(_row(bs, "Total Liabilities Net Minority Interest"))
    sales = _val(_row(inc, "Total Revenue"))
    market_cap = fs.info.get("marketCap")

    required = [total_assets, working_capital, retained_earnings, ebit, total_liabilities, sales, market_cap]
    if any(v is None for v in required) or total_assets == 0 or total_liabilities == 0:
        return None

    a = working_capital / total_assets
    b = retained_earnings / total_assets
    c = ebit / total_assets
    d = market_cap / total_liabilities
    e = sales / total_assets

    return 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e


def altman_zone(z: float | None) -> str:
    if z is None:
        return "N/D"
    if z > 2.99:
        return "Sicura"
    if z >= 1.81:
        return "Grigia"
    return "Distress"


def calculate_piotroski_f(fs: FinancialStatements) -> int | None:
    bs, inc, cf = fs.balance_sheet, fs.income_stmt, fs.cashflow

    total_assets = _row(bs, "Total Assets")
    net_income = _row(inc, "Net Income")
    op_cash_flow = _row(cf, "Operating Cash Flow")
    long_term_debt = _row(bs, "Long Term Debt")
    current_assets = _row(bs, "Current Assets")
    current_liabilities = _row(bs, "Current Liabilities")
    shares = _row(bs, "Ordinary Shares Number")
    gross_profit = _row(inc, "Gross Profit")
    revenue = _row(inc, "Total Revenue")

    # need at least 2 years of Total Assets, Net Income, Operating Cash Flow to score anything
    if total_assets is None or net_income is None or op_cash_flow is None:
        return None
    if len(total_assets) < 2 or len(net_income) < 2 or len(op_cash_flow) < 2:
        return None

    ta0, ta1 = _val(total_assets, 0), _val(total_assets, 1)
    ni0, ni1 = _val(net_income, 0), _val(net_income, 1)
    ocf0 = _val(op_cash_flow, 0)

    if None in (ta0, ta1, ni0, ni1, ocf0) or ta0 == 0 or ta1 == 0:
        return None

    roa0 = ni0 / ta0
    roa1 = ni1 / ta1

    score = 0

    # 1. Positive net income
    score += 1 if ni0 > 0 else 0
    # 2. Positive operating cash flow
    score += 1 if ocf0 > 0 else 0
    # 3. ROA improved year over year
    score += 1 if roa0 > roa1 else 0
    # 4. Operating cash flow exceeds net income (earnings quality)
    score += 1 if ocf0 > ni0 else 0

    # 5. Leverage decreased (long-term debt / total assets)
    ltd0, ltd1 = _val(long_term_debt, 0), _val(long_term_debt, 1)
    if ltd0 is not None and ltd1 is not None:
        score += 1 if (ltd0 / ta0) < (ltd1 / ta1) else 0

    # 6. Current ratio improved
    ca0, cl0 = _val(current_assets, 0), _val(current_liabilities, 0)
    ca1, cl1 = _val(current_assets, 1), _val(current_liabilities, 1)
    if None not in (ca0, cl0, ca1, cl1) and cl0 != 0 and cl1 != 0:
        score += 1 if (ca0 / cl0) > (ca1 / cl1) else 0

    # 7. No new shares issued (share count didn't increase)
    sh0, sh1 = _val(shares, 0), _val(shares, 1)
    if sh0 is not None and sh1 is not None:
        score += 1 if sh0 <= sh1 else 0

    # 8. Gross margin improved
    gp0, rev0 = _val(gross_profit, 0), _val(revenue, 0)
    gp1, rev1 = _val(gross_profit, 1), _val(revenue, 1)
    if None not in (gp0, rev0, gp1, rev1) and rev0 != 0 and rev1 != 0:
        score += 1 if (gp0 / rev0) > (gp1 / rev1) else 0

    # 9. Asset turnover improved
    if None not in (rev0, rev1) and ta0 != 0 and ta1 != 0:
        score += 1 if (rev0 / ta0) > (rev1 / ta1) else 0

    return score


def calculate_beneish_m(fs: FinancialStatements) -> float | None:
    bs, inc, cf = fs.balance_sheet, fs.income_stmt, fs.cashflow

    receivables = _row(bs, "Receivables")
    revenue = _row(inc, "Total Revenue")
    gross_profit = _row(inc, "Gross Profit")
    current_assets = _row(bs, "Current Assets")
    net_ppe = _row(bs, "Net PPE")
    total_assets = _row(bs, "Total Assets")
    depreciation = _row(cf, "Depreciation And Amortization")
    sga = _row(inc, "Selling General And Administration")
    net_income = _row(inc, "Net Income")
    op_cash_flow = _row(cf, "Operating Cash Flow")
    long_term_debt = _row(bs, "Long Term Debt")
    current_liabilities = _row(bs, "Current Liabilities")

    rows = [receivables, revenue, gross_profit, current_assets, net_ppe, total_assets,
            depreciation, sga, net_income, op_cash_flow, long_term_debt, current_liabilities]
    if any(r is None or len(r) < 2 for r in rows):
        return None

    def v(row, idx):
        return _val(row, idx)

    rec0, rec1 = v(receivables, 0), v(receivables, 1)
    rev0, rev1 = v(revenue, 0), v(revenue, 1)
    gp0, gp1 = v(gross_profit, 0), v(gross_profit, 1)
    ca0, ca1 = v(current_assets, 0), v(current_assets, 1)
    ppe0, ppe1 = v(net_ppe, 0), v(net_ppe, 1)
    ta0, ta1 = v(total_assets, 0), v(total_assets, 1)
    dep0, dep1 = v(depreciation, 0), v(depreciation, 1)
    sga0, sga1 = v(sga, 0), v(sga, 1)
    ni0 = v(net_income, 0)
    ocf0 = v(op_cash_flow, 0)
    ltd0, ltd1 = v(long_term_debt, 0), v(long_term_debt, 1)
    cl0, cl1 = v(current_liabilities, 0), v(current_liabilities, 1)

    required = [rec0, rec1, rev0, rev1, gp0, gp1, ca0, ca1, ppe0, ppe1, ta0, ta1,
                dep0, dep1, sga0, sga1, ni0, ocf0, ltd0, ltd1, cl0, cl1]
    if any(x is None for x in required) or 0 in (rev0, rev1, rec1, gp1, ta0, ta1, sga1, (dep0 + ppe0) or 1):
        return None

    try:
        dsri = (rec0 / rev0) / (rec1 / rev1)
        gmi = (gp1 / rev1) / (gp0 / rev0)
        aqi = (1 - (ca0 + ppe0) / ta0) / (1 - (ca1 + ppe1) / ta1)
        sgi = rev0 / rev1
        depi = (dep1 / (dep1 + ppe1)) / (dep0 / (dep0 + ppe0))
        sgai = (sga0 / rev0) / (sga1 / rev1)
        tata = (ni0 - ocf0) / ta0
        lvgi = ((ltd0 + cl0) / ta0) / ((ltd1 + cl1) / ta1)
    except ZeroDivisionError:
        return None

    m_score = (
        -4.84
        + 0.92 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )
    return m_score


def beneish_flag(m: float | None) -> str:
    if m is None:
        return "N/D"
    return "Possibile manipolazione" if m > -1.78 else "Nessun segnale"
