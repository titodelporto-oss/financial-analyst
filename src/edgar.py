"""SEC EDGAR: free, official US financial statements (XBRL "company facts").

Copia di ~/equity-analyst/src/edgar.py (PATTY) del 5/11/2026. Differenze: il contatto per la SEC
(User-Agent) si legge da SEC_USER_AGENT (variabile d'ambiente o .streamlit/secrets.toml), mai dal
codice; tutte le richieste passano da un limite comune di 8 al secondo, perché lo screening analizza
4 titoli alla volta. Le voci (SEC_TAGS) e le regole di lettura sono le stesse: se cambiano in PATTY,
vanno copiate anche qui.

Voci aggiunte rispetto a PATTY (5/11/2026, misurate sui 518 titoli dello screening):
- immobilizzazioni nette comprese quelle in leasing finanziario (dopo il principio ASC 842);
- crediti commerciali registrati insieme a cambiali e prestiti, o ad altri crediti;
- costo del venduto senza ammortamenti, solo come ultima scelta;
- SG&A = solo spese amministrative (più quelle di vendita, se ci sono) quando non c'è la voce unica;
- debito a lungo termine = debito totale a lungo meno la quota entro l'anno;
- passivo totale = passivo corrente + non corrente, oppure totale meno patrimonio e minoranze;
- EBIT = utile ante imposte meno interessi netti, quando mancano reddito operativo e interessi passivi.

Why EDGAR on top of Yahoo:
- 10-15 years of annual data instead of 4;
- the *filing date* of every figure, so a backtest only uses numbers once they were public;
- values *as originally reported*: for each fiscal year we keep the figure from the
  earliest 10-K that contained it, ignoring later restatements (point-in-time).

Only US filers (10-K) are covered. Requests must carry a User-Agent with a contact
email (see user_agent()) and stay under 10 requests/second.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .statements import FLOW_FIELDS, SPLIT_SENSITIVE, empty_frame, finalize

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

ANNUAL_FORMS = ("10-K", "10-K/A")

# us-gaap tags, in order of preference. Companies switch tags over the years (e.g.
# SalesRevenueNet -> RevenueFromContractWithCustomer... after ASC 606), so the
# fallback is applied year by year, not once per company.
SEC_TAGS = {
    # last: banks and brokers that publish only net revenue (Goldman Sachs, Wells Fargo)
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
                "RevenuesNetOfInterestExpense"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
                        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "tax_provision": ["IncomeTaxExpenseBenefit"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "net_income_cont_ops": ["IncomeLossFromContinuingOperations"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "d_and_a": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
                "DepreciationAndAmortization", "Depreciation"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "buyback": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends", "PaymentsOfOrdinaryDividends"],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent",
                    "AccountsNotesAndLoansReceivableNetCurrent", "AccountsAndOtherReceivablesNetCurrent"],
    "net_ppe": ["PropertyPlantAndEquipmentNet",
                "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
                "PropertyPlantAndEquipmentExcludingLessorAssetUnderOperatingLeaseAfterAccumulatedDepreciation"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "short_term_investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent",
                               "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "inventory": ["InventoryNet"],
    "shares_outstanding": ["CommonStockSharesOutstanding"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    # EPS without discontinued operations (Duke Energy 2022-23: losses on the renewables sold): used for growth
    "eps_cont_diluted": ["IncomeLossFromContinuingOperationsPerDilutedShare"],
    "goodwill": ["Goodwill"],
    "intangibles": ["IntangibleAssetsNetExcludingGoodwill"],
    "preferred_equity": ["PreferredStockValue", "PreferredStockLiquidationPreferenceValue",
                         "PreferredStockValueOutstanding"],
    "net_interest_income": ["InterestIncomeExpenseNet"],
    "noninterest_expense": ["NoninterestExpense"],
    "ni_common": ["NetIncomeLossAvailableToCommonStockholdersBasic"],
    "re_gains": ["GainsLossesOnSalesOfInvestmentRealEstate", "GainLossOnSaleOfRealEstateInvestmentProperty",
                 "GainLossOnSaleOfProperties"],
    "re_impairment": ["ImpairmentOfRealEstate"],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    # value of the shares issued to pay for acquisitions (equity statement): dilution from an acquisition
    "stock_issued_acquisitions": ["StockIssuedDuringPeriodValueAcquisitions"],
    # helpers used to derive total liabilities when "Liabilities" isn't tagged
    "_liabilities_and_equity": ["LiabilitiesAndStockholdersEquity"],
    "_equity_incl_nci": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    # some companies split SG&A in two lines (e.g. Microsoft)
    "_selling_marketing": ["SellingAndMarketingExpense"],
    "_general_admin": ["GeneralAndAdministrativeExpense"],
    "_selling_expense": ["SellingExpense"],
    # total liabilities, other ways (screening only)
    "_liabilities_noncurrent": ["LiabilitiesNoncurrent"],
    "_equity_parent": ["StockholdersEquity"],
    "_minority_interest": ["MinorityInterest"],
    # interest income - interest expense, outside operations (EBIT when interest expense is not tagged)
    "_interest_net_nonop": ["InterestIncomeExpenseNonoperatingNet"],
    # shares outstanding = issued - treasury, when not tagged directly (e.g. Coca-Cola)
    "_shares_issued": ["CommonStockSharesIssued"],
    "_treasury_shares": ["TreasuryStockCommonShares", "TreasuryStockShares"],
    # pieces of total financial debt (leases excluded)
    "_finite_intangibles": ["FiniteLivedIntangibleAssetsNet"],
    "_indefinite_intangibles": ["IndefiniteLivedIntangibleAssetsExcludingGoodwill"],
    "_debt_current": ["DebtCurrent"],
    "_ltd_current": ["LongTermDebtCurrent"],
    "_short_term_borrowings": ["ShortTermBorrowings", "CommercialPaper"],
    "_ltd_total": ["LongTermDebt"],
    # other ways companies tag their total debt (e.g. Oracle, GM, D.R. Horton): used only
    # when the tags above are missing, and only if not smaller than a known piece of debt
    "_debt_combined": ["DebtLongtermAndShorttermCombinedAmount",
                       "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities"],
    "_debt_instruments": ["DebtInstrumentCarryingAmount"],
    "_notes_payable": ["NotesPayable"],
    "_lt_notes": ["LongTermNotesPayable", "SeniorLongTermNotes", "UnsecuredLongTermDebt"],
    "_secured_debt": ["SecuredDebt"],
}
MAX_TAKEN_FIELDS = {"d_and_a"}  # tags disagree (MCD): the largest of the same 10-K is the cash-flow total
DA_DISAGREEMENT = 0.6  # smallest / largest below this -> note
DURATION_FIELDS = set(FLOW_FIELDS) | {"_selling_marketing", "_general_admin", "_selling_expense", "_interest_net_nonop"}
FYE_ANCHORS = ("revenue", "net_income", "ocf")  # tags whose 1-year periods define fiscal year ends


SECRETS_PATH = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
MIN_INTERVAL = 1 / 8  # secondi tra due richieste alla SEC (limite SEC: 10 al secondo)


def user_agent() -> str | None:
    """Contatto richiesto dalla SEC: variabile d'ambiente SEC_USER_AGENT (GitHub Actions),
    altrimenti la riga SEC_USER_AGENT di .streamlit/secrets.toml (Mac). None se manca."""
    value = os.environ.get("SEC_USER_AGENT", "").strip()
    if not value and SECRETS_PATH.exists():
        m = re.search(r'^\s*SEC_USER_AGENT\s*=\s*"([^"]*)"', SECRETS_PATH.read_text(), re.M)
        value = m.group(1).strip() if m else ""
    return value or None


_throttle_lock = threading.Lock()
_last_request = 0.0


def _get(url: str, timeout: int) -> requests.Response:
    """GET verso la SEC, rispettando il limite comune di richieste al secondo (anche tra thread)."""
    global _last_request
    agent = user_agent()
    if not agent:
        raise RuntimeError("SEC_USER_AGENT mancante: la SEC rifiuta le richieste senza un contatto")
    with _throttle_lock:
        wait = _last_request + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
    resp = requests.get(url, headers={"User-Agent": agent, "Accept-Encoding": "gzip, deflate"}, timeout=timeout)
    resp.raise_for_status()
    return resp


_directory: list | None = None
_directory_time = 0.0
_directory_lock = threading.Lock()


def ticker_directory() -> list:
    """Every SEC-registered ticker as {"ticker", "cik", "name"} (cached for a day)."""
    global _directory, _directory_time
    with _directory_lock:  # 4 titoli in parallelo: lo scarica uno solo
        if _directory is None or time.time() - _directory_time > 86400:
            resp = _get(TICKERS_URL, timeout=30)
            _directory = [{"ticker": row["ticker"].upper(), "cik": int(row["cik_str"]), "name": row["title"]}
                          for row in resp.json().values()]
            _directory_time = time.time()
        return _directory


def get_cik(ticker: str) -> int | None:
    """Ticker -> SEC CIK number, from the SEC's own mapping file."""
    by_ticker = {row["ticker"]: row["cik"] for row in ticker_directory()}
    t = ticker.upper()
    return by_ticker.get(t) or by_ticker.get(t.replace("-", "."))


def fetch_company_facts(cik: int) -> dict:
    return _get(FACTS_URL.format(cik=cik), timeout=60).json()


def fetch_submissions(cik: int) -> dict:
    """Company profile: name, SIC industry code, etc."""
    return _get(SUBMISSIONS_URL.format(cik=cik), timeout=30).json()


def _records(facts: dict, tag: str) -> pd.DataFrame:
    """All annual-report records for one us-gaap tag as a DataFrame."""
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag)
    if not node:
        return pd.DataFrame()
    rows = []
    for unit, items in node.get("units", {}).items():
        if unit not in ("USD", "shares", "USD/shares"):
            continue
        rows.extend(items)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["form"].isin(ANNUAL_FORMS)].copy()
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"])
        df["days"] = (df["end"] - df["start"]).dt.days
    else:
        df["days"] = np.nan
    return df


def _as_reported(df: pd.DataFrame, duration: bool) -> pd.DataFrame:
    """One value per period end: the one from the *earliest* filing (as originally published)."""
    if df.empty:
        return df
    if duration:
        df = df[(df["days"] >= 350) & (df["days"] <= 380)]  # full fiscal years only, no quarters
    else:
        df = df[df["days"].isna()]
    return df.sort_values("filed").drop_duplicates(subset="end", keep="first").set_index("end")


DEI_SHARES_MAX_DAYS = 180  # cover-page date: weeks after the fiscal year end, before the next one


def _dei_shares(facts: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Shares outstanding from the cover page of each 10-K (dei:EntityCommonStockSharesOutstanding), the
    total for companies that tag the balance-sheet count only by share class (audit D16: WMT, LLY...).
    The count is dated a few weeks after the fiscal year end: each fiscal year takes the first cover-page
    count after it. Columns "val" and "filed"."""
    node = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding")
    rows = [o for o in (node or {}).get("units", {}).get("shares", []) if o.get("form") in ANNUAL_FORMS]
    out = pd.DataFrame({"val": np.nan, "filed": pd.NaT}, index=index)
    if not rows:
        return out
    df = pd.DataFrame(rows)
    df["end"], df["filed"] = pd.to_datetime(df["end"]), pd.to_datetime(df["filed"])
    # one figure per cover date (classes reported separately are summed), as first published
    df["filing"] = df["accn"] if "accn" in df.columns else df["filed"].astype(str)
    df = (df.groupby(["filing", "end"]).agg(val=("val", "sum"), filed=("filed", "min")).reset_index()
          .sort_values("filed").drop_duplicates(subset="end", keep="first").sort_values("end"))
    for fye in index:
        after = df[(df["end"] >= fye) & ((df["end"] - fye).dt.days <= DEI_SHARES_MAX_DAYS)]
        if not after.empty:
            out.loc[fye, ["val", "filed"]] = [after["val"].iloc[0], after["filed"].iloc[0]]
    return out


def parse_annual_statements(facts: dict, max_years: int = 15, notes: list | None = None) -> pd.DataFrame:
    """Company-facts JSON -> canonical annual frame (see statements.py), point-in-time.
    Warnings about the data of the last fiscal year are appended to `notes`."""
    notes = [] if notes is None else notes
    per_field: dict[str, list[pd.DataFrame]] = {}
    for field_name, tags in SEC_TAGS.items():
        duration = field_name in DURATION_FIELDS
        per_field[field_name] = [_as_reported(_records(facts, tag), duration) for tag in tags]

    fye = set()
    for anchor in FYE_ANCHORS:
        for df in per_field[anchor]:
            if not df.empty:
                fye.update(df.index)
    if not fye:
        return empty_frame()

    index = pd.DatetimeIndex(sorted(fye))[-max_years:]
    out = pd.DataFrame(index=index, dtype=float)
    filed_by_field = {}
    for field_name, frames in per_field.items():
        # Across all tags, keep for each year the value filed *first* (tag order only
        # breaks ties): a company that renames a tag shouldn't make us pick the
        # figure re-published a year later in the next 10-K's comparative column.
        tagged = [df.assign(priority=i) for i, df in enumerate(frames) if not df.empty]
        if tagged:
            allrec = pd.concat(tagged).reset_index()
            if field_name in MAX_TAKEN_FIELDS:
                # among the tags of the *first* filing of each year, the largest value
                first = allrec.groupby("end")["filed"].transform("min")
                same = allrec[allrec["filed"] == first]
                best = (same.sort_values(["val", "priority"], ascending=[False, True])
                        .drop_duplicates(subset="end", keep="first").set_index("end").sort_index())
                spread = same.groupby("end")["val"].agg(lambda v: v.min() / v.max() if v.max() > 0 else 1.0)
                if len(index) and spread.reindex(index).iloc[-1] < DA_DISAGREEMENT:
                    notes.append("Ammortamenti: le voci SEC dell'ultimo esercizio non concordano (differenza oltre "
                                 "il 40%); usato il valore più alto, quello del rendiconto finanziario.")
            else:
                best = (allrec.sort_values(["filed", "priority"])
                        .drop_duplicates(subset="end", keep="first").set_index("end").sort_index())
            if field_name in DURATION_FIELDS:
                aligned = best.reindex(index)
            else:
                # balance-sheet dates are sometimes tagged a few days off the fiscal year end
                aligned = best.reindex(index, method="nearest", tolerance=pd.Timedelta(days=7))
            out[field_name] = aligned["val"].astype(float)
            filed_by_field[field_name] = aligned["filed"]
        else:
            out[field_name] = np.nan

    # a year counts as public only once *all* of its figures had been filed
    filed = pd.DataFrame(filed_by_field).max(axis=1)

    derived_liab = out["_liabilities_and_equity"] - out["_equity_incl_nci"]
    out["total_liabilities"] = out["total_liabilities"].fillna(derived_liab)
    out["total_liabilities"] = out["total_liabilities"].fillna(out["current_liabilities"] + out["_liabilities_noncurrent"])
    out["total_liabilities"] = out["total_liabilities"].fillna(
        out["_liabilities_and_equity"] - out["_equity_parent"] - out["_minority_interest"].fillna(0))
    out["sga"] = out["sga"].fillna(out["_selling_marketing"] + out["_general_admin"])
    # no single SG&A line (often R&D + G&A companies): G&A, plus selling expenses when tagged
    out["sga"] = out["sga"].fillna(out["_general_admin"] + out["_selling_expense"].fillna(0))
    out["long_term_debt"] = out["long_term_debt"].fillna(out["_ltd_total"] - out["_ltd_current"])
    # EBIT = pretax income + net interest cost, only when neither operating income nor interest expense is
    # tagged (finalize prefers operating income, then this, then pretax + interest expense)
    out["ebit"] = (out["pretax_income"] - out["_interest_net_nonop"]).where(
        out["operating_income"].isna() & out["interest_expense"].isna())
    # shares outstanding: balance-sheet count, else issued - treasury, else the 10-K cover page (D16). Each
    # fallback keeps its own filing date, so a later stock split is applied to it (split_adjust).
    shares_filed = filed_by_field.get("shares_outstanding", pd.Series(pd.NaT, index=index))
    issued_treasury = out["_shares_issued"] - out["_treasury_shares"]
    it_filed = pd.concat([filed_by_field.get("_shares_issued", pd.Series(pd.NaT, index=index)),
                          filed_by_field.get("_treasury_shares", pd.Series(pd.NaT, index=index))], axis=1).max(axis=1)
    dei = _dei_shares(facts, index)
    for fallback, when in ((issued_treasury, it_filed), (dei["val"], dei["filed"])):
        use = out["shares_outstanding"].isna() & fallback.notna()
        out.loc[use, "shares_outstanding"] = fallback[use]
        shares_filed = shares_filed.where(~use, when)
    filed_by_field["shares_outstanding"] = shares_filed
    out["total_debt"] = total_debt(out)
    if len(out) and pd.isna(primary_debt(out).iloc[-1]) and pd.notna(out["total_debt"].iloc[-1]):
        notes.append("Debito dell'ultimo esercizio: letto da una voce SEC alternativa (le voci standard mancano).")
    split_intangibles = out["_finite_intangibles"].fillna(0) + out["_indefinite_intangibles"].fillna(0)
    has_split = out["_finite_intangibles"].notna() | out["_indefinite_intangibles"].notna()
    out["intangibles"] = out["intangibles"].fillna(split_intangibles.where(has_split))
    out = out.drop(columns=[c for c in out.columns if c.startswith("_")])
    out["filed"] = filed
    for f in SPLIT_SENSITIVE:
        if f in filed_by_field:
            out[f"filed_{f}"] = filed_by_field[f]
    return finalize(out)


def primary_debt(df: pd.DataFrame) -> pd.Series:
    """Financial debt = long-term (non-current) part + current part, leases excluded.

    The current part is "DebtCurrent" when tagged, otherwise current portion of
    long-term debt + short-term borrowings. When only the all-in "LongTermDebt" tag
    exists, it already includes its current portion. NaN when nothing is tagged."""
    st_borrow = df["_short_term_borrowings"].fillna(0)
    current = df["_debt_current"].fillna(df["_ltd_current"].fillna(0) + st_borrow)
    from_noncurrent = df["long_term_debt"] + current
    from_total = df["_ltd_total"] + st_borrow
    return from_noncurrent.fillna(from_total)


def total_debt(df: pd.DataFrame) -> pd.Series:
    """Standard tags first (primary_debt), then the alternative ways of tagging the total.

    Audit D15: an alternative total is accepted only if it is not smaller than a piece
    of debt known for the same year (DebtInstrumentCarryingAmount is sometimes a single
    loan); pieces alone are never summed into a total (it would understate the debt).
    A zero is not believed when another tag shows debt that year, or when it sits
    alone between two years with debt (Oracle 2022: "LongTermDebt" = 0)."""
    primary = primary_debt(df)
    st_borrow = df["_short_term_borrowings"].fillna(0)
    current = df["_debt_current"].fillna(df["_ltd_current"].fillna(0) + st_borrow)
    pieces = df[["_debt_current", "_ltd_current", "_short_term_borrowings", "_secured_debt", "_lt_notes"]].max(axis=1)
    candidates = [df["_debt_combined"], df["_debt_instruments"], df["_notes_payable"] + st_borrow,
                  df["_lt_notes"] + current]
    alternative = pd.Series(np.nan, index=df.index)
    for c in candidates:
        alternative = alternative.fillna(c.where(~(c < pieces * 0.999)))
    any_debt = pd.concat([pieces, *candidates], axis=1).max(axis=1) > 0
    primary = primary.mask((primary == 0) & any_debt)
    prev, nxt = primary.shift(1), primary.shift(-1)
    primary = primary.mask((primary == 0) & (prev > 0) & (nxt > 0))
    return primary.fillna(alternative)
