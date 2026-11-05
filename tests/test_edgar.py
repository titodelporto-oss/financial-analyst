"""Bilanci dalla SEC (Fase B): lettura dei dati SEC, split, ripiego su Yahoo, contatto e limite di richieste.

Nessuna richiesta vera: la SEC e Yahoo sono sostituiti da versioni finte. I primi test sono copiati da
quelli di PATTY (~/equity-analyst/tests/test_data_sources.py), perché il codice SEC è lo stesso."""

import threading
import time

import pandas as pd
import pytest

from src import edgar, screener
from src.edgar import parse_annual_statements
from src.screener import _market_value_at_fye, load_statements
from src.statements import split_adjust
from tests.conftest import make_annual


def fact(start, end, val, filed, form="10-K"):
    d = {"end": end, "val": val, "filed": filed, "form": form, "fy": int(filed[:4]), "fp": "FY"}
    if start:
        d["start"] = start
    return d


FACTS = {"facts": {"us-gaap": {
    # old tag up to FY2018, new tag afterwards (ASC 606 switch)
    "SalesRevenueNet": {"units": {"USD": [
        fact("2017-01-01", "2017-12-31", 900, "2018-02-20"),
        fact("2018-01-01", "2018-12-31", 1000, "2019-02-20"),
    ]}},
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        fact("2018-01-01", "2018-12-31", 990, "2020-02-20"),   # restated in the next 10-K
        fact("2019-01-01", "2019-12-31", 1100, "2020-02-20"),
        fact("2019-10-01", "2019-12-31", 300, "2020-02-20"),   # a quarter inside the 10-K: ignore
        fact("2019-01-01", "2019-12-31", 1105, "2019-11-01", form="10-Q"),  # not an annual report
    ]}},
    "Assets": {"units": {"USD": [
        fact(None, "2018-12-31", 5000, "2019-02-20"),
        fact(None, "2019-12-31", 5500, "2020-02-20"),
        fact(None, "2019-12-31", 5600, "2021-02-20"),          # later restatement: ignore
    ]}},
}}}


# --- lettura dei dati SEC (come PATTY) ------------------------------------------------------------

def test_edgar_keeps_values_as_originally_reported():
    df = parse_annual_statements(FACTS)
    assert list(df.index.year) == [2017, 2018, 2019]
    assert df.loc["2018-12-31", "revenue"] == 1000  # not the 990 restated a year later
    assert df.loc["2019-12-31", "revenue"] == 1100  # not the quarter, not the 10-Q
    assert df.loc["2019-12-31", "total_assets"] == 5500
    assert df.loc["2018-12-31", "filed"] == pd.Timestamp("2019-02-20")


def test_balance_sheet_date_a_day_off_is_matched():
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [fact("2021-01-01", "2021-12-31", 100, "2022-02-07")]}},
        "Assets": {"units": {"USD": [fact(None, "2021-12-30", 500, "2022-02-07")]}},
    }}}
    df = parse_annual_statements(facts)
    assert df.loc["2021-12-31", "total_assets"] == 500


def test_sga_from_two_lines_and_liabilities_derived():
    """Microsoft: SG&A in due righe; senza "Liabilities": passivo = totale - patrimonio con le minoranze."""
    y = "2024"
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [fact(f"{y}-01-01", f"{y}-12-31", 1000, "2025-02-10")]}},
        "SellingAndMarketingExpense": {"units": {"USD": [fact(f"{y}-01-01", f"{y}-12-31", 70, "2025-02-10")]}},
        "GeneralAndAdministrativeExpense": {"units": {"USD": [fact(f"{y}-01-01", f"{y}-12-31", 30, "2025-02-10")]}},
        "LiabilitiesAndStockholdersEquity": {"units": {"USD": [fact(None, f"{y}-12-31", 900, "2025-02-10")]}},
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": {"units": {"USD": [fact(None, f"{y}-12-31", 400, "2025-02-10")]}},
    }}}
    df = parse_annual_statements(facts)
    assert df["sga"].iloc[-1] == 100
    assert df["total_liabilities"].iloc[-1] == 500


def test_ebit_prefers_operating_income():
    """Audit D13 di PATTY (Alphabet): utile ante imposte + interessi contiene i guadagni sugli investimenti."""
    a = make_annual({"2024-12-31": {"operating_income": 100, "pretax_income": 160, "interest_expense": 10},
                     "2025-12-31": {"pretax_income": 160, "interest_expense": 10}})
    assert list(a["ebit"]) == [100, 170]  # il ripiego solo quando manca il reddito operativo


# --- split azionari -------------------------------------------------------------------------------

def test_split_adjustment_like_nvidia():
    # 10-K FY2024 depositato prima dello split 10:1 di giugno 2024: 2,464 mld di azioni come pubblicato.
    annual = make_annual({"2024-01-28": {"shares_outstanding": 2.464e9},
                          "2025-01-26": {"shares_outstanding": 24.477e9}})
    annual["filed"] = pd.to_datetime(["2024-02-21", "2025-02-26"])
    splits = pd.Series([4.0, 10.0], index=pd.to_datetime(["2021-07-20", "2024-06-10"]).tz_localize("America/New_York"))
    adj = split_adjust(annual, splits)
    assert adj["shares_outstanding"].iloc[0] == pytest.approx(24.64e9)
    assert adj["shares_outstanding"].iloc[1] == pytest.approx(24.477e9)  # già dopo lo split


def test_split_adjustment_uses_the_shares_own_filing_date():
    annual = make_annual({"2021-12-31": {"shares_outstanding": 3.1e9}})
    annual["filed"] = pd.to_datetime(["2022-02-07"])
    annual["filed_shares_outstanding"] = pd.to_datetime(["2023-01-31"])  # ripubblicato dopo lo split
    adj = split_adjust(annual, pd.Series([3.0], index=pd.to_datetime(["2022-08-25"])))
    assert adj["shares_outstanding"].iloc[0] == pytest.approx(3.1e9)


def test_no_splits_or_unknown_filing_date_changes_nothing():
    annual = make_annual({"2024-12-31": {"shares_outstanding": 1e9}})
    assert split_adjust(annual, None)["shares_outstanding"].iloc[0] == 1e9
    assert split_adjust(annual, pd.Series([2.0], index=pd.to_datetime(["2025-06-01"])))["shares_outstanding"].iloc[0] == 1e9


def _history(splits_on=None):
    idx = pd.date_range("2024-01-02", "2024-12-31", freq="B", tz="America/New_York")
    h = pd.DataFrame({"Close": 50.0, "Stock Splits": 0.0}, index=idx)
    if splits_on:
        h.loc[pd.Timestamp(splits_on, tz="America/New_York"), "Stock Splits"] = 2.0
    return h


def test_market_value_sec_shares_are_not_multiplied_twice():
    """Split 2:1 dopo la chiusura dell'esercizio: con la SEC le azioni sono già corrette, con Yahoo no."""
    annual = make_annual({"2024-06-28": {"shares_outstanding": 1e9}})
    hist = _history(splits_on="2024-09-03")
    assert _market_value_at_fye(hist, annual) == pytest.approx(50 * 2 * 1e9)  # Yahoo: come prima
    assert _market_value_at_fye(hist, annual, shares_on_today_basis=True) == pytest.approx(50 * 1e9)


# --- da dove arrivano i bilanci: SEC, altrimenti Yahoo --------------------------------------------

class FakeYahooTicker:
    """yf.Ticker finto: bilanci Yahoo minimi, per vedere quando si usa il ripiego."""

    def __init__(self):
        self.calls = 0
        d = pd.to_datetime(["2024-12-31"])
        self._bs = pd.DataFrame({d[0]: [777.0]}, index=["Total Assets"])
        self._inc = pd.DataFrame({d[0]: [100.0, 10.0]}, index=["Total Revenue", "Net Income"])

    @property
    def balance_sheet(self):
        self.calls += 1
        return self._bs

    @property
    def income_stmt(self):
        self.calls += 1
        return self._inc

    @property
    def cashflow(self):
        self.calls += 1
        return pd.DataFrame()


class FakeSec:
    def __init__(self, cik=320193, facts=FACTS, sic="3571", fail=None):
        self.cik, self.facts, self.sic, self.fail = cik, facts, sic, fail

    def install(self, monkeypatch):
        monkeypatch.setattr(edgar, "user_agent", lambda: "prova contatto@example.com")
        monkeypatch.setattr(edgar, "get_cik", lambda t: self.cik)
        monkeypatch.setattr(edgar, "fetch_company_facts", self._facts)
        monkeypatch.setattr(edgar, "fetch_submissions", lambda cik: {"sic": self.sic, "sicDescription": "Computers"})

    def _facts(self, cik):
        if self.fail:
            raise RuntimeError(self.fail)
        return self.facts


def test_statements_from_sec_with_sic(monkeypatch, capsys):
    FakeSec(sic="6021").install(monkeypatch)  # una banca: Altman e Beneish non si applicano
    yahoo = FakeYahooTicker()
    annual, sic, desc, from_sec = load_statements("JPM", yahoo, _history())
    assert from_sec and sic == "6021" and desc == "Computers"
    assert annual.loc["2019-12-31", "revenue"] == 1100
    assert yahoo.calls == 0  # nessuna richiesta a Yahoo per i bilanci
    assert "BILANCI DA YAHOO" not in capsys.readouterr().out


@pytest.mark.parametrize("sec, reason", [
    (FakeSec(cik=None), "ticker non trovato alla SEC"),
    (FakeSec(facts={"facts": {}}), "nessun bilancio annuale (10-K) alla SEC"),
    (FakeSec(fail="403 Forbidden"), "errore SEC (403 Forbidden)"),
])
def test_yahoo_fallback_when_sec_has_nothing(monkeypatch, capsys, sec, reason):
    sec.install(monkeypatch)
    yahoo = FakeYahooTicker()
    annual, sic, _, from_sec = load_statements("SHEL", yahoo, _history())
    assert not from_sec and sic is None
    assert annual["total_assets"].iloc[-1] == 777  # i bilanci di Yahoo
    assert f"BILANCI DA YAHOO SHEL: {reason}" in capsys.readouterr().out


def test_yahoo_fallback_without_sec_contact(monkeypatch, capsys):
    monkeypatch.setattr(edgar, "user_agent", lambda: None)
    annual, _, _, from_sec = load_statements("AAPL", FakeYahooTicker(), _history())
    assert not from_sec and annual["total_assets"].iloc[-1] == 777
    assert "SEC_USER_AGENT mancante" in capsys.readouterr().out


# --- contatto SEC e limite di richieste -----------------------------------------------------------

def test_user_agent_from_environment_or_secrets_file(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets.toml"
    monkeypatch.setattr(edgar, "SECRETS_PATH", secrets)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    assert edgar.user_agent() is None  # né variabile né file

    secrets.write_text('ALPACA_API_KEY_ID = "x"\nSEC_USER_AGENT = "screening prova@example.com"\n')
    assert edgar.user_agent() == "screening prova@example.com"

    monkeypatch.setenv("SEC_USER_AGENT", "github prova@example.com")  # GitHub Actions: vince la variabile
    assert edgar.user_agent() == "github prova@example.com"


def test_no_request_without_contact(monkeypatch):
    monkeypatch.setattr(edgar, "user_agent", lambda: None)
    monkeypatch.setattr(edgar.requests, "get", lambda *a, **k: pytest.fail("richiesta senza contatto"))
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT mancante"):
        edgar._get("https://data.sec.gov/x", timeout=1)


def test_requests_stay_under_sec_limit_across_threads(monkeypatch):
    """4 titoli in parallelo: le richieste alla SEC restano distanziate di almeno 1/8 di secondo."""
    times = []

    class Ok:
        def raise_for_status(self):
            pass

    def fake_get(url, headers, timeout):
        assert "@" in headers["User-Agent"]
        times.append(time.monotonic())
        return Ok()

    monkeypatch.setattr(edgar, "user_agent", lambda: "prova contatto@example.com")
    monkeypatch.setattr(edgar.requests, "get", fake_get)
    monkeypatch.setattr(edgar, "MIN_INTERVAL", 0.02)
    threads = [threading.Thread(target=lambda: [edgar._get("u", 1) for _ in range(3)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    times.sort()
    assert len(times) == 12
    assert min(b - a for a, b in zip(times, times[1:])) >= 0.02 * 0.95


def test_analyze_ticker_uses_sec_sic_for_sector(monkeypatch):
    """Il codice SIC della SEC decide la categoria (qui una utility), come in PATTY."""
    FakeSec(sic="4911").install(monkeypatch)
    annual, sic, desc, _ = load_statements("DUK", FakeYahooTicker(), _history())
    assert screener.classify(sic, desc, "Technology", "Software").category == "utility"



# --- voci mancanti alla SEC: prese da Yahoo solo se servono -----------------------------------------

from src.screener import drop_share_unit_errors, fill_missing_from_yahoo  # noqa: E402
from src.sector import SectorInfo  # noqa: E402

COMPLETE = {"receivables": 50, "revenue": 1000, "cost_of_revenue": 600, "current_assets": 400, "net_ppe": 300,
            "total_assets": 1500, "d_and_a": 40, "sga": 100, "long_term_debt": 200, "current_liabilities": 250,
            "ocf": 120, "net_income": 80, "retained_earnings": 500, "operating_income": 150,
            "total_liabilities": 700, "equity": 800, "shares_outstanding": 1e8}


def _sec_years(**changes):
    """Due esercizi SEC completi; changes = {voce: (valore 2023, valore 2024)} per creare buchi."""
    y23, y24 = dict(COMPLETE), dict(COMPLETE)
    for f, (a, b) in changes.items():
        y23[f], y24[f] = a, b
    return make_annual({"2023-12-31": y23, "2024-12-31": y24})


class StatementsYahoo:
    """yf.Ticker finto che conta quali prospetti vengono scaricati."""

    def __init__(self, balance=None, income=None, fail=False):
        self.requested = []
        self.fail = fail
        d = pd.to_datetime(["2024-12-31", "2023-12-31"])
        self._bs = pd.DataFrame(balance or {}, index=d).T
        self._inc = pd.DataFrame(income or {}, index=d).T

    def _get(self, name, frame):
        self.requested.append(name)
        if self.fail:
            raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")
        return frame

    @property
    def balance_sheet(self):
        return self._get("balance_sheet", self._bs)

    @property
    def income_stmt(self):
        return self._get("income_stmt", self._inc)

    @property
    def cashflow(self):
        return self._get("cashflow", pd.DataFrame())


def test_complete_sec_data_asks_nothing_to_yahoo(services):
    yahoo = StatementsYahoo()
    out = fill_missing_from_yahoo(_sec_years(), "X", yahoo, services)
    assert yahoo.requested == []
    assert out.equals(_sec_years())


def test_missing_receivables_only_balance_sheet_requested_both_years_from_yahoo(services, capsys):
    yahoo = StatementsYahoo(balance={"Accounts Receivable": [60.0, 55.0]})
    out = fill_missing_from_yahoo(_sec_years(receivables=(50, None)), "BSX", yahoo, services)
    assert yahoo.requested == ["balance_sheet"]  # niente conto economico né flussi di cassa
    # entrambi gli anni da Yahoo, anche il 2023 che la SEC aveva: Beneish non mescola le fonti
    assert list(out["receivables"]) == [55.0, 60.0]
    assert "VOCI DA YAHOO BSX: receivables" in capsys.readouterr().out


def test_yahoo_with_one_year_only_leaves_nd(services, capsys):
    yahoo = StatementsYahoo(income={"Selling General And Administration": [110.0, None]})
    out = fill_missing_from_yahoo(_sec_years(sga=(None, None)), "AMZN", yahoo, services)
    assert yahoo.requested == ["income_stmt"]
    assert out["sga"].isna().all()
    assert "restano N/D sga" in capsys.readouterr().out


def test_yahoo_block_does_not_fail_the_ticker(services, capsys):
    yahoo = StatementsYahoo(fail=True)
    out = fill_missing_from_yahoo(_sec_years(net_ppe=(None, None)), "ALB", yahoo, services)
    assert out["net_ppe"].isna().all()
    assert "VOCI DA YAHOO NON DISPONIBILI ALB (Too Many Requests" in capsys.readouterr().out


def test_no_yahoo_for_banks_or_utilities(bank):
    utility = SectorInfo("utility", "SIC (SEC)", "SIC 4911")
    for sector in (bank, utility):
        yahoo = StatementsYahoo()
        fill_missing_from_yahoo(_sec_years(receivables=(None, None)), "X", yahoo, sector)
        assert yahoo.requested == []  # Altman e Beneish non si applicano: nessuna richiesta


def test_working_capital_recomputed_after_fill(manufacturer):
    yahoo = StatementsYahoo(balance={"Current Assets": [420.0, 410.0], "Current Liabilities": [260.0, 255.0]})
    out = fill_missing_from_yahoo(_sec_years(current_assets=(None, None), current_liabilities=(None, None)),
                                  "PHM", yahoo, manufacturer)
    assert list(out["working_capital"]) == [410 - 255, 420 - 260]


def test_shares_from_yahoo_are_split_adjusted_after_year_end(manufacturer):
    yahoo = StatementsYahoo(balance={"Ordinary Shares Number": [5e7, 5e7]})
    splits = pd.Series([2.0], index=pd.to_datetime(["2025-03-03"]))
    out = fill_missing_from_yahoo(_sec_years(shares_outstanding=(None, None)), "AOS", yahoo, manufacturer, splits)
    assert list(out["shares_outstanding"]) == [1e8, 1e8]  # split 2:1 dopo la fine dei due esercizi


def test_share_unit_error_becomes_nd(capsys):
    """PKG 2025 nei dati SEC: 89 miliardi di azioni invece di 89 milioni."""
    a = make_annual({"2023-12-31": {"shares_outstanding": 89.6e6}, "2024-12-31": {"shares_outstanding": 89.8e6},
                     "2025-12-31": {"shares_outstanding": 89.2e9}})
    out = drop_share_unit_errors(a, "PKG")
    assert out["shares_outstanding"].isna().tolist() == [False, False, True]
    assert "AZIONI SOSPETTE PKG: numero di azioni 2025" in capsys.readouterr().out
    normal = make_annual({"2024-12-31": {"shares_outstanding": 1e9}, "2025-12-31": {"shares_outstanding": 1.1e9}})
    assert drop_share_unit_errors(normal).equals(normal)


def test_share_unit_error_old_years_in_thousands_like_rtx():
    """RTX alla SEC: 2011-2021 in migliaia, dal 2022 giusti. Il riferimento sono le azioni di oggi."""
    a = make_annual({f"{y}-12-31": {"shares_outstanding": v} for y, v in
                     ((2019, 864_366), (2020, 1_519_439), (2021, 1_493_280), (2022, 1.466e9), (2023, 1.327e9))})
    out = drop_share_unit_errors(a, "RTX", reference=1.34e9)
    assert out["shares_outstanding"].isna().tolist() == [True, True, True, False, False]


def test_share_unit_error_does_not_discard_the_correct_year_after():
    """Un anno sbagliato in mezzo (2019): si scarta solo quello, non il 2020 che torna normale."""
    a = make_annual({f"{y}-12-31": {"shares_outstanding": v} for y, v in
                     ((2017, 2.5e9), (2018, 2.5e9), (2019, 2.5e6), (2020, 2.45e9), (2021, 2.4e9))})
    out = drop_share_unit_errors(a)
    assert out["shares_outstanding"].isna().tolist() == [False, False, True, False, False]
