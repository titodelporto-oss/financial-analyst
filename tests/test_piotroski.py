"""Piotroski F-Score, verified by hand.

"Good" company, total assets 1000 every year:
  ROA t = 80/1000 = 0.08 > 0, and > ROA t-1 = 50/1000 = 0.05
  CFO 120 > 0 and > net income 80
  leverage t = 200/1000 = 0.20 < t-1 = 300/1000 = 0.30
  current ratio 400/200 = 2.0 > 300/200 = 1.5
  shares 98 <= 100
  gross margin 450/900 = 0.50 > 360/800 = 0.45
  asset turnover 900/1000 = 0.9 > 800/1000 = 0.8
"""

import pytest

from src.models import piotroski_f
from tests.conftest import make_annual

T2 = {"total_assets": 1000, "revenue": 700, "net_income": 30}
T1 = {"total_assets": 1000, "net_income": 50, "ocf": 60, "long_term_debt": 300, "current_assets": 300,
      "current_liabilities": 200, "shares_outstanding": 100, "gross_profit": 360, "revenue": 800}
T_GOOD = {"total_assets": 1000, "net_income": 80, "ocf": 120, "long_term_debt": 200, "current_assets": 400,
          "current_liabilities": 200, "shares_outstanding": 98, "gross_profit": 450, "revenue": 900}
T_BAD = {"total_assets": 1000, "net_income": -10, "ocf": -20, "long_term_debt": 400, "current_assets": 200,
         "current_liabilities": 200, "shares_outstanding": 110, "gross_profit": 300, "revenue": 700}


def annual(t, t1=T1, t2=T2):
    return make_annual({"2022-12-31": t2, "2023-12-31": t1, "2024-12-31": t})


def test_all_nine_criteria_pass():
    res = piotroski_f(annual(T_GOOD))
    assert res.value == 9
    assert res.variant == "9/9 criteri valutabili"
    assert res.label == "Forte"


def test_all_nine_criteria_fail():
    # ROA -0.01; CFO -20 (< 0 and < NI -10); leverage 0.4 > 0.3; current ratio 1.0 < 1.5;
    # shares 110 > 100; margin 0.43 < 0.45; turnover 0.7 < 0.8
    res = piotroski_f(annual(T_BAD))
    assert res.value == 0
    assert res.variant == "9/9 criteri valutabili"
    assert res.label == "Debole"


def test_missing_item_is_not_counted_as_a_fail():
    t = {k: v for k, v in T_GOOD.items() if k != "long_term_debt"}
    res = piotroski_f(annual(t))
    assert res.value == 8  # was scored 0 for that criterion by the old app
    assert res.variant == "8/9 criteri valutabili"
    assert res.label == "Forte (parziale)"
    assert any("Leva" in m for m in res.missing)


def test_debt_free_company_passes_leverage_test():
    res = piotroski_f(annual(dict(T_GOOD, long_term_debt=0), dict(T1, long_term_debt=0)))
    assert res.value == 9


def test_years_are_matched_by_date_and_must_be_consecutive():
    res = piotroski_f(make_annual({"2020-12-31": T1, "2024-12-31": T_GOOD}))
    assert res.value is None
    assert res.label == "N/D"


def test_without_third_year_beginning_assets_tests_are_nd():
    # ROA t-1, leverage t-1 and turnover t-1 need total assets at the start of t-1
    res = piotroski_f(make_annual({"2023-12-31": T1, "2024-12-31": T_GOOD}))
    assert res.variant == "6/9 criteri valutabili"
    assert res.value == 6


def test_components_show_both_years():
    res = piotroski_f(annual(T_GOOD))
    gm = next(c for c in res.components if "Margine lordo" in c["Criterio"])
    assert gm["Anno t"] == pytest.approx(0.5)
    assert gm["Anno t-1"] == pytest.approx(0.45)
