"""Beneish M-Score, verified by hand.

M = -4.84 + 0.920 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI + 0.115 DEPI
         - 0.172 SGAI + 4.679 TATA - 0.327 LVGI

When every index equals 1, the constant part is
  -4.84 + 0.920 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 - 0.327 = -2.48
"""

import pytest

from src.models import NOT_APPLICABLE, beneish_m
from tests.conftest import make_annual

PREV = {"receivables": 100, "revenue": 1000, "gross_profit": 400, "current_assets": 300, "net_ppe": 400,
        "total_assets": 1000, "d_and_a": 50, "sga": 200, "long_term_debt": 200, "current_liabilities": 100}


def annual(cur, prev=PREV):
    return make_annual({"2023-12-31": prev, "2024-12-31": cur})


def test_stable_company_no_signal(services):
    # Same figures both years -> all indices 1; TATA = (100 - 150) / 1000 = -0.05
    # M = -2.48 + 4.679 * -0.05 = -2.71395
    cur = dict(PREV, net_income=100, ocf=150)
    res = beneish_m(annual(cur), services)
    assert res.value == pytest.approx(-2.71395)
    assert res.label == "Nessun segnale"
    values = {c["Indice"]: c["Valore"] for c in res.components}
    assert values["DSRI"] == pytest.approx(1) and values["AQI"] == pytest.approx(1)


def test_manipulation_profile_flagged(services):
    # Revenue +50% (SGI 1.5), receivables/revenue doubled (DSRI 2), every other ratio unchanged,
    # TATA = (150 - 75) / 1500 = +0.05
    # M = -4.84 + 0.92*2 + 0.528 + 0.404 + 0.892*1.5 + 0.115 - 0.172 + 4.679*0.05 - 0.327 = -0.88005
    cur = {"receivables": 300, "revenue": 1500, "gross_profit": 600, "current_assets": 450, "net_ppe": 600,
           "total_assets": 1500, "d_and_a": 75, "sga": 300, "long_term_debt": 300, "current_liabilities": 150,
           "net_income": 150, "ocf": 75}
    res = beneish_m(annual(cur), services)
    values = {c["Indice"]: c["Valore"] for c in res.components}
    assert values["DSRI"] == pytest.approx(2)
    assert values["SGI"] == pytest.approx(1.5)
    assert values["DEPI"] == pytest.approx(1)
    assert values["TATA"] == pytest.approx(0.05)
    assert res.value == pytest.approx(-0.88005)
    assert res.label == "Possibile manipolazione"


def test_prefers_income_from_continuing_operations(services):
    # TATA = (130 - 150) / 1000 = -0.02 -> M = -2.48 - 0.09358 = -2.57358
    cur = dict(PREV, net_income=100, net_income_cont_ops=130, ocf=150)
    assert beneish_m(annual(cur), services).value == pytest.approx(-2.57358)


def test_missing_item_gives_nd(services):
    cur = dict(PREV, net_income=100, ocf=150)
    del cur["sga"]
    res = beneish_m(annual(cur), services)
    assert res.value is None
    assert res.label == "N/D"
    assert any("Spese generali" in m for m in res.missing)


def test_not_applicable_to_banks(bank):
    res = beneish_m(annual(dict(PREV, net_income=100, ocf=150)), bank)
    assert res.label == NOT_APPLICABLE and res.value is None


def test_growth_driven_flag_gets_explained(services):
    # Only revenue growth changes: every line item scales by 2.2, so all other indices stay at 1.
    # SGI = 2.2 -> M = -2.48 + 0.892 * 1.2 + 4.679 * TATA(-0.05) = -1.64355 (> -1.78)
    # Without the growth term: -1.64355 - 1.0704 = -2.71395 -> below threshold -> explained
    k = 2.2
    cur = {key: v * k for key, v in PREV.items()}
    cur.update(net_income=100 * k, ocf=150 * k)
    res = beneish_m(annual(cur), services)
    assert res.value == pytest.approx(-1.64355)
    assert res.label == "Possibile manipolazione"
    assert any("crescita dei ricavi" in n for n in res.notes)
