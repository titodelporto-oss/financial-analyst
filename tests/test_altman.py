"""Altman Z-Score, verified by hand.

Base case: total assets 1000, working capital 200 (500 - 300), retained earnings 300,
EBIT 100, total liabilities 500, sales 1500, book equity 500, market value 1000.
  X1 = 0.2, X2 = 0.3, X3 = 0.1, X4 (market) = 2.0, X4 (book) = 1.0, X5 = 1.5
"""

import pytest

from src.models import NOT_APPLICABLE, altman_z
from src.sector import SectorInfo
from tests.conftest import make_annual

BASE = {"total_assets": 1000, "current_assets": 500, "current_liabilities": 300, "retained_earnings": 300,
        "ebit": 100, "total_liabilities": 500, "revenue": 1500, "equity": 500}


def test_original_z_for_manufacturers(manufacturer):
    # 1.2*0.2 + 1.4*0.3 + 3.3*0.1 + 0.6*2.0 + 1.0*1.5 = 0.24 + 0.42 + 0.33 + 1.2 + 1.5 = 3.69
    res = altman_z(make_annual({"2024-12-31": BASE}), manufacturer, market_value_equity=1000)
    assert res.value == pytest.approx(3.69)
    assert res.label == "Sicura"
    assert "originale" in res.variant
    assert len(res.components) == 5


def test_z_double_prime_for_services(services):
    # 6.56*0.2 + 3.26*0.3 + 6.72*0.1 + 1.05*1.0 = 1.312 + 0.978 + 0.672 + 1.05 = 4.012
    res = altman_z(make_annual({"2024-12-31": BASE}), services)
    assert res.value == pytest.approx(4.012)
    assert res.label == "Sicura"
    assert "Z''" in res.variant
    assert len(res.components) == 4  # no sales/assets term in Z''


def test_z_double_prime_distress(services):
    # X1 = -100/1000, X2 = -200/1000, X3 = -50/1000, X4 = 100/900
    # 6.56*-0.1 + 3.26*-0.2 + 6.72*-0.05 + 1.05*0.1111 = -0.656 - 0.652 - 0.336 + 0.11667 = -1.52733
    data = {"total_assets": 1000, "current_assets": 200, "current_liabilities": 300,
            "retained_earnings": -200, "ebit": -50, "total_liabilities": 900, "equity": 100}
    res = altman_z(make_annual({"2024-12-31": data}), services)
    assert res.value == pytest.approx(-1.52733, abs=1e-4)
    assert res.label == "Distress"


def test_grey_zone_boundaries_differ_by_variant(manufacturer, services):
    # Same Z value of 2.0 is grey for the original model (1.81-2.99) and grey for Z'' (1.10-2.60),
    # while 2.7 is grey for the original but safe for Z''.
    from src.models import ALTMAN_ORIGINAL, ALTMAN_Z2, altman_zone
    assert altman_zone(2.0, ALTMAN_ORIGINAL) == "Grigia"
    assert altman_zone(2.7, ALTMAN_ORIGINAL) == "Grigia"
    assert altman_zone(2.7, ALTMAN_Z2) == "Sicura"
    assert altman_zone(1.5, ALTMAN_ORIGINAL) == "Distress"
    assert altman_zone(1.5, ALTMAN_Z2) == "Grigia"


def test_not_applicable_to_banks(bank):
    res = altman_z(make_annual({"2024-12-31": BASE}), bank, market_value_equity=1000)
    assert res.value is None
    assert res.label == NOT_APPLICABLE


def test_not_applicable_to_insurers():
    insurer = SectorInfo("insurance", "SIC (SEC)", "SIC 6331")
    assert altman_z(make_annual({"2024-12-31": BASE}), insurer).label == NOT_APPLICABLE


@pytest.mark.parametrize("category", ["reit", "utility"])
def test_not_applicable_to_reits_and_utilities(category):
    res = altman_z(make_annual({"2024-12-31": BASE}), SectorInfo(category, "Yahoo Finance", "-"), 1000)
    assert res.value is None and res.label == NOT_APPLICABLE
    assert any("non adatto" in n for n in res.notes)


def test_missing_market_value_is_nd_not_a_guess(manufacturer):
    res = altman_z(make_annual({"2024-12-31": BASE}), manufacturer, market_value_equity=None)
    assert res.value is None
    assert res.label == "N/D"
    assert any("Capitalizzazione" in m for m in res.missing)


def test_uses_latest_year(services):
    old = dict(BASE, ebit=-500)
    res = altman_z(make_annual({"2023-12-31": old, "2024-12-31": BASE}), services)
    assert res.value == pytest.approx(4.012)
    assert res.fiscal_year == "31/12/2024"
