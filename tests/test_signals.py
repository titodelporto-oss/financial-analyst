"""Signal gate and divergence detection of the screening site."""

from types import SimpleNamespace

import pandas as pd

from src.indicators import detect_rsi_divergence
from src.narrative import build_narrative
from src.signals import _fundamentals_buy_gate, _quality_reason


def ta(**overrides):
    base = dict(ticker="TEST", name="Test Inc.", piotroski_f=8, piotroski_evaluable=9, altman_zone="Sicura",
                altman_z=3.5, altman_variant="Z originale", beneish_flag="Nessun segnale", beneish_m=-2.5,
                target_upside_pct=20.0, volume_ratio=1.0, stop_loss=None,
                sector_category="manufacturing")
    base.update(overrides)
    return SimpleNamespace(**base)


def test_gate_passes_when_everything_is_verified():
    assert _fundamentals_buy_gate(ta())


def test_missing_data_never_passes_the_gate():
    assert not _fundamentals_buy_gate(ta(altman_zone="N/D"))
    assert not _fundamentals_buy_gate(ta(beneish_flag="N/D"))
    assert not _fundamentals_buy_gate(ta(piotroski_f=None))
    assert not _fundamentals_buy_gate(ta(piotroski_evaluable=5, piotroski_f=5))  # too few criteria known
    assert not _fundamentals_buy_gate(ta(target_upside_pct=None))


def test_red_flags_block_the_gate():
    assert not _fundamentals_buy_gate(ta(altman_zone="Distress"))
    assert not _fundamentals_buy_gate(ta(beneish_flag="Possibile manipolazione"))
    assert not _fundamentals_buy_gate(ta(altman_zone="Non applicabile"))  # financials


def _series(n):
    close, rsi = [100.0] * n, [50.0] * n
    for i, (c, r) in zip(range(8, 13), [(97, 45), (94, 35), (90, 25), (94, 35), (97, 45)]):
        close[i], rsi[i] = c, r
    for i, (c, r) in zip(range(18, 23), [(96, 45), (92, 38), (85, 30), (92, 38), (96, 45)]):
        close[i], rsi[i] = c, r
    return pd.Series(close), pd.Series(rsi)


def test_divergence_only_after_confirmation_and_while_fresh():
    # second low at bar 20, confirmed at bar 23 (3 bars later), valid for 5 more bars
    assert detect_rsi_divergence(*_series(23)) is None
    res = detect_rsi_divergence(*_series(24))
    assert res["type"] == "bullish" and res["bars_ago"] == 0
    assert (res["price_low_1"], res["price_low_2"]) == (90, 85)
    assert detect_rsi_divergence(*_series(29))["bars_ago"] == 5
    assert detect_rsi_divergence(*_series(30)) is None  # stale


def test_narrative_states_missing_scores_and_handles_missing_target():
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    hist = pd.DataFrame({"Close": 100.0, "High": 101.0, "Low": 99.0, "Volume": 1e6}, index=idx)
    rsi = pd.Series(50.0, index=idx)
    macd = pd.DataFrame({"histogram": 0.0}, index=idx)
    stoch = pd.DataFrame({"k": 50.0, "d": 50.0}, index=idx)
    t = ta(beneish_flag="N/D", beneish_m=None, altman_zone="N/D", altman_z=None, target_upside_pct=None)
    text = " ".join(build_narrative(t, {"numberOfAnalystOpinions": 10}, hist, rsi, macd, stoch, "SELL"))
    assert "Beneish M-Score: non calcolabile" in text
    assert "Altman Z-Score: non calcolabile" in text
    assert "non rileva segnali" not in text


def test_piotroski_threshold_is_proportional_to_evaluable_criteria():
    assert _fundamentals_buy_gate(ta(piotroski_f=7, piotroski_evaluable=9))
    assert not _fundamentals_buy_gate(ta(piotroski_f=6, piotroski_evaluable=9))
    assert not _fundamentals_buy_gate(ta(piotroski_f=6, piotroski_evaluable=8))  # 6/8 < 7/9
    assert _fundamentals_buy_gate(ta(piotroski_f=6, piotroski_evaluable=7))  # 6/7 >= 7/9
    assert not _fundamentals_buy_gate(ta(piotroski_f=5, piotroski_evaluable=7))
    assert _fundamentals_buy_gate(ta(piotroski_f=5, piotroski_evaluable=6))
    assert not _fundamentals_buy_gate(ta(piotroski_f=5, piotroski_evaluable=5))


def test_reits_and_utilities_pass_without_models_that_do_not_apply_to_them():
    reit = dict(sector_category="reit", altman_zone="Non applicabile", altman_z=None,
                beneish_flag="Non applicabile", beneish_m=None, piotroski_f=6, piotroski_evaluable=7)
    assert _fundamentals_buy_gate(ta(**reit))
    assert "REIT" in _quality_reason(ta(**reit)) and "non applicabile" in _quality_reason(ta(**reit))
    utility = dict(sector_category="utility", altman_zone="Non applicabile", altman_z=None,
                   beneish_flag="Non applicabile", beneish_m=None)
    assert _fundamentals_buy_gate(ta(**utility))
    # a real missing value still blocks, whatever the type of company
    assert not _fundamentals_buy_gate(ta(**dict(utility, beneish_flag="N/D")))
    # weak balance sheet still blocks
    assert not _fundamentals_buy_gate(ta(**dict(reit, piotroski_f=4)))


def test_not_applicable_does_not_let_other_companies_through():
    assert not _fundamentals_buy_gate(ta(sector_category="bank", altman_zone="Non applicabile",
                                         beneish_flag="Non applicabile"))
    assert not _fundamentals_buy_gate(ta(sector_category="non_manufacturing", altman_zone="Non applicabile"))
