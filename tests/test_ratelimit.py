"""Pausa comune per il blocco di Yahoo e nuovi tentativi in analyze_ticker.

Nessuna richiesta a Yahoo: un "Yahoo finto" sostituisce l'analisi del singolo titolo e un orologio
finto sostituisce il tempo vero, così le pause da 30/60/120 s durano zero secondi."""

import pytest

from src import screener
from src.ratelimit import RateLimitGate
from src.screener import STOPPED_BY_DEADLINE, TickerAnalysis, _empty, analyze_ticker, is_rate_limited

YAHOO_LIMIT = "Too Many Requests. Rate limited. Try after a while."


class FakeClock:
    """Orologio finto: dormire fa solo avanzare il tempo, e ogni pausa viene annotata."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeYahoo:
    """Yahoo finto: blocca le richieste fino al momento `blocked_until` dell'orologio finto."""

    def __init__(self, clock, blocked_until=float("inf"), other_error=None):
        self.clock = clock
        self.blocked_until = blocked_until
        self.other_error = other_error
        self.calls = []

    def analyze(self, ticker):
        self.calls.append((ticker, self.clock()))
        if self.other_error:
            return _empty(ticker, self.other_error)
        if self.clock() < self.blocked_until:
            return _empty(ticker, YAHOO_LIMIT)
        return TickerAnalysis(ticker, f"{ticker} Inc.", 100.0, 1e9, 50.0, "Neutrale/Sano", "Rialzista",
                              1.0, 10.0, 3.5, "Sicura", 7, -2.5, "Nessun segnale", 3.0)


@pytest.fixture
def clock():
    return FakeClock()


def _gate(clock, deadline=None):
    return RateLimitGate(deadline=deadline, clock=clock, sleep=clock.sleep, log=lambda msg: None)


# --- la pausa comune ------------------------------------------------------------------------------

def test_pauses_grow_30_60_120_and_stop_growing(clock):
    gate = _gate(clock)
    for _ in range(4):
        gate.report_limit()
        assert gate.wait()
    assert clock.sleeps == [30.0, 60.0, 120.0, 120.0]


def test_good_answer_restarts_from_shortest_pause(clock):
    gate = _gate(clock)
    gate.report_limit(); gate.wait()
    gate.report_limit(); gate.wait()
    gate.report_ok()
    gate.report_limit(); gate.wait()
    assert clock.sleeps == [30.0, 60.0, 30.0]


def test_same_block_reported_by_many_titles_counts_once(clock):
    gate = _gate(clock)
    for _ in range(4):  # 4 titoli in parallelo trovano lo stesso blocco
        gate.report_limit()
    assert gate.pauses_taken == 1
    assert gate.wait()
    assert clock.sleeps == [30.0]


def test_pause_past_deadline_is_not_waited(clock):
    gate = _gate(clock, deadline=20.0)
    gate.report_limit()  # pausa fino a 30 s, ma il tempo massimo è a 20 s
    assert gate.wait() is False
    assert clock.sleeps == []  # nessuna attesa inutile


# --- analyze_ticker con la pausa comune (screening notturno) --------------------------------------

def test_with_gate_waits_for_block_to_end(monkeypatch, clock):
    yahoo = FakeYahoo(clock, blocked_until=50.0)
    monkeypatch.setattr(screener, "_analyze_ticker_once", yahoo.analyze)

    r = analyze_ticker("SPG", retries=4, gate=_gate(clock))

    assert r.error is None and r.ticker == "SPG"
    assert clock.sleeps == [30.0, 60.0]  # bloccato a 0 s e a 30 s, buono a 90 s
    assert [t for _, t in yahoo.calls] == [0.0, 30.0, 90.0]


def test_with_gate_other_errors_are_not_retried(monkeypatch, clock):
    yahoo = FakeYahoo(clock, other_error="Nessun dato storico trovato")
    monkeypatch.setattr(screener, "_analyze_ticker_once", yahoo.analyze)
    gate = _gate(clock)

    r = analyze_ticker("PG", retries=4, gate=gate)

    assert r.error == "Nessun dato storico trovato"
    assert len(yahoo.calls) == 1
    assert gate.pauses_taken == 0


def test_with_gate_gives_up_at_deadline_keeping_yahoo_message(monkeypatch, clock):
    yahoo = FakeYahoo(clock)  # bloccato per sempre
    monkeypatch.setattr(screener, "_analyze_ticker_once", yahoo.analyze)

    r = analyze_ticker("UBER", retries=4, gate=_gate(clock, deadline=100.0))

    # tentativi a 0 e 30 e 90 s; la pausa successiva (120 s) finirebbe oltre i 100 s: stop
    assert [t for _, t in yahoo.calls] == [0.0, 30.0, 90.0]
    assert r.error == YAHOO_LIMIT


def test_with_gate_after_deadline_no_request_at_all(monkeypatch, clock):
    yahoo = FakeYahoo(clock, blocked_until=0.0)
    monkeypatch.setattr(screener, "_analyze_ticker_once", yahoo.analyze)
    clock.now = 10.0

    r = analyze_ticker("AAPL", retries=4, gate=_gate(clock, deadline=5.0))

    assert yahoo.calls == []
    assert r.error == STOPPED_BY_DEADLINE
    assert is_rate_limited(r.error)  # così il secondo giro lo riconosce come bloccato


# --- analyze_ticker senza pausa comune (sito Streamlit): come prima -------------------------------

def test_without_gate_behaves_as_before(monkeypatch, clock):
    yahoo = FakeYahoo(clock)  # bloccato per sempre
    monkeypatch.setattr(screener, "_analyze_ticker_once", yahoo.analyze)
    monkeypatch.setattr(screener.time, "sleep", clock.sleep)

    r = analyze_ticker("MSFT")

    assert len(yahoo.calls) == 3
    assert clock.sleeps == [1.5, 3.0, 4.5]
    assert r.error == YAHOO_LIMIT


# --- l'errore vero di yfinance viene riconosciuto -------------------------------------------------

class _BlockedTicker:
    """yf.Ticker finto: qualsiasi richiesta risponde come yfinance durante il blocco."""

    def __init__(self, symbol):
        self.symbol = symbol

    @property
    def info(self):
        raise RuntimeError(YAHOO_LIMIT)


def test_yfinance_block_is_recognised(monkeypatch, clock):
    monkeypatch.setattr(screener.yf, "Ticker", _BlockedTicker)
    gate = _gate(clock, deadline=40.0)

    r = analyze_ticker("SPG", retries=4, gate=gate)

    assert is_rate_limited(r.error)
    assert gate.pauses_taken >= 1


def test_is_rate_limited():
    assert is_rate_limited(YAHOO_LIMIT)
    assert is_rate_limited("HTTP Error 401: Invalid Crumb")
    assert not is_rate_limited("Nessun dato storico trovato")
    assert not is_rate_limited("")
    assert not is_rate_limited(None)
