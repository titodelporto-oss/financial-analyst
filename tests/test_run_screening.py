"""Il giro notturno salva i titoli falliti (voce "failed") senza toccare le altre voci del JSON.

Nessuna richiesta a Yahoo: l'analisi del singolo titolo (un "Yahoo finto") e l'elenco dei titoli sono
sostituiti da versioni finte. analyze_ticker e la pausa comune sono quelli veri, con pause brevissime."""

import json
import threading
import time

import run_screening
from src import screener
from src.screener import TickerAnalysis, _empty

YAHOO_LIMIT = "Too Many Requests. Rate limited. Try after a while."


def _ok(ticker: str, score: float) -> TickerAnalysis:
    return TickerAnalysis(ticker, f"{ticker} Inc.", 100.0, 1e9, 50.0, "Neutrale/Sano", "Rialzista",
                          1.0, 10.0, 3.5, "Sicura", 7, -2.5, "Nessun segnale", score)


FAKE_ERRORS = {
    "PEP": "Too Many Requests. Rate limited. Try after a while.",
    "PG": "Nessun dato storico trovato",
    "XOM": "riga uno\n   riga due   ",  # messaggio su più righe: va messo su una riga sola
    "ZTS": "",  # eccezione senza testo: prima finiva tra i risultati buoni
}


# punteggi tutti diversi: a parità di punteggio l'ordine dipenderebbe da chi finisce prima
FAKE_SCORES = {"GOOGL": 5.0, "AAPL": 4.0, "MSFT": 3.0}


def _fake_analyze(ticker: str) -> TickerAnalysis:
    if ticker in FAKE_ERRORS:
        return _empty(ticker, FAKE_ERRORS[ticker])
    return _ok(ticker, score=FAKE_SCORES[ticker])


def _run(monkeypatch, tmp_path, tickers, yahoo=_fake_analyze, pauses=(0.0,)):
    monkeypatch.setattr(run_screening, "get_core_universe", lambda: list(tickers))
    monkeypatch.setattr(screener, "_analyze_ticker_once", yahoo)
    monkeypatch.setattr(run_screening, "STAGGER_SECONDS", 0)
    monkeypatch.setattr(run_screening, "GATE_PAUSES", pauses)
    monkeypatch.setattr(run_screening, "SECOND_PASS_START_PAUSE", 0)
    monkeypatch.setattr(run_screening, "SECOND_PASS_SPACING", 0)
    monkeypatch.setattr(run_screening, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_screening, "LATEST_PATH", tmp_path / "latest_screening.json")
    run_screening.run()
    return json.loads((tmp_path / "latest_screening.json").read_text())


def test_failed_tickers_are_saved_with_reason(monkeypatch, tmp_path, capsys):
    data = _run(monkeypatch, tmp_path, ["AAPL", "PEP", "PG", "MSFT", "XOM", "ZTS"])

    assert data["failed"] == [
        {"ticker": "PEP", "error": "Too Many Requests. Rate limited. Try after a while."},
        {"ticker": "PG", "error": "Nessun dato storico trovato"},
        {"ticker": "XOM", "error": "riga uno riga due"},
        {"ticker": "ZTS", "error": "errore senza messaggio"},
    ]
    assert data["errors"] == 4
    assert data["analyzed_ok"] == 2
    assert data["universe_size"] == 6

    log = capsys.readouterr().out
    assert "ERRORE PEP: Too Many Requests" in log
    assert "Titoli non analizzati (4): PEP PG XOM ZTS" in log


def test_existing_fields_unchanged(monkeypatch, tmp_path):
    """Sistema di trading e PATTY leggono "generated_at" e "results": devono restare come prima."""
    data = _run(monkeypatch, tmp_path, ["AAPL", "PEP", "MSFT", "GOOGL"])

    assert set(data) == {"generated_at", "universe_size", "analyzed_ok", "errors",
                         "elapsed_seconds", "results", "failed"}
    assert [r["ticker"] for r in data["results"]] == ["GOOGL", "AAPL", "MSFT"]  # per punteggio
    assert all(r["error"] is None for r in data["results"])
    assert set(data["results"][0]) == set(_ok("X", 0).__dataclass_fields__)


def test_no_failures_gives_empty_list(monkeypatch, tmp_path, capsys):
    data = _run(monkeypatch, tmp_path, ["AAPL", "MSFT"])

    assert data["failed"] == []
    assert data["errors"] == 0
    assert "Titoli non analizzati" not in capsys.readouterr().out


def test_short_error_truncates_long_messages():
    assert len(run_screening._short_error("x" * 1000)) == 300
    assert run_screening._short_error("x" * 1000).endswith("...")


# --- Fase 2: pausa comune e secondo giro lento -----------------------------------------------------

class CountingYahoo:
    """Yahoo finto: i titoli in `blocked` rispondono "Too Many Requests" per le prime `fail_calls` richieste."""

    def __init__(self, blocked=(), fail_calls=0, errors=None):
        self.blocked = set(blocked)
        self.fail_calls = fail_calls
        self.errors = errors or {}
        self.calls = {}
        self._lock = threading.Lock()

    def __call__(self, ticker):
        with self._lock:
            self.calls[ticker] = self.calls.get(ticker, 0) + 1
            n = self.calls[ticker]
        if ticker in self.errors:
            return _empty(ticker, self.errors[ticker])
        if ticker in self.blocked and n <= self.fail_calls:
            return _empty(ticker, YAHOO_LIMIT)
        return _ok(ticker, score=float(len(self.calls)))


def test_second_pass_recovers_blocked_tickers(monkeypatch, tmp_path, capsys):
    """Come il 5/11: un blocco di titoli (SPG…UBER) bloccato per tutto il primo giro."""
    attempts = run_screening.ATTEMPTS_PER_TICKER
    yahoo = CountingYahoo(blocked={"SPG", "TSLA", "UBER"}, fail_calls=attempts)
    data = _run(monkeypatch, tmp_path, ["AAPL", "MSFT", "SPG", "TSLA", "UBER", "V"], yahoo=yahoo)

    assert data["failed"] == []
    assert data["errors"] == 0
    assert data["analyzed_ok"] == 6
    assert {r["ticker"] for r in data["results"]} == {"AAPL", "MSFT", "SPG", "TSLA", "UBER", "V"}
    assert yahoo.calls["SPG"] == attempts + 1  # tutti i tentativi del primo giro, poi uno nel secondo
    assert yahoo.calls["AAPL"] == 1

    log = capsys.readouterr().out
    assert "SECONDO GIRO lento su 3 titoli bloccati da Yahoo: SPG TSLA UBER" in log
    assert "RECUPERATO TSLA" in log
    assert "Secondo giro: 3 recuperati su 3" in log


def test_second_pass_only_for_rate_limited_and_keeps_last_error(monkeypatch, tmp_path, capsys):
    yahoo = CountingYahoo(blocked={"SPG"}, fail_calls=10**6,  # bloccato per sempre
                          errors={"PG": "Nessun dato storico trovato"})
    data = _run(monkeypatch, tmp_path, ["AAPL", "PG", "SPG"], yahoo=yahoo)

    assert data["failed"] == [
        {"ticker": "PG", "error": "Nessun dato storico trovato"},
        {"ticker": "SPG", "error": YAHOO_LIMIT},
    ]
    assert data["errors"] == 2
    assert yahoo.calls["PG"] == 1  # errore che non dipende dal blocco: nessun nuovo tentativo
    assert yahoo.calls["SPG"] == 2 * run_screening.ATTEMPTS_PER_TICKER  # primo e secondo giro
    log = capsys.readouterr().out
    assert "ANCORA ERRORE SPG: Too Many Requests" in log
    assert "Secondo giro: 0 recuperati su 1" in log


def test_no_second_pass_without_block(monkeypatch, tmp_path, capsys):
    _run(monkeypatch, tmp_path, ["AAPL", "MSFT"])
    assert "SECONDO GIRO" not in capsys.readouterr().out


def test_screening_ends_anyway_when_block_outlasts_time_limit(monkeypatch, tmp_path, capsys):
    """Yahoo bloccato per sempre e pause più lunghe del tempo massimo: il JSON esce comunque, subito."""
    tickers = [f"T{i:03d}" for i in range(40)]
    yahoo = CountingYahoo(blocked=set(tickers), fail_calls=10**6)

    t0 = time.monotonic()
    data = _run(monkeypatch, tmp_path, tickers, yahoo=yahoo, pauses=(10_000.0,))

    assert time.monotonic() - t0 < 10  # nessuna pausa vera da 10.000 s
    assert data["analyzed_ok"] == 0
    assert data["errors"] == 40
    assert all("Too Many Requests" in f["error"] for f in data["failed"])
    assert sum(yahoo.calls.values()) <= 2 * run_screening.MAX_WORKERS  # niente raffica di richieste
    assert "Tempo massimo finito" in capsys.readouterr().out


def test_shared_pause_lets_parallel_tickers_through_short_block(monkeypatch, tmp_path, capsys):
    """Blocco breve (0,1 s veri) durante il primo giro: la pausa comune basta, niente secondo giro."""
    state = {"start": None}
    lock = threading.Lock()

    def yahoo(ticker):
        with lock:
            if state["start"] is None:
                state["start"] = time.monotonic()
        if time.monotonic() - state["start"] < 0.1:
            return _empty(ticker, YAHOO_LIMIT)
        return _ok(ticker, score=float(sum(map(ord, ticker))))

    tickers = [f"T{i:02d}" for i in range(12)]
    data = _run(monkeypatch, tmp_path, tickers, yahoo=yahoo, pauses=(0.05, 0.1, 0.2))

    assert data["failed"] == []
    assert data["analyzed_ok"] == 12
    log = capsys.readouterr().out
    assert "pause per il blocco di Yahoo" in log
    assert "SECONDO GIRO" not in log
