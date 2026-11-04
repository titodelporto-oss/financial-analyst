"""Il giro notturno salva i titoli falliti (voce "failed") senza toccare le altre voci del JSON.

Nessuna richiesta a Yahoo: analyze_ticker e l'elenco dei titoli sono sostituiti da versioni finte."""

import json

import run_screening
from src.screener import TickerAnalysis, _empty


def _ok(ticker: str, score: float) -> TickerAnalysis:
    return TickerAnalysis(ticker, f"{ticker} Inc.", 100.0, 1e9, 50.0, "Neutrale/Sano", "Rialzista",
                          1.0, 10.0, 3.5, "Sicura", 7, -2.5, "Nessun segnale", score)


FAKE_ERRORS = {
    "PEP": "Too Many Requests. Rate limited. Try after a while.",
    "PG": "Nessun dato storico trovato",
    "XOM": "riga uno\n   riga due   ",  # messaggio su più righe: va messo su una riga sola
    "ZTS": "",  # eccezione senza testo: prima finiva tra i risultati buoni
}


def _fake_analyze(ticker: str) -> TickerAnalysis:
    if ticker in FAKE_ERRORS:
        return _empty(ticker, FAKE_ERRORS[ticker])
    return _ok(ticker, score=len(ticker))


def _run(monkeypatch, tmp_path, tickers):
    monkeypatch.setattr(run_screening, "get_core_universe", lambda: list(tickers))
    monkeypatch.setattr(run_screening, "analyze_ticker", _fake_analyze)
    monkeypatch.setattr(run_screening, "STAGGER_SECONDS", 0)
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
