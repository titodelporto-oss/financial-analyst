"""Batch job: screen the entire US common-stock universe and cache the results to disk.

Meant to run once a day after US market close (scheduled via macOS launchd).
The Streamlit dashboard (app.py) reads the latest cached file to display results
instantly instead of recomputing on every page load.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.ratelimit import DEFAULT_PAUSES, RateLimitGate
from src.screener import STOPPED_BY_DEADLINE, analyze_ticker, is_rate_limited
from src.universe import get_core_universe

RESULTS_DIR = Path(__file__).resolve().parent / "cache"
LATEST_PATH = RESULTS_DIR / "latest_screening.json"
MAX_WORKERS = 4
STAGGER_SECONDS = 0.1  # spacing between submitted requests, to stay polite with Yahoo Finance

# Blocco di Yahoo ("Too Many Requests"): pausa comune a tutti i titoli (30/60/120 s), poi un secondo
# giro lento, un titolo alla volta, sui soli titoli rimasti bloccati. Uno screening normale dura ~5 min:
# i tempi massimi servono solo a garantire che il JSON venga scritto comunque.
GATE_PAUSES = DEFAULT_PAUSES
ATTEMPTS_PER_TICKER = len(DEFAULT_PAUSES) + 1  # un tentativo per ogni pausa, più il primo
FIRST_PASS_MAX_MINUTES = 40   # oltre questo tempo il primo giro non ritenta più
TOTAL_MAX_MINUTES = 60        # oltre questo tempo nemmeno il secondo giro chiede altro a Yahoo
SECOND_PASS_START_PAUSE = 60.0  # riposo prima del secondo giro, per far scadere il blocco
SECOND_PASS_SPACING = 2.0       # secondi tra un titolo e l'altro nel secondo giro


def run(limit: int | None = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tickers = get_core_universe()
    if limit:
        tickers = tickers[:limit]

    print(f"[{datetime.now()}] Avvio screening su {len(tickers)} titoli...", flush=True)
    start = time.time()

    start_clock = time.monotonic()  # orologio per i tempi massimi (non cambia con l'ora di sistema)
    first_gate = RateLimitGate(pauses=GATE_PAUSES, deadline=start_clock + FIRST_PASS_MAX_MINUTES * 60)

    results = []
    failed = {}  # titoli non analizzati -> motivo: finiscono nel log e nel JSON
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for t in tickers:
            futures[ex.submit(analyze_ticker, t, retries=ATTEMPTS_PER_TICKER, gate=first_gate)] = t
            time.sleep(STAGGER_SECONDS)
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if r.error is not None:
                failed[r.ticker] = r.error
                print(f"  ERRORE {r.ticker}: {_short_error(r.error)}", flush=True)
            else:
                results.append(r)
            if done % 500 == 0:
                elapsed = time.time() - start
                print(f"  ...{done}/{len(tickers)} completati ({elapsed:.0f}s, {len(failed)} errori finora)", flush=True)

    if first_gate.pauses_taken:
        print(f"  Primo giro: {first_gate.pauses_taken} pause per il blocco di Yahoo", flush=True)

    blocked = [t for t in dict.fromkeys(tickers) if is_rate_limited(failed.get(t))]
    if blocked:
        results.extend(_second_pass(blocked, failed, deadline=start_clock + TOTAL_MAX_MINUTES * 60))

    errors = len(failed)
    failed_list = [{"ticker": t, "error": _short_error(e)} for t, e in sorted(failed.items())]

    results.sort(key=lambda r: r.score, reverse=True)
    elapsed = time.time() - start

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(tickers),
        "analyzed_ok": len(results),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "results": [asdict(r) for r in results],
        # voce aggiunta il 4/11/2026: chi legge solo "results" e "generated_at" non cambia nulla
        "failed": failed_list,
    }

    LATEST_PATH.write_text(json.dumps(payload, indent=2))
    print(
        f"[{datetime.now()}] Fatto in {elapsed/60:.1f} min. "
        f"{len(results)} titoli analizzati, {errors} errori. Salvato in {LATEST_PATH}",
        flush=True,
    )
    if failed_list:
        print(f"Titoli non analizzati ({len(failed_list)}): {' '.join(f['ticker'] for f in failed_list)}", flush=True)


def _second_pass(blocked: list[str], failed: dict, deadline: float) -> list:
    """Ritenta lentamente, un titolo alla volta, i titoli bloccati da Yahoo nel primo giro.

    I titoli recuperati escono da `failed` e vengono restituiti; per gli altri resta l'ultimo errore."""
    print(f"  SECONDO GIRO lento su {len(blocked)} titoli bloccati da Yahoo: {' '.join(blocked)}", flush=True)
    gate = RateLimitGate(pauses=GATE_PAUSES, deadline=deadline)
    time.sleep(max(0.0, min(SECOND_PASS_START_PAUSE, deadline - time.monotonic())))

    recovered = []
    for i, t in enumerate(blocked):
        if i:
            time.sleep(SECOND_PASS_SPACING)
        r = analyze_ticker(t, retries=ATTEMPTS_PER_TICKER, gate=gate)
        if r.error == STOPPED_BY_DEADLINE:  # tempo finito: gli altri restano col loro errore di Yahoo
            print(f"  Tempo massimo finito: {len(blocked) - i} titoli non ritentati", flush=True)
            break
        if r.error is None:
            del failed[t]
            recovered.append(r)
            print(f"  RECUPERATO {t}", flush=True)
        else:
            failed[t] = r.error
            print(f"  ANCORA ERRORE {t}: {_short_error(r.error)}", flush=True)
    print(f"  Secondo giro: {len(recovered)} recuperati su {len(blocked)}, {gate.pauses_taken} pause", flush=True)
    return recovered


def _short_error(error: str, max_len: int = 300) -> str:
    """Il messaggio di errore su una riga sola e non troppo lungo, per il log e il JSON."""
    text = " ".join(str(error).split()) or "errore senza messaggio"
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


if __name__ == "__main__":
    limit_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(limit=limit_arg)
