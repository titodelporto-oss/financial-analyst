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

from src.screener import analyze_ticker
from src.universe import get_core_universe

RESULTS_DIR = Path(__file__).resolve().parent / "cache"
LATEST_PATH = RESULTS_DIR / "latest_screening.json"
MAX_WORKERS = 4
STAGGER_SECONDS = 0.1  # spacing between submitted requests, to stay polite with Yahoo Finance


def run(limit: int | None = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tickers = get_core_universe()
    if limit:
        tickers = tickers[:limit]

    print(f"[{datetime.now()}] Avvio screening su {len(tickers)} titoli...", flush=True)
    start = time.time()

    results = []
    errors = 0
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for t in tickers:
            futures[ex.submit(analyze_ticker, t)] = t
            time.sleep(STAGGER_SECONDS)
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if r.error:
                errors += 1
            else:
                results.append(r)
            if done % 500 == 0:
                elapsed = time.time() - start
                print(f"  ...{done}/{len(tickers)} completati ({elapsed:.0f}s, {errors} errori finora)", flush=True)

    results.sort(key=lambda r: r.score, reverse=True)
    elapsed = time.time() - start

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(tickers),
        "analyzed_ok": len(results),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "results": [asdict(r) for r in results],
    }

    LATEST_PATH.write_text(json.dumps(payload, indent=2))
    print(
        f"[{datetime.now()}] Fatto in {elapsed/60:.1f} min. "
        f"{len(results)} titoli analizzati, {errors} errori. Salvato in {LATEST_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    limit_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(limit=limit_arg)
