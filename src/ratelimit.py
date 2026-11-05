"""Pausa comune quando Yahoo Finance blocca le richieste ("Too Many Requests").

Lo screening analizza 4 titoli alla volta. Se ognuno ritentasse per conto suo dopo pochi secondi,
il blocco di Yahoo (che dura decine di secondi o qualche minuto) li farebbe fallire tutti. Con il
semaforo, appena uno trova il blocco si fermano tutti insieme: prima 30 s, poi 60, poi 120. La pausa
riparte da 30 s dopo una risposta buona. Un tempo massimo garantisce che lo screening finisca comunque.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

DEFAULT_PAUSES = (30.0, 60.0, 120.0)


class RateLimitGate:
    def __init__(
        self,
        pauses: tuple = DEFAULT_PAUSES,
        deadline: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = lambda msg: print(msg, flush=True),
    ):
        self._pauses = tuple(pauses)
        self.deadline = deadline  # momento (secondo l'orologio `clock`) oltre il quale non si chiede più nulla
        self._clock = clock
        self._sleep = sleep
        self._log = log
        self._lock = threading.Lock()
        self._pause_until = 0.0
        self._level = 0
        self.pauses_taken = 0

    def time_left(self) -> bool:
        return self.deadline is None or self._clock() < self.deadline

    def wait(self) -> bool:
        """Aspetta la fine della pausa in corso. False se il tempo massimo è finito (o finirebbe in pausa)."""
        with self._lock:
            until = self._pause_until
        if self.deadline is not None and until >= self.deadline:
            return False
        remaining = until - self._clock()
        if remaining > 0:
            self._sleep(remaining)
        return self.time_left()

    def report_limit(self) -> None:
        """Un titolo ha trovato il blocco: parte una pausa comune, se non ce n'è già una in corso."""
        with self._lock:
            now = self._clock()
            if now < self._pause_until:
                return  # stesso blocco già segnalato da un altro titolo
            pause = self._pauses[min(self._level, len(self._pauses) - 1)]
            self._level += 1
            self._pause_until = now + pause
            self.pauses_taken += 1
        self._log(f"  PAUSA di {pause:.0f} s: Yahoo blocca le richieste (pausa n. {self.pauses_taken})")

    def report_ok(self) -> None:
        """Risposta buona: la prossima pausa riparte dalla più corta."""
        with self._lock:
            self._level = 0
