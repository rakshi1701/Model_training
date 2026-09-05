"""Small Qt helpers shared by the controllers."""
from __future__ import annotations

import traceback
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class _WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)


class Worker(QRunnable):
    """Runs a callable off the GUI thread.

    The callable may accept a ``progress`` keyword — a callback that emits
    ``progress`` with a status line, which is how long uploads and downloads
    stream their state into the UI.
    """

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            self._emit(self.signals.failed, traceback.format_exc(limit=6))
        else:
            self._emit(self.signals.finished, result)

    @staticmethod
    def _emit(signal, payload):
        # The app can shut down while a worker is still in flight; the signal
        # object is gone by then and there is nobody left to notify.
        try:
            signal.emit(payload)
        except RuntimeError:
            pass


# Qt takes ownership of a started QRunnable, but the Python Worker (and the
# QObject carrying its signals) must be kept alive until it finishes, or the
# callbacks are dropped with "Signal source has been deleted".
_LIVE: set = set()


def run_async(fn, on_done=None, on_error=None, on_progress=None, *args, **kwargs):
    """Fire-and-forget background call with signals delivered on the GUI thread."""
    w = Worker(fn, *args, **kwargs)
    _LIVE.add(w)
    w.signals.finished.connect(lambda _r, w=w: _LIVE.discard(w))
    w.signals.failed.connect(lambda _e, w=w: _LIVE.discard(w))
    if on_done:
        w.signals.finished.connect(on_done)
    if on_error:
        w.signals.failed.connect(on_error)
    if on_progress:
        w.signals.progress.connect(on_progress)
    QThreadPool.globalInstance().start(w)
    return w


def human_size(n_bytes: float) -> str:
    mb = n_bytes / (1024 * 1024)
    return f"{mb/1024:.2f} GB" if mb >= 1024 else f"{mb:.1f} MB"
