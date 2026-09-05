"""Hyperparameter tuning: `yolo tune` plus the best_hyperparameters.yaml reader."""
from __future__ import annotations

import re
import signal
from pathlib import Path
from typing import Any, Dict, List

import yaml
from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot

from . import core

PROGRESS_RE = re.compile(r"\d+%\|")


class TuneController(QObject):
    stateChanged = Signal()
    logChanged = Signal()
    bestChanged = Signal()
    toast = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = None
        self._logs: List[str] = []
        self._active = False
        self._run_name = "tune1"
        self._best: Dict[str, Any] = {}
        self._best_source = ""

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._reload_best()

    @Property(bool, notify=stateChanged)
    def active(self):
        return self._active

    @Property(str, notify=stateChanged)
    def runName(self):
        return self._run_name

    @Property(str, notify=logChanged)
    def logText(self):
        if not self._logs:
            return "Ready for hyperparameter tuning. Click 'Start Tuning'."
        return "".join(self._logs[-400:])

    @Property("QVariantList", notify=bestChanged)
    def bestParams(self):
        return [{"key": k, "value": str(v)} for k, v in sorted(self._best.items())]

    @Property("QVariantMap", notify=bestChanged)
    def bestParamsMap(self):
        return self._best

    @Property(str, notify=bestChanged)
    def bestSource(self):
        return self._best_source

    @Slot("QVariantMap")
    def start(self, cfg):
        if self._active:
            return
        if not cfg.get("data"):
            self.toast.emit("Select a dataset before tuning.", "warn")
            return
        name = cfg.get("name") or self._run_name
        cmd = core.yolo_cli() + [
            "tune",
            f"model={cfg.get('model')}",
            f"data={cfg.get('data')}",
            f"epochs={int(cfg.get('epochs', 10))}",
            f"iterations={int(cfg.get('iterations', 15))}",
            f"optimizer={cfg.get('optimizer', 'auto')}",
            f"device={cfg.get('device', 'cpu')}",
            f"project={core.TUNE_DIR}",
            f"name={name}",
        ]
        self._logs = [f"$ {' '.join(cmd)}\n"]
        self.logChanged.emit()
        try:
            self._proc = core.spawn_process(cmd)
        except FileNotFoundError:
            self.toast.emit("`yolo` CLI not found in this environment.", "error")
            return
        core.reader_thread(self._proc, self._append, PROGRESS_RE)
        self._active = True
        self._run_name = name
        self.stateChanged.emit()
        self.toast.emit(f"Tuning started → tune/{name}", "success")

    @Slot()
    def stop(self):
        if not self._active:
            return
        if not core.signal_group(self._proc, signal.SIGTERM) and self._proc:
            self._proc.terminate()
        self._active = False
        self.stateChanged.emit()
        self.toast.emit("Tuning stopped.", "warn")

    def _append(self, line):
        self._logs.append(line)
        if len(self._logs) > 4000:
            del self._logs[:-4000]

    def _tick(self):
        if self._logs:
            self.logChanged.emit()
        if self._active and self._proc and self._proc.poll() is not None:
            self._active = False
            self.stateChanged.emit()
            self.toast.emit("🎉 Hyperparameter optimization finished!", "success")
        self._reload_best()

    def _reload_best(self):
        best_yaml = core.TUNE_DIR / self._run_name / "best_hyperparameters.yaml"
        if not best_yaml.exists():
            best_yaml = next(core.TUNE_DIR.rglob("best_hyperparameters.yaml"), None)
        if not best_yaml or not best_yaml.exists():
            return
        try:
            data = yaml.safe_load(best_yaml.read_text()) or {}
        except Exception:
            return
        if data != self._best or self._best_source != best_yaml.parent.name:
            self._best = data
            self._best_source = best_yaml.parent.name
            self.bestChanged.emit()
