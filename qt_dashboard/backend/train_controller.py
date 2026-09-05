"""Local training: launch, signal-control and live metrics off results.csv."""
from __future__ import annotations

import re
import signal
import time
from pathlib import Path
from typing import Any, Dict, List

from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot

from . import core

PROGRESS_RE = re.compile(r"\d+%\|")
MAX_LOG_LINES = 4000


def _series(df, cols) -> List[Dict[str, Any]]:
    """Turns selected results.csv columns into QML-friendly chart series."""
    import pandas as pd
    out = []
    if df is None or df.empty:
        return out
    x = df["epoch"] if "epoch" in df.columns else pd.Series(range(1, len(df) + 1))
    for c in cols:
        pts = []
        for xv, yv in zip(x, df[c]):
            try:
                fx, fy = float(xv), float(yv)
            except (TypeError, ValueError):
                continue
            if fy == fy:  # skip NaN
                pts.append({"x": fx, "y": fy})
        if pts:
            out.append({"name": c, "points": pts})
    return out


class TrainController(QObject):
    """Owns the `yolo train` subprocess and everything the Training tab shows."""

    stateChanged = Signal()
    logChanged = Signal()
    metricsChanged = Signal()
    toast = Signal(str, str)

    def __init__(self, monitor=None, parent=None):
        super().__init__(parent)
        self._monitor = monitor
        self._proc = None
        self._logs: List[str] = []
        self._active = False
        self._paused = False
        self._run_name = core.next_free_exp_name()
        self._target_epochs = 100
        self._start_time = None
        self._metrics: Dict[str, Any] = {}
        self._curves: Dict[str, List] = {"loss": [], "acc": [], "lr": []}
        self._artifacts: Dict[str, Any] = {}

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._refresh_metrics()

    # -- state -------------------------------------------------------------
    @Property(bool, notify=stateChanged)
    def active(self):
        return self._active

    @Property(bool, notify=stateChanged)
    def paused(self):
        return self._paused

    @Property(str, notify=stateChanged)
    def runName(self):
        return self._run_name

    @runName.setter
    def runName(self, v):
        if v and v != self._run_name:
            self._run_name = v
            self.stateChanged.emit()
            self._refresh_metrics()

    @Property(int, notify=stateChanged)
    def targetEpochs(self):
        return self._target_epochs

    @targetEpochs.setter
    def targetEpochs(self, v):
        if v and v != self._target_epochs:
            self._target_epochs = int(v)
            self.stateChanged.emit()
            self._refresh_metrics()

    @Property(str, notify=stateChanged)
    def statusText(self):
        if self._active and self._paused:
            return "⏸ Training Paused"
        if self._active:
            return f"🟢 Training: {self._run_name}"
        return "⚪ Engine Ready"

    @Property(str, notify=stateChanged)
    def statusKind(self):
        return "paused" if (self._active and self._paused) else (
            "running" if self._active else "idle")

    @Property(str, notify=stateChanged)
    def elapsedText(self):
        if not self._start_time:
            return ""
        secs = int(time.time() - self._start_time)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

    # -- log ---------------------------------------------------------------
    @Property(str, notify=logChanged)
    def logText(self):
        if not self._logs:
            return "Ready to train. Output stream will appear here."
        return "".join(self._logs[-400:])

    # -- metrics -----------------------------------------------------------
    @Property("QVariantMap", notify=metricsChanged)
    def metrics(self):
        return self._metrics

    @Property("QVariantList", notify=metricsChanged)
    def lossSeries(self):
        return self._curves["loss"]

    @Property("QVariantList", notify=metricsChanged)
    def accSeries(self):
        return self._curves["acc"]

    @Property("QVariantList", notify=metricsChanged)
    def lrSeries(self):
        return self._curves["lr"]

    @Property("QVariantMap", notify=metricsChanged)
    def artifacts(self):
        return self._artifacts

    # -- lifecycle ---------------------------------------------------------
    @Slot("QVariantMap")
    def start(self, cfg):
        if self._active:
            return
        data_yaml = cfg.get("data")
        if not data_yaml:
            self.toast.emit("Select a dataset before starting a run.", "warn")
            return

        run_name = cfg.get("name") or self._run_name
        model_name = cfg.get("model")
        cmd = core.yolo_cli() + [
            "train",
            f"model={model_name}",
            f"data={data_yaml}",
            f"epochs={int(cfg.get('epochs', 100))}",
            f"batch={int(cfg.get('batch', 16))}",
            f"imgsz={int(cfg.get('imgsz', 640))}",
            f"optimizer={cfg.get('optimizer', 'AdamW')}",
            f"lr0={float(cfg.get('lr0', 0.005))}",
            f"patience={int(cfg.get('patience', 50))}",
            f"device={cfg.get('device', 'cpu')}",
            f"workers={int(cfg.get('workers', 8))}",
            f"resume={bool(cfg.get('resume', False))}",
            f"cache={cfg.get('cache', 'False')}",
            f"cos_lr={bool(cfg.get('cos_lr', True))}",
            f"weight_decay={float(cfg.get('weight_decay', 0.0005))}",
            f"freeze={int(cfg.get('freeze', 0))}",
            f"project={core.RUNS_DIR}",
            f"name={run_name}",
        ]

        self._logs = [f"$ {' '.join(cmd)}\n"]
        self.logChanged.emit()
        try:
            self._proc = core.spawn_process(cmd)
        except FileNotFoundError:
            self.toast.emit("`yolo` CLI not found in this environment.", "error")
            return

        core.reader_thread(self._proc, self._append_log, PROGRESS_RE)
        self._active, self._paused = True, False
        self._run_name = run_name
        self._target_epochs = int(cfg.get("epochs", 100))
        self._start_time = time.time()
        if self._monitor:
            self._monitor.track_process(self._proc.pid)
        self.stateChanged.emit()
        self.toast.emit(f"Training started → runs/{run_name}", "success")

    @Slot()
    def pause(self):
        if self._active and not self._paused and core.signal_group(self._proc, signal.SIGSTOP):
            self._paused = True
            self.stateChanged.emit()
            self.toast.emit("Training paused (SIGSTOP).", "info")

    @Slot()
    def resume(self):
        if self._active and self._paused and core.signal_group(self._proc, signal.SIGCONT):
            self._paused = False
            self.stateChanged.emit()
            self.toast.emit("Training resumed.", "info")

    @Slot()
    def terminate(self):
        if not self._active:
            return
        if self._paused:
            core.signal_group(self._proc, signal.SIGCONT)
        if not core.signal_group(self._proc, signal.SIGTERM) and self._proc:
            self._proc.terminate()
        self._active = self._paused = False
        if self._monitor:
            self._monitor.track_process(None)
        self.stateChanged.emit()
        self.toast.emit("Termination signal sent to the training group.", "warn")

    # -- polling -----------------------------------------------------------
    def _append_log(self, line):
        self._logs.append(line)
        if len(self._logs) > MAX_LOG_LINES:
            del self._logs[:-MAX_LOG_LINES]

    def _tick(self):
        if self._logs:
            self.logChanged.emit()
        if self._active:
            self.stateChanged.emit()
            if self._proc and self._proc.poll() is not None:
                ret = self._proc.poll()
                self._active = self._paused = False
                if self._monitor:
                    self._monitor.track_process(None)
                self.stateChanged.emit()
                if ret == 0:
                    self.toast.emit("🎉 Training completed successfully!", "success")
                else:
                    self.toast.emit(f"Training exited with code {ret}.", "error")
        self._refresh_metrics()

    @Slot()
    def _refresh_metrics(self):
        df, run_dir = core.load_run_results(self._run_name)
        if df is None or df.empty:
            if self._metrics:
                self._metrics, self._artifacts = {}, {}
                self._curves = {"loss": [], "acc": [], "lr": []}
                self.metricsChanged.emit()
            return

        import pandas as pd
        last = df.iloc[-1]
        cur_epoch = int(last.get("epoch", len(df)))

        def stat(col):
            if col not in df.columns:
                return None, None
            try:
                v = float(last[col])
            except (TypeError, ValueError):
                return None, None
            d = None
            if len(df) > 1:
                try:
                    d = v - float(df.iloc[-2][col])
                except (TypeError, ValueError):
                    d = None
            return v, d

        box, box_d = stat("val/box_loss")
        if box is None:
            box, box_d = stat("train/box_loss")
        cls, cls_d = stat("val/cls_loss")
        if cls is None:
            cls, cls_d = stat("train/cls_loss")
        m50, m50_d = stat("metrics/mAP50(B)")
        m95, m95_d = stat("metrics/mAP50-95(B)")
        prec, _ = stat("metrics/precision(B)")
        rec, _ = stat("metrics/recall(B)")

        eta = "Calculating…"
        if len(df) > 1 and "time" in df.columns:
            avg = df["time"].diff().mean()
            if pd.notnull(avg) and avg > 0:
                rem = max(0, self._target_epochs - cur_epoch) * avg
                mins, secs = divmod(int(rem), 60)
                hrs, mins = divmod(mins, 60)
                eta = f"{hrs}h {mins}m" if hrs else f"{mins}m {secs}s"

        total = max(1, self._target_epochs)
        self._metrics = {
            "epoch": cur_epoch, "totalEpochs": total,
            "progress": min(1.0, max(0.0, cur_epoch / total)),
            "eta": eta,
            "boxLoss": box, "boxDelta": box_d,
            "clsLoss": cls, "clsDelta": cls_d,
            "map50": m50, "map50Delta": m50_d,
            "map5095": m95, "map5095Delta": m95_d,
            "precision": prec, "recall": rec,
        }

        self._curves = {
            "loss": _series(df, [c for c in df.columns if "loss" in c.lower()]),
            "acc": _series(df, [c for c in df.columns if any(
                m in c.lower() for m in ("map", "precision", "recall", "accuracy"))]),
            "lr": _series(df, [c for c in df.columns if "lr/" in c.lower()]),
        }

        art: Dict[str, Any] = {}
        if run_dir and run_dir.exists():
            art["dir"] = str(run_dir)
            for tag, p in (("best", run_dir / "weights" / "best.pt"),
                           ("last", run_dir / "weights" / "last.pt")):
                if p.exists():
                    art[tag] = str(p)
                    art[f"{tag}Size"] = f"{p.stat().st_size / (1024*1024):.1f} MB"
            png = run_dir / "results.png"
            if png.exists():
                art["plot"] = str(png)
                art["plotStamp"] = int(png.stat().st_mtime)
        self._artifacts = art
        self.metricsChanged.emit()

    @Slot(result="QStringList")
    def runNames(self):
        if not core.RUNS_DIR.exists():
            return []
        return sorted((d.name for d in core.RUNS_DIR.iterdir()
                       if d.is_dir() and d.name != "kaggle_models"),
                      key=str.lower)
