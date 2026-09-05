"""Run history table and multi-run curve overlays."""
from __future__ import annotations

from typing import Any, Dict, List

from PySide6.QtCore import Property, QObject, Signal, Slot

from . import core

COMPARE_METRICS = ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
                   "train/box_loss", "val/box_loss"]


class HistoryController(QObject):
    runsChanged = Signal()
    overlayChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._runs: List[Dict[str, Any]] = []
        self._overlay: List[Dict[str, Any]] = []
        self.refresh()

    @Property("QVariantList", notify=runsChanged)
    def runs(self):
        return self._runs

    @Property("QStringList", notify=runsChanged)
    def runNames(self):
        return [r["name"] for r in self._runs]

    @Property("QStringList", constant=True)
    def compareMetrics(self):
        return COMPARE_METRICS

    @Property("QVariantList", notify=overlayChanged)
    def overlay(self):
        return self._overlay

    @Slot()
    def refresh(self):
        records = []
        if core.RUNS_DIR.exists():
            dirs = [d for d in core.RUNS_DIR.iterdir()
                    if d.is_dir() and d.name != "kaggle_models"]
            for rd in sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True):
                df, _ = core.load_run_results(rd.name)
                epochs = len(df) if df is not None else 0
                m50 = float(df["metrics/mAP50(B)"].max()) if (
                    df is not None and "metrics/mAP50(B)" in df.columns) else 0.0
                m95 = float(df["metrics/mAP50-95(B)"].max()) if (
                    df is not None and "metrics/mAP50-95(B)" in df.columns) else 0.0
                best = rd / "weights" / "best.pt"
                records.append({
                    "name": rd.name,
                    "epochs": epochs,
                    "map50": f"{m50:.4f}" if m50 > 0 else "N/A",
                    "map5095": f"{m95:.4f}" if m95 > 0 else "N/A",
                    "hasWeights": best.exists(),
                    "weights": str(best) if best.exists() else "",
                    "dir": str(rd),
                    "owner": self._owner(rd),
                })
        self._runs = records
        self.runsChanged.emit()

    @staticmethod
    def _owner(run_dir):
        try:
            import kaggle_bridge
            return kaggle_bridge.run_dir_owner(run_dir) or ""
        except Exception:
            return ""

    @Slot("QStringList", str)
    def buildOverlay(self, run_names, metric):
        series = []
        for name in run_names:
            df, _ = core.load_run_results(name)
            if df is None or metric not in df.columns:
                continue
            x = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)
            pts = []
            for xv, yv in zip(x, df[metric]):
                try:
                    fx, fy = float(xv), float(yv)
                except (TypeError, ValueError):
                    continue
                if fy == fy:
                    pts.append({"x": fx, "y": fy})
            if pts:
                series.append({"name": name, "points": pts})
        self._overlay = series
        self.overlayChanged.emit()

    @Slot(str, result="QVariantList")
    def runFigures(self, run_name):
        """Confusion matrix, PR curve and validation batch previews for a run."""
        from pathlib import Path
        out = []
        for r in self._runs:
            if r["name"] != run_name:
                continue
            d = Path(r["dir"])
            for png in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
                out.append({"name": png.name, "path": str(png)})
        return out
