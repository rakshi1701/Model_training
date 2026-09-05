#!/usr/bin/env python3
"""
YOLO Studio — Qt/QML desktop dashboard.

Same workspace, same engine and the same seven tabs as the Streamlit app in
``app.py``; this front-end just drives them from a native window.

Run with:  python qt_dashboard/main.py
"""
from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # so `import kaggle_bridge` resolves
sys.path.insert(0, str(HERE))

from PySide6.QtCore import QUrl, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QIcon  # noqa: E402
from PySide6.QtQml import QQmlApplicationEngine  # noqa: E402
from PySide6.QtQuickControls2 import QQuickStyle  # noqa: E402

from backend.app_state import AppState  # noqa: E402
from backend.dataset_controller import DatasetController  # noqa: E402
from backend.export_controller import ExportController  # noqa: E402
from backend.history_controller import HistoryController  # noqa: E402
from backend.inference_controller import InferenceController  # noqa: E402
from backend.system_monitor import SystemMonitor  # noqa: E402
from backend.train_controller import TrainController  # noqa: E402
from backend.tune_controller import TuneController  # noqa: E402


def main() -> int:
    # Ctrl+C in the launching terminal should close the window.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    QGuiApplication.setOrganizationName("YOLO Studio")
    QGuiApplication.setApplicationName("YOLO Studio")
    app = QGuiApplication(sys.argv)
    QQuickStyle.setStyle("Basic")      # the custom dark theme styles Basic

    monitor = SystemMonitor()
    state = AppState()
    train = TrainController(monitor=monitor)
    datasets = DatasetController()
    inference = InferenceController()
    export = ExportController()
    tune = TuneController()
    history = HistoryController()

    # Kaggle is optional: a missing/broken kaggle package must not stop the app.
    kaggle = None
    kaggle_error = ""
    try:
        from backend.kaggle_controller import KaggleController
        kaggle = KaggleController()
    except Exception as e:                       # pragma: no cover - env dependent
        kaggle_error = f"{type(e).__name__}: {e}"

    engine = QQmlApplicationEngine()
    ctx = engine.rootContext()
    ctx.setContextProperty("App", state)
    ctx.setContextProperty("Monitor", monitor)
    ctx.setContextProperty("Train", train)
    ctx.setContextProperty("Datasets", datasets)
    ctx.setContextProperty("Inference", inference)
    ctx.setContextProperty("Exporter", export)
    ctx.setContextProperty("Tuner", tune)
    ctx.setContextProperty("History", history)
    ctx.setContextProperty("Kaggle", kaggle)
    ctx.setContextProperty("KaggleError", kaggle_error)

    engine.addImportPath(str(HERE / "qml"))
    engine.load(QUrl.fromLocalFile(str(HERE / "qml" / "Main.qml")))
    if not engine.rootObjects():
        print("Failed to load Main.qml", file=sys.stderr)
        return 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
