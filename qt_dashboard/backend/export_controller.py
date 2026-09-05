"""Model Export Studio: compile checkpoints into deployment runtimes."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import Property, QObject, Signal, Slot

from . import core
from .app_state import url_to_path
from .util import human_size, run_async

FORMATS = [
    ("onnx", "ONNX (Cross-platform, Triton, OpenCV DNN)"),
    ("engine", "TensorRT (NVIDIA GPU Maximum Throughput)"),
    ("openvino", "OpenVINO (Intel CPU & iGPU Optimized)"),
    ("torchscript", "TorchScript (C++ PyTorch Runtime)"),
    ("coreml", "CoreML (Apple iOS / macOS Devices)"),
    ("tflite", "TFLite (Mobile, Android & Edge TPUs)"),
]


def export_checkpoint(ckpt: Path, fmt: str, imgsz: int, half: bool,
                      dynamic: bool, simplify: bool, int8: bool, device: str):
    """Runs the Ultralytics export and zips directory outputs for portability."""
    from ultralytics import YOLO
    out = YOLO(str(ckpt)).export(format=fmt, imgsz=imgsz, half=half,
                                 dynamic=dynamic, simplify=simplify,
                                 int8=int8, device=device)
    p = Path(out)
    if p.is_dir():
        zip_dst = core.EXPORTS_DIR / f"{p.name}.zip"
        shutil.make_archive(str(zip_dst.with_suffix("")), "zip", str(p))
        return {"path": str(zip_dst), "raw": str(p), "packaged": True,
                "size": human_size(zip_dst.stat().st_size)}
    return {"path": str(p), "raw": str(p), "packaged": False,
            "size": human_size(p.stat().st_size) if p.exists() else "—"}


class ExportController(QObject):
    resultChanged = Signal()
    busyChanged = Signal()
    toast = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = ""
        self._result: Dict[str, Any] = {}

    @Property("QVariantList", constant=True)
    def formats(self):
        return [{"key": k, "label": v} for k, v in FORMATS]

    @Property(str, notify=busyChanged)
    def busy(self):
        return self._busy

    @Property("QVariantMap", notify=resultChanged)
    def result(self):
        return self._result

    @Slot("QVariantMap")
    def exportModel(self, cfg):
        ckpt = core.resolve_model_path(url_to_path(cfg.get("model", "")))
        if not ckpt.exists():
            self.toast.emit("Invalid model checkpoint selected.", "error")
            return
        fmt = cfg.get("format", "onnx")
        self._busy = f"Compiling to {fmt.upper()} …"
        self._result = {}
        self.busyChanged.emit()
        self.resultChanged.emit()

        def work():
            return export_checkpoint(
                ckpt, fmt, int(cfg.get("imgsz", 640)), bool(cfg.get("half", False)),
                bool(cfg.get("dynamic", False)), bool(cfg.get("simplify", True)),
                bool(cfg.get("int8", False)), cfg.get("device", "cpu"))

        def done(res):
            self._busy = ""
            self._result = {**res, "format": fmt, "ok": True}
            self.busyChanged.emit()
            self.resultChanged.emit()
            self.toast.emit(f"Exported {fmt.upper()} → {res['path']}", "success")

        def fail(err):
            self._busy = ""
            msg = err.strip().splitlines()[-1]
            self._result = {"ok": False, "error": msg, "format": fmt}
            self.busyChanged.emit()
            self.resultChanged.emit()
            self.toast.emit(f"Export failed: {msg}", "error")

        run_async(work, on_done=done, on_error=fail)
