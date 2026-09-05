"""Inference playground: images, validation samples, webcam frames and video."""
from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

from PySide6.QtCore import Property, QObject, Signal, Slot

from . import core
from .app_state import url_to_path
from .util import run_async

OUT_DIR = core.TEMP_DIR / "inference"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class InferenceController(QObject):
    """Runs YOLO predictions off the GUI thread and hands QML image paths back."""

    modelChanged = Signal()
    resultsChanged = Signal()
    busyChanged = Signal()
    videoProgress = Signal(float, str)
    toast = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._model_info: Dict[str, Any] = {}
        self._model_path = ""
        self._results: List[Dict[str, Any]] = []
        self._busy = ""
        self._video_out = ""

    # -- properties --------------------------------------------------------
    @Property("QVariantMap", notify=modelChanged)
    def modelInfo(self):
        return self._model_info

    @Property("QVariantList", notify=resultsChanged)
    def results(self):
        return self._results

    @Property(str, notify=busyChanged)
    def busy(self):
        return self._busy

    @Property(str, notify=busyChanged)
    def videoOutput(self):
        return self._video_out

    def _set_busy(self, msg):
        self._busy = msg
        self.busyChanged.emit()

    # -- model -------------------------------------------------------------
    def _load(self, path: str):
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        from ultralytics import YOLO
        model = YOLO(key)
        self._cache[key] = model
        while len(self._cache) > 3:
            self._cache.popitem(last=False)
        return model

    @Slot(str)
    def loadModel(self, path):
        p = core.resolve_model_path(url_to_path(path))
        if not p.exists():
            self._model_info = {}
            self.modelChanged.emit()
            return
        self._model_path = str(p)
        self._set_busy(f"Loading {p.name} …")

        def work():
            m = self._load(str(p))
            return {"name": p.name, "path": str(p), "task": str(m.task),
                    "classes": len(m.names)}

        def done(info):
            self._set_busy("")
            self._model_info = info
            self.modelChanged.emit()

        def fail(err):
            self._set_busy("")
            self._model_info = {}
            self.modelChanged.emit()
            self.toast.emit(f"Could not load model: {err.strip().splitlines()[-1]}", "error")

        run_async(work, on_done=done, on_error=fail)

    # -- image inference ---------------------------------------------------
    def _predict_one(self, img_path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
        import cv2
        model = self._load(self._model_path)
        t0 = time.time()
        res = model.predict(
            source=str(img_path),
            conf=float(cfg.get("conf", 0.25)),
            iou=float(cfg.get("iou", 0.45)),
            imgsz=int(cfg.get("imgsz", 640)),
            device=cfg.get("device", "cpu"),
            boxes=bool(cfg.get("boxes", True)),
            show_labels=bool(cfg.get("labels", True)),
            show_conf=bool(cfg.get("conf_labels", True)),
            line_width=int(cfg.get("lineWidth", 2)),
            verbose=False,
        )[0]
        ms = (time.time() - t0) * 1000

        stamp = int(time.time() * 1000)
        out = OUT_DIR / f"det_{stamp}_{img_path.stem}.png"
        cv2.imwrite(str(out), res.plot())

        dets = []
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes:
                cid = int(b.cls[0].item())
                dets.append({
                    "cls": model.names.get(cid, str(cid)),
                    "conf": f"{float(b.conf[0].item()) * 100:.1f}%",
                    "coords": ", ".join(f"{float(c):.1f}" for c in b.xyxy[0].tolist()),
                })
        speed = res.speed or {}
        return {
            "name": img_path.name,
            "original": str(img_path),
            "annotated": str(out),
            "ms": round(ms, 1),
            "preprocess": round(float(speed.get("preprocess") or 0), 1),
            "inference": round(float(speed.get("inference") or 0), 1),
            "postprocess": round(float(speed.get("postprocess") or 0), 1),
            "count": len(dets),
            "detections": dets,
        }

    @Slot("QStringList", "QVariantMap")
    def runImages(self, paths, cfg):
        if not self._model_path:
            self.toast.emit("Select a model checkpoint first.", "warn")
            return
        files = [Path(url_to_path(p)) for p in paths]
        files = [f for f in files if f.exists()]
        if not files:
            return
        self._set_busy(f"Running inference on {len(files)} image(s) …")

        def work():
            return [self._predict_one(f, dict(cfg)) for f in files]

        def done(results):
            self._set_busy("")
            self._results = results
            self.resultsChanged.emit()

        def fail(err):
            self._set_busy("")
            self.toast.emit(f"Inference error: {err.strip().splitlines()[-1]}", "error")

        run_async(work, on_done=done, on_error=fail)

    @Slot("QVariantMap")
    def captureWebcam(self, cfg):
        """Grabs one frame from the default camera, then runs inference on it."""
        if not self._model_path:
            self.toast.emit("Select a model checkpoint first.", "warn")
            return
        self._set_busy("Capturing webcam frame …")

        def work():
            import cv2
            cap = cv2.VideoCapture(int(cfg.get("camera", 0)))
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok or frame is None:
                raise RuntimeError("No frame from the camera device.")
            snap = OUT_DIR / f"webcam_{int(time.time()*1000)}.png"
            cv2.imwrite(str(snap), frame)
            return [self._predict_one(snap, dict(cfg))]

        def done(results):
            self._set_busy("")
            self._results = results
            self.resultsChanged.emit()

        def fail(err):
            self._set_busy("")
            self.toast.emit(f"Webcam capture failed: {err.strip().splitlines()[-1]}", "error")

        run_async(work, on_done=done, on_error=fail)

    # -- video -------------------------------------------------------------
    @Slot(str, "QVariantMap")
    def runVideo(self, video_url, cfg):
        if not self._model_path:
            self.toast.emit("Select a model checkpoint first.", "warn")
            return
        src = Path(url_to_path(video_url))
        if not src.exists():
            self.toast.emit("Video file not found.", "error")
            return
        self._video_out = ""
        self._set_busy(f"Processing {src.name} …")

        def work():
            import cv2
            model = self._load(self._model_path)
            cap = cv2.VideoCapture(str(src))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
            out_path = OUT_DIR / f"annotated_{src.stem}.mp4"
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            idx = 0
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                res = model.predict(
                    source=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    conf=float(cfg.get("conf", 0.25)),
                    iou=float(cfg.get("iou", 0.45)),
                    imgsz=int(cfg.get("imgsz", 640)),
                    device=cfg.get("device", "cpu"),
                    verbose=False,
                )[0]
                writer.write(res.plot())
                idx += 1
                if idx % 5 == 0 or idx == total:
                    self.videoProgress.emit(min(idx / total, 1.0),
                                            f"Processed frame {idx}/{total}")
            cap.release()
            writer.release()
            return str(out_path)

        def done(path):
            self._set_busy("")
            self._video_out = path
            self.videoProgress.emit(1.0, "Complete")
            self.busyChanged.emit()
            self.toast.emit("Video processing complete.", "success")

        def fail(err):
            self._set_busy("")
            self.videoProgress.emit(0.0, "")
            self.toast.emit(f"Video inference failed: {err.strip().splitlines()[-1]}", "error")

        run_async(work, on_done=done, on_error=fail)

    @Slot()
    def clear(self):
        self._results = []
        self.resultsChanged.emit()

    @Slot(str, result="QStringList")
    def imagesIn(self, folder):
        return [str(p) for p in core.list_split_images(url_to_path(folder))]
