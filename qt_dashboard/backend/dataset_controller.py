"""Dataset Hub: split browsing and the ground-truth annotation inspector."""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

from PySide6.QtCore import Property, QObject, Signal, Slot

from . import core
from .app_state import url_to_path

PREVIEW_DIR = core.TEMP_DIR / "gt_preview"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


class DatasetController(QObject):
    """Renders ground-truth boxes for one image of a split at a time."""

    imagesChanged = Signal()
    previewChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._images: List[Path] = []
        self._class_names: dict = {}
        self._preview = ""
        self._boxes = 0
        self._index = 0

    @Property(int, notify=imagesChanged)
    def imageCount(self):
        return len(self._images)

    @Property(str, notify=previewChanged)
    def previewPath(self):
        return self._preview

    @Property(int, notify=previewChanged)
    def boxCount(self):
        return self._boxes

    @Property(str, notify=previewChanged)
    def currentName(self):
        if 0 <= self._index < len(self._images):
            return self._images[self._index].name
        return ""

    @Slot(str, "QVariantMap")
    def loadSplit(self, split_dir, class_names=None):
        self._class_names = self._coerce(class_names)
        self._images = core.list_split_images(url_to_path(split_dir)) if split_dir else []
        self.imagesChanged.emit()
        if self._images:
            self.showIndex(0)
        else:
            self._preview, self._boxes = "", 0
            self.previewChanged.emit()

    @staticmethod
    def _coerce(class_names) -> dict:
        """QML hands class ids over as strings; the label writer wants ints."""
        names = {}
        for k, v in dict(class_names or {}).items():
            try:
                names[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        return names

    @Slot(int)
    @Slot(int, "QVariantMap")
    def showIndex(self, index, class_names=None):
        if not self._images:
            return
        self._index = max(0, min(index, len(self._images) - 1))
        img = self._images[self._index]
        names = self._coerce(class_names) if class_names else self._class_names
        out = PREVIEW_DIR / f"gt_{int(time.time()*1000)}.png"
        for stale in PREVIEW_DIR.glob("gt_*.png"):
            try:
                stale.unlink()
            except OSError:
                pass
        n = core.draw_ground_truth_boxes(img, names, out)
        if n is None:
            self._preview, self._boxes = "", 0
        else:
            self._preview, self._boxes = str(out), n
        self.previewChanged.emit()

    @Slot(int, result=str)
    def imagePath(self, index):
        if 0 <= index < len(self._images):
            return str(self._images[index])
        return ""
