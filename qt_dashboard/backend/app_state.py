"""Shared application state: hardware, the active dataset, presets, model list."""
from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from PySide6.QtCore import (Property, QObject, QUrl, Signal, Slot)

from . import core
from .util import run_async


def url_to_path(url) -> str:
    """QML file dialogs hand back file:// URLs; controllers want plain paths."""
    if isinstance(url, QUrl):
        return url.toLocalFile()
    s = str(url)
    return QUrl(s).toLocalFile() if s.startswith("file:") else s


class AppState(QObject):
    """The sidebar's dataset selection plus everything the tabs read from it."""

    datasetsChanged = Signal()
    activeDatasetChanged = Signal()
    modelsChanged = Signal()
    tunedParamsChanged = Signal()
    busyChanged = Signal()
    toast = Signal(str, str)          # message, level: info|success|warn|error

    def __init__(self, parent=None):
        super().__init__(parent)
        self._datasets: Dict[str, Path] = {}
        self._active_key = ""
        self._info: Dict[str, Any] = {}
        self._models: List[str] = []
        self._tuned: Dict[str, Any] = {}
        self._busy = ""
        self._extracting = False
        self.refreshDatasets()
        self.refreshModels()

    # -- hardware ----------------------------------------------------------
    @Property(bool, constant=True)
    def cudaAvailable(self):
        return core.CUDA_AVAILABLE

    @Property(int, constant=True)
    def gpuCount(self):
        return core.GPU_COUNT

    @Property("QStringList", constant=True)
    def gpuNames(self):
        return core.GPU_NAMES

    @Property("QStringList", constant=True)
    def deviceOptions(self):
        """Labels for the compute selector; index maps to deviceValues."""
        opts = []
        for i, name in enumerate(core.GPU_NAMES):
            opts.append(f"GPU {i}: {name}")
        if core.GPU_COUNT > 1:
            opts.append(f"All GPUs (0-{core.GPU_COUNT - 1})")
        opts.append("CPU")
        return opts

    @Property("QStringList", constant=True)
    def deviceValues(self):
        vals = [str(i) for i in range(core.GPU_COUNT)]
        if core.GPU_COUNT > 1:
            vals.append(",".join(str(i) for i in range(core.GPU_COUNT)))
        vals.append("cpu")
        return vals

    @Property(str, constant=True)
    def projectRoot(self):
        return str(core.PROJECT_ROOT)

    @Property(str, constant=True)
    def runsDir(self):
        return str(core.RUNS_DIR)

    # -- presets & architecture menus --------------------------------------
    @Property("QStringList", constant=True)
    def presetNames(self):
        return list(core.load_presets().keys())

    @Slot(str, result="QVariantMap")
    def preset(self, name):
        return core.load_presets().get(name, {})

    @Property("QStringList", constant=True)
    def modelFamilies(self):
        return core.MODEL_FAMILIES

    @Slot(str, result="QStringList")
    def modelSizes(self, family):
        return core.SIZE_MAP.get(family, [])

    @Slot(str, result="QStringList")
    def kaggleWeights(self, family):
        return core.KAGGLE_SIZE_MAP.get(family, [])

    @Slot(str, result=str)
    def taskSuffix(self, task):
        return core.TASK_SUFFIX.get(task, "")

    # -- datasets ----------------------------------------------------------
    @Property("QStringList", notify=datasetsChanged)
    def datasetNames(self):
        return list(self._datasets.keys())

    @Property(str, notify=activeDatasetChanged)
    def activeDatasetKey(self):
        return self._active_key

    @Property(str, notify=activeDatasetChanged)
    def dataYamlPath(self):
        p = self._datasets.get(self._active_key)
        return str(p) if p else ""

    @Property("QVariantMap", notify=activeDatasetChanged)
    def datasetInfo(self):
        """{name, yaml, classes:[{id,name}], counts:[{split,count}], splits:[...]}"""
        return self._info

    @Property(bool, notify=activeDatasetChanged)
    def hasDataset(self):
        return bool(self._info)

    @Slot()
    def refreshDatasets(self):
        self._datasets = core.discover_all_datasets()
        if self._active_key not in self._datasets:
            self._active_key = next(iter(self._datasets), "")
        self.datasetsChanged.emit()
        self._reload_info()

    @Slot(str)
    def selectDataset(self, key):
        if key in self._datasets:
            self._active_key = key
            self._reload_info()

    @Slot(str)
    def selectDatasetPath(self, path):
        """Adds an arbitrary data.yaml (the 'custom path' route) and activates it."""
        p = Path(url_to_path(path))
        if not p.exists():
            self.toast.emit(f"No such file: {p}", "error")
            return
        key = f"Custom Path: {p.parent.name} ({p.name})"
        self._datasets[key] = p.resolve()
        self._active_key = key
        self.datasetsChanged.emit()
        self._reload_info()

    def _reload_info(self):
        yaml_path = self._datasets.get(self._active_key)
        info = core.get_dataset_info(yaml_path) if yaml_path else None
        if not info:
            self._info = {}
        else:
            root = Path(info.get("root", info["yaml_path"]))
            self._info = {
                "name": info.get("name", root.name),
                "kind": info.get("kind", "detect"),
                "root": str(root),
                "needsSplit": bool(info.get("needs_split")),
                "removable": core.DATASET_DIR.resolve() in root.resolve().parents,
                "yaml": str(info["yaml_path"]),
                "classes": [{"id": k, "name": v} for k, v in sorted(info["classes"].items())],
                "counts": [{"split": k.capitalize(), "count": v}
                           for k, v in info["counts"].items()],
                "splits": list(info["splits"].keys()),
                "splitPaths": {k: str(v) for k, v in info["splits"].items()},
                "classNames": {str(k): v for k, v in info["classes"].items()},
            }
        self.activeDatasetChanged.emit()

    # -- zip import --------------------------------------------------------
    @Property(str, notify=busyChanged)
    def busy(self):
        return self._busy

    def _set_busy(self, msg):
        self._busy = msg
        self.busyChanged.emit()

    @Slot(str)
    def importDatasetZip(self, zip_url):
        """Extracts a dataset .zip into the workspace and activates it."""
        src = Path(url_to_path(zip_url))
        if not src.exists():
            self.toast.emit("Zip file not found.", "error")
            return
        if self._extracting:
            self.toast.emit("An extraction is already running.", "warn")
            return
        self._extracting = True
        self._set_busy(f"Extracting {src.name} …")

        def work():
            # Unpack beside the workspace and swap the finished tree in, so a
            # failed or interrupted import never leaves the active dataset
            # half-deleted.
            stamp = time.strftime("%Y%m%d-%H%M%S")
            staging = core.TEMP_DIR / f"extract_{stamp}"
            try:
                staging.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(src, "r") as z:
                    z.extractall(staging)
                # Datasets sit side by side: a new upload is added under its own
                # folder, it never replaces what is already in the workspace.
                dest = core.install_extracted_dataset(staging, src.name)
            finally:
                core.safe_rmtree(staging)
            yamls = list(dest.rglob("*.yaml")) + list(dest.rglob("*.yml"))
            has_yaml = bool(yamls)
            info = core.get_dataset_info(yamls[0] if has_yaml else dest) or {}
            return {"dest": str(dest), "name": dest.name,
                    "kind": info.get("kind") if info else None,
                    "classes": len(info.get("classes", {})) if info else 0,
                    "usable": has_yaml or bool(info)}

        def done(res):
            self._extracting = False
            self._set_busy("")
            self.refreshDatasets()
            self.selectDatasetByPath(res["dest"])
            if not res["usable"]:
                self.toast.emit(
                    f"Extracted to {res['name']}, but it has no data.yaml and no class "
                    "folders — training needs a YOLO-format zip or a folder-per-class set.",
                    "warn")
            elif res["kind"] == "classify":
                self.toast.emit(f"Classification dataset {res['name']} added "
                                f"({res['classes']} classes).", "success")
            else:
                self.toast.emit(f"Dataset {res['name']} extracted and activated.", "success")

        def fail(err):
            self._extracting = False
            self._set_busy("")
            self.toast.emit(f"Extraction failed: {err.strip().splitlines()[-1]}", "error")

        run_async(work, on_done=done, on_error=fail)

    @Slot(str)
    def selectDatasetByPath(self, path):
        """Activates whichever discovered dataset lives under `path`."""
        target = str(Path(url_to_path(path)).resolve())
        for key, p in self._datasets.items():
            if str(p).startswith(target):
                self._active_key = key
                self._reload_info()
                return

    @Slot()
    def removeActiveDataset(self):
        """Deletes the active dataset's folder — workspace datasets only."""
        if not self._info or not self._info.get("removable"):
            self.toast.emit("Only datasets inside yolo_workspace/dataset/ can be removed.",
                            "warn")
            return
        root = Path(self._info["root"])
        name = root.name
        core.safe_rmtree(root)
        self._active_key = ""
        self.refreshDatasets()
        self.toast.emit(f"Removed dataset {name}.", "info")

    @Slot(int)
    def splitActiveDataset(self, val_percent):
        """Creates train/val folders for a flat classification dataset."""
        if not self._info or not self._info.get("needsSplit"):
            return
        root = Path(self._info["root"])
        self._set_busy(f"Splitting {root.name} …")

        def work():
            return core.make_classify_split(root, max(1, min(50, val_percent)) / 100.0)

        def done(res):
            n_train, n_val = res
            self._set_busy("")
            self._reload_info()
            self.toast.emit(f"Split complete: {n_train} train / {n_val} val.", "success")

        def fail(err):
            self._set_busy("")
            self.toast.emit(f"Split failed: {err.strip().splitlines()[-1]}", "error")

        run_async(work, on_done=done, on_error=fail)

    # -- models ------------------------------------------------------------
    @Property("QStringList", notify=modelsChanged)
    def models(self):
        return self._models

    @Property(int, notify=modelsChanged)
    def defaultModelIndex(self):
        """First checkpoint that is actually on disk, so pages preselect it."""
        for i, m in enumerate(self._models):
            if core.resolve_model_path(m).exists():
                return i
        return 0

    @Slot()
    def refreshModels(self):
        self._models = core.get_available_models()
        self.modelsChanged.emit()

    # -- tuned hyperparameters hand-off ------------------------------------
    @Property("QVariantMap", notify=tunedParamsChanged)
    def tunedParams(self):
        return self._tuned

    @Property(bool, notify=tunedParamsChanged)
    def hasTunedParams(self):
        return bool(self._tuned)

    @Slot("QVariantMap")
    def setTunedParams(self, params):
        self._tuned = dict(params or {})
        self.tunedParamsChanged.emit()

    # -- misc helpers used across pages ------------------------------------
    @Slot(str, result=str)
    def basename(self, path):
        return Path(path).name if path else ""

    @Slot(str, result=bool)
    def exists(self, path):
        return bool(path) and Path(path).exists()

    @Slot(str, str, result=bool)
    def saveCopy(self, src, dest_url):
        """Desktop equivalent of the web app's download buttons."""
        import shutil
        try:
            dest = Path(url_to_path(dest_url))
            shutil.copy2(Path(url_to_path(src)), dest)
            self.toast.emit(f"Saved to {dest}", "success")
            return True
        except Exception as e:
            self.toast.emit(f"Save failed: {e}", "error")
            return False

    @Slot(str)
    def revealPath(self, path):
        """Opens a file or folder in the desktop file manager / default app."""
        from PySide6.QtGui import QDesktopServices
        p = Path(url_to_path(path))
        target = p if p.is_dir() else p.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    @Slot(str)
    def openUrl(self, url):
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(url))
