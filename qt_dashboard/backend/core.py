"""
Shared, UI-free helpers for the Qt/QML dashboard.

Everything here is a straight port of the helper layer in ``app.py`` with the
Streamlit calls stripped out, so both front-ends read the same workspace and
produce identical results.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# The project root is the parent of qt_dashboard/, so the Qt app shares the
# exact workspace the Streamlit app uses.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKDIR = PROJECT_ROOT / "yolo_workspace"
DATASET_DIR = WORKDIR / "dataset"
RUNS_DIR = WORKDIR / "runs"
TUNE_DIR = WORKDIR / "tune"
EXPORTS_DIR = WORKDIR / "exports"
TEMP_DIR = WORKDIR / "temp"
PRESETS_FILE = WORKDIR / "presets.json"

for _d in (WORKDIR, RUNS_DIR, TUNE_DIR, EXPORTS_DIR, TEMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

MODEL_FAMILIES = ["YOLO11", "YOLOv8", "YOLOv9", "YOLOv10"]
SIZE_MAP = {
    "YOLO11": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
    "YOLOv8": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
    "YOLOv9": ["yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e"],
    "YOLOv10": ["yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"],
}
KAGGLE_SIZE_MAP = {
    "YOLO11": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
    "YOLOv8": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
    "YOLOv9": ["yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt"],
    "YOLOv10": ["yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt",
                "yolov10l.pt", "yolov10x.pt"],
}
TASK_SUFFIX = {"detect": "", "segment": "-seg", "classify": "-cls",
               "pose": "-pose", "obb": "-obb"}

DEFAULT_PRESETS = {
    "⚡ Fast Prototype (10 Epochs, Nano)": {
        "model_family": "YOLO11", "model_size": "yolo11n", "task": "detect",
        "epochs": 10, "batch": 16, "imgsz": 320, "optimizer": "auto", "lr0": 0.01,
        "cos_lr": False, "patience": 10, "weight_decay": 0.0005, "freeze": 0,
    },
    "⚖️ Standard Balanced (100 Epochs, Small)": {
        "model_family": "YOLO11", "model_size": "yolo11s", "task": "detect",
        "epochs": 100, "batch": 16, "imgsz": 640, "optimizer": "AdamW", "lr0": 0.005,
        "cos_lr": True, "patience": 50, "weight_decay": 0.0005, "freeze": 0,
    },
    "\U0001f3af High Precision Production (200 Epochs, Medium)": {
        "model_family": "YOLO11", "model_size": "yolo11m", "task": "detect",
        "epochs": 200, "batch": 8, "imgsz": 640, "optimizer": "AdamW", "lr0": 0.002,
        "cos_lr": True, "patience": 50, "weight_decay": 0.001, "freeze": 0,
    },
}

if not PRESETS_FILE.exists():
    PRESETS_FILE.write_text(json.dumps(DEFAULT_PRESETS, indent=2))


# ---------------------------------------------------------------------------
# Hardware probe
# ---------------------------------------------------------------------------
def probe_cuda():
    """Returns (cuda_available, gpu_count, [device names])."""
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            return True, n, [torch.cuda.get_device_name(i) for i in range(n)]
    except Exception:
        pass
    return False, 0, []


CUDA_AVAILABLE, GPU_COUNT, GPU_NAMES = probe_cuda()


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
def load_presets() -> Dict[str, Any]:
    try:
        return json.loads(PRESETS_FILE.read_text())
    except Exception:
        return DEFAULT_PRESETS


# ---------------------------------------------------------------------------
# Filesystem / dataset helpers
# ---------------------------------------------------------------------------
def safe_rmtree(path):
    """Removes a directory tree without throwing Errno 39."""
    if not path or not Path(path).exists():
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
    if Path(path).exists():
        try:
            subprocess.run(["rm", "-rf", str(path)], check=False)
        except Exception:
            pass


def _has_images(d: Path) -> bool:
    """True if the directory holds at least one image directly."""
    try:
        return any(f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in d.iterdir())
    except OSError:
        return False


def _is_detection_layout(d: Path) -> bool:
    """True for a YOLO detection folder — images/ (+ labels/) rather than classes."""
    try:
        names = {c.name.lower() for c in d.iterdir() if c.is_dir()}
    except OSError:
        return False
    return "images" in names or "labels" in names


def classify_splits(root: Path) -> Optional[Dict[str, Path]]:
    """Split map for a classification-style dataset (folder per class), else None.

    Handles the Ultralytics layout (root/train/<class>, root/val/<class>) and a
    flat export such as PetImages/Cat, PetImages/Dog that has no split yet.
    A YOLO detection tree also has train/ and val/ folders, so a split holding
    images/ or labels/ rules classification out.
    """
    if not root.is_dir():
        return None
    if _is_detection_layout(root):
        return None
    splits: Dict[str, Path] = {}
    for name in ("train", "val", "valid", "test"):
        sub = root / name
        if not sub.is_dir() or _is_detection_layout(sub):
            continue
        if any(c.is_dir() and _has_images(c) for c in sub.iterdir()):
            splits[name] = sub
    if splits:
        return splits
    classes = [c for c in root.iterdir() if c.is_dir() and _has_images(c)]
    return {"all": root} if len(classes) >= 2 else None


def discover_all_datasets() -> Dict[str, Path]:
    """Every dataset in the workspace, keyed by the label shown in the selector.

    Detection sets are found by their data.yaml; a classification set has none,
    so a folder of class subdirectories counts as one too. Datasets live side by
    side under yolo_workspace/dataset/ and are all listed.
    """
    datasets: Dict[str, Path] = {}
    if DATASET_DIR.exists():
        for y in list(DATASET_DIR.rglob("*.yaml")) + list(DATASET_DIR.rglob("*.yml")):
            datasets[f"Extracted Dataset ({y.parent.name})"] = y.resolve()
        for child in sorted(DATASET_DIR.iterdir()):
            if not child.is_dir():
                continue
            if list(child.rglob("*.yaml")) or list(child.rglob("*.yml")):
                continue        # already listed by its yaml
            if classify_splits(child):
                datasets[f"Classification Dataset ({child.name})"] = child.resolve()
    for root_dir in PROJECT_ROOT.glob("*/"):
        if (root_dir.is_dir() and root_dir.name != "yolo_workspace"
                and not root_dir.name.startswith(".")):
            for y in list(root_dir.glob("*.yaml")) + list(root_dir.glob("*.yml")):
                datasets[f"Workspace Folder: {root_dir.name} ({y.name})"] = y.resolve()
    return datasets


def unique_dataset_dir(name: str) -> Path:
    """A free folder under dataset/ for `name`, suffixed if it is taken."""
    import re
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "dataset"
    dest = DATASET_DIR / base
    i = 2
    while dest.exists():
        dest = DATASET_DIR / f"{base}-{i}"
        i += 1
    return dest


def install_extracted_dataset(staging: Path, zip_name: str) -> Path:
    """Moves a staged extraction into its own folder under dataset/.

    A zip that wraps everything in one top-level folder keeps that folder's
    name; anything else is named after the zip.
    """
    entries = [e for e in staging.iterdir() if not e.name.startswith("__MACOSX")]
    if len(entries) == 1 and entries[0].is_dir():
        src_root, base = entries[0], entries[0].name
    else:
        src_root, base = staging, Path(zip_name).stem
    dest = unique_dataset_dir(base)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    src_root.rename(dest)
    normalize_dataset_yaml(dest)
    return dest


def normalize_dataset_yaml(root: Path) -> bool:
    """Rewrites split paths that escape the dataset folder to stay inside it.

    Roboflow exports ship `train: ../train/images`, which resolves outside the
    dataset once it is extracted into its own folder — with several datasets
    side by side that silently points at a neighbour. Ultralytics resolves
    these paths itself at training time, so they are fixed on disk rather than
    only in the dashboard's reading of them.
    """
    changed_any = False
    for y in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
        try:
            cfg = yaml.safe_load(y.read_text()) or {}
        except Exception:
            continue
        if not isinstance(cfg, dict) or "path" in cfg:
            continue        # an explicit `path:` is the author's own decision
        changed = False
        for split in ("train", "val", "valid", "test"):
            rel = cfg.get(split)
            if not isinstance(rel, str) or Path(rel).is_absolute():
                continue
            p = Path(rel)
            if ".." not in p.parts:
                continue
            stripped = Path(*[part for part in p.parts if part != ".."])
            if (y.parent / stripped).exists():
                cfg[split] = stripped.as_posix()
                changed = True
        if changed:
            try:
                y.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
                changed_any = True
            except OSError:
                pass
    return changed_any


def make_classify_split(root: Path, val_fraction: float = 0.2, progress=None):
    """Turns root/<class>/*.jpg into root/train/<class> + root/val/<class>.

    Ultralytics' classify task needs the split on disk; a flat class-folder
    export has none. Files are moved, so this stays cheap for large sets.
    """
    class_dirs = [c for c in sorted(root.iterdir())
                  if c.is_dir() and c.name not in ("train", "val", "valid", "test")]
    n_train = n_val = 0
    for cdir in class_dirs:
        images = sorted(f for f in cdir.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        if not images:
            continue
        cut = max(1, int(len(images) * (1 - val_fraction)))
        for split, files in (("train", images[:cut]), ("val", images[cut:])):
            target = root / split / cdir.name
            target.mkdir(parents=True, exist_ok=True)
            for f in files:
                try:
                    f.rename(target / f.name)
                except OSError:
                    continue
            if split == "train":
                n_train += len(files)
            else:
                n_val += len(files)
        if progress:
            progress(f"{cdir.name}: {len(images)} images split")
        safe_rmtree(cdir)
    return n_train, n_val


def _resolve_split(base: Path, yaml_dir: Path, rel: str,
                   explicit_base: bool = False) -> Optional[Path]:
    """Finds a split directory named by a data.yaml entry.

    Roboflow exports write `../train/images`, which escapes the dataset folder.
    With several datasets side by side under dataset/, that stray path can land
    on a *neighbour's* split, so a candidate inside the dataset's own folder
    always wins unless the yaml set `path:` explicitly.
    """
    p = Path(rel)
    if p.is_absolute():
        return p if p.exists() else None
    stripped = Path(*[part for part in p.parts if part != ".."]) if ".." in p.parts else None

    inside = []
    if stripped is not None:
        inside.append((yaml_dir / stripped).resolve())
    inside.append((yaml_dir / p).resolve())
    declared = (base / p).resolve()

    candidates = ([declared] + inside) if explicit_base else (inside + [declared])
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand.exists():
            return cand
    return None


def _classify_dataset_info(root: Path) -> Optional[Dict[str, Any]]:
    """Stats for a class-folder dataset, shaped like the yaml version."""
    splits = classify_splits(root)
    if not splits:
        return None
    class_names: List[str] = []
    counts, split_paths = {}, {}
    for split, sdir in splits.items():
        for c in sorted(x for x in sdir.iterdir() if x.is_dir() and _has_images(x)):
            if c.name not in class_names:
                class_names.append(c.name)
        counts[split] = sum(1 for f in sdir.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        split_paths[split] = sdir
    return {
        "config": {}, "classes": {i: n for i, n in enumerate(sorted(class_names))},
        "counts": counts, "splits": split_paths, "yaml_path": root,
        "name": root.name, "kind": "classify", "root": root,
        "needs_split": list(splits) == ["all"],
    }


def get_dataset_info(yaml_path) -> Optional[Dict[str, Any]]:
    """Parses a dataset yaml — or a class-folder dataset — into stats."""
    if not yaml_path:
        return None
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return None
    if yaml_path.is_dir():
        return _classify_dataset_info(yaml_path)
    try:
        cfg = yaml.safe_load(yaml_path.read_text()) or {}
        base = Path(cfg.get("path", yaml_path.parent))
        names = cfg.get("names", {})
        if isinstance(names, dict):
            classes = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            classes = {i: str(v) for i, v in enumerate(names)}
        else:
            classes = {}

        counts, splits = {}, {}
        for split in ("train", "val", "valid", "test"):
            rel = cfg.get(split)
            if not rel:
                continue
            p = _resolve_split(base, yaml_path.parent, str(rel),
                               explicit_base="path" in cfg)
            if p:
                counts[split] = sum(1 for f in p.rglob("*")
                                    if f.suffix.lower() in IMAGE_EXTS)
                splits[split] = p
        return {"config": cfg, "classes": classes, "counts": counts,
                "splits": splits, "yaml_path": yaml_path,
                "name": yaml_path.parent.name, "kind": "detect",
                "root": yaml_path.parent, "needs_split": False}
    except Exception:
        return None


def list_split_images(split_dir) -> List[Path]:
    return sorted(p for p in Path(split_dir).rglob("*")
                  if p.suffix.lower() in IMAGE_EXTS)


def draw_ground_truth_boxes(img_path: Path, class_names: dict, out_path: Path):
    """Renders an image with its YOLO ground-truth boxes drawn on. Returns box count."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    label_path = None
    if "images" in str(img_path):
        cand = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        if cand.exists():
            label_path = cand
    if not label_path:
        cand = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if cand.exists():
            label_path = cand

    n = 0
    if label_path and label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = (float(x) for x in parts[1:5])
            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w, int((xc + bw / 2) * w))
            y2 = min(h, int((yc + bh / 2) * h))
            cname = class_names.get(cls_id, f"Class {cls_id}")
            color = (163, 229, 0)  # emerald, BGR
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(cname, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - 20)),
                          (x1 + tw + 6, max(20, y1)), color, -1)
            cv2.putText(img, cname, (x1 + 3, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            n += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return n


# ---------------------------------------------------------------------------
# Models & runs
# ---------------------------------------------------------------------------
def get_available_models() -> List[str]:
    """Base weights in the project root plus every checkpoint under runs/."""
    models = [str(pt.name) for pt in PROJECT_ROOT.glob("*.pt") if pt.is_file()]
    for pt in RUNS_DIR.rglob("*.pt"):
        if pt.is_file():
            models.append(f"yolo_workspace/{pt.relative_to(WORKDIR)}")
    for d in ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
              "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"):
        if d not in models:
            models.append(d)
    return sorted(set(models))


def resolve_model_path(name: str) -> Path:
    """Model entries are project-root relative; absolute paths pass through."""
    p = Path(name)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_run_results(run_name: str):
    """Loads a run's results.csv. Returns (DataFrame|None, run_dir|None)."""
    import pandas as pd
    candidates = [RUNS_DIR / run_name] + [
        RUNS_DIR / t / run_name for t in
        ("detect", "segment", "classify", "pose", "obb")
    ]
    csv_path = actual_dir = None
    for d in candidates:
        if (d / "results.csv").exists():
            csv_path, actual_dir = d / "results.csv", d
            break
    if not csv_path:
        for p in RUNS_DIR.rglob("results.csv"):
            if run_name in str(p.parent):
                csv_path, actual_dir = p, p.parent
                break
    if csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            return df, actual_dir
        except Exception:
            return None, actual_dir
    return None, actual_dir


def next_free_exp_name(prefix: str = "exp") -> str:
    i = 1
    while (RUNS_DIR / f"{prefix}{i}").exists():
        i += 1
    return f"{prefix}{i}"


# ---------------------------------------------------------------------------
# Subprocess plumbing (shared by training and tuning)
# ---------------------------------------------------------------------------
def spawn_process(cmd: List[str]):
    """Starts a command in its own process group with merged stdout/stderr."""
    kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                  text=True, bufsize=1, cwd=str(PROJECT_ROOT))
    if os.name != "nt":
        kwargs["preexec_fn"] = os.setsid
    return subprocess.Popen(cmd, **kwargs)


def reader_thread(proc, on_line, progress_re=None):
    """Pumps a process's output into ``on_line``, skipping tqdm redraw noise."""
    def _pump():
        try:
            for line in proc.stdout:
                if progress_re and progress_re.search(line) and "100%|" not in line:
                    continue
                on_line(line)
        except Exception:
            pass
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    return t


def signal_group(proc, sig) -> bool:
    """Signals the whole process group so dataloader workers follow along."""
    if os.name == "nt" or proc is None:
        return False
    try:
        os.killpg(os.getpgid(proc.pid), sig)
        return True
    except Exception:
        return False


def yolo_cli() -> List[str]:
    """The `yolo` entry point of the interpreter running this app."""
    exe = Path(sys.executable).parent / "yolo"
    if exe.exists():
        return [str(exe)]
    return ["yolo"]
