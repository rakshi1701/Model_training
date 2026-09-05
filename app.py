"""
YOLO Studio v2.5: Training, Evaluation, Inference & Deployment Hub
Run with: streamlit run app.py
Requires: pip install streamlit ultralytics pyyaml pandas Pillow opencv-python-headless onnx onnxruntime
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import streamlit as st
import subprocess
import zipfile
import shutil
import re
import os
import signal
import threading
import time
import yaml
import json
import tempfile
from pathlib import Path
import pandas as pd
from PIL import Image
import cv2
import psutil

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except Exception:
    CUDA_AVAILABLE = False
    GPU_COUNT = 0

from ultralytics import YOLO
import kaggle_bridge

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLO Vision Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS: Modern Glassmorphism & High-Contrast Visuals
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(0, 210, 160, 0.5);
        transform: translateY(-2px);
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #00e5a3 !important;
    }
    div[data-testid="stMetricLabel"] > div > p {
        font-size: 0.85rem !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Pipeline Status Ribbon */
    .pipeline-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 18px;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
    }
    .pipeline-step.active {
        color: #00e5a3;
    }
    .pipeline-badge {
        background: rgba(0, 229, 163, 0.15);
        color: #00e5a3;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.75rem;
        border: 1px solid rgba(0, 229, 163, 0.3);
    }
    .pipeline-arrow {
        color: rgba(255, 255, 255, 0.2);
        font-size: 0.9rem;
    }

    /* Status Indicator Pulses */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-running {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-paused {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-idle {
        background: rgba(100, 116, 139, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }

    /* Terminal Console Window */
    .terminal-box {
        background: #090d16;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.82rem;
        color: #38bdf8;
        max-height: 380px;
        overflow-y: auto;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Workspace Directories
# ---------------------------------------------------------------------------
WORKDIR = Path("yolo_workspace")
DATASET_DIR = WORKDIR / "dataset"
RUNS_DIR = WORKDIR / "runs"
TUNE_DIR = WORKDIR / "tune"
EXPORTS_DIR = WORKDIR / "exports"
TEMP_DIR = WORKDIR / "temp"
PRESETS_FILE = WORKDIR / "presets.json"

for d in [WORKDIR, RUNS_DIR, TUNE_DIR, EXPORTS_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Default Presets
DEFAULT_PRESETS = {
    "⚡ Fast Prototype (10 Epochs, Nano)": {
        "model_family": "YOLO11", "model_size": "yolo11n", "task": "detect",
        "epochs": 10, "batch": 16, "imgsz": 320, "optimizer": "auto", "lr0": 0.01,
        "cos_lr": False, "patience": 10, "weight_decay": 0.0005, "freeze": 0
    },
    "⚖️ Standard Balanced (100 Epochs, Small)": {
        "model_family": "YOLO11", "model_size": "yolo11s", "task": "detect",
        "epochs": 100, "batch": 16, "imgsz": 640, "optimizer": "AdamW", "lr0": 0.005,
        "cos_lr": True, "patience": 50, "weight_decay": 0.0005, "freeze": 0
    },
    "🎯 High Precision Production (200 Epochs, Medium)": {
        "model_family": "YOLO11", "model_size": "yolo11m", "task": "detect",
        "epochs": 200, "batch": 8, "imgsz": 640, "optimizer": "AdamW", "lr0": 0.002,
        "cos_lr": True, "patience": 50, "weight_decay": 0.001, "freeze": 0
    }
}

if not PRESETS_FILE.exists():
    with open(PRESETS_FILE, "w") as f:
        json.dump(DEFAULT_PRESETS, f, indent=2)

def load_presets():
    try:
        with open(PRESETS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_PRESETS

# ---------------------------------------------------------------------------
# Helper Functions: Dataset, Visual Ground Truth & Metrics
# ---------------------------------------------------------------------------
def safe_rmtree(path: Path):
    """Safely removes a directory tree without throwing Errno 39."""
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


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _has_images(d: Path) -> bool:
    """True if the directory holds at least one image directly."""
    try:
        return any(f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES for f in d.iterdir())
    except OSError:
        return False


def _is_detection_layout(d: Path) -> bool:
    """True for a YOLO detection folder — images/ (+ labels/) rather than classes."""
    try:
        names = {c.name.lower() for c in d.iterdir() if c.is_dir()}
    except OSError:
        return False
    return "images" in names or "labels" in names


def classify_splits(root: Path):
    """Split map for a classification-style dataset (a folder per class), else None.

    Handles both the Ultralytics layout (root/train/<class>/, root/val/<class>/)
    and a flat export such as PetImages/Cat, PetImages/Dog with no split yet.
    A YOLO detection tree also has train/ and val/, so a split holding images/
    or labels/ rules classification out.
    """
    if not root.is_dir():
        return None
    if _is_detection_layout(root):
        return None
    splits = {}
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


def make_classify_split(root: Path, val_fraction: float = 0.2, progress=None):
    """Turns root/<class>/*.jpg into root/train/<class> + root/val/<class>.

    Ultralytics' classify task needs the split on disk; a flat class-folder
    export (Cat/, Dog/) has none. Files are moved, so this is cheap even for
    tens of thousands of images.
    """
    class_dirs = [c for c in sorted(root.iterdir())
                  if c.is_dir() and c.name not in ("train", "val", "valid", "test")]
    if not class_dirs:
        return 0, 0
    n_train = n_val = 0
    for cdir in class_dirs:
        images = sorted(f for f in cdir.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
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


def resolve_split_dir(base: Path, yaml_dir: Path, rel: str, explicit_base: bool = False):
    """Finds a split directory named by a data.yaml entry.

    Roboflow exports write `../train/images`, which escapes the dataset folder.
    Datasets now sit side by side under dataset/, so that stray path can land on
    a *neighbour's* split — a candidate inside the dataset's own folder wins
    unless the yaml set `path:` explicitly.
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

    seen = set()
    for cand in ([declared] + inside) if explicit_base else (inside + [declared]):
        if cand in seen:
            continue
        seen.add(cand)
        if cand.exists():
            return cand
    return None


def normalize_dataset_yaml(root: Path) -> bool:
    """Rewrites split paths that escape the dataset folder to stay inside it.

    Ultralytics resolves these paths itself at training time, so they are fixed
    on disk rather than only in how the dashboard reads them.
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


def unique_dataset_dir(name: str) -> Path:
    """A free folder under dataset/ for `name`, suffixed if it is taken."""
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


def discover_all_datasets():
    """Every dataset in the workspace, keyed by the label shown in the selector.

    Detection sets are found by their data.yaml; classification sets have no
    yaml, so a folder of class subdirectories counts as one too. Datasets sit
    side by side under yolo_workspace/dataset/ and are listed together.
    """
    datasets = {}
    if DATASET_DIR.exists():
        for y_path in list(DATASET_DIR.rglob("*.yaml")) + list(DATASET_DIR.rglob("*.yml")):
            datasets[f"Extracted Dataset ({y_path.parent.name})"] = y_path.resolve()
        for child in sorted(DATASET_DIR.iterdir()):
            if not child.is_dir():
                continue
            if list(child.rglob("*.yaml")) or list(child.rglob("*.yml")):
                continue    # already listed above by its yaml
            if classify_splits(child):
                datasets[f"Classification Dataset ({child.name})"] = child.resolve()
    for root_dir in Path(".").glob("*/"):
        if root_dir.is_dir() and root_dir.name != "yolo_workspace" and not root_dir.name.startswith("."):
            for y_path in list(root_dir.glob("*.yaml")) + list(root_dir.glob("*.yml")):
                datasets[f"Workspace Folder: {root_dir.name} ({y_path.name})"] = y_path.resolve()
    return datasets


def _classify_dataset_info(root: Path):
    """Stats for a class-folder dataset, shaped like the yaml version."""
    splits = classify_splits(root)
    if not splits:
        return None
    class_names, counts, split_paths = [], {}, {}
    for split, sdir in splits.items():
        class_dirs = sorted(c for c in sdir.iterdir() if c.is_dir() and _has_images(c))
        for c in class_dirs:
            if c.name not in class_names:
                class_names.append(c.name)
        counts[split] = sum(1 for f in sdir.rglob("*")
                            if f.suffix.lower() in IMAGE_SUFFIXES)
        split_paths[split] = sdir
    return {
        "config": {}, "classes": {i: n for i, n in enumerate(sorted(class_names))},
        "counts": counts, "splits": split_paths, "yaml_path": root,
        "name": root.name, "kind": "classify", "root": root,
        "needs_split": list(splits) == ["all"],
    }


def get_dataset_info(yaml_path: Path):
    """Parses a dataset yaml (or a class-folder dataset) and returns statistics."""
    if not yaml_path or not Path(yaml_path).exists():
        return None
    yaml_path = Path(yaml_path)
    if yaml_path.is_dir():
        return _classify_dataset_info(yaml_path)
    try:
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        base = Path(cfg.get("path", yaml_path.parent))
        names = cfg.get("names", {})
        if isinstance(names, dict):
            class_list = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            class_list = {i: str(v) for i, v in enumerate(names)}
        else:
            class_list = {}

        counts = {}
        split_paths = {}
        for split in ["train", "val", "valid", "test"]:
            rel = cfg.get(split)
            if rel:
                p = resolve_split_dir(base, yaml_path.parent, str(rel),
                                      explicit_base="path" in cfg)
                if p:
                    n_imgs = sum(1 for img in p.rglob("*") if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"])
                    counts[split] = n_imgs
                    split_paths[split] = p

        return {
            "config": cfg,
            "classes": class_list,
            "counts": counts,
            "splits": split_paths,
            "yaml_path": yaml_path,
            "name": yaml_path.parent.name,
            "kind": "detect",
            "root": yaml_path.parent,
            "needs_split": False,
        }
    except Exception:
        return None


def draw_ground_truth_boxes(img_path: Path, class_names: dict):
    """Draws ground truth YOLO bounding boxes onto image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    # Look for matching label in labels/ directory
    label_path = None
    if "images" in str(img_path):
        candidate = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        if candidate.exists():
            label_path = candidate
    if not label_path:
        label_cand = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if label_cand.exists():
            label_path = label_cand

    box_count = 0
    if label_path and label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                    x1 = max(0, int((xc - bw / 2) * w))
                    y1 = max(0, int((yc - bh / 2) * h))
                    x2 = min(w, int((xc + bw / 2) * w))
                    y2 = min(h, int((yc + bh / 2) * h))

                    cname = class_names.get(cls_id, f"Class {cls_id}")
                    color = (0, 229, 163)  # Emerald green

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{cname}"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), color, -1)
                    cv2.putText(img, label_text, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                    box_count += 1

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), box_count


def get_available_models():
    """Scans for all available models."""
    models = []
    for pt in Path(".").glob("*.pt"):
        if pt.is_file():
            models.append(str(pt.name))
    for pt in RUNS_DIR.rglob("*.pt"):
        if pt.is_file():
            rel = pt.relative_to(WORKDIR)
            models.append(f"{WORKDIR}/{rel}")
    defaults = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    for d in defaults:
        if d not in models:
            models.append(d)
    return sorted(list(set(models)))


def load_run_results(run_name: str):
    """Loads results.csv and args.yaml for a run."""
    target_dirs = [
        RUNS_DIR / run_name,
        RUNS_DIR / "detect" / run_name,
        RUNS_DIR / "segment" / run_name,
        RUNS_DIR / "classify" / run_name,
        RUNS_DIR / "pose" / run_name,
        RUNS_DIR / "obb" / run_name,
    ]
    csv_path, actual_dir = None, None
    for d in target_dirs:
        p = d / "results.csv"
        if p.exists():
            csv_path, actual_dir = p, d
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


@st.cache_resource(max_entries=3)
def load_yolo_cached(model_path: str):
    return YOLO(model_path)


def get_system_utilization():
    """Collects live system utilization stats (CPU, RAM, Disk, GPU)."""
    stats = {}

    # CPU
    stats["cpu_percent"] = psutil.cpu_percent(interval=0)
    stats["cpu_per_core"] = psutil.cpu_percent(interval=0, percpu=True)
    stats["cpu_count_logical"] = psutil.cpu_count(logical=True)
    stats["cpu_count_physical"] = psutil.cpu_count(logical=False)
    try:
        stats["cpu_freq"] = psutil.cpu_freq().current  # MHz
    except Exception:
        stats["cpu_freq"] = None

    # Memory
    mem = psutil.virtual_memory()
    stats["ram_total_gb"] = mem.total / (1024 ** 3)
    stats["ram_used_gb"] = mem.used / (1024 ** 3)
    stats["ram_available_gb"] = mem.available / (1024 ** 3)
    stats["ram_percent"] = mem.percent

    # Swap
    swap = psutil.swap_memory()
    stats["swap_total_gb"] = swap.total / (1024 ** 3)
    stats["swap_used_gb"] = swap.used / (1024 ** 3)
    stats["swap_percent"] = swap.percent

    # Disk
    disk = psutil.disk_usage("/")
    stats["disk_total_gb"] = disk.total / (1024 ** 3)
    stats["disk_used_gb"] = disk.used / (1024 ** 3)
    stats["disk_percent"] = disk.percent

    # GPU via nvidia-smi (if available)
    stats["gpus"] = []
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            text=True, timeout=3
        ).strip()
        for line in smi_out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpu = {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util_percent": float(parts[2]) if parts[2] not in ("[N/A]", "") else None,
                    "mem_used_mb": float(parts[3]) if parts[3] not in ("[N/A]", "") else None,
                    "mem_total_mb": float(parts[4]) if parts[4] not in ("[N/A]", "") else None,
                    "temp_c": float(parts[5]) if parts[5] not in ("[N/A]", "") else None,
                    "power_w": float(parts[6]) if len(parts) > 6 and parts[6] not in ("[N/A]", "") else None,
                    "power_limit_w": float(parts[7]) if len(parts) > 7 and parts[7] not in ("[N/A]", "") else None,
                }
                if gpu["mem_total_mb"] and gpu["mem_used_mb"]:
                    gpu["mem_percent"] = (gpu["mem_used_mb"] / gpu["mem_total_mb"]) * 100
                else:
                    gpu["mem_percent"] = None
                stats["gpus"].append(gpu)
    except Exception:
        pass

    # Process-specific stats for the training subprocess
    stats["train_proc"] = None
    if st.session_state.train_proc and st.session_state.train_active:
        try:
            proc = psutil.Process(st.session_state.train_proc.pid)
            children = proc.children(recursive=True)
            total_rss = proc.memory_info().rss
            total_cpu = proc.cpu_percent(interval=0)
            for child in children:
                try:
                    total_rss += child.memory_info().rss
                    total_cpu += child.cpu_percent(interval=0)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            stats["train_proc"] = {
                "pid": proc.pid,
                "rss_gb": total_rss / (1024 ** 3),
                "cpu_percent": total_cpu,
                "num_threads": proc.num_threads() + sum(c.num_threads() for c in children if c.is_running()),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return stats


# ---------------------------------------------------------------------------
# Background Process Manager
# ---------------------------------------------------------------------------
def _reader_thread(proc, buf, progress_re=None):
    for line in proc.stdout:
        if progress_re and progress_re.search(line) and "100%|" not in line:
            continue
        buf.append(line)
    proc.stdout.close()


def _signal_group(proc, sig):
    if os.name == "nt" or proc is None:
        return False
    try:
        os.killpg(os.getpgid(proc.pid), sig)
        return True
    except Exception as e:
        st.error(f"Signal error: {e}")
        return False


# Session State Initialization
st.session_state.setdefault("train_proc", None)
st.session_state.setdefault("train_logs", [])
st.session_state.setdefault("train_active", False)
st.session_state.setdefault("train_paused", False)
st.session_state.setdefault("train_run_name", "exp1")
st.session_state.setdefault("train_start_time", None)

st.session_state.setdefault("tune_proc", None)
st.session_state.setdefault("tune_logs", [])
st.session_state.setdefault("tune_active", False)
st.session_state.setdefault("tune_run_name", "tune1")

st.session_state.setdefault("active_dataset_path", None)
st.session_state.setdefault("tuned_params", None)
st.session_state.setdefault("kaggle_active_kernel", None)
st.session_state.setdefault("kaggle_status_info", None)

# ---------------------------------------------------------------------------
# Sidebar: Dataset Selector & Hardware Status
# ---------------------------------------------------------------------------
st.sidebar.markdown("### ⚡ YOLO Control Center")

# Hardware Status Badge
if CUDA_AVAILABLE:
    st.sidebar.markdown(f"""
    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px;">
        <div style="color: #10b981; font-weight: 700; font-size: 0.85rem;">⚡ NVIDIA CUDA Accelerated</div>
        <div style="color: #94a3b8; font-size: 0.78rem;">{GPU_COUNT}x GPU Device(s) Online</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px;">
        <div style="color: #f59e0b; font-weight: 700; font-size: 0.85rem;">⚠️ CPU Compute Mode</div>
        <div style="color: #94a3b8; font-size: 0.78rem;">Training will use host CPU cores</div>
    </div>
    """, unsafe_allow_html=True)

# Dataset Hub in Sidebar
st.sidebar.markdown("#### 📁 Active Dataset")
discovered_datasets = discover_all_datasets()
dataset_options = list(discovered_datasets.keys()) + ["➕ Upload New Dataset (.zip)", "🔍 Enter Custom Path"]

# A dataset that was just extracted becomes the active one on the next run.
_pending = st.session_state.pop("pending_dataset", None)
if _pending:
    for label, path in discovered_datasets.items():
        if str(path).startswith(_pending):
            st.session_state.ds_source_select = label
            break
# Drop a selection that points at a dataset which no longer exists.
if st.session_state.get("ds_source_select") not in dataset_options:
    st.session_state.pop("ds_source_select", None)

selected_ds_source = st.sidebar.selectbox(
    "Dataset Source", dataset_options, key="ds_source_select"
)

data_yaml_path = None
if selected_ds_source == "➕ Upload New Dataset (.zip)":
    uploaded_zip = st.sidebar.file_uploader("Upload .zip (up to 50GB)", type=["zip"])
    st.sidebar.caption("Datasets are kept side by side — a new upload is added, not replacing "
                       "what is already there. Remove one from the list below when you're done.")
    if uploaded_zip and st.sidebar.button("📦 Extract & Activate", width="stretch"):
        # A big zip takes long enough that Streamlit can start a second script
        # run over the top of this one. Two runs wiping and filling the same
        # folder used to race (rm -rf walking the tree the other was extracting
        # into, taking its upload.zip with it), so: refuse to re-enter, stage
        # everything outside DATASET_DIR, and only swap it in once it's whole.
        if st.session_state.get("ds_extracting"):
            st.sidebar.warning("An extraction is already running — give it a moment.")
        else:
            st.session_state.ds_extracting = True
            stamp = time.strftime("%Y%m%d-%H%M%S")
            zip_path = TEMP_DIR / f"upload_{stamp}.zip"
            staging = TEMP_DIR / f"extract_{stamp}"
            try:
                with st.spinner("Extracting dataset — this can take a while for large zips..."):
                    staging.mkdir(parents=True, exist_ok=True)
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_zip.getbuffer())
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(staging)

                    dest = install_extracted_dataset(staging, uploaded_zip.name)

                info = get_dataset_info(dest) or {}
                if list(dest.rglob("*.yaml")) or list(dest.rglob("*.yml")):
                    st.session_state.pending_dataset = str(dest)
                    st.sidebar.success(f"Dataset `{dest.name}` extracted and activated!")
                elif info.get("kind") == "classify":
                    st.session_state.pending_dataset = str(dest)
                    st.sidebar.success(
                        f"Classification dataset `{dest.name}` added "
                        f"({len(info.get('classes', {}))} classes)."
                    )
                else:
                    st.sidebar.warning(
                        f"Extracted to `{dest.name}`, but it has no data.yaml and no class "
                        "folders. Training needs a YOLO-format zip (data.yaml + images/ + "
                        "labels/) or a folder-per-class classification set."
                    )
            except zipfile.BadZipFile:
                st.sidebar.error("That file is not a readable .zip archive.")
            except Exception as e:
                st.sidebar.error(f"Extraction failed: {e}")
            finally:
                zip_path.unlink(missing_ok=True)
                safe_rmtree(staging)
                st.session_state.ds_extracting = False
            st.rerun()

elif selected_ds_source == "🔍 Enter Custom Path":
    custom_path_input = st.sidebar.text_input("Absolute Path to data.yaml", value="")
    if custom_path_input and Path(custom_path_input).exists():
        data_yaml_path = Path(custom_path_input).resolve()
        st.session_state.active_dataset_path = str(data_yaml_path)
elif selected_ds_source in discovered_datasets:
    data_yaml_path = discovered_datasets[selected_ds_source]
    st.session_state.active_dataset_path = str(data_yaml_path)

# Active Dataset Stats Card in Sidebar
ds_info = get_dataset_info(data_yaml_path) if data_yaml_path else None
if ds_info:
    counts_str = " · ".join([f"{k}: {v}" for k, v in ds_info["counts"].items()])
    classes_str = ", ".join(list(ds_info["classes"].values())[:5])
    if len(ds_info["classes"]) > 5:
        classes_str += "..."
    st.sidebar.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 10px; font-size: 0.8rem; margin-bottom: 12px;">
        <div style="font-weight: 700; color: #38bdf8;">📄 {ds_info.get('name', ds_info['yaml_path'].parent.name)}{' · classify' if ds_info.get('kind') == 'classify' else ''}</div>
        <div style="color: #cbd5e1; margin-top: 4px;">🖼️ <b>Images:</b> {counts_str or 'None detected'}</div>
        <div style="color: #cbd5e1;">🏷️ <b>Classes ({len(ds_info['classes'])}):</b> {classes_str or 'None'}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.info("Select or upload a dataset to begin training.")

# Remove a dataset once its training is done. Only datasets that live inside
# yolo_workspace/dataset/ can be deleted — a custom path outside the workspace
# is somebody else's data.
if ds_info:
    _root = Path(ds_info.get("root", ds_info["yaml_path"]))
    _removable = DATASET_DIR.resolve() in _root.resolve().parents
    if _removable:
        _confirm_key = f"confirm_del_ds_{_root}"
        if st.session_state.get(_confirm_key):
            st.sidebar.warning(f"Delete `{_root.name}` and everything in it?")
            c_yes, c_no = st.sidebar.columns(2)
            if c_yes.button("🗑 Delete", type="primary", width="stretch"):
                safe_rmtree(_root)
                st.session_state.pop(_confirm_key, None)
                st.session_state.pop("ds_source_select", None)
                st.toast(f"Removed dataset {_root.name}", icon="🗑")
                st.rerun()
            if c_no.button("Cancel", width="stretch"):
                st.session_state.pop(_confirm_key, None)
                st.rerun()
        elif st.sidebar.button(f"🗑 Remove `{_root.name}` from workspace", width="stretch"):
            st.session_state[_confirm_key] = True
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("🚀 **YOLO Vision Studio v2.5** • Google Deepmind AGY")

# ---------------------------------------------------------------------------
# Interactive Top Pipeline Tracker
# ---------------------------------------------------------------------------
pipeline_ds_badge = "Ready ✅" if ds_info else "Waiting 📁"
pipeline_train_badge = "Active 🟢" if st.session_state.train_active else ("Paused ⏸" if st.session_state.train_paused else "Idle ⚪")
active_model_badge = "Configured ⚡"

st.markdown(f"""
<div class="pipeline-container">
    <div class="pipeline-step {'active' if ds_info else ''}">
        <span>📁 1. Dataset</span>
        <span class="pipeline-badge">{pipeline_ds_badge}</span>
    </div>
    <span class="pipeline-arrow">➔</span>
    <div class="pipeline-step active">
        <span>⚙️ 2. Model & Tuning</span>
        <span class="pipeline-badge">{active_model_badge}</span>
    </div>
    <span class="pipeline-arrow">➔</span>
    <div class="pipeline-step {'active' if st.session_state.train_active else ''}">
        <span>🏋️ 3. Training & Logs</span>
        <span class="pipeline-badge">{pipeline_train_badge}</span>
    </div>
    <span class="pipeline-arrow">➔</span>
    <div class="pipeline-step active">
        <span>🧪 4. Inference</span>
        <span class="pipeline-badge">Interactive</span>
    </div>
    <span class="pipeline-arrow">➔</span>
    <div class="pipeline-step active">
        <span>📦 5. Export</span>
        <span class="pipeline-badge">Multi-Format</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------------------------
tab_train, tab_kaggle, tab_ds, tab_infer, tab_export, tab_tune, tab_history = st.tabs([
    "🏋️ Local Training & Metrics",
    "☁️ Kaggle Cloud GPU Training",
    "📂 Dataset Hub & Visual Inspector",
    "🧪 Inference & Testing Playground",
    "📦 Model Export Studio",
    "🎛️ Hyperparameter Tuning",
    "📊 Experiment History & Analytics",
])

# ===========================================================================
# TAB 1: 🏋️ TRAINING & LIVE METRIC DASHBOARD
# ===========================================================================
with tab_train:
    st.markdown("### 🏋️ Model Training Studio")

    # Hyperparameter Profile & Presets Bar
    presets = load_presets()
    preset_col1, preset_col2 = st.columns([2, 1])

    with preset_col1:
        selected_preset = st.selectbox("🎯 Quick Training Preset Profile", list(presets.keys()), index=1)
        p_cfg = presets[selected_preset]

    with preset_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.session_state.tuned_params:
            if st.button("✨ Apply Optuna Tuned Hyperparameters", type="primary", width="stretch"):
                st.toast("Tuned hyperparameters loaded into training settings!", icon="🎯")

    # Expandable Hyperparameter Settings
    with st.expander("⚙️ Customize Hyperparameters & Hardware Configuration", expanded=not st.session_state.train_active):
        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)

        with cfg_col1:
            st.markdown("##### 1. Architecture")
            fam_idx = ["YOLO11", "YOLOv8", "YOLOv9", "YOLOv10"].index(p_cfg.get("model_family", "YOLO11"))
            model_family = st.selectbox("Family", ["YOLO11", "YOLOv8", "YOLOv9", "YOLOv10"], index=fam_idx)
            size_map = {
                "YOLO11": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
                "YOLOv8": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                "YOLOv9": ["yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e"],
                "YOLOv10": ["yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"],
            }
            cur_sizes = size_map[model_family]
            size_idx = cur_sizes.index(p_cfg.get("model_size", cur_sizes[0])) if p_cfg.get("model_size") in cur_sizes else 0
            model_size = st.selectbox("Model Size", cur_sizes, index=size_idx)

            task_opts = ["detect", "segment", "classify", "pose", "obb"]
            task = st.selectbox("Task", task_opts, index=task_opts.index(p_cfg.get("task", "detect")))
            task_suffix = {"detect": "", "segment": "-seg", "classify": "-cls", "pose": "-pose", "obb": "-obb"}[task]
            pretrained = st.checkbox("Start from Pretrained Weights", value=True)
            model_name = f"{model_size}{task_suffix}.pt" if pretrained else f"{model_size}{task_suffix}.yaml"

        with cfg_col2:
            st.markdown("##### 2. Optimization")
            epochs = st.number_input("Total Epochs", min_value=1, max_value=5000, value=int(p_cfg.get("epochs", 100)))
            batch = st.selectbox("Batch Size", [-1, 2, 4, 8, 16, 32, 64, 128], index=4, help="-1 automatically optimizes batch size to fill 60% of GPU VRAM")
            imgsz = st.selectbox("Image Size (px)", [320, 416, 512, 640, 768, 960, 1280], index=3)
            optimizer = st.selectbox("Optimizer", ["auto", "AdamW", "SGD", "Adam", "NAdam", "RMSProp"], index=1)

        with cfg_col3:
            st.markdown("##### 3. Learning Rate & Decay")
            def_lr = float(st.session_state.tuned_params.get("lr0", p_cfg.get("lr0", 0.005))) if st.session_state.tuned_params else float(p_cfg.get("lr0", 0.005))
            lr0 = st.number_input("Initial LR (lr0)", min_value=0.00001, max_value=1.0, value=def_lr, format="%.5f")
            patience = st.number_input("Early Stopping (epochs)", min_value=0, max_value=500, value=int(p_cfg.get("patience", 50)))
            cos_lr = st.checkbox("Cosine LR Schedule", value=bool(p_cfg.get("cos_lr", True)))
            cache = st.selectbox("Image Caching", ["False", "ram", "disk"], index=0)

        with cfg_col4:
            st.markdown("##### 4. Compute & Checkpoints")
            device_opts = []
            device_lbls = {}
            if CUDA_AVAILABLE:
                for i in range(GPU_COUNT):
                    lbl = f"GPU {i}: {torch.cuda.get_device_name(i)}"
                    device_opts.append(lbl)
                    device_lbls[lbl] = str(i)
                if GPU_COUNT > 1:
                    lbl = f"All GPUs (0-{GPU_COUNT-1})"
                    device_opts.append(lbl)
                    device_lbls[lbl] = ",".join(str(i) for i in range(GPU_COUNT))
            device_opts.append("CPU")
            device_lbls["CPU"] = "cpu"

            device_sel = st.selectbox("Target Compute", device_opts, index=0)
            device = device_lbls[device_sel]
            workers = st.number_input("Dataloader Workers", min_value=0, max_value=32, value=8)
            run_name = st.text_input("Experiment Run Name", value=st.session_state.train_run_name)
            weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=float(p_cfg.get("weight_decay", 0.0005)), format="%.5f")
            freeze = st.number_input("Freeze Backbone Layers (0=None)", min_value=0, max_value=50, value=int(p_cfg.get("freeze", 0)))
            resume = st.checkbox("Resume Checkpoint", value=False)

    # Action Buttons Bar
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([1.6, 1, 1, 1, 1.8])
    start_disabled = data_yaml_path is None or st.session_state.train_active

    with btn_col1:
        if st.button("🚀 Start Training Run", type="primary", disabled=start_disabled, width="stretch"):
            cmd = [
                "yolo", "train",
                f"model={model_name}",
                f"data={data_yaml_path}",
                f"epochs={epochs}",
                f"batch={batch}",
                f"imgsz={imgsz}",
                f"optimizer={optimizer}",
                f"lr0={lr0}",
                f"patience={patience}",
                f"device={device}",
                f"workers={workers}",
                f"resume={resume}",
                f"cache={cache}",
                f"cos_lr={cos_lr}",
                f"weight_decay={weight_decay}",
                f"freeze={freeze}",
                f"project={RUNS_DIR}",
                f"name={run_name}",
            ]
            st.session_state.train_logs = []
            st.session_state.train_start_time = time.time()
            popen_kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            if os.name != "nt":
                popen_kwargs["preexec_fn"] = os.setsid
            proc = subprocess.Popen(cmd, **popen_kwargs)
            prog_re = re.compile(r"\d+%\|")
            threading.Thread(
                target=_reader_thread, args=(proc, st.session_state.train_logs, prog_re), daemon=True
            ).start()
            st.session_state.train_proc = proc
            st.session_state.train_active = True
            st.session_state.train_paused = False
            st.session_state.train_run_name = run_name
            st.rerun()

    with btn_col2:
        if st.button("⏸ Pause", disabled=not st.session_state.train_active or st.session_state.train_paused, width="stretch"):
            if _signal_group(st.session_state.train_proc, signal.SIGSTOP):
                st.session_state.train_paused = True
            st.rerun()

    with btn_col3:
        if st.button("▶ Resume", disabled=not st.session_state.train_active or not st.session_state.train_paused, width="stretch"):
            if _signal_group(st.session_state.train_proc, signal.SIGCONT):
                st.session_state.train_paused = False
            st.rerun()

    with btn_col4:
        if st.button("🛑 Terminate", disabled=not st.session_state.train_active, width="stretch"):
            if st.session_state.train_paused:
                _signal_group(st.session_state.train_proc, signal.SIGCONT)
            if os.name == "nt":
                st.session_state.train_proc.terminate()
            else:
                _signal_group(st.session_state.train_proc, signal.SIGTERM)
            st.session_state.train_active = False
            st.session_state.train_paused = False
            st.rerun()

    with btn_col5:
        if st.session_state.train_active:
            if st.session_state.train_paused:
                st.markdown('<span class="status-badge status-paused">⏸ Training Paused</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="status-badge status-running">🟢 Training: {st.session_state.train_run_name}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-idle">⚪ Engine Ready</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Real-Time Metrics & Interactive Charts
    current_run = st.session_state.train_run_name
    df_results, actual_run_dir = load_run_results(current_run)

    if df_results is not None and not df_results.empty:
        last_row = df_results.iloc[-1]
        cur_epoch = int(last_row.get("epoch", len(df_results)))

        # Calculate ETA
        eta_str = "Calculating..."
        if len(df_results) > 1 and "time" in df_results.columns:
            avg_sec_epoch = df_results["time"].diff().mean()
            if pd.notnull(avg_sec_epoch) and avg_sec_epoch > 0:
                rem_epochs = max(0, epochs - cur_epoch)
                rem_sec = int(rem_epochs * avg_sec_epoch)
                mins, secs = divmod(rem_sec, 60)
                hrs, mins = divmod(mins, 60)
                eta_str = f"{hrs}h {mins}m" if hrs > 0 else f"{mins}m {secs}s"

        # Epoch Progress Bar
        progress_pct = min(1.0, max(0.0, cur_epoch / epochs))
        st.progress(progress_pct, text=f"Epoch Progress: {cur_epoch}/{epochs} ({progress_pct*100:.1f}%) • Estimated ETA: {eta_str}")

        # Metric KPI cards
        k1, k2, k3, k4, k5, k6 = st.columns(6)

        def get_stat(col):
            if col in df_results.columns:
                v = float(last_row[col])
                d = (v - float(df_results.iloc[-2][col])) if len(df_results) > 1 else None
                return v, d
            return None, None

        box_val, box_d = get_stat("val/box_loss")
        if box_val is None:
            box_val, box_d = get_stat("train/box_loss")

        cls_val, cls_d = get_stat("val/cls_loss")
        if cls_val is None:
            cls_val, cls_d = get_stat("train/cls_loss")

        map50_val, map50_d = get_stat("metrics/mAP50(B)")
        map95_val, map95_d = get_stat("metrics/mAP50-95(B)")
        prec_val, _ = get_stat("metrics/precision(B)")
        rec_val, _ = get_stat("metrics/recall(B)")

        k1.metric("Epoch", f"{cur_epoch}/{epochs}")
        k2.metric("Box Loss", f"{box_val:.4f}" if box_val is not None else "N/A", delta=f"{box_d:.4f}" if box_d else None, delta_color="inverse")
        k3.metric("Class Loss", f"{cls_val:.4f}" if cls_val is not None else "N/A", delta=f"{cls_d:.4f}" if cls_d else None, delta_color="inverse")
        k4.metric("mAP@50", f"{map50_val:.4f}" if map50_val is not None else "N/A", delta=f"{map50_d:+.4f}" if map50_d else None)
        k5.metric("mAP@50-95", f"{map95_val:.4f}" if map95_val is not None else "N/A", delta=f"{map95_d:+.4f}" if map95_d else None)
        k6.metric("Prec / Recall", f"{(prec_val or 0):.2f} / {(rec_val or 0):.2f}")

        # Metric Curves Tabs
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📉 Training & Validation Losses", "🎯 mAP & Accuracy Dynamics", "⚡ Learning Rate Schedule"])

        with chart_tab1:
            loss_cols = [c for c in df_results.columns if "loss" in c.lower()]
            if loss_cols:
                pdf = df_results.set_index("epoch")[loss_cols] if "epoch" in df_results.columns else df_results[loss_cols]
                st.line_chart(pdf, height=300)

        with chart_tab2:
            acc_cols = [c for c in df_results.columns if any(m in c.lower() for m in ["map", "precision", "recall", "accuracy"])]
            if acc_cols:
                pdf = df_results.set_index("epoch")[acc_cols] if "epoch" in df_results.columns else df_results[acc_cols]
                st.line_chart(pdf, height=300)

        with chart_tab3:
            lr_cols = [c for c in df_results.columns if "lr/" in c.lower()]
            if lr_cols:
                pdf = df_results.set_index("epoch")[lr_cols] if "epoch" in df_results.columns else df_results[lr_cols]
                st.line_chart(pdf, height=300)
    else:
        st.info("💡 Real-time metric curves and KPI badges will render here as soon as the first epoch finishes training.")

    # Live Console & Artifacts
    col_con, col_art = st.columns([1.3, 0.7])

    with col_con:
        st.markdown("##### 🖥️ Live Terminal Log Stream")
        st.code("".join(st.session_state.train_logs[-40:]) if st.session_state.train_logs else "Ready to train. Output stream will appear here.", language="bash")

    with col_art:
        st.markdown("##### 🏆 Checkpoints & Artifacts")
        if actual_run_dir and actual_run_dir.exists():
            b_pt = actual_run_dir / "weights" / "best.pt"
            l_pt = actual_run_dir / "weights" / "last.pt"

            if b_pt.exists():
                st.success(f"✨ `best.pt` ready ({b_pt.stat().st_size / (1024*1024):.1f} MB)")
                with open(b_pt, "rb") as f:
                    st.download_button("📥 Download best.pt Weights", f, file_name=f"{current_run}_best.pt", key="dl_best_w", width="stretch")

            if l_pt.exists():
                with open(l_pt, "rb") as f:
                    st.download_button("📥 Download last.pt Weights", f, file_name=f"{current_run}_last.pt", key="dl_last_w", width="stretch")

            res_png = actual_run_dir / "results.png"
            if res_png.exists():
                st.image(str(res_png), caption="Training Results Plot", width="stretch")
        else:
            st.caption("Training checkpoints and summary plots will appear here.")

    # -------------------------------------------------------------------
    # 📊 System Utilization Monitor (auto-refreshes every 2 seconds)
    # -------------------------------------------------------------------
    st.markdown("---")

    @st.fragment(run_every=2)
    def _system_utilization_monitor():
        st.markdown("##### 📊 System Utilization Monitor — 🔴 Live")

        sys_stats = get_system_utilization()

        # ---- Row 1: CPU, RAM, Swap, Disk overview metrics ----
        hw_k1, hw_k2, hw_k3, hw_k4 = st.columns(4)
        cpu_freq_str = f" @ {sys_stats['cpu_freq']:.0f} MHz" if sys_stats.get('cpu_freq') else ""
        hw_k1.metric(
            f"CPU ({sys_stats['cpu_count_physical']}P/{sys_stats['cpu_count_logical']}L{cpu_freq_str})",
            f"{sys_stats['cpu_percent']:.1f}%"
        )
        hw_k2.metric(
            "RAM Usage",
            f"{sys_stats['ram_used_gb']:.1f} / {sys_stats['ram_total_gb']:.1f} GB",
            delta=f"{sys_stats['ram_percent']:.1f}% used",
            delta_color="inverse" if sys_stats['ram_percent'] > 85 else "off"
        )
        hw_k3.metric(
            "Swap",
            f"{sys_stats['swap_used_gb']:.1f} / {sys_stats['swap_total_gb']:.1f} GB",
            delta=f"{sys_stats['swap_percent']:.1f}% used" if sys_stats['swap_total_gb'] > 0 else "N/A",
            delta_color="inverse" if sys_stats['swap_percent'] > 50 else "off"
        )
        hw_k4.metric(
            "Disk (/)",
            f"{sys_stats['disk_used_gb']:.0f} / {sys_stats['disk_total_gb']:.0f} GB",
            delta=f"{sys_stats['disk_percent']:.1f}% used",
            delta_color="inverse" if sys_stats['disk_percent'] > 90 else "off"
        )

        # ---- Row 2: CPU per-core utilization bar + RAM visual bar ----
        util_col1, util_col2 = st.columns(2)

        with util_col1:
            cores = sys_stats["cpu_per_core"]
            core_html_rows = ""
            for i, pct in enumerate(cores):
                if pct < 50:
                    bar_color = f"rgb({int(pct * 5.1)}, 229, 163)"
                elif pct < 80:
                    bar_color = f"rgb(245, {int(229 - (pct - 50) * 5.7)}, {int(163 - (pct - 50) * 5.4)})"
                else:
                    bar_color = f"rgb(239, {max(68, int(163 - (pct - 80) * 4.75))}, {max(68, int(100 - (pct - 80) * 1.6))})"
                core_html_rows += (
                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">'
                    f'<span style="min-width:42px;font-size:0.72rem;color:#94a3b8;font-family:monospace;">C{i:02d}</span>'
                    f'<div style="flex:1;height:14px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;">'
                    f'<div style="width:{pct}%;height:100%;background:{bar_color};border-radius:4px;transition:width 0.3s ease;"></div></div>'
                    f'<span style="min-width:38px;text-align:right;font-size:0.72rem;color:#cbd5e1;font-family:monospace;">{pct:.0f}%</span>'
                    f'</div>'
                )
            cpu_html = (
                '<div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:12px 14px;max-height:260px;overflow-y:auto;">'
                '<div style="font-weight:700;font-size:0.8rem;color:#38bdf8;margin-bottom:8px;">🧠 CPU Core Utilization</div>'
                f'{core_html_rows}'
                '</div>'
            )
            st.markdown(cpu_html, unsafe_allow_html=True)

        with util_col2:
            ram_pct = sys_stats["ram_percent"]
            if ram_pct < 60:
                ram_color, ram_glow = "#00e5a3", "rgba(0,229,163,0.3)"
            elif ram_pct < 85:
                ram_color, ram_glow = "#f59e0b", "rgba(245,158,11,0.3)"
            else:
                ram_color, ram_glow = "#ef4444", "rgba(239,68,68,0.3)"

            train_proc_html = ""
            if sys_stats["train_proc"]:
                tp = sys_stats["train_proc"]
                proc_ram_pct = (tp['rss_gb'] / sys_stats['ram_total_gb']) * 100 if sys_stats['ram_total_gb'] > 0 else 0
                train_proc_html = (
                    '<div style="background:rgba(0,229,163,0.08);border:1px solid rgba(0,229,163,0.2);border-radius:8px;padding:10px;margin-top:10px;">'
                    f'<div style="font-size:0.75rem;color:#00e5a3;font-weight:700;margin-bottom:6px;">🏋️ Training Process (PID {tp["pid"]})</div>'
                    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.75rem;">'
                    f'<div style="color:#94a3b8;">Resident Memory</div><div style="color:#cbd5e1;text-align:right;font-weight:600;">{tp["rss_gb"]:.2f} GB ({proc_ram_pct:.1f}%)</div>'
                    f'<div style="color:#94a3b8;">CPU Usage</div><div style="color:#cbd5e1;text-align:right;font-weight:600;">{tp["cpu_percent"]:.1f}%</div>'
                    f'<div style="color:#94a3b8;">Threads</div><div style="color:#cbd5e1;text-align:right;font-weight:600;">{tp["num_threads"]}</div>'
                    '</div></div>'
                )

            ram_html = (
                '<div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:14px;">'
                '<div style="font-weight:700;font-size:0.8rem;color:#38bdf8;margin-bottom:10px;">💾 Memory Allocation</div>'
                '<div style="margin-bottom:14px;">'
                '<div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#94a3b8;margin-bottom:4px;">'
                f'<span>RAM</span><span style="color:{ram_color};font-weight:700;">{sys_stats["ram_used_gb"]:.1f} GB / {sys_stats["ram_total_gb"]:.1f} GB ({ram_pct:.1f}%)</span></div>'
                f'<div style="height:22px;background:rgba(255,255,255,0.06);border-radius:6px;overflow:hidden;box-shadow:0 0 8px {ram_glow};">'
                f'<div style="width:{ram_pct}%;height:100%;background:linear-gradient(90deg,{ram_color}cc,{ram_color});border-radius:6px;transition:width 0.4s ease;"></div></div></div>'
                '<div style="margin-bottom:14px;">'
                '<div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#94a3b8;margin-bottom:4px;">'
                f'<span>Available</span><span style="color:#10b981;font-weight:600;">{sys_stats["ram_available_gb"]:.1f} GB free</span></div></div>'
                f'{train_proc_html}'
                '</div>'
            )
            st.markdown(ram_html, unsafe_allow_html=True)

        # ---- Row 3: GPU Cards (if NVIDIA GPUs detected) ----
        if sys_stats["gpus"]:
            gpu_cols = st.columns(len(sys_stats["gpus"]))
            for idx, gpu in enumerate(sys_stats["gpus"]):
                with gpu_cols[idx]:
                    gpu_util = gpu.get("util_percent", 0) or 0
                    gpu_mem_pct = gpu.get("mem_percent", 0) or 0
                    gpu_temp = gpu.get("temp_c")
                    gpu_power = gpu.get("power_w")
                    gpu_power_limit = gpu.get("power_limit_w")
                    gpu_mem_used = gpu.get("mem_used_mb", 0) or 0
                    gpu_mem_total = gpu.get("mem_total_mb", 0) or 0

                    if gpu_temp and gpu_temp > 80:
                        temp_color = "#ef4444"
                    elif gpu_temp and gpu_temp > 65:
                        temp_color = "#f59e0b"
                    else:
                        temp_color = "#10b981"

                    if gpu_util > 80:
                        util_color = "#00e5a3"
                    elif gpu_util > 40:
                        util_color = "#38bdf8"
                    else:
                        util_color = "#94a3b8"

                    if gpu_mem_pct > 85:
                        vram_color = "#ef4444"
                    elif gpu_mem_pct > 60:
                        vram_color = "#f59e0b"
                    else:
                        vram_color = "#00e5a3"

                    power_html = ""
                    if gpu_power and gpu_power_limit:
                        power_html = (
                            f'<div style="display:flex;justify-content:space-between;margin-top:6px;">'
                            f'<span style="color:#94a3b8;">⚡ Power</span>'
                            f'<span style="color:#cbd5e1;font-weight:600;">{gpu_power:.0f}W / {gpu_power_limit:.0f}W</span></div>'
                        )

                    temp_str = f'{gpu_temp:.0f}°C' if gpu_temp else 'N/A'

                    gpu_html = (
                        '<div style="background:rgba(15,23,42,0.7);border:1px solid rgba(56,189,248,0.2);border-radius:10px;padding:14px;">'
                        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
                        f'<span style="font-weight:700;font-size:0.85rem;color:#38bdf8;">🎮 GPU {gpu["index"]}</span>'
                        f'<span style="font-size:0.72rem;color:#94a3b8;background:rgba(255,255,255,0.06);padding:2px 8px;border-radius:12px;">{gpu["name"]}</span></div>'
                        '<div style="font-size:0.78rem;">'
                        '<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                        f'<span style="color:#94a3b8;">GPU Compute</span><span style="color:{util_color};font-weight:700;">{gpu_util:.0f}%</span></div>'
                        f'<div style="height:16px;background:rgba(255,255,255,0.06);border-radius:5px;overflow:hidden;margin-bottom:10px;">'
                        f'<div style="width:{gpu_util}%;height:100%;background:linear-gradient(90deg,{util_color}aa,{util_color});border-radius:5px;"></div></div>'
                        '<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                        f'<span style="color:#94a3b8;">VRAM</span><span style="color:{vram_color};font-weight:700;">{gpu_mem_used:.0f} / {gpu_mem_total:.0f} MB ({gpu_mem_pct:.1f}%)</span></div>'
                        f'<div style="height:16px;background:rgba(255,255,255,0.06);border-radius:5px;overflow:hidden;margin-bottom:8px;">'
                        f'<div style="width:{gpu_mem_pct}%;height:100%;background:linear-gradient(90deg,{vram_color}aa,{vram_color});border-radius:5px;"></div></div>'
                        f'<div style="display:flex;justify-content:space-between;">'
                        f'<span style="color:#94a3b8;">🌡️ Temperature</span><span style="color:{temp_color};font-weight:700;">{temp_str}</span></div>'
                        f'{power_html}'
                        '</div></div>'
                    )
                    st.markdown(gpu_html, unsafe_allow_html=True)

    _system_utilization_monitor()

    # Polling loop
    if st.session_state.train_active:
        proc = st.session_state.train_proc
        if proc and proc.poll() is not None:
            st.session_state.train_active = False
            ret = proc.poll()
            if ret == 0:
                st.toast("🎉 Training completed successfully!", icon="✅")
            else:
                st.toast(f"⚠️ Training exited with code {ret}", icon="❌")
            st.rerun()
        else:
            time.sleep(1)
            st.rerun()


# ===========================================================================
# TAB 1.5: ☁️ KAGGLE CLOUD GPU TRAINING
# ===========================================================================
with tab_kaggle:
    st.markdown("### ☁️ Kaggle Cloud GPU Training Hub")
    st.caption("Offload training to Kaggle's free compute cluster (Dual NVIDIA Tesla T4 32GB VRAM / P100) and synchronize models automatically.")

    # 1. Authentication Status Card
    is_auth, kaggle_user, auth_err = kaggle_bridge.is_authenticated()

    auth_col1, auth_col2 = st.columns([2, 1])
    with auth_col1:
        if is_auth:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 12px 18px; margin-bottom: 12px;">
                <span style="font-weight: 700; color: #10b981; font-size: 1rem;">🟢 Kaggle API Connected:</span>
                <span style="color: #f8fafc; font-weight: 600;">@{kaggle_user}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 12px 18px; margin-bottom: 12px;">
                <span style="font-weight: 700; color: #f59e0b; font-size: 1rem;">⚠️ Kaggle API Not Connected</span>
                <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 4px;">Configure your API token below to enable 1-click cloud GPU training.</div>
            </div>
            """, unsafe_allow_html=True)

    with auth_col2:
        with st.expander("🔑 Kaggle API Credentials Setup", expanded=not is_auth):
            st.markdown("""
            **Get your API token:**
            1. Sign in to [Kaggle](https://www.kaggle.com).
            2. Go to **Account Settings** -> **API** -> Click **Create New Token** (`kaggle.json`).
            """)
            uploaded_k_json = st.file_uploader("Upload kaggle.json", type=["json"], key="k_json_uploader")
            if uploaded_k_json:
                try:
                    k_data = json.load(uploaded_k_json)
                    u = k_data.get("username", "")
                    k = k_data.get("key", "")
                    if u and k:
                        ok, msg = kaggle_bridge.save_credentials(u, k)
                        if ok:
                            st.success(msg)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(msg)
                except Exception as e:
                    st.error(f"Invalid JSON file: {e}")

            st.markdown("*Or enter manually:*")
            k_user_input = st.text_input("Kaggle Username (handle, not email)", value=kaggle_user or "", placeholder="e.g. rakshithr1701")
            k_key_input = st.text_input("Kaggle API Key", type="password", placeholder="e.g. 38d9c...")
            if st.button("💾 Save Credentials & Connect", type="primary", use_container_width=True):
                if k_user_input and k_key_input:
                    ok, msg = kaggle_bridge.save_credentials(k_user_input, k_key_input)
                    if ok:
                        st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please provide both Username and API Key.")

    # 2. GPU quota tracker.
    # Kaggle publishes no quota API, so this is an estimate over the jobs this
    # dashboard dispatched — runs started on kaggle.com are not counted.
    quota = kaggle_bridge.estimate_weekly_gpu_usage()
    res1, res2, res3, res4 = st.columns(4)
    with res1:
        st.metric("⚡ Remote GPU Cluster", "2x T4 / P100", "32GB VRAM (DDP)")
    with res2:
        st.metric(
            "⏱️ Est. GPU Time Used",
            f"{quota['used_hours']:.1f} h",
            f"{quota['pct_used']:.0f}% of {quota['quota_hours']:.0f}h weekly",
            delta_color="inverse",
        )
    with res3:
        st.metric("🔋 Est. Remaining", f"{quota['remaining_hours']:.1f} h", "rolling 7 days")
    with res4:
        st.metric("📦 Local Ingestion", "Auto-Sync", "best.pt & results.csv")

    st.progress(min(1.0, quota["pct_used"] / 100.0))
    _q_note = (
        f"Estimated from {quota['jobs_counted']} tracked job(s) in the last 7 days"
        + (f"; {quota['jobs_unknown']} with unknown runtime" if quota["jobs_unknown"] else "")
        + (f"; {quota['jobs_ongoing']} still running" if quota["jobs_ongoing"] else "")
        + ". Kaggle has no quota API — jobs launched on kaggle.com are not counted. "
    )
    st.caption(_q_note + "Confirm at [kaggle.com/settings](https://www.kaggle.com/settings).")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Training Job Dispatcher
    st.markdown("#### 🚀 Configure Remote Training Job")
    k_col1, k_col2 = st.columns(2)

    with k_col1:
        st.markdown("##### 1. Dataset & Identification")
        datasets_map = discover_all_datasets()
        if datasets_map:
            k_selected_ds = st.selectbox("Select Dataset to Train On", list(datasets_map.keys()), key="k_ds_select")
            k_ds_yaml_path = datasets_map[k_selected_ds]
            # A classification dataset is a directory, not a yaml — staging must
            # upload that folder, not everything under dataset/.
            k_ds_folder = (k_ds_yaml_path if k_ds_yaml_path.is_dir()
                           else k_ds_yaml_path.parent)
        else:
            st.warning("No datasets detected in workspace. Upload or extract a dataset in Dataset Hub.")
            k_ds_yaml_path = None
            k_ds_folder = None

        k_ds_title = st.text_input("Kaggle Dataset Title", value=k_ds_folder.name if k_ds_folder else "cctv-yolo-dataset", help="Remote dataset name on Kaggle")
        k_existing_ds = st.text_input(
            "Or use existing Kaggle dataset ref(s)",
            value="",
            placeholder="user/my-dataset  or  user/ds-p0, user/ds-p1",
            help="Comma-separated owner/slug refs. If set, the app skips uploading and "
                 "attaches these directly — use this when you've uploaded the dataset "
                 "on kaggle.com yourself.",
        )
        k_job_title = st.text_input("Kaggle Kernel Job Title", value="yolo11-cloud-training", help="Remote kernel notebook name")
        # Defaults to the first unused name so a new cloud run never ingests
        # over a previous one's results.
        k_target_exp = st.text_input(
            "Local Target Run Name", value=kaggle_bridge.next_free_run_name(),
            help="Folder under yolo_workspace/runs/ for this run's weights and metrics. "
                 "Auto-incremented so runs don't overwrite each other.")

    with k_col2:
        st.markdown("##### 2. Remote Hyperparameters")
        k_arch_col1, k_arch_col2 = st.columns(2)
        with k_arch_col1:
            k_family = st.selectbox("Model Architecture", ["YOLO11", "YOLOv8", "YOLOv9", "YOLOv10"], key="k_family")
            k_size_map = {
                "YOLO11": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
                "YOLOv8": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                "YOLOv9": ["yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt"],
                "YOLOv10": ["yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10l.pt", "yolov10x.pt"],
            }
        with k_arch_col2:
            k_model_weight = st.selectbox("Weights / Scale", k_size_map.get(k_family, ["yolo11n.pt"]), key="k_model_weight")

        k_p_col1, k_p_col2 = st.columns(2)
        with k_p_col1:
            k_epochs = st.slider("Epochs", 1, 300, 100, step=5, key="k_epochs")
            k_batch = st.select_slider("Batch Size (GPU)", options=[8, 16, 32, 64, 128], value=32, key="k_batch")
            k_imgsz = st.select_slider("Image Size (imgsz)", options=[320, 416, 512, 640, 768, 1024, 1280], value=640, key="k_imgsz")
        with k_p_col2:
            k_opt = st.selectbox("Optimizer", ["AdamW", "SGD", "Adam", "auto"], key="k_opt")
            k_lr0 = st.number_input("Initial Learning Rate (lr0)", value=0.005, format="%.4f", step=0.001, key="k_lr0")
            k_patience = st.number_input("Early Stopping Patience", min_value=0, max_value=100, value=20, key="k_patience")

        k_dual_gpu = st.checkbox("⚡ Leverage Dual-GPU Distributed Data Parallel (2x T4)", value=True)

        k_max_hours = st.slider(
            "⏳ Max runtime (hours)", min_value=1.0, max_value=11.5, value=11.0, step=0.5,
            key="k_max_hours",
            help="Kaggle kills a session at 12h. Training stops at this cap and packages "
                 "last.pt so the run can be resumed in a follow-up job.",
        )

        # Resume: continue a previous kernel that hit the runtime cap.
        _resumable = [
            j for j in kaggle_bridge.list_recent_jobs_history()
            if j.get("resumable") or j.get("remote_state") == "timecapped"
        ]
        _resume_opts = ["(start a fresh run)"] + [j["kernel_ref"] for j in _resumable]
        k_resume_pick = st.selectbox(
            "♻️ Resume from previous job", _resume_opts, key="k_resume_pick",
            help="Mounts that kernel's output and continues training from its last.pt.",
        )
        k_resume_from = None if k_resume_pick == _resume_opts[0] else k_resume_pick
        if k_resume_from:
            st.caption(f"Will continue `{k_resume_from}` from its last checkpoint. "
                       "Epochs/optimizer come from that checkpoint, not the sliders above.")

    st.markdown("##### 3. After the job finishes")
    post1, post2 = st.columns(2)
    with post1:
        st.checkbox(
            "📥 Auto-ingest results when the job completes", value=True, key="k_auto_ingest",
            help="Polls tracked jobs and pulls weights + metrics into yolo_workspace/runs/ automatically.",
        )
    with post2:
        st.multiselect(
            "📦 Auto-export formats after ingest", ["onnx", "torchscript", "openvino"],
            default=[], key="k_auto_export",
            help="Compiles the ingested best.pt locally into these deployment runtimes.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Launch Button
    can_launch = is_auth and (k_ds_folder is not None)
    if st.button("🚀 Stage Dataset & Dispatch Training to Kaggle GPU", type="primary", disabled=not can_launch, use_container_width=True):
        api = kaggle_bridge.get_kaggle_api()
        with st.status("🚀 Processing Kaggle Remote Job...", expanded=True) as status_box:
            _manual_refs = [r.strip() for r in k_existing_ds.split(",") if r.strip() and "/" in r]
            if _manual_refs:
                status_box.write(f"📎 Using existing Kaggle dataset ref(s): {', '.join(_manual_refs)}")
                ds_ok, ds_msg, ds_refs = True, f"Attaching {len(_manual_refs)} existing dataset(s).", _manual_refs
            else:
                status_box.write("📦 Packaging and verifying dataset for Kaggle...")
                ds_ok, ds_msg, ds_refs = kaggle_bridge.package_and_upload_dataset(
                    dataset_path=k_ds_folder,
                    dataset_title=k_ds_title,
                    api=api,
                    progress_callback=lambda msg: status_box.write(f"  ➜ {msg}")
                )
            if not ds_ok:
                status_box.update(label="❌ Dataset upload failed", state="error")
                st.error(ds_msg)
            else:
                status_box.write(f"✅ {ds_msg}")
                if k_resume_from:
                    status_box.write(f"♻️ Mounting `{k_resume_from}` output to resume from its last.pt...")
                status_box.write("🛰️ Generating remote training script and pushing kernel to Kaggle GPU cluster...")
                disp_ok, disp_msg, kernel_ref = kaggle_bridge.dispatch_kaggle_training(
                    dataset_ref=ds_refs,
                    kernel_title=k_job_title,
                    model_name=k_model_weight,
                    epochs=k_epochs,
                    batch_size=k_batch,
                    imgsz=k_imgsz,
                    optimizer=k_opt,
                    lr0=k_lr0,
                    patience=k_patience,
                    enable_dual_gpu=k_dual_gpu,
                    api=api,
                    max_hours=k_max_hours,
                    resume_from_kernel=k_resume_from,
                )
                # The kernel is created even when attachment can't be confirmed —
                # record it either way so the dashboard can track it.
                if kernel_ref:
                    st.session_state.kaggle_active_kernel = kernel_ref
                    kaggle_bridge.save_job_to_history({
                        "kernel_ref": kernel_ref,
                        "dataset_ref": (", ".join(ds_refs) if isinstance(ds_refs, (list, tuple)) else ds_refs),
                        "dataset_parts": len(ds_refs) if isinstance(ds_refs, (list, tuple)) else 1,
                        "model_name": k_model_weight,
                        "epochs": k_epochs,
                        "target_exp": k_target_exp,
                        "max_hours": k_max_hours,
                        "resumed_from": k_resume_from,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })

                if disp_ok:
                    status_box.update(label="🎉 Job Dispatched to Kaggle GPU!", state="complete")
                    st.success(disp_msg)
                else:
                    status_box.update(label="⚠️ Dispatched with a warning", state="error")
                    st.warning(disp_msg)

    st.markdown("---")

    # 4. Remote Job Dashboard — live status, progress, GPU telemetry, ingestion.
    st.session_state.setdefault("jobs_nonce", 0)
    st.session_state.setdefault("auto_ingest_log", [])

    def _post_ingest_export(run_dir: Path, formats):
        """Compiles an ingested checkpoint into the requested deployment runtimes."""
        done, failed = [], []
        weights = [run_dir / "weights" / "best.pt", run_dir / "best.pt"]
        ckpt = next((w for w in weights if w.exists()), None)
        if not ckpt:
            return done, [("*", "no best.pt in the ingested run")]
        for fmt in formats:
            try:
                out = YOLO(str(ckpt)).export(format=fmt, imgsz=640, device="cpu")
                done.append((fmt, str(out)))
            except Exception as e:
                failed.append((fmt, str(e)))
        return done, failed

    dash_head1, dash_head2, dash_head3 = st.columns([2, 1, 1])
    with dash_head1:
        st.markdown("#### 📡 Kaggle Training Jobs")
    with dash_head2:
        k_live = st.toggle("🔄 Live polling", value=False, key="k_live_poll",
                           help="Re-checks job status every 30s and auto-ingests finished runs.")
    with dash_head3:
        if st.button("🔄 Refresh Jobs", use_container_width=True):
            st.session_state.jobs_nonce += 1
            st.rerun()

    def _run_auto_ingest():
        """Pulls finished jobs down and optionally exports them. Returns log lines."""
        lines = []
        for res in kaggle_bridge.auto_ingest_completed_jobs():
            stamp = time.strftime("%H:%M:%S")
            if res["ok"]:
                lines.append(f"[{stamp}] ✅ {res['kernel_ref'].split('/')[-1]} → runs/{res['target_exp']}")
                _fmts = st.session_state.get("k_auto_export") or []
                if _fmts and res.get("path"):
                    done, failed = _post_ingest_export(Path(res["path"]), _fmts)
                    for fmt, out in done:
                        lines.append(f"[{stamp}]    📦 exported {fmt}: {Path(out).name}")
                    for fmt, err in failed:
                        lines.append(f"[{stamp}]    ⚠️ {fmt} export failed: {err[:120]}")
            else:
                lines.append(f"[{stamp}] ⚠️ {res['kernel_ref'].split('/')[-1]}: {res['message'][:140]}")
        return lines

    # Kaggle has no cancel API and no direct URL for the Active Events panel,
    # so spell out the click path instead of only linking the kernel page.
    _STOP_STEPS = (
        "**To stop it on Kaggle:**\n"
        "1. Open [kaggle.com]({url}) — or any Kaggle page.\n"
        "2. In the left sidebar click **⧉ View Active Events**.\n"
        "3. Find this session, click the **⋯** next to it, then **⏹ Stop Session**.\n\n"
        "The kernel page's Run/Save controls do *not* stop a running batch job — "
        "Active Events is the only place with the stop control."
    )

    _CAT_COLOUR = {
        "running": "#10b981", "pending": "#f59e0b", "successful": "#22c55e",
        "terminated": "#f97316", "cancelled": "#ef4444", "failed": "#dc2626",
        "unresolved": "#64748b",
    }

    _CAT_HELP = {
        "running":    "Training on Kaggle right now.",
        "pending":    "Queued, or still starting up.",
        "successful": "Finished cleanly — the trained model is ready to download.",
        "terminated": "Cut short by the runtime cap before finishing all epochs.",
        "cancelled":  "Stopped from the Kaggle console.",
        "failed":     "Errored before producing a model.",
        "unresolved": "Session has ended; its outcome has not been read yet.",
    }

    def _when(ts):
        """'12 min ago' / 'yesterday 23:23' style timestamp for the table."""
        if not ts:
            return "—"
        try:
            t = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
        except Exception:
            return ts
        secs = time.time() - t
        if secs < 60:
            return "just now"
        if secs < 3600:
            return f"{int(secs // 60)} min ago"
        if secs < 86400:
            return f"{int(secs // 3600)} h ago"
        if secs < 172800:
            return f"yesterday {ts[11:16]}"
        return f"{int(secs // 86400)} days ago"

    def _ran(j):
        if j.get("category") in ("running", "pending"):
            return j.get("elapsed", "—")
        if j.get("gpu_hours"):
            return kaggle_bridge._duration_str(j["gpu_hours"] * 3600)
        return "—"

    def _model_cell(j):
        cat = j.get("category")
        if cat == "successful":
            return "✅ best.pt" if j.get("has_local_model") else "best.pt"
        if cat in ("terminated", "cancelled"):
            return "partial"
        return "—"

    def _render_gpu_panel(prog):
        """GPU utilisation cards + history chart from remote telemetry."""
        latest = prog.get("gpu_latest") or []
        summary = prog.get("gpu_summary") or {}
        if not latest:
            return
        st.markdown("**🖥️ Remote GPU utilisation**")
        for g in latest:
            gc = st.columns(4)
            mem_used, mem_total = g.get("mem_used") or 0, g.get("mem_total") or 0
            gc[0].metric(f"GPU {g['index']} Util", f"{g.get('util') or 0:.0f}%")
            gc[1].metric("VRAM", f"{mem_used/1024:.1f} GB",
                         f"of {mem_total/1024:.1f} GB" if mem_total else None)
            gc[2].metric("Temp", f"{g.get('temp') or 0:.0f}°C")
            gc[3].metric("Power", f"{g.get('power') or 0:.0f} W")
        if summary:
            st.caption(
                f"Avg util {summary.get('avg_util')}% · peak {summary.get('peak_util')}% · "
                f"peak VRAM {(summary.get('peak_mem_mb') or 0)/1024:.1f} GB · "
                f"{summary.get('n_samples')} samples over "
                f"{(summary.get('sampled_seconds') or 0)/60:.0f} min"
            )
        series = prog.get("gpu_series") or []
        if len(series) > 2:
            gdf = pd.DataFrame(series)
            gdf["Minutes into job"] = (gdf["t"] / 60.0).round(1)
            util_df = (gdf.pivot_table(index="Minutes into job", columns="index",
                                       values="util", aggfunc="max")
                          .rename(columns=lambda i: f"GPU {i} util %"))
            st.line_chart(util_df, height=220)

    def _render_progress(prog):
        """Epoch progress, metrics, ETA and log tail for one job."""
        if prog.get("epoch") and prog.get("total_epochs"):
            st.progress((prog.get("pct") or 0) / 100.0)
            eta = f" · ETA {prog['eta_str']}" if prog.get("eta_str") else ""
            pace = f" · {prog['sec_per_epoch']}s/epoch" if prog.get("sec_per_epoch") else ""
            st.caption(f"Epoch {prog['epoch']} / {prog['total_epochs']} · {prog.get('pct', 0)}%{pace}{eta}")
        m = prog.get("metrics") or {}
        if m:
            mc = st.columns(4)
            mc[0].metric("mAP@50", f"{m.get('mAP50', 0):.3f}")
            mc[1].metric("mAP@50-95", f"{m.get('mAP50_95', 0):.3f}")
            mc[2].metric("Precision", f"{m.get('precision', 0):.3f}")
            mc[3].metric("Recall", f"{m.get('recall', 0):.3f}")
        _render_gpu_panel(prog)
        if prog.get("error_line"):
            st.warning(f"Likely cause: {prog['error_line']}")
        if prog.get("log_available"):
            st.code(prog.get("tail") or "(log empty)", language="log")
        else:
            st.info("Kaggle only serves the run log once the session ends, so epoch detail and "
                    "GPU telemetry appear after the job finishes. The status above is live.")

    def _weight_downloads(j, partial=False):
        """Download buttons for whatever .pt files are on disk for this job.

        Driven by the filesystem, not session state, so a fetched model stays
        downloadable across reruns.
        """
        weights = kaggle_bridge.local_weights_for(j)
        if not weights:
            return False
        ref = j["kernel_ref"]
        cols = st.columns(len(weights))
        for col, (wname, wpath) in zip(cols, sorted(weights.items())):
            try:
                data = wpath.read_bytes()
            except OSError:
                continue
            tag = " (partial)" if partial else ""
            with col:
                st.download_button(
                    f"💾 Download {wname}{tag} · {len(data)/(1024*1024):.1f} MB",
                    data=data,
                    file_name=f"{ref.split('/')[-1]}_{wname}",
                    mime="application/octet-stream",
                    key=f"dl_{ref}_{wname}", width="stretch")
        return True

    def _fetch_weights(j, label, partial=False):
        """Pull .pt files from Kaggle into this job's local weights folder."""
        ref = j["kernel_ref"]
        if st.button(label, key=f"w_{ref}", type="secondary", width="stretch"):
            with st.spinner("Downloading model weights from Kaggle..."):
                ok, msg, saved = kaggle_bridge.download_weights(
                    ref, dest_dir=kaggle_bridge.job_weights_dir(j))
            if ok:
                st.success(msg)
                st.rerun()          # re-render the card so the buttons appear
            else:
                st.warning(msg)

    def _render_card(j):
        """One self-contained vertical card for a job."""
        ref = j["kernel_ref"]
        cat = j["category"]
        icon, name = kaggle_bridge.CATEGORY_LABELS[cat]
        colour = _CAT_COLOUR[cat]

        with st.container(border=True):
            # --- Header ----------------------------------------------------
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>"
                f"<span style='background:{colour}22;border:1px solid {colour}66;color:{colour};"
                f"padding:3px 12px;border-radius:20px;font-weight:700;font-size:0.82rem;'>"
                f"{icon} {name}</span>"
                f"<span style='font-weight:700;font-size:1.02rem;color:#f8fafc;'>"
                f"{ref.split('/')[-1]}</span>"
                f"<span style='color:#94a3b8;font-size:0.85rem;'>· {_when(j.get('timestamp'))}</span>"
                f"</div>", unsafe_allow_html=True)

            # --- Facts -----------------------------------------------------
            f1, f2, f3, f4 = st.columns(4)
            f1.markdown(f"**Model**<br>`{j.get('model_name') or '—'}`", unsafe_allow_html=True)
            f2.markdown(f"**Epochs**<br>{j.get('epochs') or '—'}", unsafe_allow_html=True)
            _ds = str(j.get("dataset_ref", "—")).split(",")[0].strip().split("/")[-1] or "—"
            f3.markdown(f"**Dataset**<br>`{_ds}`", unsafe_allow_html=True)
            f4.markdown(f"**Runtime**<br>{_ran(j)}", unsafe_allow_html=True)

            reason = kaggle_bridge.job_termination_reason(j)
            if cat in ("terminated", "cancelled"):
                st.warning(f"{reason}.")
            elif cat == "failed":
                st.error(f"{reason}.")
            elif cat == "unresolved":
                st.info(f"{reason}.")
            if j.get("failureMessage"):
                st.error(f"Kernel failure: {j['failureMessage']}")
            if j.get("discovered"):
                st.caption("📡 Found on your Kaggle account — not dispatched from here, so its "
                           "hyperparameters aren't known locally.")
            if j.get("resumed_from"):
                st.caption(f"♻️ Continues `{j['resumed_from']}`")

            # --- Model downloads -------------------------------------------
            if cat == "successful":
                st.markdown("**🎯 Trained model**")
                if not _weight_downloads(j):
                    st.caption("Weights are still on Kaggle — fetch them to enable the download.")
                    _fetch_weights(j, "⬇ Fetch model (.pt) from Kaggle")
                else:
                    st.caption(f"Saved in `{kaggle_bridge.job_weights_dir(j)}` — also available "
                               "in Inference and Export Studio.")
            elif cat in ("terminated", "cancelled"):
                st.markdown("**🧩 Partial checkpoint**")
                if not _weight_downloads(j, partial=True):
                    st.caption("Not a finished model, but it can seed a resume job.")
                    _fetch_weights(j, "⬇ Fetch partial checkpoint (.pt)", partial=True)

            # --- Actions ---------------------------------------------------
            acts = []
            if cat in ("running", "pending"):
                acts.append("stop")
            if cat == "successful":
                acts.append("ingest")
            acts += ["log", "remove"]
            cols = st.columns(len(acts))

            for col, act in zip(cols, acts):
                with col:
                    if act == "stop":
                        if st.button("🛑 Stop this job", key=f"stop_{ref}",
                                     type="primary", width="stretch"):
                            with st.spinner("Checking live status on Kaggle..."):
                                stopped, msg, url = kaggle_bridge.request_kernel_stop(ref)
                            (st.success if stopped else st.warning)(msg)
                            if not stopped:
                                st.markdown(_STOP_STEPS.format(url=url))
                    elif act == "ingest":
                        if st.button("📥 Ingest full run", key=f"ing_{ref}", width="stretch",
                                     help="Metrics, curves and plots into yolo_workspace/runs/"):
                            exp_name = j.get("target_exp") or kaggle_bridge.format_dataset_slug(
                                ref.split("/")[-1])
                            with st.spinner("Downloading run output from Kaggle..."):
                                ok, msg, run_path = kaggle_bridge.download_and_ingest_artifacts(
                                    ref, exp_name)
                            (st.success if ok else st.warning)(msg)
                            if ok:
                                prog = kaggle_bridge.get_training_progress(ref)
                                kaggle_bridge.record_job_runtime(ref, prog)
                                kaggle_bridge.save_job_to_history({
                                    "kernel_ref": ref, "ingested": True, "target_exp": exp_name,
                                    "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                                _fmts = st.session_state.get("k_auto_export") or []
                                if _fmts and run_path:
                                    with st.spinner(f"Exporting {', '.join(_fmts)}..."):
                                        done, failed = _post_ingest_export(Path(run_path), _fmts)
                                    for fmt, out in done:
                                        st.success(f"📦 {fmt.upper()} → `{out}`")
                                    for fmt, err in failed:
                                        st.warning(f"{fmt.upper()} export failed: {err}")
                    elif act == "log":
                        lbl = "🔎 Check outcome" if cat == "unresolved" else "📊 Log & GPU"
                        if st.button(lbl, key=f"log_{ref}", width="stretch"):
                            with st.spinner("Reading the run log from Kaggle..."):
                                st.session_state[f"prog_{ref}"] = (
                                    kaggle_bridge.resolve_job_outcome(ref))
                    elif act == "remove":
                        if st.button("🗑 Remove", key=f"del_{ref}", width="stretch"):
                            if (cat in ("running", "pending")
                                    and not st.session_state.get(f"confirm_del_{ref}")):
                                st.session_state[f"confirm_del_{ref}"] = True
                                st.warning("This job still looks active on Kaggle. Removing it "
                                           "only stops tracking — the kernel keeps running and "
                                           "keeps using your GPU quota. Press again to remove.")
                            else:
                                kaggle_bridge.delete_job_from_history(ref)
                                st.session_state.pop(f"confirm_del_{ref}", None)
                                st.session_state.jobs_nonce += 1
                                st.rerun()

            st.markdown(f"<a href='{j['url']}' target='_blank' style='color:#60a5fa;"
                        f"font-size:0.85rem;'>🔗 Open in Kaggle Console</a>",
                        unsafe_allow_html=True)

            if st.session_state.get(f"prog_{ref}"):
                with st.expander("📊 Progress, metrics, GPU telemetry & log", expanded=True):
                    _render_progress(st.session_state[f"prog_{ref}"])

    @st.fragment(run_every=30 if st.session_state.get("k_live_poll") else None)
    def _jobs_dashboard():
        if st.session_state.get("k_live_poll"):
            st.caption(f"🔄 Live — last polled {time.strftime('%H:%M:%S')} (every 30s)")
            if st.session_state.get("k_auto_ingest"):
                new_lines = _run_auto_ingest()
                if new_lines:
                    st.session_state.auto_ingest_log = (
                        st.session_state.auto_ingest_log + new_lines)[-30:]

        if st.session_state.auto_ingest_log:
            with st.expander("📥 Auto-ingest activity", expanded=True):
                st.code("\n".join(st.session_state.auto_ingest_log), language="log")

        all_jobs = kaggle_bridge.list_all_jobs()
        if not all_jobs:
            st.info("No training jobs found. Launch one above and it will appear here.")
            return

        counts = {c: 0 for c in kaggle_bridge.JOB_CATEGORIES}
        for j in all_jobs:
            counts[j["category"]] += 1

        cols = st.columns(len(kaggle_bridge.JOB_CATEGORIES))
        for col, cat in zip(cols, kaggle_bridge.JOB_CATEGORIES):
            icon, name = kaggle_bridge.CATEGORY_LABELS[cat]
            col.metric(f"{icon} {name}", counts[cat], help=_CAT_HELP[cat])

        f1, f2 = st.columns([1, 2])
        with f1:
            opts = ["All"] + [f"{kaggle_bridge.CATEGORY_LABELS[c][0]} "
                              f"{kaggle_bridge.CATEGORY_LABELS[c][1]} ({counts[c]})"
                              for c in kaggle_bridge.JOB_CATEGORIES if counts[c]]
            choice = st.selectbox("Show", opts, key="k_job_filter")
        with f2:
            query = st.text_input("Search job name or model", key="k_job_search",
                                  placeholder="e.g. yolov8, cctv, 20260904")

        shown = all_jobs
        if choice != "All":
            wanted = choice.split(" ", 1)[1].rsplit(" (", 1)[0]
            shown = [j for j in shown
                     if kaggle_bridge.CATEGORY_LABELS[j["category"]][1] == wanted]
        if query:
            q = query.lower()
            shown = [j for j in shown
                     if q in j["kernel_ref"].lower() or q in str(j.get("model_name", "")).lower()]

        if not shown:
            st.info("No jobs match that filter.")
            return

        st.caption(f"Showing {len(shown)} of {len(all_jobs)} job(s), newest first.")
        for j in shown:
            _render_card(j)

    with st.spinner("Fetching job status from Kaggle..."):
        _jobs_dashboard()

    # 5. Standalone Notebook Template Expander
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📓 Standalone Kaggle Notebook Template (.ipynb)"):
        st.markdown("""
        Prefer running manually in the Kaggle UI? Download the pre-built notebook template below, upload it directly to Kaggle, attach your dataset, and hit **Run All**.
        """)
        template_path = Path("kaggle_yolo_train_template.ipynb")
        if template_path.exists():
            with open(template_path, "r") as f:
                nb_content = f.read()
            st.download_button(
                label="📥 Download kaggle_yolo_train_template.ipynb",
                data=nb_content,
                file_name="kaggle_yolo_train_template.ipynb",
                mime="application/x-ipynb+json",
                use_container_width=True
            )


# ===========================================================================
# TAB 2: 📂 DATASET HUB & VISUAL INSPECTOR
# ===========================================================================
with tab_ds:
    st.subheader("Dataset Hub & Visual Ground-Truth Inspector")
    st.caption("Inspect class balances, explore images, and verify bounding box annotations directly from your dataset splits.")

    if not ds_info:
        st.warning("No active dataset selected. Please choose or upload a dataset in the sidebar.")
    else:
        if ds_info.get("kind") == "classify":
            st.info(
                f"**`{ds_info['name']}` is a classification dataset** "
                f"({len(ds_info['classes'])} classes, folder per class). Train it with "
                "**Task = classify** in the Training tab — Ultralytics reads the folder "
                "directly, so there is no data.yaml to select."
            )
            if ds_info.get("needs_split"):
                st.warning(
                    "It has no train/val split yet, which the classify task requires. "
                    "Split it below — images are moved into `train/` and `val/` in place."
                )
                sp1, sp2 = st.columns([1, 3])
                with sp1:
                    val_pct = st.slider("Validation %", 5, 40, 20, step=5, key="cls_val_pct")
                with sp2:
                    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                    if st.button("✂️ Create train/val split", type="primary", key="cls_split_btn"):
                        with st.status("Splitting dataset...", expanded=True) as sbox:
                            n_tr, n_va = make_classify_split(
                                Path(ds_info["root"]), val_pct / 100.0,
                                progress=lambda m: sbox.write(f"  ➜ {m}"))
                            sbox.update(label=f"Split complete: {n_tr} train / {n_va} val",
                                        state="complete")
                        st.rerun()

        ds_col1, ds_col2 = st.columns([1, 2])

        with ds_col1:
            st.markdown("##### 📊 Dataset Statistics")
            split_df = pd.DataFrame([{"Split": k.capitalize(), "Image Count": v} for k, v in ds_info["counts"].items()])
            st.dataframe(split_df, width="stretch", hide_index=True)

            st.markdown("##### 🏷️ Defined Classes")
            class_df = pd.DataFrame([{"Class ID": k, "Class Name": v} for k, v in ds_info["classes"].items()])
            st.dataframe(class_df, width="stretch", hide_index=True)

            st.markdown("##### 📄 Config File")
            st.code(str(ds_info["yaml_path"]), language="bash")

        with ds_col2:
            st.markdown("##### 🖼️ Visual Annotation Inspector")
            available_splits = list(ds_info["splits"].keys())

            if not available_splits:
                st.info("No image split folders found on disk.")
            else:
                sel_split = st.selectbox("Select Split to Inspect", available_splits, index=0)
                split_p = ds_info["splits"][sel_split]
                img_files = sorted([p for p in split_p.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]])

                if not img_files:
                    st.warning(f"No images found in split `{sel_split}`.")
                else:
                    sel_img_idx = st.slider("Browse Image Index", 0, len(img_files) - 1, 0)
                    chosen_img_path = img_files[sel_img_idx]

                    annotated_rgb, n_boxes = draw_ground_truth_boxes(chosen_img_path, ds_info["classes"])

                    if annotated_rgb is not None:
                        st.image(
                            annotated_rgb,
                            caption=f"Image [{sel_img_idx+1}/{len(img_files)}]: {chosen_img_path.name} • Ground Truth Boxes: {n_boxes}",
                            width="stretch"
                        )
                    else:
                        st.error("Could not load image.")


# ===========================================================================
# TAB 3: 🧪 INFERENCE & TESTING PLAYGROUND
# ===========================================================================
with tab_infer:
    st.subheader("Inference & Model Testing Playground")
    st.caption("Test trained weights on custom images, dataset validation samples, live camera snapshots, or video files.")

    inf_c1, inf_c2 = st.columns([1, 2.2])

    with inf_c1:
        st.markdown("##### 1. Model & Inference Settings")
        models_available = get_available_models()
        model_src_type = st.radio("Model Source", ["Discovered Checkpoints", "Upload Custom .pt Weights"], horizontal=True)

        chosen_model_path = None
        if model_src_type == "Discovered Checkpoints":
            chosen_model_path = st.selectbox("Select Weights", models_available, index=0)
        else:
            up_pt = st.file_uploader("Upload .pt File", type=["pt"], key="inf_pt_up")
            if up_pt:
                custom_save = TEMP_DIR / up_pt.name
                with open(custom_save, "wb") as f:
                    f.write(up_pt.getbuffer())
                chosen_model_path = str(custom_save)

        conf_th = st.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
        iou_th = st.slider("NMS IoU Threshold", min_value=0.05, max_value=1.0, value=0.45, step=0.05)
        inf_res = st.selectbox("Inference Resolution", [320, 416, 512, 640, 768, 960, 1280], index=3)

        inf_dev = "0" if CUDA_AVAILABLE else "cpu"
        dev_choice = st.selectbox("Compute Hardware", ["Auto", "GPU (0)", "CPU"] if CUDA_AVAILABLE else ["CPU"])
        if dev_choice == "CPU":
            inf_dev = "cpu"
        elif dev_choice == "GPU (0)":
            inf_dev = "0"

        with st.expander("Visualization Settings"):
            s_boxes = st.checkbox("Show Bounding Boxes", value=True)
            s_labels = st.checkbox("Show Labels", value=True)
            s_conf = st.checkbox("Show Confidence", value=True)
            l_width = st.slider("Line Width", min_value=1, max_value=10, value=2)

    with inf_c2:
        st.markdown("##### 2. Input Mode & Detection Results")
        mode = st.radio("Input Source", ["🖼️ Single/Multi Image", "📂 Validation Dataset Sample", "📷 Live Webcam", "🎥 Video File"], horizontal=True)

        if not chosen_model_path or not Path(chosen_model_path).exists():
            st.warning("Please select or upload a valid `.pt` model checkpoint.")
        else:
            try:
                m_obj = load_yolo_cached(chosen_model_path)
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.82rem;">
                    🧠 <b>Model:</b> <code>{Path(chosen_model_path).name}</code> • <b>Task:</b> <code>{m_obj.task}</code> • <b>Classes:</b> {len(m_obj.names)}
                </div>
                """, unsafe_allow_html=True)

                if mode == "🖼️ Single/Multi Image":
                    uploaded_images = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png", "webp", "bmp"], accept_multiple_files=True)
                    if uploaded_images:
                        for u_img in uploaded_images:
                            p_img = Image.open(u_img).convert("RGB")
                            t0 = time.time()
                            res = m_obj.predict(
                                source=p_img,
                                conf=conf_th,
                                iou=iou_th,
                                imgsz=inf_res,
                                device=inf_dev,
                                boxes=s_boxes,
                                show_labels=s_labels,
                                show_conf=s_conf,
                                line_width=l_width,
                                verbose=False
                            )[0]
                            t_inf = (time.time() - t0) * 1000

                            res_rgb = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)

                            c_orig, c_pred = st.columns(2)
                            c_orig.image(p_img, caption=f"Original: {u_img.name}", width="stretch")
                            c_pred.image(res_rgb, caption=f"Detections ({t_inf:.1f} ms)", width="stretch")

                            # Table
                            dets = []
                            if res.boxes is not None and len(res.boxes) > 0:
                                for b in res.boxes:
                                    cid = int(b.cls[0].item())
                                    cn = m_obj.names.get(cid, str(cid))
                                    cf = float(b.conf[0].item())
                                    xy = [round(float(c), 1) for c in b.xyxy[0].tolist()]
                                    dets.append({"Class": cn, "Confidence": f"{cf*100:.1f}%", "Coordinates [x1, y1, x2, y2]": str(xy)})

                            if dets:
                                st.dataframe(pd.DataFrame(dets), width="stretch")

                            out_save = TEMP_DIR / f"det_{u_img.name}"
                            Image.fromarray(res_rgb).save(out_save)
                            with open(out_save, "rb") as f:
                                st.download_button(f"📥 Download {u_img.name} Result", f, file_name=f"detected_{u_img.name}", width="stretch")

                elif mode == "📂 Validation Dataset Sample":
                    val_imgs = []
                    if ds_info and "val" in ds_info["splits"]:
                        val_imgs = sorted([p for p in ds_info["splits"]["val"].rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])

                    if not val_imgs:
                        st.info("No validation split found in active dataset.")
                    else:
                        sel_v_img = st.selectbox("Pick Validation Image", val_imgs, format_func=lambda p: p.name)
                        if sel_v_img:
                            p_img = Image.open(sel_v_img).convert("RGB")
                            t0 = time.time()
                            res = m_obj.predict(
                                source=p_img,
                                conf=conf_th,
                                iou=iou_th,
                                imgsz=inf_res,
                                device=inf_dev,
                                boxes=s_boxes,
                                show_labels=s_labels,
                                show_conf=s_conf,
                                line_width=l_width,
                                verbose=False
                            )[0]
                            t_inf = (time.time() - t0) * 1000
                            res_rgb = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)

                            c_orig, c_pred = st.columns(2)
                            c_orig.image(p_img, caption=f"Validation Image: {sel_v_img.name}", width="stretch")
                            c_pred.image(res_rgb, caption=f"Detections ({t_inf:.1f} ms)", width="stretch")

                elif mode == "📷 Live Webcam":
                    cam_snap = st.camera_input("Capture Webcam Frame")
                    if cam_snap:
                        p_img = Image.open(cam_snap).convert("RGB")
                        t0 = time.time()
                        res = m_obj.predict(
                            source=p_img,
                            conf=conf_th,
                            iou=iou_th,
                            imgsz=inf_res,
                            device=inf_dev,
                            boxes=s_boxes,
                            show_labels=s_labels,
                            show_conf=s_conf,
                            line_width=l_width,
                            verbose=False
                        )[0]
                        t_inf = (time.time() - t0) * 1000
                        res_rgb = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
                        st.image(res_rgb, caption=f"Live Detection ({t_inf:.1f} ms)", width="stretch")

                elif mode == "🎥 Video File":
                    up_vid = st.file_uploader("Upload Video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
                    if up_vid:
                        v_path = TEMP_DIR / up_vid.name
                        with open(v_path, "wb") as f:
                            f.write(up_vid.getbuffer())

                        if st.button("▶ Run Video Inference", type="primary", width="stretch"):
                            out_v_path = TEMP_DIR / f"annotated_{up_vid.name}.mp4"
                            cap = cv2.VideoCapture(str(v_path))
                            w_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps_v = cap.get(cv2.CAP_PROP_FPS) or 25.0
                            tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            vw = cv2.VideoWriter(str(out_v_path), fourcc, fps_v, (w_v, h_v))

                            pbar = st.progress(0, text="Processing video frames...")
                            f_idx = 0
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                res_f = m_obj.predict(source=frame_rgb, conf=conf_th, iou=iou_th, imgsz=inf_res, device=inf_dev, verbose=False)[0]
                                vw.write(res_f.plot())
                                f_idx += 1
                                if f_idx % 5 == 0 or f_idx == tot_frames:
                                    pbar.progress(min(f_idx / tot_frames, 1.0), text=f"Processed frame {f_idx}/{tot_frames}")

                            cap.release()
                            vw.release()
                            pbar.empty()
                            st.success("✅ Video processing complete!")
                            with open(out_v_path, "rb") as f:
                                st.download_button("📥 Download Annotated Video", f, file_name=f"annotated_{up_vid.name}", width="stretch")

            except Exception as e:
                st.error(f"Inference execution error: {e}")


# ===========================================================================
# TAB 4: 📦 MODEL EXPORT STUDIO
# ===========================================================================
with tab_export:
    st.subheader("Model Export & Production Deployment Studio")
    st.caption("Convert your trained weights into high-performance edge & cloud deployment runtimes.")

    exp_col1, exp_col2 = st.columns([1, 1.5])

    with exp_col1:
        st.markdown("##### 1. Export Configuration")
        exp_models = get_available_models()
        sel_exp_model = st.selectbox("Model Checkpoint", exp_models, key="studio_exp_model")

        format_cards = {
            "onnx": "ONNX (Cross-platform, Triton, OpenCV DNN)",
            "engine": "TensorRT (NVIDIA GPU Maximum Throughput)",
            "openvino": "OpenVINO (Intel CPU & iGPU Optimized)",
            "torchscript": "TorchScript (C++ PyTorch Runtime)",
            "coreml": "CoreML (Apple iOS / macOS Devices)",
            "tflite": "TFLite (Mobile, Android & Edge TPUs)",
        }
        exp_fmt = st.selectbox("Target Runtime Format", list(format_cards.keys()), format_func=lambda k: format_cards[k])

        exp_sz = st.selectbox("Input Resolution", [320, 416, 512, 640, 768, 960, 1280], index=3, key="exp_res_sel")
        exp_fp16 = st.checkbox("FP16 Half-Precision (2x speed on GPU)", value=False)
        exp_dyn = st.checkbox("Dynamic Axes (Variable Batch & Size)", value=False)
        exp_simp = st.checkbox("Simplify ONNX Graph", value=True)
        exp_i8 = st.checkbox("INT8 Quantization", value=False)

        exp_dev = "0" if (CUDA_AVAILABLE and exp_fmt == "engine") else "cpu"

    with exp_col2:
        st.markdown("##### 2. Export Pipeline & Verification")
        st.info(f"Target: **{exp_fmt.upper()}** • Checkpoint: `{Path(sel_exp_model).name}` • Size: `{exp_sz}px`")

        if st.button("🚀 Compile & Export Model", type="primary", width="stretch"):
            if not sel_exp_model or not Path(sel_exp_model).exists():
                st.error("Invalid model selected.")
            else:
                with st.spinner(f"Compiling model to {exp_fmt.upper()}..."):
                    try:
                        em = YOLO(sel_exp_model)
                        exp_out = em.export(
                            format=exp_fmt,
                            imgsz=exp_sz,
                            half=exp_fp16,
                            dynamic=exp_dyn,
                            simplify=exp_simp,
                            int8=exp_i8,
                            device=exp_dev
                        )
                        st.success(f"🎉 Successfully exported: `{exp_out}`")
                        ep = Path(exp_out)
                        if ep.is_file():
                            sz_mb = ep.stat().st_size / (1024 * 1024)
                            st.caption(f"📦 Binary Size: **{sz_mb:.2f} MB**")
                            with open(ep, "rb") as f:
                                st.download_button(f"📥 Download {ep.name}", f, file_name=ep.name, width="stretch")
                        elif ep.is_dir():
                            zip_dst = EXPORTS_DIR / f"{ep.name}.zip"
                            shutil.make_archive(str(zip_dst.with_suffix("")), "zip", str(ep))
                            sz_mb = zip_dst.stat().st_size / (1024 * 1024)
                            st.caption(f"📦 Packaged Zip Size: **{sz_mb:.2f} MB**")
                            with open(zip_dst, "rb") as f:
                                st.download_button(f"📥 Download {zip_dst.name}", f, file_name=zip_dst.name, width="stretch")
                    except Exception as e:
                        st.error(f"Export error: {e}")


# ===========================================================================
# TAB 5: 🎛️ HYPERPARAMETER TUNING
# ===========================================================================
with tab_tune:
    st.subheader("Automated Hyperparameter Optimization Studio")
    st.caption("Search for optimal learning rates, momentum, loss multipliers, and augmentation settings using Optuna & Genetic Algorithms.")

    tc1, tc2 = st.columns([1, 1.5])

    with tc1:
        st.markdown("##### 1. Tuning Search Settings")
        t_models = get_available_models()
        t_base = st.selectbox("Base Checkpoint", t_models, index=0, key="t_base_sel")
        t_trials = st.number_input("Search Iterations / Trials", min_value=2, max_value=300, value=15)
        t_epochs = st.number_input("Epochs per Trial", min_value=1, max_value=100, value=10)
        t_opt = st.selectbox("Optimizer", ["auto", "AdamW", "SGD"], index=0, key="t_opt_sel")
        t_gpu = "0" if CUDA_AVAILABLE else "cpu"
        t_name = st.text_input("Tuning Experiment Name", value=st.session_state.tune_run_name)

        t_btn1, t_btn2 = st.columns(2)
        t_start_dis = data_yaml_path is None or st.session_state.tune_active

        with t_btn1:
            if st.button("🚀 Start Tuning", type="primary", disabled=t_start_dis, width="stretch"):
                cmd = [
                    "yolo", "tune",
                    f"model={t_base}",
                    f"data={data_yaml_path}",
                    f"epochs={t_epochs}",
                    f"iterations={t_trials}",
                    f"optimizer={t_opt}",
                    f"device={t_gpu}",
                    f"project={TUNE_DIR}",
                    f"name={t_name}",
                ]
                st.session_state.tune_logs = []
                popen_kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                if os.name != "nt":
                    popen_kwargs["preexec_fn"] = os.setsid
                proc = subprocess.Popen(cmd, **popen_kwargs)
                prog_re = re.compile(r"\d+%\|")
                threading.Thread(
                    target=_reader_thread, args=(proc, st.session_state.tune_logs, prog_re), daemon=True
                ).start()
                st.session_state.tune_proc = proc
                st.session_state.tune_active = True
                st.session_state.tune_run_name = t_name
                st.rerun()

        with t_btn2:
            if st.button("🛑 Stop Tuning", disabled=not st.session_state.tune_active, width="stretch"):
                if os.name == "nt":
                    st.session_state.tune_proc.terminate()
                else:
                    _signal_group(st.session_state.tune_proc, signal.SIGTERM)
                st.session_state.tune_active = False
                st.rerun()

    with tc2:
        st.markdown("##### 2. Live Tuning Log & Optimal Parameters")
        st.code("".join(st.session_state.tune_logs[-30:]) if st.session_state.tune_logs else "Ready for hyperparameter tuning. Click 'Start Tuning'.", language="bash")

        best_yaml = TUNE_DIR / st.session_state.tune_run_name / "best_hyperparameters.yaml"
        if not best_yaml.exists():
            for yf in TUNE_DIR.rglob("best_hyperparameters.yaml"):
                best_yaml = yf
                break

        if best_yaml.exists():
            try:
                with open(best_yaml, "r") as f:
                    b_hyp = yaml.safe_load(f)
                st.success(f"🏆 **Optimal Hyperparameters Discovered** (`{best_yaml.parent.name}`)")
                st.json(b_hyp)
                if st.button("✨ Save & Apply to Training Settings", type="primary", width="stretch"):
                    st.session_state.tuned_params = b_hyp
                    st.toast("✅ Tuned hyperparameters transferred to Training Dashboard!", icon="🎯")
            except Exception:
                pass

    if st.session_state.tune_active:
        proc = st.session_state.tune_proc
        if proc and proc.poll() is not None:
            st.session_state.tune_active = False
            st.toast("🎉 Hyperparameter optimization finished!", icon="✅")
            st.rerun()
        else:
            time.sleep(1)
            st.rerun()


# ===========================================================================
# TAB 6: 📊 EXPERIMENT HISTORY & ANALYTICS
# ===========================================================================
with tab_history:
    st.subheader("Training Run History & Cross-Experiment Comparison")

    run_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
    if not run_dirs:
        st.info("No training runs found in `yolo_workspace/runs/`.")
    else:
        history_records = []
        for rd in sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True):
            dfr, _ = load_run_results(rd.name)
            ep_done = len(dfr) if dfr is not None else 0
            m50 = dfr["metrics/mAP50(B)"].max() if dfr is not None and "metrics/mAP50(B)" in dfr.columns else 0.0
            m95 = dfr["metrics/mAP50-95(B)"].max() if dfr is not None and "metrics/mAP50-95(B)" in dfr.columns else 0.0
            has_w = (rd / "weights" / "best.pt").exists()

            history_records.append({
                "Run Name": rd.name,
                "Epochs": ep_done,
                "Best mAP@50": f"{m50:.4f}" if m50 > 0 else "N/A",
                "Best mAP@50-95": f"{m95:.4f}" if m95 > 0 else "N/A",
                "Weights Ready": "✅" if has_w else "❌",
            })

        st.dataframe(pd.DataFrame(history_records), width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("##### 🔍 Multi-Run Curve Overlay")
        selected_for_comp = st.multiselect(
            "Select Experiments to Overlay",
            [r["Run Name"] for r in history_records],
            default=[r["Run Name"] for r in history_records[:min(3, len(history_records))]]
        )

        if selected_for_comp:
            comp_m = st.selectbox("Metric to Compare", ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "train/box_loss", "val/box_loss"])
            comp_df = pd.DataFrame()
            for rn in selected_for_comp:
                dft, _ = load_run_results(rn)
                if dft is not None and comp_m in dft.columns and "epoch" in dft.columns:
                    comp_df[rn] = dft.set_index("epoch")[comp_m]

            if not comp_df.empty:
                st.line_chart(comp_df, height=350)
            else:
                st.caption(f"No metric `{comp_m}` found in selected runs.")