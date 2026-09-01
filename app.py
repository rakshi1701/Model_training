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

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except Exception:
    CUDA_AVAILABLE = False
    GPU_COUNT = 0

from ultralytics import YOLO

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
def discover_all_datasets():
    """Finds all available datasets in workspace and dataset directory."""
    datasets = {}
    if DATASET_DIR.exists():
        for y_path in list(DATASET_DIR.rglob("*.yaml")) + list(DATASET_DIR.rglob("*.yml")):
            datasets[f"Extracted Dataset ({y_path.parent.name})"] = y_path.resolve()
    for root_dir in Path(".").glob("*/"):
        if root_dir.is_dir() and root_dir.name != "yolo_workspace" and not root_dir.name.startswith("."):
            for y_path in list(root_dir.glob("*.yaml")) + list(root_dir.glob("*.yml")):
                datasets[f"Workspace Folder: {root_dir.name} ({y_path.name})"] = y_path.resolve()
    return datasets


def get_dataset_info(yaml_path: Path):
    """Parses dataset yaml and returns statistics."""
    if not yaml_path or not yaml_path.exists():
        return None
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
                p = (base / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
                if p.exists():
                    n_imgs = sum(1 for img in p.rglob("*") if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"])
                    counts[split] = n_imgs
                    split_paths[split] = p

        return {
            "config": cfg,
            "classes": class_list,
            "counts": counts,
            "splits": split_paths,
            "yaml_path": yaml_path
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

selected_ds_source = st.sidebar.selectbox("Dataset Source", dataset_options, index=0 if dataset_options else 0)

data_yaml_path = None
if selected_ds_source == "➕ Upload New Dataset (.zip)":
    uploaded_zip = st.sidebar.file_uploader("Upload .zip (up to 50GB)", type=["zip"])
    if uploaded_zip and st.sidebar.button("📦 Extract & Activate", width="stretch"):
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DATASET_DIR / "upload.zip"
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATASET_DIR)
        zip_path.unlink()

        yamls = list(DATASET_DIR.rglob("*.yaml")) + list(DATASET_DIR.rglob("*.yml"))
        if yamls:
            st.sidebar.success("Dataset extracted and activated!")
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
        <div style="font-weight: 700; color: #38bdf8;">📄 {ds_info['yaml_path'].parent.name}</div>
        <div style="color: #cbd5e1; margin-top: 4px;">🖼️ <b>Images:</b> {counts_str or 'None detected'}</div>
        <div style="color: #cbd5e1;">🏷️ <b>Classes ({len(ds_info['classes'])}):</b> {classes_str or 'None'}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.info("Select or upload a dataset to begin training.")

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
tab_train, tab_ds, tab_infer, tab_export, tab_tune, tab_history = st.tabs([
    "🏋️ Training & Real-Time Metrics",
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
# TAB 2: 📂 DATASET HUB & VISUAL INSPECTOR
# ===========================================================================
with tab_ds:
    st.subheader("Dataset Hub & Visual Ground-Truth Inspector")
    st.caption("Inspect class balances, explore images, and verify bounding box annotations directly from your dataset splits.")

    if not ds_info:
        st.warning("No active dataset selected. Please choose or upload a dataset in the sidebar.")
    else:
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