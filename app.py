"""
YOLO Model Training, Evaluation, Inference & Deployment Studio
Run with: streamlit run app.py
Requires: pip install streamlit ultralytics pyyaml pandas Pillow opencv-python-headless onnx onnxruntime
"""

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
# Page Configuration & Directories
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLO Studio - Train, Test & Deploy",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

WORKDIR = Path("yolo_workspace")
DATASET_DIR = WORKDIR / "dataset"
RUNS_DIR = WORKDIR / "runs"
TUNE_DIR = WORKDIR / "tune"
EXPORTS_DIR = WORKDIR / "exports"
TEMP_DIR = WORKDIR / "temp"

for d in [WORKDIR, RUNS_DIR, TUNE_DIR, EXPORTS_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper Functions: Dataset, Models & Results
# ---------------------------------------------------------------------------
def get_available_models():
    """Scans for all available trained and base .pt weights."""
    models = []
    # Base models in workspace root
    for pt in Path(".").glob("*.pt"):
        if pt.is_file():
            models.append(str(pt.name))
    # Trained run weights
    for pt in RUNS_DIR.rglob("*.pt"):
        if pt.is_file():
            rel = pt.relative_to(WORKDIR)
            models.append(f"{WORKDIR}/{rel}")
    # Standard common presets if not already present
    defaults = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    for d in defaults:
        if d not in models:
            models.append(d)
    return sorted(list(set(models)))


def find_split_dir(root: Path, names):
    """Finds image split directories."""
    for d in root.rglob("images"):
        if d.is_dir() and d.parent.name.lower() in names:
            return d.resolve()
    # Fallback to direct folder matching
    for d in root.rglob("*"):
        if d.is_dir() and d.name.lower() in names:
            img_sub = d / "images"
            if img_sub.is_dir():
                return img_sub.resolve()
            return d.resolve()
    return None


def load_run_results(run_name: str):
    """Reads results.csv for a given run and cleans column headers."""
    target_dirs = [
        RUNS_DIR / run_name,
        RUNS_DIR / "detect" / run_name,
        RUNS_DIR / "segment" / run_name,
        RUNS_DIR / "classify" / run_name,
        RUNS_DIR / "pose" / run_name,
        RUNS_DIR / "obb" / run_name,
    ]
    csv_path = None
    actual_dir = None
    for d in target_dirs:
        p = d / "results.csv"
        if p.exists():
            csv_path = p
            actual_dir = d
            break

    if not csv_path:
        # Recursive search in runs dir
        for p in RUNS_DIR.rglob("results.csv"):
            if run_name in str(p.parent):
                csv_path = p
                actual_dir = p.parent
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
    """Loads and caches YOLO model for inference."""
    return YOLO(model_path)


# ---------------------------------------------------------------------------
# Background Process Manager (Training / Tuning)
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


# Initialize Session State
st.session_state.setdefault("train_proc", None)
st.session_state.setdefault("train_logs", [])
st.session_state.setdefault("train_active", False)
st.session_state.setdefault("train_paused", False)
st.session_state.setdefault("train_run_name", "exp1")

st.session_state.setdefault("tune_proc", None)
st.session_state.setdefault("tune_logs", [])
st.session_state.setdefault("tune_active", False)
st.session_state.setdefault("tune_paused", False)
st.session_state.setdefault("tune_run_name", "tune1")

st.session_state.setdefault("tuned_params", None)

# ---------------------------------------------------------------------------
# Sidebar: Dataset Management & Global Hardware Status
# ---------------------------------------------------------------------------
st.sidebar.title("🛠️ YOLO Control Panel")

# Hardware indicator
if CUDA_AVAILABLE:
    gpu_names = [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)]
    st.sidebar.success(f"⚡ GPU: {GPU_COUNT} CUDA Device(s)\n\n" + "\n".join([f"- GPU {i}: {name}" for i, name in enumerate(gpu_names)]))
else:
    st.sidebar.warning("⚠️ No CUDA GPU detected (CPU mode).")

st.sidebar.markdown("---")
st.sidebar.header("📁 1. Dataset Management")
st.sidebar.caption("Upload a `.zip` containing YOLO format dataset (`images/`, `labels/`, `data.yaml`). Max upload: **50GB**.")

data_yaml_path = None
uploaded_zip = st.sidebar.file_uploader("Upload Dataset (.zip)", type=["zip"], key="dataset_zip_uploader")

if uploaded_zip and st.sidebar.button("📦 Extract Dataset", use_container_width=True):
    with st.sidebar.status("Extracting and validating dataset...", expanded=True) as status:
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
            yaml_path = yamls[0]
            try:
                with open(yaml_path, "r") as f:
                    data_cfg = yaml.safe_load(f) or {}

                train_dir = find_split_dir(DATASET_DIR, {"train"})
                val_dir = find_split_dir(DATASET_DIR, {"val", "valid", "validation"})
                test_dir = find_split_dir(DATASET_DIR, {"test"})

                found = []
                if train_dir:
                    data_cfg["train"] = str(train_dir)
                    found.append("train")
                if val_dir:
                    data_cfg["val"] = str(val_dir)
                    found.append("val")
                if test_dir:
                    data_cfg["test"] = str(test_dir)
                    found.append("test")
                data_cfg.pop("path", None)

                with open(yaml_path, "w") as f:
                    yaml.safe_dump(data_cfg, f, sort_keys=False)

                status.update(label=f"Extracted & patched: {', '.join(found)}", state="complete")
            except Exception as e:
                status.update(label=f"Extracted, yaml warning: {e}", state="error")
        else:
            status.update(label="Extracted (no data.yaml found)", state="complete")

# Resolve current dataset
if DATASET_DIR.exists():
    yamls = list(DATASET_DIR.rglob("*.yaml")) + list(DATASET_DIR.rglob("*.yml"))
    if yamls:
        data_yaml_path = yamls[0]
        st.sidebar.info(f"📄 **Active Config**: `{data_yaml_path.name}`")
        try:
            with open(data_yaml_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            names = cfg.get("names", {})
            if isinstance(names, dict):
                class_list = list(names.values())
            elif isinstance(names, list):
                class_list = names
            else:
                class_list = []

            # Count images
            counts = {}
            for split in ["train", "val", "test"]:
                split_val = cfg.get(split)
                if split_val and Path(split_val).exists():
                    p = Path(split_val)
                    n_imgs = sum(1 for img in p.rglob("*") if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
                    counts[split] = n_imgs

            summary_str = " · ".join([f"{k}: {v}" for k, v in counts.items()])
            if summary_str:
                st.sidebar.caption(f"📊 **Images**: {summary_str}")
            if class_list:
                st.sidebar.caption(f"🏷️ **Classes ({len(class_list)})**: {', '.join([str(c) for c in class_list[:8]])}{'...' if len(class_list) > 8 else ''}")
        except Exception:
            pass
    elif uploaded_zip:
        st.sidebar.warning("⚠️ No data.yaml found inside extracted dataset.")

# Check for standalone dataset folders like cctv-1
if data_yaml_path is None:
    root_yamls = list(Path(".").glob("*/data.yaml"))
    if root_yamls:
        data_yaml_path = root_yamls[0].resolve()
        st.sidebar.info(f"📄 **Found Dataset**: `{data_yaml_path.parent.name}/data.yaml`")

st.sidebar.markdown("---")
st.sidebar.caption("🚀 **YOLO Dashboard v2.0** • Ultralytics Studio")

# ---------------------------------------------------------------------------
# Main App Header & Navigation Tabs
# ---------------------------------------------------------------------------
st.title("🎯 YOLO Vision Studio")
st.caption("Complete Suite for Training, Real-time Monitoring, Inference Testing, Model Export & Hyperparameter Tuning")

tab_train, tab_infer, tab_export, tab_tune, tab_history = st.tabs([
    "🏋️ Training & Live Metrics",
    "🧪 Inference Playground",
    "📦 Model Export Studio",
    "🎛️ Hyperparameter Tuning",
    "📊 Run History & Comparisons",
])


# ===========================================================================
# TAB 1: 🏋️ TRAINING & LIVE METRIC CHARTS
# ===========================================================================
with tab_train:
    st.subheader("Model Training & Real-Time Monitoring")

    # Hyperparameter Config Row
    with st.expander("⚙️ Training Hyperparameters & Model Settings", expanded=not st.session_state.train_active):
        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)

        with cfg_col1:
            st.markdown("##### 1. Architecture")
            model_family = st.selectbox("Family", ["YOLO11", "YOLOv8", "YOLOv9", "YOLOv10"], index=0)
            size_map = {
                "YOLO11": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
                "YOLOv8": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                "YOLOv9": ["yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e"],
                "YOLOv10": ["yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"],
            }
            model_size = st.selectbox("Size", size_map[model_family], index=0)
            task = st.selectbox("Task", ["detect", "segment", "classify", "pose", "obb"], index=0)
            task_suffix = {"detect": "", "segment": "-seg", "classify": "-cls", "pose": "-pose", "obb": "-obb"}[task]
            pretrained = st.checkbox("Pretrained Weights", value=True)
            model_name = f"{model_size}{task_suffix}.pt" if pretrained else f"{model_size}{task_suffix}.yaml"

        with cfg_col2:
            st.markdown("##### 2. Optimization")
            epochs = st.number_input("Epochs", min_value=1, max_value=5000, value=100)
            batch = st.selectbox("Batch Size", [-1, 2, 4, 8, 16, 32, 64, 128], index=4, help="-1 auto-tunes batch based on GPU memory")
            imgsz = st.selectbox("Image Size (px)", [320, 416, 512, 640, 768, 960, 1280], index=3)
            optimizer = st.selectbox("Optimizer", ["auto", "AdamW", "SGD", "Adam", "NAdam", "RMSProp"], index=0)

        with cfg_col3:
            st.markdown("##### 3. Learning Rate & Control")
            default_lr = 0.01
            if st.session_state.tuned_params and "lr0" in st.session_state.tuned_params:
                default_lr = float(st.session_state.tuned_params["lr0"])
            lr0 = st.number_input("Initial LR (lr0)", min_value=0.00001, max_value=1.0, value=default_lr, format="%.5f")
            patience = st.number_input("Early Stopping Patience", min_value=0, max_value=500, value=50)
            cos_lr = st.checkbox("Cosine LR Schedule", value=False)
            cache = st.selectbox("Cache Dataset", ["False", "ram", "disk"], index=0)

        with cfg_col4:
            st.markdown("##### 4. Hardware & Run Name")
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

            device_sel = st.selectbox("Compute Device", device_opts, index=0)
            device = device_lbls[device_sel]
            workers = st.number_input("Workers", min_value=0, max_value=32, value=8)
            run_name = st.text_input("Experiment Name", value=st.session_state.train_run_name)
            weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=0.0005, format="%.5f")
            freeze = st.number_input("Freeze Layers (0=None)", min_value=0, max_value=50, value=0)
            resume = st.checkbox("Resume Checkpoint", value=False)

        if st.session_state.tuned_params:
            st.info(f"💡 Tuned Hyperparameters loaded from search: `{st.session_state.tuned_params}`")

    # Action Buttons Bar
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([1.5, 1, 1, 1, 2])
    start_disabled = data_yaml_path is None or st.session_state.train_active

    with btn_col1:
        if st.button("🚀 Start Training", type="primary", disabled=start_disabled, use_container_width=True):
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
        if st.button("⏸ Pause", disabled=not st.session_state.train_active or st.session_state.train_paused, use_container_width=True):
            if _signal_group(st.session_state.train_proc, signal.SIGSTOP):
                st.session_state.train_paused = True
            st.rerun()

    with btn_col3:
        if st.button("▶ Resume", disabled=not st.session_state.train_active or not st.session_state.train_paused, use_container_width=True):
            if _signal_group(st.session_state.train_proc, signal.SIGCONT):
                st.session_state.train_paused = False
            st.rerun()

    with btn_col4:
        if st.button("🛑 Terminate", disabled=not st.session_state.train_active, use_container_width=True):
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
                st.warning("⏸ Training Paused")
            else:
                st.success(f"🟢 Training In Progress: `{st.session_state.train_run_name}`")
        elif data_yaml_path is None:
            st.info("👈 Upload and extract a dataset to start training.")

    st.markdown("---")

    # Real-Time Metrics & Charts Section
    current_run = st.session_state.train_run_name
    df_results, actual_run_dir = load_run_results(current_run)

    st.markdown(f"#### 📈 Live Metrics Dashboard `{current_run}`")

    if df_results is not None and not df_results.empty:
        last_row = df_results.iloc[-1]
        cur_epoch = int(last_row.get("epoch", len(df_results)))

        # Metric KPI cards
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

        # Helper to get delta
        def get_val_delta(col_name):
            if col_name in df_results.columns:
                val = float(last_row[col_name])
                delta = None
                if len(df_results) > 1:
                    delta = val - float(df_results.iloc[-2][col_name])
                return val, delta
            return None, None

        box_loss_val, box_delta = get_val_delta("val/box_loss")
        if box_loss_val is None:
            box_loss_val, box_delta = get_val_delta("train/box_loss")

        cls_loss_val, cls_delta = get_val_delta("val/cls_loss")
        if cls_loss_val is None:
            cls_loss_val, cls_delta = get_val_delta("train/cls_loss")

        map50_val, map50_delta = get_val_delta("metrics/mAP50(B)")
        map50_95_val, map50_95_delta = get_val_delta("metrics/mAP50-95(B)")
        prec_val, _ = get_val_delta("metrics/precision(B)")
        rec_val, _ = get_val_delta("metrics/recall(B)")

        kpi1.metric("Epoch", f"{cur_epoch}/{epochs}")
        kpi2.metric("Box Loss", f"{box_loss_val:.4f}" if box_loss_val is not None else "N/A", delta=f"{box_delta:.4f}" if box_delta is not None else None, delta_color="inverse")
        kpi3.metric("Class Loss", f"{cls_loss_val:.4f}" if cls_loss_val is not None else "N/A", delta=f"{cls_delta:.4f}" if cls_delta is not None else None, delta_color="inverse")
        kpi4.metric("mAP@50 (B)", f"{map50_val:.4f}" if map50_val is not None else "N/A", delta=f"{map50_delta:+.4f}" if map50_delta is not None else None)
        kpi5.metric("mAP@50-95 (B)", f"{map50_95_val:.4f}" if map50_95_val is not None else "N/A", delta=f"{map50_95_delta:+.4f}" if map50_95_delta is not None else None)
        kpi6.metric("Precision / Recall", f"{(prec_val or 0):.2f} / {(rec_val or 0):.2f}")

        # Real-time Interactive Line Charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### 📉 Training & Validation Losses")
            loss_cols = [c for c in df_results.columns if "loss" in c.lower()]
            if loss_cols:
                plot_df = df_results.set_index("epoch")[loss_cols] if "epoch" in df_results.columns else df_results[loss_cols]
                st.line_chart(plot_df, height=280)

        with chart_col2:
            st.markdown("##### 🎯 Accuracy & mAP Curves")
            metric_cols = [c for c in df_results.columns if any(m in c.lower() for m in ["map", "precision", "recall", "accuracy"])]
            if metric_cols:
                plot_df = df_results.set_index("epoch")[metric_cols] if "epoch" in df_results.columns else df_results[metric_cols]
                st.line_chart(plot_df, height=280)
    else:
        st.info("Metrics charts will render live as soon as the first epoch finishes writing to `results.csv`.")

    # Live Console & Finished Artifacts
    col_log, col_artifacts = st.columns([1.2, 0.8])

    with col_log:
        st.markdown("##### 🖥️ Live Training Console")
        log_view = st.empty()
        log_view.code("".join(st.session_state.train_logs[-35:]) if st.session_state.train_logs else "Ready to train. Click 'Start Training' above.", language="bash")

    with col_artifacts:
        st.markdown("##### 🏆 Artifacts & Checkpoints")
        if actual_run_dir and actual_run_dir.exists():
            best_pt = actual_run_dir / "weights" / "best.pt"
            last_pt = actual_run_dir / "weights" / "last.pt"

            if best_pt.exists():
                st.success(f"✅ `best.pt` found ({best_pt.stat().st_size / (1024*1024):.1f} MB)")
                with open(best_pt, "rb") as f:
                    st.download_button("📥 Download best.pt", f, file_name=f"{current_run}_best.pt", key="dl_best_btn", use_container_width=True)

            if last_pt.exists():
                with open(last_pt, "rb") as f:
                    st.download_button("📥 Download last.pt", f, file_name=f"{current_run}_last.pt", key="dl_last_btn", use_container_width=True)

            results_png = actual_run_dir / "results.png"
            if results_png.exists():
                st.image(str(results_png), caption="Training Results Plot", use_container_width=True)
        else:
            st.caption("Artifacts and plots will be available here once training begins.")

    # Polling & Reactive loop for training
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
# TAB 2: 🧪 INFERENCE & TESTING PLAYGROUND
# ===========================================================================
with tab_infer:
    st.subheader("Inference & Model Testing Playground")
    st.caption("Run instant detections using your trained weights or base YOLO models on images, validation sets, webcam snapshots, or videos.")

    infer_col1, infer_col2 = st.columns([1, 2.2])

    with infer_col1:
        st.markdown("##### 1. Model & Parameters")
        available_models = get_available_models()
        model_choice_type = st.radio("Model Source", ["Discovered Models", "Upload Custom .pt File"], horizontal=True)

        selected_model_path = None
        if model_choice_type == "Discovered Models":
            selected_model_path = st.selectbox("Select Model Weights", available_models, index=0)
        else:
            uploaded_model = st.file_uploader("Upload .pt Weights", type=["pt"], key="infer_pt_uploader")
            if uploaded_model:
                custom_pt_path = TEMP_DIR / uploaded_model.name
                with open(custom_pt_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                selected_model_path = str(custom_pt_path)

        conf_thresh = st.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
        iou_thresh = st.slider("NMS IoU Threshold", min_value=0.05, max_value=1.0, value=0.45, step=0.05)
        infer_imgsz = st.selectbox("Inference Resolution", [320, 416, 512, 640, 768, 960, 1280], index=3)

        infer_device = "0" if CUDA_AVAILABLE else "cpu"
        device_infer_choice = st.selectbox("Inference Device", ["Auto", "GPU (0)", "CPU"] if CUDA_AVAILABLE else ["CPU"])
        if device_infer_choice == "CPU":
            infer_device = "cpu"
        elif device_infer_choice == "GPU (0)":
            infer_device = "0"

        with st.expander("Display Options"):
            show_boxes = st.checkbox("Show Bounding Boxes", value=True)
            show_labels = st.checkbox("Show Labels", value=True)
            show_conf = st.checkbox("Show Confidence Scores", value=True)
            line_width = st.slider("Line Width", min_value=1, max_value=10, value=2)

    with infer_col2:
        st.markdown("##### 2. Input Source & Live Results")
        input_type = st.radio("Select Input Mode", ["🖼️ Upload Image", "📂 Validation Dataset Sample", "📷 Live Webcam Snapshot", "🎥 Upload Video"], horizontal=True)

        if not selected_model_path or not Path(selected_model_path).exists():
            st.warning("Please select or upload a valid `.pt` model checkpoint.")
        else:
            try:
                model_obj = load_yolo_cached(selected_model_path)
                st.caption(f"🧠 Loaded: `{Path(selected_model_path).name}` | Task: `{model_obj.task}` | Classes: `{len(model_obj.names)}`")

                # Mode 1: Image Upload
                if input_type == "🖼️ Upload Image":
                    uploaded_img = st.file_uploader("Upload Image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp", "bmp"])
                    if uploaded_img:
                        pil_img = Image.open(uploaded_img).convert("RGB")
                        t0 = time.time()
                        results = model_obj.predict(
                            source=pil_img,
                            conf=conf_thresh,
                            iou=iou_thresh,
                            imgsz=infer_imgsz,
                            device=infer_device,
                            boxes=show_boxes,
                            show_labels=show_labels,
                            show_conf=show_conf,
                            line_width=line_width,
                            verbose=False
                        )
                        t_infer = (time.time() - t0) * 1000

                        res = results[0]
                        res_bgr = res.plot()
                        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)

                        img_c1, img_c2 = st.columns(2)
                        img_c1.image(pil_img, caption="Original Image", use_container_width=True)
                        img_c2.image(res_rgb, caption=f"Detections ({t_infer:.1f} ms)", use_container_width=True)

                        # Detections table
                        detections = []
                        if res.boxes is not None and len(res.boxes) > 0:
                            for box in res.boxes:
                                cls_id = int(box.cls[0].item())
                                cname = model_obj.names.get(cls_id, str(cls_id))
                                conf_val = float(box.conf[0].item())
                                xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]
                                detections.append({"Class": cname, "Confidence": f"{conf_val*100:.1f}%", "Box [x1, y1, x2, y2]": str(xyxy)})

                        if detections:
                            st.dataframe(pd.DataFrame(detections), use_container_width=True)
                        else:
                            st.info("No objects detected above the confidence threshold.")

                        # Download annotated image
                        save_out_path = TEMP_DIR / "annotated_detection.jpg"
                        Image.fromarray(res_rgb).save(save_out_path)
                        with open(save_out_path, "rb") as f:
                            st.download_button("📥 Download Annotated Image", f, file_name="annotated_detection.jpg", use_container_width=True)

                # Mode 2: Validation Dataset Sample
                elif input_type == "📂 Validation Dataset Sample":
                    val_images = []
                    search_dirs = [DATASET_DIR, Path("cctv-1")]
                    for s_dir in search_dirs:
                        if s_dir.exists():
                            for val_sub in ["val", "valid", "validation"]:
                                for p in (s_dir / val_sub).rglob("*"):
                                    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                                        val_images.append(p)

                    if not val_images:
                        st.info("No validation images found in current dataset. Upload and extract a dataset first.")
                    else:
                        selected_val_img_path = st.selectbox("Select Sample Image", val_images, format_func=lambda p: p.name)
                        if selected_val_img_path and selected_val_img_path.exists():
                            pil_img = Image.open(selected_val_img_path).convert("RGB")
                            t0 = time.time()
                            results = model_obj.predict(
                                source=pil_img,
                                conf=conf_thresh,
                                iou=iou_thresh,
                                imgsz=infer_imgsz,
                                device=infer_device,
                                boxes=show_boxes,
                                show_labels=show_labels,
                                show_conf=show_conf,
                                line_width=line_width,
                                verbose=False
                            )
                            t_infer = (time.time() - t0) * 1000
                            res = results[0]
                            res_bgr = res.plot()
                            res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)

                            img_c1, img_c2 = st.columns(2)
                            img_c1.image(pil_img, caption=f"Validation Image: {selected_val_img_path.name}", use_container_width=True)
                            img_c2.image(res_rgb, caption=f"Detections ({t_infer:.1f} ms)", use_container_width=True)

                # Mode 3: Live Webcam Snapshot
                elif input_type == "📷 Live Webcam Snapshot":
                    camera_photo = st.camera_input("Take a snapshot for detection")
                    if camera_photo:
                        pil_img = Image.open(camera_photo).convert("RGB")
                        t0 = time.time()
                        results = model_obj.predict(
                            source=pil_img,
                            conf=conf_thresh,
                            iou=iou_thresh,
                            imgsz=infer_imgsz,
                            device=infer_device,
                            boxes=show_boxes,
                            show_labels=show_labels,
                            show_conf=show_conf,
                            line_width=line_width,
                            verbose=False
                        )
                        t_infer = (time.time() - t0) * 1000
                        res = results[0]
                        res_bgr = res.plot()
                        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)

                        st.image(res_rgb, caption=f"Webcam Detection Result ({t_infer:.1f} ms)", use_container_width=True)

                # Mode 4: Video Upload & Processing
                elif input_type == "🎥 Upload Video":
                    uploaded_video = st.file_uploader("Upload Video File (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
                    if uploaded_video:
                        video_path = TEMP_DIR / uploaded_video.name
                        with open(video_path, "wb") as f:
                            f.write(uploaded_video.getbuffer())

                        if st.button("▶ Process Video", type="primary", use_container_width=True):
                            output_video_path = TEMP_DIR / f"annotated_{uploaded_video.name}.mp4"
                            cap = cv2.VideoCapture(str(video_path))
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

                            prog_bar = st.progress(0, text="Processing video frames...")
                            frame_idx = 0
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                res = model_obj.predict(
                                    source=frame_rgb,
                                    conf=conf_thresh,
                                    iou=iou_thresh,
                                    imgsz=infer_imgsz,
                                    device=infer_device,
                                    verbose=False
                                )[0]
                                out_frame_bgr = res.plot()
                                out.write(out_frame_bgr)
                                frame_idx += 1
                                if frame_idx % 5 == 0 or frame_idx == total_frames:
                                    prog_bar.progress(min(frame_idx / total_frames, 1.0), text=f"Processed frame {frame_idx}/{total_frames}")

                            cap.release()
                            out.release()
                            prog_bar.empty()
                            st.success("✅ Video processing complete!")
                            with open(output_video_path, "rb") as f:
                                st.download_button("📥 Download Annotated Video", f, file_name=f"annotated_{uploaded_video.name}", use_container_width=True)

            except Exception as e:
                st.error(f"Inference error: {e}")


# ===========================================================================
# TAB 3: 📦 MODEL EXPORT STUDIO
# ===========================================================================
with tab_export:
    st.subheader("Model Export & Deployment Studio")
    st.caption("Export trained YOLO models to ultra-fast inference formats (ONNX, TensorRT, OpenVINO, TorchScript, CoreML, TFLite).")

    exp_col1, exp_col2 = st.columns([1, 1.5])

    with exp_col1:
        st.markdown("##### 1. Export Configuration")
        available_exp_models = get_available_models()
        model_to_export = st.selectbox("Select Model to Export", available_exp_models, key="export_model_select")

        export_format = st.selectbox(
            "Target Format",
            ["onnx", "engine", "torchscript", "openvino", "coreml", "tflite"],
            index=0,
            help="ONNX is cross-platform; TensorRT (engine) is fastest on NVIDIA GPUs; OpenVINO is optimized for Intel CPUs."
        )

        export_imgsz = st.selectbox("Export Image Size", [320, 416, 512, 640, 768, 960, 1280], index=3)
        export_half = st.checkbox("FP16 Half-Precision (2x speed on GPU)", value=False)
        export_dynamic = st.checkbox("Dynamic Axes (Variable Batch & Size)", value=False)
        export_simplify = st.checkbox("Simplify ONNX Graph", value=True)
        export_int8 = st.checkbox("INT8 Quantization (if supported)", value=False)

        export_device = "0" if (CUDA_AVAILABLE and export_format == "engine") else "cpu"

    with exp_col2:
        st.markdown("##### 2. Export Execution & Artifacts")
        st.info(f"Target format: **{export_format.upper()}** • Input: `{Path(model_to_export).name}` • Image size: `{export_imgsz}`")

        if st.button("🚀 Export Model", type="primary", use_container_width=True):
            if not model_to_export or not Path(model_to_export).exists():
                st.error("Invalid model path. Please choose an existing model.")
            else:
                with st.spinner(f"Exporting model to {export_format.upper()}..."):
                    try:
                        exp_model = YOLO(model_to_export)
                        exported_path_str = exp_model.export(
                            format=export_format,
                            imgsz=export_imgsz,
                            half=export_half,
                            dynamic=export_dynamic,
                            simplify=export_simplify,
                            int8=export_int8,
                            device=export_device,
                        )
                        st.success(f"🎉 Successfully exported to: `{exported_path_str}`")

                        exp_p = Path(exported_path_str)
                        if exp_p.is_file():
                            file_size_mb = exp_p.stat().st_size / (1024 * 1024)
                            st.caption(f"📦 File Size: **{file_size_mb:.2f} MB**")
                            with open(exp_p, "rb") as f:
                                st.download_button(
                                    f"📥 Download {exp_p.name}",
                                    f,
                                    file_name=exp_p.name,
                                    key="dl_exported_btn",
                                    use_container_width=True
                                )
                        elif exp_p.is_dir():
                            zip_out = EXPORTS_DIR / f"{exp_p.name}.zip"
                            shutil.make_archive(str(zip_out.with_suffix("")), "zip", str(exp_p))
                            st.caption(f"📦 Packaged Directory: **{zip_out.stat().st_size / (1024 * 1024):.2f} MB**")
                            with open(zip_out, "rb") as f:
                                st.download_button(
                                    f"📥 Download {zip_out.name}",
                                    f,
                                    file_name=zip_out.name,
                                    key="dl_exported_zip_btn",
                                    use_container_width=True
                                )
                    except Exception as e:
                        st.error(f"Export error: {e}")


# ===========================================================================
# TAB 4: 🎛️ HYPERPARAMETER TUNING (OPTUNA / ULTRALYTICS TUNE)
# ===========================================================================
with tab_tune:
    st.subheader("Hyperparameter Tuning Studio")
    st.caption("Automated search (Genetic Algorithm & Optuna) to discover the optimal learning rate, momentum, loss gains, and data augmentations for your dataset.")

    tune_c1, tune_c2 = st.columns([1, 1.5])

    with tune_c1:
        st.markdown("##### 1. Tuning Configuration")
        tune_models = get_available_models()
        tune_base_model = st.selectbox("Base Model", tune_models, index=0, key="tune_base_model_sel")
        tune_iterations = st.number_input("Tuning Iterations / Trials", min_value=2, max_value=300, value=15)
        tune_epochs = st.number_input("Epochs per Trial", min_value=1, max_value=100, value=10)
        tune_optimizer = st.selectbox("Optimizer for Tuning", ["auto", "AdamW", "SGD"], index=0)
        tune_gpu = "0" if CUDA_AVAILABLE else "cpu"
        tune_run_name = st.text_input("Tuning Run Name", value=st.session_state.tune_run_name)

        tune_btn_col1, tune_btn_col2 = st.columns(2)
        tune_start_disabled = data_yaml_path is None or st.session_state.tune_active

        with tune_btn_col1:
            if st.button("🚀 Start Tuning", type="primary", disabled=tune_start_disabled, use_container_width=True):
                cmd = [
                    "yolo", "tune",
                    f"model={tune_base_model}",
                    f"data={data_yaml_path}",
                    f"epochs={tune_epochs}",
                    f"iterations={tune_iterations}",
                    f"optimizer={tune_optimizer}",
                    f"device={tune_gpu}",
                    f"project={TUNE_DIR}",
                    f"name={tune_run_name}",
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
                st.session_state.tune_paused = False
                st.session_state.tune_run_name = tune_run_name
                st.rerun()

        with tune_btn_col2:
            if st.button("🛑 Stop Tuning", disabled=not st.session_state.tune_active, use_container_width=True):
                if os.name == "nt":
                    st.session_state.tune_proc.terminate()
                else:
                    _signal_group(st.session_state.tune_proc, signal.SIGTERM)
                st.session_state.tune_active = False
                st.rerun()

    with tune_c2:
        st.markdown("##### 2. Live Tuning Console & Best Parameters")
        tune_log_ph = st.empty()
        tune_log_ph.code("".join(st.session_state.tune_logs[-30:]) if st.session_state.tune_logs else "Ready to tune hyperparameters. Click 'Start Tuning'.", language="bash")

        # Check for best_hyperparameters.yaml
        tune_best_yaml = TUNE_DIR / st.session_state.tune_run_name / "best_hyperparameters.yaml"
        if not tune_best_yaml.exists():
            for y_file in TUNE_DIR.rglob("best_hyperparameters.yaml"):
                tune_best_yaml = y_file
                break

        if tune_best_yaml.exists():
            try:
                with open(tune_best_yaml, "r") as f:
                    best_hyp = yaml.safe_load(f)
                st.success(f"🏆 **Optimal Hyperparameters Found** (`{tune_best_yaml.parent.name}`)")
                st.json(best_hyp)

                if st.button("✨ Apply Best Hyperparameters to Training Dashboard", type="primary", use_container_width=True):
                    st.session_state.tuned_params = best_hyp
                    st.toast("✅ Tuned hyperparameters applied to Training tab!", icon="🎯")
            except Exception:
                pass

    if st.session_state.tune_active:
        proc = st.session_state.tune_proc
        if proc and proc.poll() is not None:
            st.session_state.tune_active = False
            st.toast("🎉 Hyperparameter tuning completed!", icon="✅")
            st.rerun()
        else:
            time.sleep(1)
            st.rerun()


# ===========================================================================
# TAB 5: 📊 RUN HISTORY & COMPARISONS
# ===========================================================================
with tab_history:
    st.subheader("Training Run History & Multi-Run Comparisons")

    # Discover all runs
    run_folders = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
    if not run_folders:
        st.info("No training runs found in `yolo_workspace/runs/` yet.")
    else:
        history_rows = []
        for r_dir in sorted(run_folders, key=lambda p: p.stat().st_mtime, reverse=True):
            df_r, _ = load_run_results(r_dir.name)
            epochs_done = len(df_r) if df_r is not None else 0
            best_map50 = df_r["metrics/mAP50(B)"].max() if df_r is not None and "metrics/mAP50(B)" in df_r.columns else 0.0
            best_map50_95 = df_r["metrics/mAP50-95(B)"].max() if df_r is not None and "metrics/mAP50-95(B)" in df_r.columns else 0.0
            has_best_pt = (r_dir / "weights" / "best.pt").exists()

            history_rows.append({
                "Run Name": r_dir.name,
                "Epochs Done": epochs_done,
                "Best mAP@50": f"{best_map50:.4f}" if best_map50 > 0 else "N/A",
                "Best mAP@50-95": f"{best_map50_95:.4f}" if best_map50_95 > 0 else "N/A",
                "Has Weights": "✅" if has_best_pt else "❌",
                "Directory": str(r_dir.name),
            })

        st.dataframe(pd.DataFrame(history_rows), use_container_width=True)

        st.markdown("---")
        st.markdown("##### 🔍 Compare Multiple Runs")
        selected_runs_for_comp = st.multiselect(
            "Select Runs to Compare Curves",
            [r["Run Name"] for r in history_rows],
            default=[r["Run Name"] for r in history_rows[:min(3, len(history_rows))]]
        )

        if selected_runs_for_comp:
            comp_metric = st.selectbox("Metric to Compare", ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "train/box_loss", "val/box_loss"])
            comp_df = pd.DataFrame()
            for r_name in selected_runs_for_comp:
                df_temp, _ = load_run_results(r_name)
                if df_temp is not None and comp_metric in df_temp.columns and "epoch" in df_temp.columns:
                    comp_df[r_name] = df_temp.set_index("epoch")[comp_metric]

            if not comp_df.empty:
                st.line_chart(comp_df, height=350)
            else:
                st.caption(f"No '{comp_metric}' data available for the selected runs.")