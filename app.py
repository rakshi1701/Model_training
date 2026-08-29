"""
YOLO Training UI
Run with: streamlit run app.py
Requires: pip install streamlit ultralytics pyyaml
"""

import streamlit as st
import subprocess
import zipfile
import shutil
import yaml
from pathlib import Path

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except Exception:
    CUDA_AVAILABLE = False
    GPU_COUNT = 0

st.set_page_config(page_title="YOLO Training UI", layout="wide")

WORKDIR = Path("yolo_workspace")
DATASET_DIR = WORKDIR / "dataset"
RUNS_DIR = WORKDIR / "runs"
WORKDIR.mkdir(exist_ok=True)

st.title("YOLO Model Training Dashboard")

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
st.sidebar.header("1. Dataset")
st.sidebar.caption(
    "Drag-and-drop or click to browse for a .zip containing images/labels in "
    "YOLO format + a data.yaml. Limit is 50GB (set in .streamlit/config.toml "
    "— must sit next to app.py)."
)

data_yaml_path = None
uploaded_zip = st.sidebar.file_uploader("Upload dataset (.zip)", type=["zip"])

if uploaded_zip and st.sidebar.button("Extract Dataset"):
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True)
    zip_path = DATASET_DIR / "upload.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATASET_DIR)
    zip_path.unlink()

    # Find the actual train/val/test image folders on disk and write their
    # absolute paths directly into data.yaml. We don't trust the yaml's own
    # relative style (e.g. Roboflow exports often use "../train/images",
    # which assumes a folder layout that doesn't always match how the zip
    # actually extracts) — searching for the real folders sidesteps that
    # mismatch entirely.
    def _find_split_dir(root: Path, names):
        for d in root.rglob("images"):
            if d.is_dir() and d.parent.name.lower() in names:
                return d.resolve()
        return None

    yamls = list(DATASET_DIR.rglob("*.yaml")) + list(DATASET_DIR.rglob("*.yml"))
    if yamls:
        yaml_path = yamls[0]
        try:
            with open(yaml_path, "r") as f:
                data_cfg = yaml.safe_load(f)

            train_dir = _find_split_dir(DATASET_DIR, {"train"})
            val_dir = _find_split_dir(DATASET_DIR, {"val", "valid", "validation"})
            test_dir = _find_split_dir(DATASET_DIR, {"test"})

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
            data_cfg.pop("path", None)  # no longer needed once train/val are absolute

            with open(yaml_path, "w") as f:
                yaml.safe_dump(data_cfg, f, sort_keys=False)

            if train_dir and val_dir:
                st.sidebar.success(f"Dataset extracted. Resolved: {', '.join(found)}.")
            else:
                st.sidebar.warning(
                    "Extracted, but couldn't locate both train and val image folders "
                    "automatically — check your zip's folder structure."
                )
        except Exception as e:
            st.sidebar.warning(f"Extracted, but couldn't auto-patch data.yaml: {e}")
    else:
        st.sidebar.success("Dataset extracted.")

if DATASET_DIR.exists():
    yamls = list(DATASET_DIR.rglob("*.yaml")) + list(DATASET_DIR.rglob("*.yml"))
    if yamls:
        data_yaml_path = yamls[0]
        st.sidebar.info(f"Using data config: {data_yaml_path.relative_to(DATASET_DIR)}")

        # Validate that train/val image folders actually resolve and contain files,
        # so a bad dataset fails here instead of mid-training.
        try:
            with open(data_yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
            base = Path(cfg.get("path", data_yaml_path.parent))
            issues = []
            counts = {}
            for split in ("train", "val"):
                rel = cfg.get(split)
                if not rel:
                    continue
                split_path = (base / rel).resolve()
                if not split_path.exists():
                    issues.append(f"'{split}' path not found: {split_path}")
                else:
                    n = sum(
                        1 for p in split_path.rglob("*")
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
                    )
                    counts[split] = n
                    if n == 0:
                        issues.append(f"'{split}' path has no images: {split_path}")
            if issues:
                for msg in issues:
                    st.sidebar.error(msg)
            else:
                st.sidebar.caption(
                    " · ".join(f"{k}: {v} images" for k, v in counts.items())
                )
        except Exception as e:
            st.sidebar.warning(f"Could not validate dataset: {e}")
    elif uploaded_zip:
        st.sidebar.warning("No data.yaml found — include one in the zip (paths + class names).")


# ---------------------------------------------------------------------------
# 2. Model selection
# ---------------------------------------------------------------------------
st.sidebar.header("2. Model")

model_family = st.sidebar.selectbox("Model family", ["YOLOv8", "YOLO11", "YOLOv9", "YOLOv10"])

size_map = {
    "YOLOv8": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
    "YOLO11": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
    "YOLOv9": ["yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e"],
    "YOLOv10": ["yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"],
}
model_size = st.sidebar.selectbox("Model size", size_map[model_family])

task = st.sidebar.selectbox("Task", ["detect", "segment", "classify", "pose", "obb"])
task_suffix = {"detect": "", "segment": "-seg", "classify": "-cls", "pose": "-pose", "obb": "-obb"}[task]
pretrained = st.sidebar.checkbox("Start from pretrained weights", value=True)
model_name = f"{model_size}{task_suffix}.pt" if pretrained else f"{model_size}{task_suffix}.yaml"

# ---------------------------------------------------------------------------
# 3. Hyperparameters
# ---------------------------------------------------------------------------
st.sidebar.header("3. Training Parameters")

epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=2000, value=100)
batch = st.sidebar.selectbox("Batch size", [-1, 4, 8, 16, 32, 64, 128], index=3,
                              help="-1 lets Ultralytics auto-pick batch size from free VRAM (60% utilization target)")
imgsz = st.sidebar.selectbox("Image size", [320, 416, 512, 640, 768, 960, 1280], index=3)
optimizer = st.sidebar.selectbox("Optimizer", ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
lr0 = st.sidebar.number_input("Initial LR (lr0)", min_value=0.00001, max_value=1.0, value=0.01, format="%.5f")
patience = st.sidebar.number_input("Early stopping patience (epochs)", min_value=0, max_value=300, value=50)
device_options = []
device_labels = {}
if CUDA_AVAILABLE:
    for i in range(GPU_COUNT):
        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = f"GPU {i}"
        label = f"GPU {i}: {name}"
        device_options.append(label)
        device_labels[label] = str(i)
    if GPU_COUNT > 1:
        all_label = f"All GPUs (0-{GPU_COUNT - 1})"
        device_options.append(all_label)
        device_labels[all_label] = ",".join(str(i) for i in range(GPU_COUNT))
device_options.append("CPU")
device_labels["CPU"] = "cpu"

device_choice = st.sidebar.selectbox(
    "Device",
    device_options,
    index=0,
    help="GPUs are auto-detected. Falls back to CPU if none are found.",
)
device = device_labels[device_choice]

if CUDA_AVAILABLE:
    st.sidebar.caption(f"✅ {GPU_COUNT} CUDA GPU(s) detected")
else:
    st.sidebar.caption("⚠️ No CUDA GPU detected — CPU only. Training will be slower.")
workers = st.sidebar.number_input("Dataloader workers", min_value=0, max_value=32, value=8)

with st.sidebar.expander("Advanced options"):
    resume = st.checkbox("Resume from last checkpoint", value=False)
    cache = st.selectbox("Cache images in", ["False", "ram", "disk"])
    cos_lr = st.checkbox("Cosine LR scheduler", value=False)
    weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=0.0005, format="%.5f")
    freeze = st.number_input("Freeze first N layers (0 = none)", min_value=0, max_value=50, value=0)
    run_name = st.text_input("Run name", value="exp1")

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Config Summary")
    st.json({
        "model": model_name,
        "data": str(data_yaml_path) if data_yaml_path else None,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "optimizer": optimizer,
        "lr0": lr0,
        "patience": patience,
        "device": device,
        "workers": workers,
    })

with col1:
    st.subheader("Training Console")
    log_placeholder = st.empty()

    start_disabled = data_yaml_path is None
    if start_disabled:
        st.caption("Upload and extract a dataset with a data.yaml to enable training.")

    if st.button("Start Training", type="primary", disabled=start_disabled):
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
        st.code(" ".join(cmd), language="bash")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        log_lines = []
        for line in process.stdout:
            log_lines.append(line)
            log_placeholder.code("".join(log_lines[-50:]), language="bash")
        process.wait()

        if process.returncode == 0:
            st.success("Training complete.")
            best_pt = RUNS_DIR / run_name / "weights" / "best.pt"
            if best_pt.exists():
                with open(best_pt, "rb") as f:
                    st.download_button("Download best.pt", f, file_name="best.pt")
            results_png = RUNS_DIR / run_name / "results.png"
            if results_png.exists():
                st.image(str(results_png), caption="Training curves")
        else:
            st.error("Training failed — check the console output above.")