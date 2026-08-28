"""
YOLO Training UI
Run with: streamlit run app.py
Requires: pip install streamlit ultralytics pyyaml
"""

import streamlit as st
import subprocess
import zipfile
import shutil
from pathlib import Path

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

dataset_mode = st.sidebar.radio(
    "Dataset source",
    ["Local path (recommended for large datasets)", "Browser upload (.zip)"],
    help="Since this app runs on your own machine, pointing to a local folder skips "
         "the browser upload entirely and has no size limit. Use browser upload only "
         "when the dataset isn't already on this machine.",
)

data_yaml_path = None

if dataset_mode.startswith("Local path"):
    local_path = st.sidebar.text_input(
        "Path to dataset folder or data.yaml",
        placeholder="/home/you/datasets/my_dataset  or  .../data.yaml",
    )
    if local_path:
        p = Path(local_path).expanduser()
        if p.is_file() and p.suffix in (".yaml", ".yml"):
            data_yaml_path = p
            st.sidebar.success(f"Using data config: {p}")
        elif p.is_dir():
            yamls = list(p.rglob("*.yaml")) + list(p.rglob("*.yml"))
            if yamls:
                data_yaml_path = yamls[0]
                st.sidebar.success(f"Found data config: {data_yaml_path}")
            else:
                st.sidebar.warning("No data.yaml found under that folder.")
        else:
            st.sidebar.error("Path doesn't exist.")
else:
    st.sidebar.caption(
        "Upload a .zip containing images/labels in YOLO format + a data.yaml. "
        "Default limit is 200MB — see the note below to raise it."
    )
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
        st.sidebar.success("Dataset extracted.")

    if DATASET_DIR.exists():
        yamls = list(DATASET_DIR.rglob("*.yaml")) + list(DATASET_DIR.rglob("*.yml"))
        if yamls:
            data_yaml_path = yamls[0]
            st.sidebar.info(f"Using data config: {data_yaml_path.relative_to(DATASET_DIR)}")
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
device = st.sidebar.text_input("Device", value="0", help="'0' for GPU 0, 'cpu', '0,1' for multi-GPU, 'mps' for Apple Silicon")
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