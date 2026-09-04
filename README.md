# YOLO Studio: Training, Inference, Export & Tuning Dashboard

A comprehensive Streamlit workspace for managing the entire lifecycle of YOLO computer vision models — from dataset extraction and training to real-time metric visualization, interactive inference testing, hyperparameter optimization, and edge deployment exports.

---

## 🌟 Key Features

### 1. 🏋️ Model Training & Real-Time Monitoring
- **Architecture Selection**: YOLO11, YOLOv8, YOLOv9, YOLOv10 in all sizes (`n`, `s`, `m`, `l`, `x`) and tasks (`detect`, `segment`, `classify`, `pose`, `obb`).
- **Comprehensive Hyperparameters**: Epochs, batch size (including auto-batch `-1`), image size, optimizers (AdamW, SGD, Adam, NAdam, RMSProp), learning rate (`lr0`), cosine scheduler, weight decay, early stopping patience, layer freezing.
- **Hardware Management**: Auto-detects NVIDIA CUDA GPUs with single-GPU, multi-GPU, or CPU fallback.
- **Live Metrics Dashboard**: Real-time KPI summary cards (Epoch progress, Box Loss, Class Loss, mAP@50, mAP@50-95, Precision, Recall) with live-updating interactive loss and accuracy curves directly parsed from `results.csv`.
- **Training Lifecycle Controls**: Start, Pause (`SIGSTOP`), Resume (`SIGCONT`), and Terminate (`SIGTERM`) with process-group scoping to control all dataloader worker subprocesses.

### 2. 🧪 Inference & Testing Playground
- **Model Checkpoints**: Test with any trained checkpoint (`best.pt`, `last.pt`), base pre-trained models (`yolo11n.pt`, `yolov8n.pt`), or uploaded custom weights.
- **Multiple Input Sources**:
  - **Single/Multi-Image Upload**: Drag-and-drop JPG, PNG, WEBP, BMP.
  - **Validation Set Explorer**: Pick validation images directly from your extracted dataset without uploading.
  - **Live Webcam**: Snapshot testing using your camera.
  - **Video Inference**: Upload MP4, AVI, MOV videos with frame-by-frame progress rendering and download.
- **Adjustable Parameters**: Confidence threshold slider, NMS IoU threshold slider, resolution, bounding boxes, labels, confidence scores, and mask toggles.
- **Detailed Analytics**: Displays latency breakdown (preprocess, inference, postprocess ms) and structured detection coordinates table.

### 3. 📦 Model Export Studio
- **One-Click Format Export**:
  - **ONNX** (`.onnx`): Universal deployment (ONNX Runtime, OpenCV DNN, Triton).
  - **TensorRT Engine** (`.engine`): Maximum throughput on NVIDIA GPUs.
  - **OpenVINO** (`_openvino_model/`): Intel CPU & iGPU acceleration (auto-zipped for 1-click download).
  - **TorchScript** (`.torchscript`): C++ LibTorch integration.
  - **CoreML** (`.mlpackage`): Apple iOS/macOS devices.
  - **TFLite** (`.tflite`): Mobile & Edge deployment.
- **Export Options**: FP16 half-precision, dynamic axes batching, ONNX graph simplification, and INT8 quantization.

### 4. 🎛️ Hyperparameter Tuning (Optuna / Ultralytics Tune)
- Automated hyperparameter search to optimize learning rates, momentum, box/cls loss gains, and augmentation parameters.
- Configurable search iterations and trial epochs.
- Live tuning console and real-time best parameters (`best_hyperparameters.yaml`) viewer.
- **1-Click Apply**: Automatically transfer tuned parameters directly into the Training dashboard.

### 5. ☁️ Kaggle Cloud GPU Training Hub
- **Free Multi-GPU Acceleration**: Offload long training jobs to Kaggle's Dual NVIDIA Tesla T4 (32GB VRAM) or P100 GPUs via PyTorch Distributed Data Parallel (DDP).
- **1-Click Remote Dispatch**: Seamlessly packages datasets, pushes training kernels via Kaggle API, and tracks live execution status from within the dashboard.
- **Automated Checkpoint Sync**: Automatically downloads `best.pt`, `results.csv`, and validation figures into `yolo_workspace/runs/` for immediate local inference and evaluation.
- **Live Polling & Auto-Ingest**: Toggle **Live polling** to re-check every tracked job every 30s and pull finished runs into `yolo_workspace/runs/` with no clicks. Runs already on disk are never re-downloaded.
- **Remote GPU Telemetry**: The generated kernel samples `nvidia-smi` every 30s and emits `[GPUSTAT]` markers, so the dashboard shows per-GPU utilisation, VRAM, temperature, power, a utilisation-over-time chart, and epoch ETA parsed from the run log.
- **Weekly GPU Quota Tracker**: Rolling 7-day estimate of GPU hours used vs. Kaggle's 30h/week allowance, measured from each job's actual runtime.
- **12-Hour Timeout Survival**: A configurable wall-clock cap (default 11h) stops training before Kaggle kills the session and always packages `last.pt`. Pick a capped or crashed job under **Resume from previous job** to mount its output and continue from that checkpoint.
- **Crash-Safe Artifacts**: A failed run still packages whatever was checkpointed, so partial weights survive an OOM or a mid-run error.
- **Post-Training Pipeline**: Optionally auto-export the ingested `best.pt` to ONNX / TorchScript / OpenVINO as soon as a cloud run lands.
- **Standalone Jupyter Template**: Includes a pre-configured `kaggle_yolo_train_template.ipynb` for direct web notebook training.

> **Note on the quota tracker:** Kaggle publishes no quota API. The figure is estimated from jobs dispatched by this dashboard — runs you start on kaggle.com are not counted. Confirm at [kaggle.com/settings](https://www.kaggle.com/settings).

> **Stopping a cloud job:** Kaggle's public API has **no cancel endpoint** ([kaggle-api#388](https://github.com/Kaggle/kaggle-api/issues/388)), so a running kernel can only be stopped from the Kaggle web console. **🛑 Stop this job** checks live status and links straight to the session page — use **Stop Session** there. **🗑 Remove from this list** is local tracking only and warns before orphaning a job that is still running. Once you stop it on Kaggle, the dashboard picks up the cancelled status on the next refresh.

### 6. 📊 Run History & Comparisons
- Automatically tracks all training experiments in `yolo_workspace/runs/`.
- Multi-run overlay comparison charts for mAP50, mAP50-95, and loss curves across experiments.
- Download artifacts and view confusion matrices, PR curves, and validation batch predictions.

---

## 🚀 Setup & Installation

```bash
pip install -r Requirements.txt
```

### Running the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## 📁 Dataset Format & 50GB Uploads

Zip your dataset in standard YOLO format:

```
my_dataset.zip
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

- Upload limit is configured up to **50GB** in `.streamlit/config.toml`.
- Roboflow relative path discrepancies (e.g. `../train/images`) are automatically detected and patched to absolute paths during extraction.

---

## 📂 Workspace Structure

```
.
├── app.py                            # Main Streamlit Vision Studio
├── kaggle_bridge.py                  # Kaggle Cloud API Integration & Sync Engine
├── kaggle_yolo_train_template.ipynb  # Standalone Kaggle Notebook Template
├── Requirements.txt                  # Dependencies
├── .streamlit/
│   └── config.toml                   # 50GB upload configuration
└── yolo_workspace/
    ├── dataset/                      # Extracted active dataset
    ├── runs/                         # Training run outputs & checkpoints
    ├── tune/                         # Hyperparameter tuning experiments
    └── exports/                      # Exported deployment models (ONNX, TensorRT, etc.)
```