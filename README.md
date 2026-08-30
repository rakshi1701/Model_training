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

### 5. 📊 Run History & Comparisons
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
├── app.py                  # Main Streamlit Vision Studio
├── Requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml         # 50GB upload configuration
└── yolo_workspace/
    ├── dataset/            # Extracted active dataset
    ├── runs/               # Training run outputs & checkpoints
    ├── tune/               # Hyperparameter tuning experiments
    └── exports/            # Exported deployment models (ONNX, TensorRT, etc.)
```