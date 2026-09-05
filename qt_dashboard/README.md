# YOLO Studio — Qt/QML desktop dashboard

A native desktop build of the Streamlit dashboard in `app.py`. Same seven tabs,
same workspace (`yolo_workspace/`), same engine: training still shells out to
the `yolo` CLI, cloud jobs still go through `kaggle_bridge.py`, and runs
ingested by either front-end show up in the other.

```bash
pip install -r qt_dashboard/requirements.txt
python3 qt_dashboard/main.py
```

> **Linux/X11:** Qt 6.5+ needs `libxcb-cursor0`. If the app exits with
> *"Could not load the Qt platform plugin xcb"*, install it:
> `sudo apt install libxcb-cursor0`.

Run it with the same interpreter that runs `app.py` — that is where
ultralytics, torch, opencv and the kaggle client live.

---

## The seven tabs

| Tab | What it does |
| :-- | :-- |
| 🏋️ **Local Training & Metrics** | Presets, full hyperparameter grid, device picker, Start / Pause (SIGSTOP) / Resume (SIGCONT) / Terminate (SIGTERM) over the whole process group, live KPI cards, loss / mAP / LR curves, terminal stream, checkpoints, and a 2-second host utilisation monitor (per-core CPU, RAM, swap, disk, per-GPU compute/VRAM/temp/power, training-process RSS). |
| ☁️ **Kaggle Cloud GPU Training** | Credentials (import `kaggle.json` or type them), weekly GPU-quota estimate, dataset staging + kernel dispatch with a live progress log, resume-from-capped-run picker, and the job board: status counts, filter/search, one card per job with model downloads, ingest, stop, log + GPU telemetry, and removal. Live polling re-checks every 30s and auto-ingests finished runs. |
| 📂 **Dataset Hub & Visual Inspector** | Split counts, class table, config path, and a ground-truth box inspector that steps through every image in a split. |
| 🧪 **Inference & Testing Playground** | Any checkpoint (discovered or picked from disk), confidence/IoU/resolution/device controls, and four input modes: image files, a validation-split sample, a webcam frame, or a video file rendered frame-by-frame with progress. Shows latency breakdown and a detections table. |
| 📦 **Model Export Studio** | ONNX, TensorRT, OpenVINO, TorchScript, CoreML, TFLite with FP16 / dynamic / simplify / INT8 switches. Directory outputs are zipped. |
| 🎛️ **Hyperparameter Tuning** | `yolo tune` with live log, the discovered `best_hyperparameters.yaml`, and one-click transfer into the Training tab. |
| 📊 **Experiment History & Analytics** | Every run under `runs/`, multi-run curve overlays, and per-run figures (confusion matrix, PR curves, val batches). |

---

## How it is put together

```
qt_dashboard/
├── main.py                     # QApplication + QML engine; publishes the controllers
├── backend/
│   ├── core.py                 # paths, presets, dataset/model/run helpers (UI-free port of app.py's helper layer)
│   ├── util.py                 # QThreadPool worker + run_async
│   ├── app_state.py            # hardware, the active dataset, model list, tuned-params hand-off
│   ├── system_monitor.py       # 2s CPU/RAM/swap/disk/GPU telemetry
│   ├── train_controller.py     # yolo train subprocess, signals, live results.csv parsing
│   ├── dataset_controller.py   # split browsing + ground-truth rendering
│   ├── inference_controller.py # image/webcam/video prediction off the GUI thread
│   ├── export_controller.py    # Ultralytics export + zip packaging
│   ├── tune_controller.py      # yolo tune subprocess + best_hyperparameters.yaml
│   ├── history_controller.py   # run table and overlay series
│   └── kaggle_controller.py    # Qt shell over kaggle_bridge.py
└── qml/
    ├── Main.qml                # window, sidebar, pipeline tracker, tab bar, toasts
    ├── Studio/                 # design system: Theme, Card, MetricCard, inputs, LogView, MetricChart, DataTable…
    └── pages/                  # one file per tab (+ SystemMonitorPanel, JobCard)
```

Every long-running call (dataset staging, weight downloads, ingest, export,
inference, log reads) runs on a `QThreadPool` worker and reports back through
signals, so the window never blocks. Charts are drawn on a `Canvas` rather than
QtCharts, so there is no extra module to install and no GPU/driver dependency.

### Desktop equivalents of the web app's browser affordances

| Streamlit | Qt |
| :-- | :-- |
| `st.file_uploader` | native `FileDialog` |
| `st.download_button` | "Save …" `FileDialog` that copies the file, plus "Open folder" |
| `st.camera_input` | one-shot OpenCV capture from the default camera |
| `st.toast` / `st.success` | toast in the bottom-right corner + inline notes |
| Auto-rerun polling | `QTimer` (1s for training/tuning, 2s for hardware, 30s for Kaggle live polling) |

### Dataset handling

Uploads are additive: each zip is extracted into its own folder under
`yolo_workspace/dataset/`, added to the selector, and made active — existing
datasets stay put. **🗑 Remove “name”** in the sidebar deletes one dataset
(confirmation required, workspace datasets only). Folder-per-class sets are
discovered without a `data.yaml` and marked **Classification Dataset**; the
Dataset Hub offers a train/val split for flat exports that need one. All of
this matches `app.py` — both front-ends share the layout on disk.

## One deliberate difference from `app.py`

Dataset splits resolve more forgivingly. Roboflow exports write
`train: ../train/images`, which only resolves when the yaml sits one level
below the images; the Qt build also tries the yaml's own folder, so
`yolo_workspace/dataset/cctv-1` lists its 3060/526/151 images and the
inspector works instead of reporting "none detected". Everything else
behaves as the Streamlit app does, including the `?` placeholders Kaggle
jobs discovered on your account carry for unknown hyperparameters.
