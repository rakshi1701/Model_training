# YOLO Training UI

A Streamlit dashboard for configuring and launching YOLO training runs without
touching the command line — upload a dataset, pick a model, set your
hyperparameters, and watch training logs live.

## Features

- **Dataset upload**: drag-and-drop or browse for a `.zip` containing your
  YOLO-format dataset (images/labels + `data.yaml`). No practical size limit
  when run on localhost (see Configuration below).
- **Model selection**: YOLOv8 / YOLOv9 / YOLOv10 / YOLO11, all sizes (n/s/m/l/x),
  and task type (detect / segment / classify / pose / obb).
- **Hyperparameters**: epochs, batch size, image size, optimizer, learning rate,
  early stopping patience, device, dataloader workers, plus advanced options
  (resume, cache mode, cosine LR, weight decay, layer freezing).
- **Live training console**: streams `yolo train` output directly into the page
  as it trains.
- **Results**: download link for `best.pt` and the training curves plot once
  a run finishes.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and (for GPU training) a CUDA-capable GPU with the
matching PyTorch build already installed — `ultralytics` will pull in a CPU
build of PyTorch by default if none is present.

## Running

From the project folder (the one containing `app.py` and `.streamlit/`):

```bash
streamlit run app.py
```

This opens the dashboard in your browser at `http://localhost:8501`.

> **Important:** `.streamlit/config.toml` must sit next to `app.py` in the
> directory you run the command from. If you run `streamlit run` from a
> different folder, the upload size settings in that config won't be picked
> up and you'll hit Streamlit's 200MB default.

## Preparing a dataset

Zip your dataset so it looks like this inside the archive:

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

`data.yaml` should follow the standard Ultralytics format:

```yaml
path: .          # relative to this file's location once extracted
train: images/train
val: images/val
names:
  0: class_one
  1: class_two
```

The app scans the extracted zip for the first `.yaml`/`.yml` file it finds
and uses that as the data config.

## Usage

1. Open the app and, in the sidebar, upload your dataset `.zip` and click
   **Extract Dataset**.
2. Pick a model family, size, and task — this determines the weight/config
   file used (e.g. `yolov8s.pt`, `yolo11n-seg.pt`).
3. Set epochs, batch size, image size, and any other hyperparameters you need.
4. Click **Start Training**. The console panel streams live output from the
   underlying `yolo train` command.
5. When training finishes, download `best.pt` and view the results plot
   directly in the app.

## Configuration

`.streamlit/config.toml` controls upload limits:

```toml
[server]
maxUploadSize = 51200    # MB — 50GB, effectively unlimited for localhost use
maxMessageSize = 51200   # must be raised alongside maxUploadSize
```

Both values need to move together — `maxUploadSize` alone isn't enough,
since uploads ride over the websocket connection capped by `maxMessageSize`.

If you move this app off localhost (e.g. onto a shared server accessed over
a network), consider lowering these back down and testing actual upload
behavior, since very large uploads over a real network can hit timeouts
regardless of the configured limit.

## Project structure

```
.
├── app.py                  # Streamlit app
├── requirements.txt
└── .streamlit/
    └── config.toml         # upload size limits
```

## Notes

- Training runs via the `yolo` CLI (installed with `ultralytics`) as a
  subprocess, so any `yolo train` argument can be added to the command list
  in `app.py` if you need a parameter that isn't exposed in the sidebar yet.
- Runs are written to `yolo_workspace/runs/<run name>/`, matching Ultralytics'
  normal output layout (`weights/best.pt`, `weights/last.pt`, `results.png`,
  confusion matrix, etc.).