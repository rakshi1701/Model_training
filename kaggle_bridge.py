"""
Kaggle Bridge Module for YOLO Vision Studio
Handles:
- Authentication & Credentials Management (~/.kaggle/kaggle.json & ~/.config/kaggle/kaggle.json)
- Dataset packaging and remote upload/versioning via Kaggle Dataset API (with dir_mode='zip' support)
- Dynamic training script generation (PyTorch DDP, Dual T4 GPU support, Ultralytics YOLO)
- Remote kernel dispatching and real-time status polling
- Automatic artifact ingestion (best.pt, results.csv, confusion matrix) into yolo_workspace/runs/
"""

import os
import sys
import json
import shutil
import zipfile
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

KAGGLE_DIR = Path.home() / ".kaggle"
KAGGLE_CONFIG_DIR = Path.home() / ".config" / "kaggle"
KAGGLE_JSON = KAGGLE_DIR / "kaggle.json"
KAGGLE_CONFIG_JSON = KAGGLE_CONFIG_DIR / "kaggle.json"
KAGGLE_ACCESS_TOKEN = KAGGLE_DIR / "access_token"
KAGGLE_CONFIG_ACCESS_TOKEN = KAGGLE_CONFIG_DIR / "access_token"

WORKSPACE_DIR = Path(__file__).parent / "yolo_workspace"
RUNS_DIR = WORKSPACE_DIR / "runs"
KAGGLE_STAGING_DIR = WORKSPACE_DIR / "temp" / "kaggle_staging"


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


def sanitize_username(username: str) -> str:
    """If username is an email (user@gmail.com), extracts the username handle portion."""
    if not username:
        return ""
    username = username.strip()
    if "@" in username:
        username = username.split("@")[0]
    return username


def is_bearer_token(key: str) -> bool:
    """New-style Kaggle API tokens are opaque strings prefixed with 'KGAT_'.

    These only authenticate via 'Authorization: Bearer <token>', which the kaggle
    client uses only when the KAGGLE_API_TOKEN env var is set. Legacy keys are a
    bare 32-char hex string and use HTTP Basic auth (username + key).
    """
    return bool(key) and key.strip().lower().startswith("kgat_")


def _apply_auth_env(username: str, key: str):
    """Exports the right env vars so the kaggle client picks the correct auth mode.

    kagglesdk.KaggleHttpClient checks KAGGLE_API_TOKEN first (Bearer), then falls
    back to KAGGLE_USERNAME/KAGGLE_KEY (Basic). For KGAT_ tokens we must set
    KAGGLE_API_TOKEN or every write call 401s on blobs/upload.
    """
    if username:
        os.environ["KAGGLE_USERNAME"] = username
    if key:
        os.environ["KAGGLE_KEY"] = key
    if is_bearer_token(key):
        os.environ["KAGGLE_API_TOKEN"] = key.strip()
    else:
        os.environ.pop("KAGGLE_API_TOKEN", None)


def _read_username_hint() -> str:
    """Best-effort Kaggle handle from kaggle.json or env (not required for auth)."""
    for p in [KAGGLE_JSON, KAGGLE_CONFIG_JSON]:
        if p.exists():
            try:
                with open(p, "r") as f:
                    u = sanitize_username(json.load(f).get("username", ""))
                if u:
                    return u
            except Exception:
                pass
    return sanitize_username(os.environ.get("KAGGLE_USERNAME", ""))


def _load_stored_credentials() -> Optional[Tuple[str, str]]:
    """Reads (username, key) from env, the access_token file, or kaggle.json.

    'key' may be a legacy hex key or a new-style 'KGAT_' bearer token. The
    dedicated token mechanism (KAGGLE_API_TOKEN env / access_token file) is the
    modern one and takes precedence over a possibly-stale kaggle.json 'key', so a
    half-migrated setup still authenticates. The access_token file holds only the
    token, so the username is sourced separately.
    """
    username_hint = _read_username_hint()

    # 1. Explicit env override.
    env_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_token:
        return username_hint, env_token

    env_u = sanitize_username(os.environ.get("KAGGLE_USERNAME", ""))
    env_k = os.environ.get("KAGGLE_KEY", "").strip()
    if env_u and env_k:
        return env_u, env_k

    # 2. Dedicated token file.
    for p in [KAGGLE_ACCESS_TOKEN, KAGGLE_CONFIG_ACCESS_TOKEN]:
        if p.exists():
            try:
                tok = p.read_text().strip()
                if tok:
                    return username_hint, tok
            except Exception:
                pass

    # 3. Legacy kaggle.json username + key.
    for p in [KAGGLE_JSON, KAGGLE_CONFIG_JSON]:
        if p.exists():
            try:
                with open(p, "r") as f:
                    creds = json.load(f)
                u = sanitize_username(creds.get("username", ""))
                k = creds.get("key", "").strip()
                if u and k:
                    return u, k
            except Exception:
                pass

    return None


def get_kaggle_api():
    """Initializes and returns an authenticated KaggleApi client, or None."""
    creds = _load_stored_credentials()
    if not creds:
        return None

    u, k = creds
    _apply_auth_env(u, k)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception:
        return None


def _verify_api(api) -> Optional[str]:
    """Makes one cheap authenticated call. Returns None if OK, else an error string.

    api.authenticate() never contacts the server, so a revoked/expired token or a
    wrong auth mode only surfaces on the first real request (e.g. a 401 on
    blobs/upload during upload). This catches it up front.
    """
    try:
        api.competitions_list(page=1)
        return None
    except Exception as e:
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg or "Unauthenticated" in msg:
            return ("Kaggle rejected the token (401). Create a new token at "
                    "kaggle.com/settings > API and re-enter it.")
        return msg


def is_authenticated() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Checks if Kaggle credentials exist and are valid.
    Returns: (is_valid, username, error_message)
    """
    creds = _load_stored_credentials()
    if not creds:
        return False, None, "No credentials found. Please configure username and API key."

    username, key = creds
    try:
        api = get_kaggle_api()
        if api is None:
            return False, username, "Credentials present, but Kaggle API authentication failed."
        err = _verify_api(api)
        if err:
            return False, username, err
        return True, username, None
    except Exception as e:
        return False, username, str(e)


def save_credentials(username: str, key: str) -> Tuple[bool, str]:
    """
    Saves username and API key to ~/.kaggle/kaggle.json and ~/.config/kaggle/kaggle.json with 0600 permissions.
    """
    username = sanitize_username(username)
    key = key.strip()
    if not username or not key:
        return False, "Username and API key cannot be empty."

    try:
        data = {"username": username, "key": key}

        # Save to ~/.kaggle/
        KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
        os.chmod(KAGGLE_DIR, 0o700)
        with open(KAGGLE_JSON, "w") as f:
            json.dump(data, f, indent=2)
        os.chmod(KAGGLE_JSON, 0o600)

        # Save to ~/.config/kaggle/
        KAGGLE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        os.chmod(KAGGLE_CONFIG_DIR, 0o700)
        with open(KAGGLE_CONFIG_JSON, "w") as f:
            json.dump(data, f, indent=2)
        os.chmod(KAGGLE_CONFIG_JSON, 0o600)

        # New-style 'KGAT_' tokens are also written to the access_token file the
        # kaggle client reads, and drive Bearer auth via KAGGLE_API_TOKEN.
        if is_bearer_token(key):
            for tok_path in [KAGGLE_ACCESS_TOKEN, KAGGLE_CONFIG_ACCESS_TOKEN]:
                with open(tok_path, "w") as f:
                    f.write(key)
                os.chmod(tok_path, 0o600)
        else:
            for tok_path in [KAGGLE_ACCESS_TOKEN, KAGGLE_CONFIG_ACCESS_TOKEN]:
                if tok_path.exists():
                    tok_path.unlink()

        _apply_auth_env(username, key)

        # Verify authentication with a real authenticated request.
        api = get_kaggle_api()
        if api is None:
            return False, "Credentials saved, but authentication failed. Please double check your API token."
        err = _verify_api(api)
        if err:
            return False, f"Credentials saved, but Kaggle rejected them: {err}"
        return True, f"Successfully authenticated with Kaggle as @{username}!"
    except Exception as e:
        return False, f"Failed to save credentials: {str(e)}"


def format_dataset_slug(name: str) -> str:
    """Generates a clean, valid Kaggle dataset slug (lowercase alphanumeric with hyphens)."""
    slug = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "yolo-dataset"


def package_and_upload_dataset(
    dataset_path: Path,
    dataset_title: str,
    api=None,
    progress_callback=None
) -> Tuple[bool, str, Optional[str]]:
    """
    Validates a local YOLO dataset directory, creates dataset-metadata.json,
    and uploads or versions it on Kaggle using dir_mode='zip' to preserve subfolders.
    Returns: (success, message, dataset_ref)
    """
    if api is None:
        api = get_kaggle_api()
        if api is None:
            return False, "Kaggle API not authenticated.", None

    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        return False, f"Dataset path {dataset_path} does not exist.", None

    yaml_file = None
    for yf in ["data.yaml", "dataset.yaml", "data.yml"]:
        candidate = dataset_path / yf
        if candidate.exists():
            yaml_file = candidate
            break

    if not yaml_file:
        return False, "No data.yaml found in dataset directory.", None

    auth_ok, username, _ = is_authenticated()
    if not username:
        return False, "Could not determine Kaggle username.", None

    dataset_slug = format_dataset_slug(dataset_title)
    dataset_ref = f"{username}/{dataset_slug}"

    staging_dir = KAGGLE_STAGING_DIR / "dataset" / dataset_slug
    safe_rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Staging dataset files for Kaggle...")

    # Copy files to staging
    for item in dataset_path.iterdir():
        if item.name.startswith("."):
            continue
        dest = staging_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # Generate dataset-metadata.json
    metadata = {
        "title": dataset_title,
        "id": dataset_ref,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(staging_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if progress_callback:
        progress_callback("Uploading dataset to Kaggle (zipping folders)...")

    try:
        existing_datasets = [str(d) for d in api.dataset_list(user=username)]
        dataset_exists = any(dataset_ref.lower() in d.lower() for d in existing_datasets)

        if dataset_exists:
            api.dataset_create_version(
                folder=str(staging_dir),
                version_notes=f"Updated from YOLO Studio at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                quiet=False,
                dir_mode="zip"
            )
            return True, f"Dataset '{dataset_ref}' updated with new version.", dataset_ref
        else:
            api.dataset_create_new(
                folder=str(staging_dir),
                public=False,
                quiet=False,
                dir_mode="zip"
            )
            return True, f"Private dataset '{dataset_ref}' successfully created on Kaggle.", dataset_ref
    except Exception as e:
        err_msg = str(e)
        if "already exists" in err_msg.lower() or "duplicate" in err_msg.lower():
            try:
                api.dataset_create_version(
                    folder=str(staging_dir),
                    version_notes="Updated from YOLO Studio",
                    quiet=False,
                    dir_mode="zip"
                )
                return True, f"Dataset '{dataset_ref}' updated with a new version.", dataset_ref
            except Exception as e2:
                return False, f"Dataset upload failed: {str(e2)}", None
        return False, f"Dataset upload error: {err_msg}", None


def generate_remote_training_script(
    model_name: str,
    dataset_slug: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    optimizer: str,
    lr0: float,
    patience: int,
    enable_dual_gpu: bool = True
) -> str:
    """
    Generates a bulletproof Python script to run headless training on Kaggle GPUs.
    Handles dual-GPU DDP, automatic path resolution, and artifact packaging.
    """
    return f'''"""
Auto-generated Remote YOLO Training Job for Kaggle Cloud GPUs
Triggered by YOLO Vision Studio
"""
import os
import sys
import shutil
import glob
import subprocess
from pathlib import Path

# 1. Install & Verify Ultralytics
print("=" * 60)
print("🚀 Initializing YOLO Training Environment on Kaggle GPU")
print("=" * 60)

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics", "pyyaml"])

import torch
import yaml
from ultralytics import YOLO

cuda_avail = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if cuda_avail else 0
print(f"🖥️ CUDA Available: {{cuda_avail}} | GPU Count: {{gpu_count}}")
if gpu_count > 0:
    for i in range(gpu_count):
        print(f"  - GPU {{i}}: {{torch.cuda.get_device_name(i)}}")

# Determine device config
if {enable_dual_gpu} and gpu_count >= 2:
    devices = [0, 1]
    print(f"⚡ Utilizing Dual-GPU DDP Mode: {{devices}}")
elif gpu_count >= 1:
    devices = 0
    print(f"⚡ Utilizing Single-GPU Mode: device=0")
else:
    devices = "cpu"
    print("⚠️ No GPU detected. Falling back to CPU.")

# 2. Locate Dataset
input_root = Path("/kaggle/input")
dataset_dir = None

# Search for data.yaml in /kaggle/input
for root, dirs, files in os.walk(input_root):
    if "data.yaml" in files or "dataset.yaml" in files:
        dataset_dir = Path(root)
        break

if not dataset_dir:
    raise FileNotFoundError(f"Could not locate data.yaml under {{input_root}}")

yaml_file = dataset_dir / ("data.yaml" if (dataset_dir / "data.yaml").exists() else "dataset.yaml")
print(f"📁 Located Dataset YAML: {{yaml_file}}")

# Read and fix paths in data.yaml for Kaggle runtime
with open(yaml_file, "r") as f:
    data_cfg = yaml.safe_load(f)

# Ensure absolute paths pointing to /kaggle/input
data_cfg["path"] = str(dataset_dir)
patched_yaml = Path("/kaggle/working/data_kaggle.yaml")
with open(patched_yaml, "w") as f:
    yaml.dump(data_cfg, f, default_flow_style=False)

print(f"✅ Patched Kaggle Data Config written to {{patched_yaml}}")

# 3. Model Training
project_dir = Path("/kaggle/working/yolo_runs")
exp_name = "train_output"

print("=" * 60)
print("🏋️ Starting Training: Model={model_name} | Epochs={epochs} | Batch={batch_size} | ImgSz={imgsz}")
print("=" * 60)

model = YOLO("{model_name}")

try:
    results = model.train(
        data=str(patched_yaml),
        epochs={epochs},
        batch={batch_size},
        imgsz={imgsz},
        optimizer="{optimizer}",
        lr0={lr0},
        patience={patience},
        device=devices,
        project=str(project_dir),
        name=exp_name,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True
    )
    print("🎉 Training Completed Successfully!")
except Exception as e:
    print(f"❌ Training Failed: {{e}}")
    raise e

# 4. Package Artifacts for 1-Click Download
run_output_dir = project_dir / exp_name
output_archive_dir = Path("/kaggle/working/output")
output_archive_dir.mkdir(parents=True, exist_ok=True)

if run_output_dir.exists():
    print("📦 Packaging weights, metrics, and plots for local synchronization...")
    for item in run_output_dir.iterdir():
        dest = output_archive_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    print("✅ All artifacts staged in /kaggle/working/output/")
else:
    print("⚠️ Run directory not found. Archiving entire project folder.")
    shutil.copytree(project_dir, output_archive_dir, dirs_exist_ok=True)

print("=" * 60)
print("🏁 Kaggle Job Completed. Ready for local download!")
print("=" * 60)
'''


def dispatch_kaggle_training(
    dataset_ref: str,
    kernel_title: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    optimizer: str,
    lr0: float,
    patience: int,
    enable_dual_gpu: bool = True,
    api=None
) -> Tuple[bool, str, Optional[str]]:
    """
    Generates remote training script, kernel-metadata.json, and pushes kernel to Kaggle.
    Returns: (success, message, kernel_ref)
    """
    if api is None:
        api = get_kaggle_api()
        if api is None:
            return False, "Kaggle API authentication failed. Please configure API token.", None

    auth_ok, username, _ = is_authenticated()
    if not username:
        return False, "Unable to get Kaggle username.", None

    kernel_slug = format_dataset_slug(kernel_title)
    kernel_ref = f"{username}/{kernel_slug}"

    staging_dir = KAGGLE_STAGING_DIR / "kernel" / kernel_slug
    safe_rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate train_remote.py
    script_content = generate_remote_training_script(
        model_name=model_name,
        dataset_slug=dataset_ref.split("/")[-1],
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        optimizer=optimizer,
        lr0=lr0,
        patience=patience,
        enable_dual_gpu=enable_dual_gpu
    )
    script_path = staging_dir / "train_remote.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    # 2. Generate kernel-metadata.json
    metadata = {
        "id": kernel_ref,
        "title": kernel_title,
        "code_file": "train_remote.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [dataset_ref]
    }
    with open(staging_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 3. Push kernel (kernels_push is the library call; kernels_push_cli is a
    # thin CLI wrapper whose 'timeout' arg has no default in kaggle >=1.7).
    try:
        api.kernels_push(folder=str(staging_dir), timeout=None)
        return True, f"Training job successfully dispatched to Kaggle GPU cluster! (Kernel: {kernel_ref})", kernel_ref
    except Exception as e:
        return False, f"Failed to push kernel to Kaggle: {str(e)}", None


def get_kernel_status(kernel_ref: str, api=None) -> Dict[str, Any]:
    """
    Fetches the current runtime status of a Kaggle kernel.
    Returns dictionary with: {'status': str, 'error': Optional[str], 'url': str}
    """
    if api is None:
        api = get_kaggle_api()
        if api is None:
            return {"status": "error", "message": "Kaggle API not authenticated."}

    try:
        # kaggle >=1.7: method is kernels_status (plural) and returns an object
        # with .status / .failure_message, not a dict.
        result = api.kernels_status(kernel_ref)
        if isinstance(result, dict):
            raw_status = result.get("status", "unknown")
            failure_msg = result.get("failureMessage") or result.get("failure_message")
        else:
            raw_status = getattr(result, "status", "unknown")
            failure_msg = getattr(result, "failure_message", None)
        # status may be an enum (e.g. KernelWorkerStatus.COMPLETE) or a string.
        status = str(getattr(raw_status, "name", raw_status)).split(".")[-1].lower()
        kernel_url = f"https://www.kaggle.com/code/{kernel_ref}"
        return {
            "status": status,
            "failureMessage": failure_msg,
            "url": kernel_url
        }
    except Exception as e:
        msg = str(e)
        # kaggle 1.7.4.x's /kernels/status endpoint 404s for kernels without an
        # active session (confirmed: the `kaggle kernels status` CLI 404s too).
        # Don't surface that as a run failure — the job may still be fine.
        if "404" in msg or "Not Found" in msg:
            return {
                "status": "unknown",
                "message": "Status polling unavailable (Kaggle API). Open the web link to check progress, then use Ingest Checkpoints once it finishes.",
                "url": f"https://www.kaggle.com/code/{kernel_ref}",
            }
        return {
            "status": "error",
            "message": msg,
            "url": f"https://www.kaggle.com/code/{kernel_ref}"
        }


def download_and_ingest_artifacts(
    kernel_ref: str,
    target_exp_name: str,
    api=None
) -> Tuple[bool, str, Optional[Path]]:
    """
    Downloads the output files of a completed Kaggle kernel and extracts them
    directly into yolo_workspace/runs/<target_exp_name>/.
    """
    if api is None:
        api = get_kaggle_api()
        if api is None:
            return False, "Kaggle API not authenticated.", None

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    target_run_dir = RUNS_DIR / target_exp_name
    target_run_dir.mkdir(parents=True, exist_ok=True)

    temp_download_dir = KAGGLE_STAGING_DIR / "downloads" / target_exp_name
    safe_rmtree(temp_download_dir)
    temp_download_dir.mkdir(parents=True, exist_ok=True)

    try:
        api.kernels_output_cli(kernel_ref, path=str(temp_download_dir))

        output_sub = temp_download_dir / "output"
        source_dir = output_sub if output_sub.exists() and any(output_sub.iterdir()) else temp_download_dir

        for item in source_dir.iterdir():
            dest = target_run_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

        has_weights = (target_run_dir / "weights" / "best.pt").exists() or (target_run_dir / "best.pt").exists()
        has_metrics = (target_run_dir / "results.csv").exists()

        status_detail = []
        if has_weights:
            status_detail.append("🎯 Model Checkpoint (best.pt)")
        if has_metrics:
            status_detail.append("📊 Metrics (results.csv)")

        msg = f"Artifacts successfully downloaded to {target_run_dir}!"
        if status_detail:
            msg += f" Found: {', '.join(status_detail)}."

        return True, msg, target_run_dir
    except Exception as e:
        return False, f"Failed to download kernel output: {str(e)}", None


def list_recent_jobs_history() -> List[Dict[str, Any]]:
    """Loads history of Kaggle training jobs from presets / local store."""
    history_file = WORKSPACE_DIR / "kaggle_jobs.json"
    if not history_file.exists():
        return []
    try:
        with open(history_file, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_job_to_history(job_info: Dict[str, Any]):
    """Appends or updates a job in local history."""
    history_file = WORKSPACE_DIR / "kaggle_jobs.json"
    jobs = list_recent_jobs_history()
    updated = False
    for i, j in enumerate(jobs):
        if j.get("kernel_ref") == job_info.get("kernel_ref"):
            jobs[i] = job_info
            updated = True
            break
    if not updated:
        jobs.insert(0, job_info)

    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, "w") as f:
        json.dump(jobs[:25], f, indent=2)
