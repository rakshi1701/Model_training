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
import re
import json
import shutil
import zipfile
import tarfile
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
JOBS_HISTORY_FILE = WORKSPACE_DIR / "kaggle_jobs.json"
DATASET_MAP_FILE = WORKSPACE_DIR / "kaggle_dataset_map.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
# Files that must never be shipped to Kaggle inside the dataset.
STAGE_IGNORE = shutil.ignore_patterns(
    "*.cache", ".*", "__pycache__", "*.pyc", "runs", "wandb", "*.zip",
)


def _load_json(path: Path, default):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _save_json(path: Path, obj):
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=2))
    except Exception:
        pass


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


def _resp_error(resp) -> Optional[str]:
    """dataset_create_* / kernels_push return a response object instead of raising
    on server-side errors (title clash, permission denied, invalid owner, ...)."""
    if resp is None:
        return None
    status = str(getattr(resp, "status", "") or "")
    err = str(getattr(resp, "error", "") or getattr(resp, "errorNullable", "") or "")
    if err:
        return err
    if status and status.lower() not in ("ok", "created", "success", "complete"):
        return f"Kaggle returned status '{status}'."
    return None


def _owned_dataset_refs(api, username: str) -> List[str]:
    """Lowercase 'owner/slug' refs of datasets this account owns."""
    refs: List[str] = []
    if api is None:
        return refs
    for kwargs in ({"mine": True}, {"user": username}):
        try:
            ds_list = api.dataset_list(**kwargs)
            for d in ds_list:
                ref = getattr(d, "ref", None)
                if not ref:
                    s = str(d)
                    if '"ref": "' in s:
                        ref = s.split('"ref": "')[1].split('"')[0]
                    elif "/" in s:
                        ref = s.strip()
                if ref:
                    refs.append(str(ref).strip("/").lower())
            if refs:
                break
        except Exception:
            continue
    return refs


def _count_images(root: Path) -> int:
    return sum(1 for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _dataset_fingerprint(root: Path) -> str:
    """Cheap change-detector: image count + newest mtime across images & yaml."""
    root = Path(root)
    n, newest = 0, 0.0
    for p in root.rglob("*"):
        sfx = p.suffix.lower()
        if sfx in IMAGE_EXTS or sfx in (".yaml", ".yml", ".txt"):
            try:
                newest = max(newest, p.stat().st_mtime)
                if sfx in IMAGE_EXTS:
                    n += 1
            except OSError:
                pass
    return f"{n}:{int(newest)}"


_SKIP_DIRS = {"runs", "wandb", "__pycache__", ".git"}
_SKIP_SUFFIX = {".cache", ".zip"}


def _dataset_status(ref: str, api=None) -> str:
    """Returns 'ready', 'error', 'initializing', or raw progress state like 'blobs_decompressed'."""
    if api is None:
        api = get_kaggle_api()
        if api is None:
            return "?"
    try:
        raw = api.dataset_status(ref)
        st = str(raw).strip().lower()
        if st in ("ready", "complete", "success"):
            return "ready"
        if st in ("failed", "error", "deleted"):
            return "error"
        return st
    except Exception as e:
        msg = str(e)
        if "403" in msg or "404" in msg:
            # During initial creation or replica sync, Kaggle's status endpoint
            # returns 403 or 404 until the record is indexed across replicas.
            return "initializing"
        return "?"


def _collect_dataset_files(dataset_path: Path) -> List[Path]:
    out: List[Path] = []
    for p in dataset_path.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(dataset_path)
        if (any(part.startswith(".") or part in _SKIP_DIRS for part in rel.parts)
                or p.name == "dataset-metadata.json"
                or p.suffix.lower() in _SKIP_SUFFIX):
            continue
        out.append(p)
    return out


def _stage_clean_dataset(staging_dir: Path, dataset_path: Path, progress_callback=None):
    """Packages dataset into a fast zip archive (dataset.zip) and root data.yaml.
    Uploading an archive with dir_mode='skip' prevents Kaggle's server-side decompressor
    from stalling on thousands of individual image files, enabling immediate 'ready' status."""
    safe_rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    yaml_file = None
    for yf in ("data.yaml", "dataset.yaml", "data.yml"):
        if (dataset_path / yf).exists():
            yaml_file = dataset_path / yf
            break

    import yaml
    if yaml_file:
        try:
            with open(yaml_file, "r") as f:
                cfg = yaml.safe_load(f) or {}
            # Normalize relative paths like ../train/images -> train/images
            for key in ["train", "val", "valid", "test"]:
                val = cfg.get(key)
                if isinstance(val, str):
                    cfg[key] = val.replace("../", "").strip("/")
            with open(staging_dir / "data.yaml", "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
        except Exception:
            shutil.copy2(yaml_file, staging_dir / "data.yaml")

    # Fast archive creation using uncompressed tar (linear speed, zero compression CPU overhead)
    # Kaggle does not decompress .tar archives server-side, bypassing the 1,000-file limit
    # and allowing the dataset to be verified as 'ready' on Kaggle in seconds.
    t0 = time.time()
    tar_path = staging_dir / "dataset.tar"
    with tarfile.open(tar_path, "w") as tf:
        for item in dataset_path.iterdir():
            if item.name.startswith(".") or item.name in _SKIP_DIRS:
                continue
            if item.is_file():
                if item.suffix.lower() not in _SKIP_SUFFIX and item.name != "dataset-metadata.json":
                    tf.add(item, arcname=item.name)
            elif item.is_dir():
                for sub in item.rglob("*"):
                    if sub.is_file():
                        if sub.suffix.lower() in _SKIP_SUFFIX or any(p.startswith(".") for p in sub.parts):
                            continue
                        tf.add(sub, arcname=sub.relative_to(dataset_path).as_posix())

    sz_mb = tar_path.stat().st_size / (1024 * 1024)
    if progress_callback:
        progress_callback(f"Packaged {sz_mb:.1f} MB archive (dataset.tar) in {time.time()-t0:.1f}s.")


def package_and_upload_dataset(
    dataset_path: Path,
    dataset_title: str,
    api=None,
    progress_callback=None,
) -> Tuple[bool, str, Optional[List[str]]]:
    """
    Uploads a local YOLO dataset to Kaggle as a native dataset archive with immediate verification.
    Reuses existing dataset if unchanged (by fingerprint).
    Returns: (success, message, [dataset_ref]).
    """
    if api is None:
        api = get_kaggle_api()
        if api is None:
            return False, "Kaggle API not authenticated.", None

    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        return False, f"Dataset path {dataset_path} does not exist.", None

    if not any((dataset_path / y).exists() for y in ("data.yaml", "dataset.yaml", "data.yml")):
        return False, f"No data.yaml in {dataset_path}.", None

    files = _collect_dataset_files(dataset_path)
    images = [f for f in files if f.suffix.lower() in IMAGE_EXTS]
    if not images:
        return False, f"No images found under {dataset_path}.", None
    total_bytes = sum(f.stat().st_size for f in files)
    size_mb = total_bytes // (1024 * 1024)

    username = _read_username_hint()
    if not username:
        _, username, err = is_authenticated()
        if not username:
            return False, err or "Could not determine Kaggle username.", None

    base_slug = format_dataset_slug(dataset_title)
    fingerprint = _dataset_fingerprint(dataset_path)
    ds_map = _load_json(DATASET_MAP_FILE, {})
    entry = ds_map.get(base_slug) or {}

    cached_ref = entry.get("ref") or (entry.get("parts", {}).get("0") if isinstance(entry.get("parts"), dict) else None)
    if entry.get("fingerprint") == fingerprint and cached_ref:
        if _dataset_status(cached_ref, api) == "ready":
            return True, f"Reusing Kaggle dataset '{cached_ref}' (unchanged, {len(images)} images).", [cached_ref]

    owned = set(_owned_dataset_refs(api, username))
    target_ref = f"{username}/{base_slug}"

    # If the user already owns this exact slug, we update its version.
    # Otherwise, append a unique timestamp to prevent global Kaggle title/slug clashes.
    if target_ref.lower() in owned:
        dataset_ref = target_ref
        dataset_title_clean = dataset_title
        already_exists = True
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dataset_ref = f"{username}/{base_slug}-{stamp}"
        dataset_title_clean = f"{dataset_title} ({stamp})"
        already_exists = False

    slug_part = dataset_ref.split("/")[-1]
    staging_dir = KAGGLE_STAGING_DIR / "dataset" / slug_part

    if progress_callback:
        progress_callback(f"Staging {len(images)} images ({size_mb} MB) into fast dataset archive...")

    _stage_clean_dataset(staging_dir, dataset_path, progress_callback)
    _save_json(
        staging_dir / "dataset-metadata.json",
        {"title": dataset_title_clean[:50], "id": dataset_ref, "licenses": [{"name": "CC0-1.0"}]}
    )

    if progress_callback:
        action_name = "Updating version of existing" if already_exists else "Uploading new"
        progress_callback(f"{action_name} dataset '{dataset_ref}' on Kaggle...")

    upload_err = None
    if already_exists:
        try:
            resp = api.dataset_create_version(
                folder=str(staging_dir),
                version_notes=f"Updated from YOLO Studio at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                quiet=True,
                dir_mode="skip"
            )
            upload_err = _resp_error(resp)
        except Exception as e:
            upload_err = str(e)
    else:
        try:
            resp = api.dataset_create_new(
                folder=str(staging_dir),
                public=False,
                quiet=True,
                dir_mode="skip"
            )
            upload_err = _resp_error(resp)
        except Exception as e:
            upload_err = str(e)

        if upload_err and any(phrase in upload_err.lower() for phrase in ["already in use", "duplicate", "already exists"]):
            if progress_callback:
                progress_callback(f"Title in use on Kaggle; re-trying with unique title...")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            dataset_ref = f"{username}/{base_slug}-{stamp}"
            dataset_title_clean = f"{dataset_title} ({stamp})"
            _save_json(
                staging_dir / "dataset-metadata.json",
                {"title": dataset_title_clean[:50], "id": dataset_ref, "licenses": [{"name": "CC0-1.0"}]}
            )
            try:
                resp = api.dataset_create_new(
                    folder=str(staging_dir),
                    public=False,
                    quiet=True,
                    dir_mode="skip"
                )
                upload_err = _resp_error(resp)
            except Exception as e2:
                upload_err = str(e2)

    if upload_err:
        return False, f"Failed to upload dataset to Kaggle: {upload_err}", None

    if progress_callback:
        progress_callback(f"Upload complete. Verifying Kaggle dataset '{dataset_ref}'...")

    ready = False
    for poll in range(20):
        time.sleep(4)
        st = _dataset_status(dataset_ref, api)
        if st == "ready":
            ready = True
            break
        elif st == "error":
            return False, f"Kaggle marked dataset '{dataset_ref}' as failed.", None
        else:
            if progress_callback:
                status_desc = "initializing/syncing" if st == "initializing" else st
                progress_callback(f"  ➜ Kaggle processing state: {status_desc} (elapsed: {(poll+1)*4}s)...")

    if not ready:
        return False, f"Dataset '{dataset_ref}' upload completed, but verification timed out. Check: https://www.kaggle.com/datasets/{dataset_ref}", None

    ds_map[base_slug] = {
        "ref": dataset_ref,
        "parts": {"0": dataset_ref},
        "fingerprint": fingerprint,
        "n_parts": 1
    }
    _save_json(DATASET_MAP_FILE, ds_map)

    if progress_callback:
        progress_callback(f"Dataset '{dataset_ref}' is ready on Kaggle!")

    return True, f"Dataset '{dataset_ref}' uploaded and verified ready on Kaggle ({len(images)} images).", [dataset_ref]


def generate_remote_training_script(
    model_name: str,
    dataset_slug: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    optimizer: str,
    lr0: float,
    patience: int,
    enable_dual_gpu: bool = True,
    max_hours: float = 11.0,
    resume: bool = False,
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

# --- GPU / PyTorch ABI guard -------------------------------------------------
# Kaggle's current base image ships a CUDA 12.8 PyTorch build that dropped
# support for older GPUs (Tesla P100 = compute capability sm_60). API-pushed
# GPU kernels are frequently scheduled onto a P100, where that torch silently
# refuses the device. If we detect a P100, install a torch build that still
# supports sm_60 BEFORE importing torch.
try:
    _gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30
    ).stdout.strip()
except Exception:
    _gpu_name = ""
print(f"🖥️ Detected accelerator: {{_gpu_name or 'none / CPU'}}")

if "P100" in _gpu_name:
    print("🔧 P100 detected — installing a P100-compatible PyTorch (cu121)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "torch==2.4.1", "torchvision==0.19.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics", "pyyaml"])

import torch
import yaml
from ultralytics import YOLO

# Verify torch can actually use the assigned GPU; fall back loudly if not.
if torch.cuda.is_available():
    _cap = torch.cuda.get_device_capability(0)
    _tag = f"sm_{{_cap[0]}}{{_cap[1]}}"
    _arch = torch.cuda.get_arch_list()
    print(f"🔩 torch {{torch.__version__}} | device {{_tag}} | supported {{_arch}}")
    if _tag not in _arch:
        print(f"⚠️ torch build does not list {{_tag}} — training may fall back to CPU.")

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

# 1b. GPU telemetry sampler.
# Emits machine-readable [GPUSTAT] lines the local dashboard parses out of the
# run log, so utilisation/VRAM/temp are visible without opening the Kaggle UI.
import threading
import time as _time

JOB_T0 = _time.time()
MAX_SECONDS = {max_hours} * 3600.0


def _gpu_sampler(interval=30):
    query = "index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    while True:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={{query}}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
            for line in out.splitlines():
                if line.strip():
                    elapsed = int(_time.time() - JOB_T0)
                    print(f"[GPUSTAT] t={{elapsed}} {{line.strip()}}", flush=True)
        except Exception:
            pass
        _time.sleep(interval)


if gpu_count > 0:
    threading.Thread(target=_gpu_sampler, daemon=True).start()
    print(f"📈 GPU telemetry sampler started (every 30s). Runtime cap: {max_hours}h.")

# 2. Locate Dataset
import zipfile
input_root = Path("/kaggle/input")
dataset_dir = None

print("📂 Contents of /kaggle/input:")
_listing = sorted(str(p) for p in input_root.rglob("*"))
for _p in _listing[:60]:
    print("   ", _p)
if not _listing:
    print("    <empty> — no dataset is attached to this kernel!")

def _find_yaml_dirs(base):
    hits = []
    for root, dirs, files in os.walk(base):
        if "data.yaml" in files or "dataset.yaml" in files:
            hits.append(Path(root))
    return hits

# 2. Locate Dataset (support raw directories, tar archives, and zip archives)
import tarfile
all_archives = sorted(list(input_root.rglob("*.tar")) + list(input_root.rglob("*.tar.gz")) + list(input_root.rglob("*.zip")))
if all_archives:
    print(f"📦 Found {{len(all_archives)}} dataset archive(s). Unpacking into /kaggle/working/_extracted...")
    extract_root = Path("/kaggle/working/_extracted")
    extract_root.mkdir(parents=True, exist_ok=True)
    for a in all_archives:
        print(f"   Extracting {{a.name}} ({{a.stat().st_size / (1024*1024):.1f}} MB)...")
        if a.name.endswith(".zip"):
            with zipfile.ZipFile(a) as zf:
                zf.extractall(extract_root)
        else:
            with tarfile.open(a) as tf:
                tf.extractall(extract_root)
    yaml_dirs = _find_yaml_dirs(extract_root)
else:
    yaml_dirs = _find_yaml_dirs(input_root)

if not yaml_dirs:
    raise FileNotFoundError(
        f"Could not locate data.yaml under {{input_root}}. "
        f"No dataset attached / archives did not extract. Check the Input panel."
    )

print(f"📁 Dataset mount(s) with data.yaml: {{[str(d) for d in yaml_dirs]}}")
primary = yaml_dirs[0]
yaml_file = primary / ("data.yaml" if (primary / "data.yaml").exists() else "dataset.yaml")
with open(yaml_file, "r") as f:
    data_cfg = yaml.safe_load(f)

# Merge every mount's train/valid/test into one writable tree via symlinks.
merged_root = Path("/kaggle/working/dataset")
safe_split = {{"train": "train", "val": "valid", "test": "test"}}
counts = {{}}
for key, folder in safe_split.items():
    for sub in ("images", "labels"):
        (merged_root / folder / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for mount in yaml_dirs:
        for cand in (mount / folder, mount / key):
            if not cand.is_dir():
                continue
            for sub in ("images", "labels"):
                srcd = cand / sub
                if not srcd.is_dir():
                    continue
                for fp in srcd.iterdir():
                    if not fp.is_file() or fp.suffix.lower() == ".cache":
                        continue
                    dst = merged_root / folder / sub / fp.name
                    if not dst.exists():
                        try:
                            os.symlink(fp, dst)
                        except OSError:
                            shutil.copy2(fp, dst)
                        if sub == "images":
                            n += 1
    counts[key] = n
    if n:
        data_cfg[key] = str(merged_root / folder / "images")
    elif key in ("train", "val"):
        raise FileNotFoundError(f"No '{{folder}}' images found across any mount.")
    else:
        data_cfg.pop(key, None)
    data_cfg.pop(folder if folder != key else "___", None)

data_cfg["path"] = str(merged_root)
print(f"🔗 Merged image counts: {{counts}}")

patched_yaml = Path("/kaggle/working/data_kaggle.yaml")
with open(patched_yaml, "w") as f:
    yaml.dump(data_cfg, f, default_flow_style=False)
print(f"✅ Patched Kaggle Data Config written to {{patched_yaml}}")

# 3. Model Training
project_dir = Path("/kaggle/working/yolo_runs")
exp_name = "train_output"
run_dir = project_dir / exp_name

# 3a. Resume support — a previous kernel's output is mounted as a kernel source
# when this job continues a run that hit Kaggle's 12-hour session limit.
# Ultralytics resumes from the run directory, so restore it before training.
RESUME = {resume}
resume_ckpt = None
if RESUME:
    prev_ckpts = []
    for pat in ("**/output/weights/last.pt", "**/weights/last.pt", "**/last.pt"):
        prev_ckpts += sorted(input_root.glob(pat))
    if prev_ckpts:
        src_ckpt = max(prev_ckpts, key=lambda p: p.stat().st_size)
        # Rebuild the run directory the checkpoint expects to continue into.
        prev_run = src_ckpt.parent.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        for item in prev_run.iterdir():
            dest = run_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        resume_ckpt = run_dir / "weights" / "last.pt"
        if not resume_ckpt.exists():
            resume_ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_ckpt, resume_ckpt)
        print(f"♻️ Resuming from previous kernel checkpoint: {{src_ckpt}}")
    else:
        print("⚠️ Resume requested but no last.pt found in any mounted kernel output. "
              "Starting a fresh run instead.")

print("=" * 60)
print("🏋️ Starting Training: Model={model_name} | Epochs={epochs} | Batch={batch_size} | ImgSz={imgsz}")
print(f"⏳ Wall-clock cap: {max_hours}h (Kaggle terminates sessions at 12h)")
print("=" * 60)

# Runtime cap as a stop-callback, NOT Ultralytics' `time=` argument.
# `time=` means "train FOR this long": trainer.py recomputes
#   self.epochs = ceil(time * 3600 / mean_epoch_time)
# every epoch, which silently overrides the requested epoch count (a 6-epoch
# job became ~1180). This callback leaves `epochs` authoritative and only
# truncates a run that would otherwise overrun Kaggle's 12h session limit.
hit_time_cap = False


def _cap_runtime(trainer):
    global hit_time_cap
    if (_time.time() - JOB_T0) > MAX_SECONDS:
        hit_time_cap = True
        trainer.stop = True
        print(f"⏳ Runtime cap of {max_hours}h reached — stopping after this epoch "
              f"so artifacts are packaged before Kaggle ends the session.", flush=True)


def _run_training(dev):
    if resume_ckpt is not None:
        # On resume Ultralytics reads epochs/optimizer/data from the checkpoint.
        model = YOLO(str(resume_ckpt))
        model.add_callback("on_fit_epoch_end", _cap_runtime)
        return model.train(resume=True, device=dev)
    model = YOLO("{model_name}")
    model.add_callback("on_fit_epoch_end", _cap_runtime)
    return model.train(
        data=str(patched_yaml),
        epochs={epochs},
        batch={batch_size},
        imgsz={imgsz},
        optimizer="{optimizer}",
        lr0={lr0},
        patience={patience},
        device=dev,
        project=str(project_dir),
        name=exp_name,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )


if isinstance(devices, list):
    # Ultralytics reconstructs the trainer from args alone inside DDP
    # subprocesses, so custom callbacks do not reach them.
    print("ℹ️ DDP mode: the runtime cap is best-effort — set epochs so the run "
          "comfortably fits inside Kaggle's 12h session limit.")

train_ok = False
try:
    try:
        results = _run_training(devices)
    except Exception as e:
        # Multi-GPU DDP can be flaky inside a Kaggle script kernel; retry on 1 GPU.
        if isinstance(devices, list):
            print(f"⚠️ Multi-GPU run failed ({{e}}); retrying on a single GPU...")
            results = _run_training(0)
        else:
            raise
    train_ok = True
    print("🎉 Training Completed Successfully!")
except Exception as e:
    # Package whatever was checkpointed before re-raising, so a crashed or
    # time-capped run can still be resumed from last.pt locally.
    print(f"❌ Training Failed: {{e}}")
    train_error = e
else:
    train_error = None

# 4. Package artifacts into /kaggle/working/output/ for 1-click download
run_output_dir = run_dir
output_archive_dir = Path("/kaggle/working/output")
output_archive_dir.mkdir(parents=True, exist_ok=True)

src_dir = run_output_dir if run_output_dir.exists() else project_dir
if src_dir.exists():
    print(f"📦 Packaging artifacts from {{src_dir}} ...")
    for item in src_dir.iterdir():
        dest = output_archive_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    # Guarantee best.pt / last.pt sit at a predictable top level too.
    for w in src_dir.rglob("*.pt"):
        if w.name in ("best.pt", "last.pt"):
            shutil.copy2(w, output_archive_dir / w.name)
else:
    print(f"⚠️ No run directory at {{src_dir}} — nothing to package.")

found = sorted(p.name for p in output_archive_dir.rglob("*") if p.is_file())
print(f"✅ Staged {{len(found)}} files in /kaggle/working/output/: {{found[:20]}}")

# 5. Completion summary the local dashboard parses out of the log.
elapsed_h = (_time.time() - JOB_T0) / 3600.0
has_last = (output_archive_dir / "last.pt").exists()
state = "ok" if train_ok else "failed"
if train_ok and hit_time_cap:
    state = "timecapped"
print(f"[YS-SUMMARY] state={{state}} elapsed_hours={{elapsed_h:.3f}} resumable={{int(has_last)}}")
if state == "timecapped":
    print("⏳ Hit the configured runtime cap before finishing all epochs. "
          "Dispatch a resume job from the dashboard to continue from last.pt.")
print("=" * 60)
print("🏁 Kaggle Job Completed. Ready for local download!")
print("=" * 60)

if train_error is not None:
    raise train_error
'''


def dispatch_kaggle_training(
    dataset_ref,
    kernel_title: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    optimizer: str,
    lr0: float,
    patience: int,
    enable_dual_gpu: bool = True,
    api=None,
    max_hours: float = 11.0,
    resume_from_kernel: Optional[str] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Generates remote training script, kernel-metadata.json, and pushes kernel to
    Kaggle. `dataset_ref` may be a single 'owner/slug' or a list of them (the
    training script merges multi-part datasets).

    `resume_from_kernel` mounts a previous kernel's output as a kernel source so
    the new job continues that run from last.pt — the escape hatch for Kaggle's
    12-hour session limit. `max_hours` caps training wall-clock so artifacts are
    always packaged before Kaggle kills the session.

    Returns: (success, message, kernel_ref)
    """
    if api is None:
        api = get_kaggle_api()
        if api is None:
            return False, "Kaggle API authentication failed. Please configure API token.", None

    username = _read_username_hint()
    if not username:
        _, username, err = is_authenticated()
        if not username:
            return False, err or "Unable to get Kaggle username.", None

    dataset_refs = [dataset_ref] if isinstance(dataset_ref, str) else list(dataset_ref or [])
    dataset_refs = [r for r in dataset_refs if r and "/" in r]
    if not dataset_refs:
        return False, f"Invalid dataset reference(s): {dataset_ref!r}", None

    # Unique slug per dispatch so each run is its own trackable kernel (Kaggle
    # otherwise overwrites the previous kernel + its logs/output).
    stamp = time.strftime("%Y%m%d-%H%M%S")
    kernel_slug = f"{format_dataset_slug(kernel_title)}-{stamp}"
    kernel_ref = f"{username}/{kernel_slug}"

    staging_dir = KAGGLE_STAGING_DIR / "kernel" / kernel_slug
    safe_rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate train_remote.py
    script_content = generate_remote_training_script(
        model_name=model_name,
        dataset_slug=dataset_refs[0].split("/")[-1],
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        optimizer=optimizer,
        lr0=lr0,
        patience=patience,
        enable_dual_gpu=enable_dual_gpu,
        max_hours=max_hours,
        resume=bool(resume_from_kernel),
    )
    script_path = staging_dir / "train_remote.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    # 2. Generate kernel-metadata.json
    metadata = {
        "id": kernel_ref,
        "title": kernel_slug[:50],
        "code_file": "train_remote.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": list(dataset_refs),
    }
    if resume_from_kernel:
        # Kaggle mounts the referenced kernel's output under /kaggle/input/,
        # which is where the training script looks for last.pt.
        metadata["kernel_sources"] = [resume_from_kernel.strip("/")]
    _save_json(staging_dir / "kernel-metadata.json", metadata)

    want = {r.strip("/").lower() for r in dataset_refs}

    # 3a. Wait for ALL dataset parts to be visible on the account BEFORE pushing
    #     - Kaggle validates datasources at push time and silently drops any it
    #     cannot resolve yet, so pushing too early yields dataset_sources: [].
    n_visible = 0
    for _ in range(36):                          # up to ~6 min
        owned = set(_owned_dataset_refs(api, username))
        n_visible = sum(1 for r in want if r in owned)
        if n_visible == len(want):
            break
        time.sleep(10)

    # 3b. Push, then POLL the stored metadata until every datasource lands.
    def _push() -> Optional[str]:
        try:
            resp = api.kernels_push(folder=str(staging_dir), timeout=None)
            return _resp_error(resp)
        except Exception as e:
            return str(e)

    push_err = _push()
    if push_err:
        return False, f"Failed to push kernel to Kaggle: {push_err}", None

    attached, repushes = False, 0
    for i in range(18):                          # up to ~3 min of polling
        time.sleep(10)
        state = _kernel_dataset_state(api, kernel_ref, dataset_refs)
        if state is True:
            attached = True
            break
        if state is False and repushes < 3 and i % 4 == 3:
            _push()
            repushes += 1

    parts_txt = (f"{len(dataset_refs)} dataset part(s)" if len(dataset_refs) > 1
                 else f"dataset '{dataset_refs[0]}'")
    if attached:
        return True, f"Dispatched to Kaggle GPU with {parts_txt} attached. (Kernel: {kernel_ref})", kernel_ref

    if state is None:
        return True, f"Dispatched to Kaggle GPU (Kernel: {kernel_ref}). Kernel pushed with {parts_txt}; attach verification deferred.", kernel_ref

    hint = ("dataset part(s) still processing on Kaggle" if n_visible < len(want)
            else "Kaggle dropped the datasource(s) at push time")
    return False, (
        f"Kernel '{kernel_ref}' was pushed but {parts_txt} not attached ({hint}). "
        f"Wait ~5 min, then hit Dispatch again - it skips the re-upload and just "
        f"re-attaches. Or add the dataset(s) under the kernel's Input panel on Kaggle."
    ), kernel_ref


def _kernel_dataset_state(api, kernel_ref: str, dataset_refs) -> Optional[bool]:
    """True = all datasources present, False = one confirmed absent, None = can't check."""
    if isinstance(dataset_refs, str):
        dataset_refs = [dataset_refs]
    want = {r.strip("/").lower() for r in dataset_refs}
    try:
        probe_dir = KAGGLE_STAGING_DIR / "kernel_probe"
        safe_rmtree(probe_dir)
        probe_dir.mkdir(parents=True, exist_ok=True)
        api.kernels_pull(kernel_ref, path=str(probe_dir), metadata=True)
        meta_path = probe_dir / "kernel-metadata.json"
        if not meta_path.exists():
            return None
        meta = _load_json(meta_path, {})
        have = {str(s).lower().strip("/") for s in meta.get("dataset_sources", [])}
        return want.issubset(have)
    except Exception:
        return None


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
        if "cancel" in status:
            status = "cancelled"
        elif status in ("notstarted", "not_started"):
            status = "queued"
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

    temp_download_dir = KAGGLE_STAGING_DIR / "downloads" / format_dataset_slug(target_exp_name)
    safe_rmtree(temp_download_dir)
    temp_download_dir.mkdir(parents=True, exist_ok=True)

    try:
        api.kernels_output(kernel_ref, path=str(temp_download_dir))
    except Exception as e:
        return False, f"Failed to download kernel output: {e}", None

    # The training script stages everything under 'output/'; fall back to the
    # whole download tree if that folder is absent.
    output_sub = temp_download_dir / "output"
    source_dir = output_sub if (output_sub.exists() and any(output_sub.iterdir())) else temp_download_dir

    copied = 0
    for item in source_dir.iterdir():
        if item.name.endswith(".log"):
            continue
        dest = target_run_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
        copied += 1

    weights = _find_weight_files(target_run_dir)
    have = []
    if weights.get("best.pt"):
        have.append("best.pt")
    if weights.get("last.pt"):
        have.append("last.pt")
    if (target_run_dir / "results.csv").exists():
        have.append("results.csv")

    if copied == 0:
        return False, (
            "Kernel produced no downloadable output yet. If it just finished, "
            "wait a moment and retry; if it errored, check the log."
        ), None

    msg = f"Ingested {copied} item(s) into {target_run_dir}"
    if have:
        msg += f" — found {', '.join(have)}"
    return True, msg, target_run_dir


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\[K|\[[0-9]+m")


def _kernel_log_text(kernel_ref: str, api) -> Optional[str]:
    """Best-effort download of a kernel's run log; returns plain text or None.

    Kaggle usually only serves the log once the session ends, so this can return
    None (or a stale/partial log) while a job is still running.
    """
    try:
        dl_dir = KAGGLE_STAGING_DIR / "logs" / format_dataset_slug(kernel_ref)
        safe_rmtree(dl_dir)
        dl_dir.mkdir(parents=True, exist_ok=True)
        api.kernels_output(kernel_ref, path=str(dl_dir))
    except Exception:
        return None

    log_file = None
    for p in Path(dl_dir).rglob("*.log"):
        log_file = p
        break
    if not log_file:
        return None

    raw = log_file.read_text(errors="replace")
    # The log file is a JSON array of {stream_name, time, data} entries.
    try:
        entries = json.loads(raw)
        raw = "".join(e.get("data", "") for e in entries)
    except Exception:
        pass
    return _ANSI_RE.sub("", raw)


def _parse_training_log(text: str) -> Dict[str, Any]:
    """Pulls epoch / loss / mAP progress out of an Ultralytics training log."""
    out: Dict[str, Any] = {
        "epoch": None, "total_epochs": None, "pct": None,
        "metrics": {}, "phase": None, "tail": "",
    }
    if not text:
        return out

    _noise = (
        "nbconvert", "mistune.py", "SyntaxWarning", "invalid escape sequence",
        "[NbConvertApp]", "filter_links.py", "re.sub(",
    )
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    signal = [ln for ln in lines if not any(n in ln for n in _noise)]
    out["tail"] = "\n".join((signal or lines)[-40:])

    # Surface the actual exception message for failed runs.
    err_line = next((ln.strip() for ln in reversed(lines)
                     if re.search(r"(Error|Exception|assert|Traceback)", ln)
                     and "re.sub(" not in ln), None)
    if err_line:
        out["error_line"] = err_line

    # Epoch progress: lines like "   12/100   2.1G   1.23   2.34   1.11   42   640"
    epoch_re = re.compile(r"^\s*(\d+)/(\d+)\s+[\d.]+G\b")
    for ln in lines:
        m = epoch_re.match(ln)
        if m:
            out["epoch"], out["total_epochs"] = int(m.group(1)), int(m.group(2))

    # Validation summary: "  all   50   120   0.81   0.73   0.80   0.51"
    val_re = re.compile(
        r"^\s*all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
    )
    for ln in lines:
        m = val_re.match(ln)
        if m:
            out["metrics"] = {
                "precision": float(m.group(1)), "recall": float(m.group(2)),
                "mAP50": float(m.group(3)), "mAP50_95": float(m.group(4)),
            }

    if out["epoch"] and out["total_epochs"]:
        out["pct"] = round(100 * out["epoch"] / out["total_epochs"], 1)

    # GPU telemetry emitted by the remote sampler:
    #   [GPUSTAT] t=<secs> <index>, <util%>, <mem_used>, <mem_total>, <temp>, <power>
    gpu_re = re.compile(r"\[GPUSTAT\]\s+t=(\d+)\s+(.+)$")
    samples: List[Dict[str, Any]] = []
    for ln in lines:
        m = gpu_re.search(ln)
        if not m:
            continue
        cols = [c.strip() for c in m.group(2).split(",")]
        if len(cols) < 5:
            continue

        def _num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        samples.append({
            "t": int(m.group(1)),
            "index": int(_num(cols[0]) or 0),
            "util": _num(cols[1]),
            "mem_used": _num(cols[2]),
            "mem_total": _num(cols[3]),
            "temp": _num(cols[4]),
            "power": _num(cols[5]) if len(cols) > 5 else None,
        })

    if samples:
        latest_t = max(s["t"] for s in samples)
        out["gpu_latest"] = sorted(
            (s for s in samples if s["t"] == latest_t), key=lambda s: s["index"]
        )
        utils = [s["util"] for s in samples if s["util"] is not None]
        mems = [s["mem_used"] for s in samples if s["mem_used"] is not None]
        out["gpu_summary"] = {
            "n_samples": len(samples),
            "avg_util": round(sum(utils) / len(utils), 1) if utils else None,
            "peak_util": max(utils) if utils else None,
            "peak_mem_mb": max(mems) if mems else None,
            "sampled_seconds": latest_t,
        }
        out["gpu_series"] = samples

    # Remote completion summary: [YS-SUMMARY] state=ok elapsed_hours=1.234 resumable=1
    sm = None
    for ln in lines:
        if "[YS-SUMMARY]" in ln:
            sm = ln
    if sm:
        fields = dict(
            kv.split("=", 1) for kv in sm.split("[YS-SUMMARY]", 1)[1].split() if "=" in kv
        )
        out["remote_state"] = fields.get("state")
        try:
            out["elapsed_hours"] = float(fields.get("elapsed_hours", 0))
        except ValueError:
            out["elapsed_hours"] = None
        out["resumable"] = fields.get("resumable") == "1"

    # Ultralytics prints "N epochs completed in X.XXX hours" on a clean finish;
    # fall back to the remote summary's own wall-clock measurement.
    dur = re.search(r"epochs completed in\s+([\d.]+)\s+hours", text)
    if dur:
        try:
            out["train_hours"] = float(dur.group(1))
        except ValueError:
            pass
    if out.get("train_hours") is None and out.get("elapsed_hours"):
        out["train_hours"] = out["elapsed_hours"]

    low = text.lower()
    if "training completed successfully" in low or "results saved to" in low or "epochs completed in" in low:
        out["phase"] = "done"
    elif "❌ training failed" in low or "traceback (most recent call last)" in low:
        out["phase"] = "failed"
    elif "starting training for" in low or out["epoch"]:
        out["phase"] = "training"
    elif "installing" in low or "collecting " in low or "initializing yolo" in low:
        out["phase"] = "setup"

    # The remote script's own verdict is authoritative when it got that far —
    # a time-capped run finishes cleanly but still has epochs left to do.
    if out.get("remote_state") == "timecapped":
        out["phase"] = "timecapped"
    elif out.get("remote_state") == "failed":
        out["phase"] = "failed"
    return out


def get_training_progress(kernel_ref: str, api=None) -> Dict[str, Any]:
    """Live-ish training progress for a kernel, parsed from its run log.

    Returns {status, epoch, total_epochs, pct, metrics, phase, tail, log_available}.
    log_available is False when Kaggle has not served any log yet (common while
    a job is still running) - callers should fall back to status + elapsed.
    """
    if api is None:
        api = get_kaggle_api()
        if api is None:
            return {"status": "error", "log_available": False,
                    "message": "Kaggle API not authenticated."}

    info = get_kernel_status(kernel_ref, api=api)
    text = _kernel_log_text(kernel_ref, api)
    parsed = _parse_training_log(text or "")
    parsed["status"] = info.get("status", "unknown")
    parsed["failureMessage"] = info.get("failureMessage")
    parsed["url"] = info.get("url", f"https://www.kaggle.com/code/{kernel_ref}")
    parsed["log_available"] = bool(text)

    # ETA from observed pace: the GPU sampler timestamps give elapsed seconds
    # even when Ultralytics' own progress bar is not in the served log yet.
    elapsed = (parsed.get("gpu_summary") or {}).get("sampled_seconds")
    epoch, total = parsed.get("epoch"), parsed.get("total_epochs")
    if elapsed and epoch and total and epoch > 0 and total > epoch:
        per_epoch = elapsed / epoch
        parsed["sec_per_epoch"] = round(per_epoch, 1)
        parsed["eta_seconds"] = int(per_epoch * (total - epoch))
        parsed["eta_str"] = _duration_str(parsed["eta_seconds"])
    return parsed


def _find_weight_files(root: Path) -> Dict[str, Path]:
    """Locate best.pt / last.pt anywhere under a downloaded kernel-output tree."""
    found: Dict[str, List[Path]] = {"best.pt": [], "last.pt": []}
    for p in Path(root).rglob("*.pt"):
        if p.name in found:
            found[p.name].append(p)

    def _pick(cands: List[Path]) -> Optional[Path]:
        if not cands:
            return None
        # Prefer the copy the training script staged under output/ or the run dir.
        for key in ("output", "train_output", "weights"):
            for c in cands:
                if key in c.parts:
                    return c
        return max(cands, key=lambda c: c.stat().st_size)

    return {k: v for k, v in
            {"best.pt": _pick(found["best.pt"]), "last.pt": _pick(found["last.pt"])}.items()
            if v is not None}


def download_weights(
    kernel_ref: str,
    dest_dir: Optional[Path] = None,
    api=None,
) -> Tuple[bool, str, Dict[str, Path]]:
    """Downloads a finished kernel's output and extracts best.pt + last.pt.

    Copies them into dest_dir (default: yolo_workspace/runs/<slug>/weights/) and
    returns {name: local_path} so the UI can also offer direct downloads.
    """
    if api is None:
        api = get_kaggle_api()
        if api is None:
            return False, "Kaggle API not authenticated.", {}

    slug = format_dataset_slug(kernel_ref)
    tmp_dir = KAGGLE_STAGING_DIR / "weights_dl" / slug
    safe_rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        api.kernels_output(kernel_ref, path=str(tmp_dir))
    except Exception as e:
        return False, f"Could not download kernel output: {e}", {}

    weights = _find_weight_files(tmp_dir)
    if not weights:
        return False, (
            "No best.pt / last.pt found in the kernel output yet. "
            "If the job is still running or just finished, try again shortly."
        ), {}

    if dest_dir is None:
        dest_dir = RUNS_DIR / slug / "weights"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}
    for name, src in weights.items():
        dst = dest_dir / name
        shutil.copy2(src, dst)
        saved[name] = dst

    return True, f"Saved {', '.join(saved)} to {dest_dir}", saved


def _duration_str(secs: float) -> str:
    secs = max(0, int(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def _elapsed_str(timestamp: str) -> str:
    try:
        started = time.mktime(time.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))
        secs = max(0, int(time.time() - started))
    except Exception:
        return "-"
    return _duration_str(secs)


def _age_hours(timestamp: str) -> float:
    try:
        started = time.mktime(time.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))
        return max(0.0, (time.time() - started) / 3600.0)
    except Exception:
        return 1e9


def list_all_jobs(api=None) -> List[Dict[str, Any]]:
    """Local job history enriched with live Kaggle status, newest first.

    Each item: history fields + {status, failureMessage, url, elapsed,
    is_ongoing, is_stale}.
    """
    if api is None:
        api = get_kaggle_api()

    enriched: List[Dict[str, Any]] = []
    for j in list_recent_jobs_history():
        ref = j.get("kernel_ref")
        item = dict(j)
        info = get_kernel_status(ref, api=api) if (api and ref) else {"status": "unknown"}
        status = info.get("status", "unknown")
        age = _age_hours(j.get("timestamp", ""))
        # 'unknown' == Kaggle's status endpoint 404s (no live session). Fresh ->
        # probably still spinning up; old -> the session is gone, treat as done.
        stale = status == "unknown" and age >= 6
        item["status"] = status
        item["failureMessage"] = info.get("failureMessage")
        item["url"] = info.get("url", f"https://www.kaggle.com/code/{ref}")
        item["elapsed"] = _elapsed_str(j.get("timestamp", ""))
        item["is_stale"] = stale
        item["is_ongoing"] = status in ("queued", "running") or (status == "unknown" and not stale)
        item["gpu_hours"] = j.get("gpu_hours")
        item["ingested"] = bool(j.get("ingested"))
        item["resumable"] = bool(j.get("resumable"))
        item["remote_state"] = j.get("remote_state")
        enriched.append(item)
    return enriched


def list_recent_jobs_history() -> List[Dict[str, Any]]:
    """Loads history of Kaggle training jobs from the local store, newest first."""
    jobs = _load_json(JOBS_HISTORY_FILE, [])
    return jobs if isinstance(jobs, list) else []


def save_job_to_history(job_info: Dict[str, Any]):
    """Appends or updates a job (keyed by kernel_ref) in local history."""
    jobs = list_recent_jobs_history()
    for i, j in enumerate(jobs):
        if j.get("kernel_ref") == job_info.get("kernel_ref"):
            jobs[i] = {**j, **job_info}
            break
    else:
        jobs.insert(0, job_info)
    _save_json(JOBS_HISTORY_FILE, jobs[:25])


WEEKLY_GPU_QUOTA_HOURS = 30.0


def _job_gpu_hours(job: Dict[str, Any]) -> Optional[float]:
    """GPU hours a job consumed, if known.

    Prefers the measured runtime recorded at ingest/finish time; falls back to
    elapsed wall-clock for a job still running.
    """
    measured = job.get("gpu_hours")
    if isinstance(measured, (int, float)) and measured > 0:
        return float(measured)
    if job.get("is_ongoing"):
        age = _age_hours(job.get("timestamp", ""))
        return min(age, 12.0) if age < 1e6 else None
    return None


def estimate_weekly_gpu_usage(jobs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Rolling 7-day GPU-hour estimate from locally tracked jobs.

    Kaggle exposes no quota API, so this is an estimate over jobs this app
    dispatched — runs started from kaggle.com are not counted. Treat it as a
    guide and confirm against kaggle.com/settings.
    """
    if jobs is None:
        jobs = list_recent_jobs_history()

    used, counted, unknown, ongoing = 0.0, 0, 0, 0
    for j in jobs:
        age = _age_hours(j.get("timestamp", ""))
        if age > 24 * 7:
            continue
        if j.get("is_ongoing"):
            ongoing += 1
        hours = _job_gpu_hours(j)
        if hours is None:
            unknown += 1
            continue
        used += hours
        counted += 1

    used = round(used, 2)
    remaining = max(0.0, round(WEEKLY_GPU_QUOTA_HOURS - used, 2))
    return {
        "quota_hours": WEEKLY_GPU_QUOTA_HOURS,
        "used_hours": used,
        "remaining_hours": remaining,
        "pct_used": round(100 * used / WEEKLY_GPU_QUOTA_HOURS, 1),
        "jobs_counted": counted,
        "jobs_unknown": unknown,
        "jobs_ongoing": ongoing,
        "is_estimate": True,
    }


def record_job_runtime(kernel_ref: str, progress: Dict[str, Any]):
    """Persists measured GPU runtime + telemetry for a finished job."""
    patch: Dict[str, Any] = {"kernel_ref": kernel_ref}
    if progress.get("train_hours"):
        patch["gpu_hours"] = round(float(progress["train_hours"]), 3)
    summary = progress.get("gpu_summary") or {}
    if summary:
        patch["gpu_avg_util"] = summary.get("avg_util")
        patch["gpu_peak_util"] = summary.get("peak_util")
        patch["gpu_peak_mem_mb"] = summary.get("peak_mem_mb")
    if progress.get("remote_state"):
        patch["remote_state"] = progress["remote_state"]
    if progress.get("resumable") is not None:
        patch["resumable"] = progress["resumable"]
    if len(patch) > 1:
        save_job_to_history(patch)


def auto_ingest_completed_jobs(api=None) -> List[Dict[str, Any]]:
    """Ingests any tracked job that finished but has not been pulled down yet.

    Called by the dashboard's polling fragment so a completed cloud run lands in
    yolo_workspace/runs/ without the user having to click anything. Returns one
    result dict per job it attempted.
    """
    if api is None:
        api = get_kaggle_api()
        if api is None:
            return []

    results: List[Dict[str, Any]] = []
    for job in list_all_jobs(api=api):
        ref = job.get("kernel_ref")
        if not ref or job.get("ingested") or job.get("is_ongoing"):
            continue
        # 'stale' means Kaggle's status endpoint no longer knows the session —
        # the run ended, so its output should be fetchable.
        if job.get("status") not in ("complete", "error", "cancelled") and not job.get("is_stale"):
            continue

        exp_name = job.get("target_exp") or format_dataset_slug(ref.split("/")[-1])

        # Already on disk (e.g. ingested by hand, or before this flag existed):
        # record it and skip the re-download rather than re-pulling every run
        # the first time polling is switched on.
        dest = RUNS_DIR / exp_name
        if (dest / "weights" / "best.pt").exists() or (dest / "best.pt").exists():
            save_job_to_history({"kernel_ref": ref, "ingested": True, "target_exp": exp_name})
            continue

        ok, msg, path = download_and_ingest_artifacts(ref, exp_name, api=api)

        if ok:
            progress = get_training_progress(ref, api=api)
            record_job_runtime(ref, progress)
            save_job_to_history({"kernel_ref": ref, "ingested": True,
                                 "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "target_exp": exp_name})
        results.append({"kernel_ref": ref, "ok": ok, "message": msg,
                        "path": str(path) if path else None, "target_exp": exp_name})
    return results


#: Buckets the dashboard groups jobs into, in display order.
JOB_CATEGORIES = ("ongoing", "successful", "cancelled", "failed")


def classify_job(job: Dict[str, Any]) -> str:
    """Sorts an enriched job into one of JOB_CATEGORIES.

    'cancelled' covers both a stop requested on kaggle.com and a run that was
    terminated without finishing - a session that expired, or one truncated by
    the runtime cap. 'successful' means Kaggle reported completion, which is the
    only case that should be offering a trained model for download.
    """
    if job.get("is_ongoing"):
        return "ongoing"
    status = job.get("status")
    if status == "cancelled" or job.get("remote_state") == "timecapped" or job.get("is_stale"):
        return "cancelled"
    if status == "complete":
        return "successful"
    return "failed"


def job_termination_reason(job: Dict[str, Any]) -> str:
    """Human-readable why for a job in the cancelled/terminated bucket."""
    if job.get("status") == "cancelled":
        return "Stopped from the Kaggle console"
    if job.get("remote_state") == "timecapped":
        return "Terminated by the runtime cap before all epochs finished"
    if job.get("is_stale"):
        return "Session ended without reporting completion (expired or terminated by Kaggle)"
    return "Terminated"


def kernel_session_url(kernel_ref: str) -> str:
    """Direct link to a kernel's session page, where Kaggle's Stop button lives."""
    return f"https://www.kaggle.com/code/{kernel_ref.strip('/')}"


def request_kernel_stop(kernel_ref: str, api=None) -> Tuple[bool, str, str]:
    """Attempts to stop a running kernel; reports honestly when it cannot.

    Kaggle's backend supports cancellation (KernelWorkerStatus has
    CANCEL_REQUESTED / CANCEL_ACKNOWLEDGED) but the public API exposes no RPC
    for it - see https://github.com/Kaggle/kaggle-api/issues/388. Only the web
    UI can trigger a stop, so this reports live status and hands back the URL
    to do it, rather than pretending the job was cancelled.

    Returns: (already_stopped, message, session_url)
    """
    url = kernel_session_url(kernel_ref)
    if api is None:
        api = get_kaggle_api()

    info = get_kernel_status(kernel_ref, api=api) if api else {"status": "unknown"}
    status = info.get("status", "unknown")

    if status in ("complete", "error", "cancelled"):
        return True, f"Job is already finished on Kaggle (status: {status}). Nothing to stop.", url
    if status == "unknown":
        return False, (
            "Kaggle's status endpoint has no live session for this kernel, which usually "
            "means it already ended. If it is still shown as running on Kaggle, stop it "
            "from the session page."
        ), url
    return False, (
        f"This job is {status.upper()} on Kaggle and is still consuming your GPU quota. "
        "Kaggle's public API has no cancel endpoint, so it must be stopped from the web "
        "console: open the session page and use **Stop Session** (the ⏹ control on the "
        "run). The dashboard will pick up the cancelled status on the next refresh."
    ), url


def delete_job_from_history(kernel_ref: str):
    """Removes a job from local history.

    Local-only: this does NOT stop the kernel on Kaggle. Callers must warn the
    user when the job is still running, or it silently leaves a job burning GPU
    quota with nothing tracking it.
    """
    jobs = [j for j in list_recent_jobs_history() if j.get("kernel_ref") != kernel_ref]
    _save_json(JOBS_HISTORY_FILE, jobs)
