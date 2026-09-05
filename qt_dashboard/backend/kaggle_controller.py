"""Kaggle Cloud GPU hub — a thin Qt shell over the UI-free ``kaggle_bridge``."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot

from . import core
from .app_state import url_to_path
from .util import run_async

sys.path.insert(0, str(core.PROJECT_ROOT))
import kaggle_bridge  # noqa: E402

# The kaggle package pulls in requests/urllib3 and a pile of C extensions.
# Importing that from a worker thread while the QML engine is still loading has
# been seen to abort the process, so warm it here on the main thread.
try:
    import kaggle.api.kaggle_api_extended  # noqa: F401,E402
except Exception:
    pass

CAT_COLOUR = {
    "running": "#10b981", "pending": "#f59e0b", "successful": "#22c55e",
    "terminated": "#f97316", "cancelled": "#ef4444", "failed": "#dc2626",
    "unresolved": "#64748b",
}
CAT_HELP = {
    "running": "Training on Kaggle right now.",
    "pending": "Queued, or still starting up.",
    "successful": "Finished cleanly — the trained model is ready to download.",
    "terminated": "Cut short by the runtime cap before finishing all epochs.",
    "cancelled": "Stopped from the Kaggle console.",
    "failed": "Errored before producing a model.",
    "unresolved": "Session has ended; its outcome has not been read yet.",
}

# Kaggle exposes no cancel endpoint and no direct URL for the Active Events
# panel, so the UI spells out the click path instead.
STOP_STEPS = (
    "To stop it on Kaggle:\n"
    "1. Open the kernel page — or any Kaggle page.\n"
    "2. In the left sidebar click  ⧉ View Active Events.\n"
    "3. Find this session, click the ⋯ next to it, then ⏹ Stop Session.\n\n"
    "The kernel page's Run/Save controls do not stop a running batch job — "
    "Active Events is the only place with the stop control."
)


def _when(ts: Optional[str]) -> str:
    """'12 min ago' / 'yesterday 23:23' style stamp for job cards."""
    if not ts:
        return "—"
    try:
        t = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return ts
    secs = time.time() - t
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    if secs < 172800:
        return f"yesterday {ts[11:16]}"
    return f"{int(secs // 86400)} days ago"


def _ran(j: Dict[str, Any]) -> str:
    if j.get("category") in ("running", "pending"):
        return j.get("elapsed") or "—"
    if j.get("gpu_hours"):
        return kaggle_bridge._duration_str(j["gpu_hours"] * 3600)
    return "—"


class KaggleController(QObject):
    """Auth, quota, dispatch and the job dashboard."""

    authChanged = Signal()
    quotaChanged = Signal()
    jobsChanged = Signal()
    dispatchLogChanged = Signal()
    busyChanged = Signal()
    progressChanged = Signal(str)      # kernel_ref whose detail panel updated
    toast = Signal(str, str)
    stopInstructions = Signal(str, str)  # message, url

    def __init__(self, parent=None):
        super().__init__(parent)
        self._auth = {"connected": False, "user": "", "error": ""}
        self._quota: Dict[str, Any] = {}
        self._jobs: List[Dict[str, Any]] = []
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._dispatch_log: List[str] = []
        self._ingest_log: List[str] = []
        self._busy = ""
        self._live = False
        self._auto_ingest = True
        self._auto_export: List[str] = []
        self._last_poll = ""

        self._poll = QTimer(self)
        self._poll.setInterval(30000)
        self._poll.timeout.connect(self._live_tick)

        self.refreshAuth()
        self.refreshQuota()
        self.refreshJobs()

    # -- busy --------------------------------------------------------------
    def _set_busy(self, msg):
        self._busy = msg
        self.busyChanged.emit()

    @Property(str, notify=busyChanged)
    def busy(self):
        return self._busy

    # -- auth --------------------------------------------------------------
    @Property("QVariantMap", notify=authChanged)
    def auth(self):
        return self._auth

    @Slot()
    def refreshAuth(self):
        def work():
            ok, user, err = kaggle_bridge.is_authenticated()
            return {"connected": bool(ok), "user": user or "", "error": err or ""}

        def done(res):
            self._auth = res
            self.authChanged.emit()

        run_async(work, on_done=done, on_error=lambda e: None)

    @Slot(str, str)
    def saveCredentials(self, username, key):
        if not username or not key:
            self.toast.emit("Provide both a Kaggle username and API key.", "warn")
            return
        self._set_busy("Verifying Kaggle credentials …")

        def work():
            return kaggle_bridge.save_credentials(username.strip(), key.strip())

        def done(res):
            ok, msg = res
            self._set_busy("")
            self.toast.emit(msg, "success" if ok else "error")
            if ok:
                self.refreshAuth()
                self.refreshJobs()

        def fail(err):
            self._set_busy("")
            self.toast.emit(err.strip().splitlines()[-1], "error")

        run_async(work, on_done=done, on_error=fail)

    @Property("QStringList", notify=authChanged)
    def credentialPaths(self):
        """Credential files that a disconnect would delete."""
        return [str(p) for p in kaggle_bridge.stored_credential_paths()]

    @Slot()
    def disconnectAccount(self):
        """Removes this machine's stored Kaggle credentials."""
        ok, msg = kaggle_bridge.clear_credentials()
        self.toast.emit(msg, "success" if ok else "error")
        self._quota = {}
        self.quotaChanged.emit()
        self.refreshAuth()
        self.refreshJobs()

    @Slot(str)
    def importKaggleJson(self, url):
        try:
            data = json.loads(Path(url_to_path(url)).read_text())
        except Exception as e:
            self.toast.emit(f"Invalid kaggle.json: {e}", "error")
            return
        self.saveCredentials(data.get("username", ""), data.get("key", ""))

    # -- quota -------------------------------------------------------------
    @Property("QVariantMap", notify=quotaChanged)
    def quota(self):
        return self._quota

    @Slot()
    def refreshQuota(self):
        def work():
            q = kaggle_bridge.estimate_weekly_gpu_usage()
            note = (f"Estimated from {q['jobs_counted']} tracked job(s) in the last 7 days"
                    + (f"; {q['jobs_unknown']} with unknown runtime" if q["jobs_unknown"] else "")
                    + (f"; {q['jobs_ongoing']} still running" if q["jobs_ongoing"] else "")
                    + ". Kaggle has no quota API — jobs launched on kaggle.com are not "
                      "counted. Confirm at kaggle.com/settings.")
            return {**q, "note": note}

        def done(q):
            self._quota = q
            self.quotaChanged.emit()

        run_async(work, on_done=done, on_error=lambda e: None)

    # -- dispatch defaults -------------------------------------------------
    @Slot(result=str)
    def nextRunName(self):
        return kaggle_bridge.next_free_run_name()

    @Property("QVariantList", constant=True)
    def categoryLabels(self):
        return [{"key": c, "icon": kaggle_bridge.CATEGORY_LABELS[c][0],
                 "name": kaggle_bridge.CATEGORY_LABELS[c][1], "help": CAT_HELP[c],
                 "colour": CAT_COLOUR[c]}
                for c in kaggle_bridge.JOB_CATEGORIES]

    @Slot(result="QStringList")
    def resumableJobs(self):
        return [j["kernel_ref"] for j in kaggle_bridge.list_recent_jobs_history()
                if j.get("resumable") or j.get("remote_state") == "timecapped"]

    @Property("QStringList", notify=dispatchLogChanged)
    def dispatchLog(self):
        return self._dispatch_log

    def _dlog(self, line):
        self._dispatch_log.append(line)
        self.dispatchLogChanged.emit()

    @Slot("QVariantMap")
    def dispatch(self, cfg):
        """Stages the dataset (or attaches existing refs) and pushes the kernel."""
        if not self._auth.get("connected"):
            self.toast.emit("Connect your Kaggle API credentials first.", "warn")
            return
        ds_yaml = cfg.get("datasetYaml")
        manual = [r.strip() for r in str(cfg.get("existingRefs", "")).split(",")
                  if r.strip() and "/" in r]
        if not ds_yaml and not manual:
            self.toast.emit("Select a dataset, or give an existing Kaggle ref.", "warn")
            return

        self._dispatch_log = []
        self.dispatchLogChanged.emit()
        self._set_busy("Dispatching job to Kaggle …")
        # A classification dataset is a directory, not a yaml — stage that
        # folder rather than everything under dataset/.
        _p = Path(ds_yaml) if ds_yaml else None
        ds_folder = (_p if (_p and _p.is_dir()) else (_p.parent if _p else None))

        def work():
            api = kaggle_bridge.get_kaggle_api()
            if manual:
                self._dlog(f"📎 Using existing Kaggle dataset ref(s): {', '.join(manual)}")
                ds_ok, ds_msg, ds_refs = True, f"Attaching {len(manual)} existing dataset(s).", manual
            else:
                self._dlog("📦 Packaging and verifying dataset for Kaggle …")
                ds_ok, ds_msg, ds_refs = kaggle_bridge.package_and_upload_dataset(
                    dataset_path=ds_folder,
                    dataset_title=cfg.get("datasetTitle") or ds_folder.name,
                    api=api,
                    progress_callback=lambda m: self._dlog(f"  ➜ {m}"))
            if not ds_ok:
                return {"ok": False, "message": ds_msg, "kernel": None}
            self._dlog(f"✅ {ds_msg}")

            resume_from = cfg.get("resumeFrom") or None
            if resume_from:
                self._dlog(f"♻️ Mounting `{resume_from}` output to resume from its last.pt …")
            self._dlog("🛰️ Generating remote training script and pushing kernel …")
            ok, msg, kernel_ref = kaggle_bridge.dispatch_kaggle_training(
                dataset_ref=ds_refs,
                kernel_title=cfg.get("jobTitle") or "yolo-cloud-training",
                model_name=cfg.get("model", "yolo11n.pt"),
                epochs=int(cfg.get("epochs", 100)),
                batch_size=int(cfg.get("batch", 32)),
                imgsz=int(cfg.get("imgsz", 640)),
                optimizer=cfg.get("optimizer", "AdamW"),
                lr0=float(cfg.get("lr0", 0.005)),
                patience=int(cfg.get("patience", 20)),
                enable_dual_gpu=bool(cfg.get("dualGpu", True)),
                api=api,
                max_hours=float(cfg.get("maxHours", 11.0)),
                resume_from_kernel=resume_from,
            )
            # The kernel exists even when attachment can't be confirmed, so
            # record it either way and let the dashboard track it.
            if kernel_ref:
                kaggle_bridge.save_job_to_history({
                    "kernel_ref": kernel_ref,
                    "dataset_ref": ", ".join(ds_refs) if isinstance(ds_refs, (list, tuple)) else ds_refs,
                    "dataset_parts": len(ds_refs) if isinstance(ds_refs, (list, tuple)) else 1,
                    "model_name": cfg.get("model"),
                    "epochs": int(cfg.get("epochs", 100)),
                    "target_exp": cfg.get("targetExp"),
                    "max_hours": float(cfg.get("maxHours", 11.0)),
                    "resumed_from": resume_from,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
            return {"ok": ok, "message": msg, "kernel": kernel_ref}

        def done(res):
            self._set_busy("")
            if res["kernel"]:
                self._dlog(f"🎉 Kernel: {res['kernel']}")
            self.toast.emit(res["message"], "success" if res["ok"] else "warn")
            self.refreshJobs()
            self.refreshQuota()

        def fail(err):
            self._set_busy("")
            msg = err.strip().splitlines()[-1]
            self._dlog(f"❌ {msg}")
            self.toast.emit(f"Dispatch failed: {msg}", "error")

        run_async(work, on_done=done, on_error=fail)

    # -- job dashboard -----------------------------------------------------
    @Property("QVariantList", notify=jobsChanged)
    def jobs(self):
        return self._jobs

    @Property("QVariantMap", notify=jobsChanged)
    def counts(self):
        counts = {c: 0 for c in kaggle_bridge.JOB_CATEGORIES}
        for j in self._jobs:
            counts[j["category"]] = counts.get(j["category"], 0) + 1
        return counts

    @Property(str, notify=jobsChanged)
    def lastPolled(self):
        return self._last_poll

    @Property("QStringList", notify=jobsChanged)
    def ingestLog(self):
        return self._ingest_log

    @Property(bool, notify=jobsChanged)
    def livePolling(self):
        return self._live

    @Slot(bool)
    def setLivePolling(self, on):
        self._live = bool(on)
        if self._live:
            self._poll.start()
            self._live_tick()
        else:
            self._poll.stop()
        self.jobsChanged.emit()

    @Slot(bool)
    def setAutoIngest(self, on):
        self._auto_ingest = bool(on)

    @Slot("QStringList")
    def setAutoExport(self, formats):
        self._auto_export = list(formats)

    def _decorate(self, j: Dict[str, Any]) -> Dict[str, Any]:
        cat = j.get("category", "unresolved")
        icon, name = kaggle_bridge.CATEGORY_LABELS[cat]
        ds = str(j.get("dataset_ref", "—")).split(",")[0].strip().split("/")[-1] or "—"
        weights = kaggle_bridge.local_weights_for(j)
        item = dict(j)
        item.update({
            "ref": j["kernel_ref"],
            "shortName": j["kernel_ref"].split("/")[-1],
            "catIcon": icon, "catName": name, "catColour": CAT_COLOUR[cat],
            "whenText": _when(j.get("timestamp")),
            "runtimeText": _ran(j),
            "datasetShort": ds,
            "reason": kaggle_bridge.job_termination_reason(j),
            "weightsDir": str(kaggle_bridge.job_weights_dir(j)),
            "weights": [{"name": n, "path": str(p),
                         "size": f"{p.stat().st_size / (1024*1024):.1f} MB"}
                        for n, p in sorted(weights.items()) if p.exists()],
            "canStop": cat in ("running", "pending"),
            "canIngest": cat == "successful",
            "isPartial": cat in ("terminated", "cancelled"),
            "logLabel": "🔎 Check outcome" if cat == "unresolved" else "📊 Log & GPU",
        })
        # QML reads strings; keep the optional message fields non-null.
        item["failureMessage"] = j.get("failureMessage") or ""
        item["resumed_from"] = j.get("resumed_from") or ""
        item["model_name"] = j.get("model_name") or "—"
        item["epochs"] = j.get("epochs") or "—"
        return item

    @Slot()
    def refreshJobs(self):
        self._set_busy("Fetching job status from Kaggle …")

        def work():
            return [self._decorate(j) for j in kaggle_bridge.list_all_jobs()]

        def done(jobs):
            self._set_busy("")
            self._jobs = jobs
            self._last_poll = time.strftime("%H:%M:%S")
            self.jobsChanged.emit()

        def fail(err):
            self._set_busy("")
            self.toast.emit(f"Job refresh failed: {err.strip().splitlines()[-1]}", "warn")

        run_async(work, on_done=done, on_error=fail)

    def _live_tick(self):
        if self._auto_ingest:
            run_async(self._auto_ingest_pass,
                      on_done=lambda lines: self._merge_ingest_log(lines),
                      on_error=lambda e: None)
        self.refreshJobs()
        self.refreshQuota()

    def _auto_ingest_pass(self):
        lines = []
        for res in kaggle_bridge.auto_ingest_completed_jobs():
            stamp = time.strftime("%H:%M:%S")
            short = res["kernel_ref"].split("/")[-1]
            if res["ok"]:
                lines.append(f"[{stamp}] ✅ {short} → runs/{res['target_exp']}")
                if self._auto_export and res.get("path"):
                    for line in self._post_export(Path(res["path"]), stamp):
                        lines.append(line)
            else:
                lines.append(f"[{stamp}] ⚠️ {short}: {res['message'][:140]}")
        return lines

    def _post_export(self, run_dir: Path, stamp: str) -> List[str]:
        """Compiles an ingested checkpoint into the requested runtimes."""
        from .export_controller import export_checkpoint
        out = []
        ckpt = next((w for w in (run_dir / "weights" / "best.pt", run_dir / "best.pt")
                     if w.exists()), None)
        if not ckpt:
            return [f"[{stamp}]    ⚠️ no best.pt in the ingested run"]
        for fmt in self._auto_export:
            try:
                res = export_checkpoint(ckpt, fmt, 640, False, False, True, False, "cpu")
                out.append(f"[{stamp}]    📦 exported {fmt}: {Path(res['path']).name}")
            except Exception as e:
                out.append(f"[{stamp}]    ⚠️ {fmt} export failed: {str(e)[:120]}")
        return out

    def _merge_ingest_log(self, lines):
        if lines:
            self._ingest_log = (self._ingest_log + lines)[-30:]
            self.jobsChanged.emit()

    # -- per-job actions ---------------------------------------------------
    @Slot(str)
    def fetchWeights(self, ref):
        job = self._job(ref)
        if not job:
            return
        self._set_busy(f"Downloading model weights for {ref.split('/')[-1]} …")

        def work():
            return kaggle_bridge.download_weights(
                ref, dest_dir=kaggle_bridge.job_weights_dir(job))

        def done(res):
            ok, msg, _saved = res
            self._set_busy("")
            self.toast.emit(msg, "success" if ok else "warn")
            self.refreshJobs()

        def fail(err):
            self._set_busy("")
            self.toast.emit(f"Weight download failed: {err.strip().splitlines()[-1]}", "error")

        run_async(work, on_done=done, on_error=fail)

    @Slot(str)
    def ingestRun(self, ref):
        job = self._job(ref)
        if not job:
            return
        exp = job.get("target_exp") or kaggle_bridge.format_dataset_slug(ref.split("/")[-1])
        self._set_busy(f"Downloading run output → runs/{exp} …")

        def work():
            ok, msg, run_path = kaggle_bridge.download_and_ingest_artifacts(ref, exp)
            extra = []
            if ok:
                prog = kaggle_bridge.get_training_progress(ref)
                kaggle_bridge.record_job_runtime(ref, prog)
                kaggle_bridge.save_job_to_history({
                    "kernel_ref": ref, "ingested": True, "target_exp": exp,
                    "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                if self._auto_export and run_path:
                    extra = self._post_export(Path(run_path), time.strftime("%H:%M:%S"))
            return ok, msg, extra

        def done(res):
            ok, msg, extra = res
            self._set_busy("")
            self.toast.emit(msg, "success" if ok else "warn")
            self._merge_ingest_log(extra)
            self.refreshJobs()
            self.refreshQuota()

        def fail(err):
            self._set_busy("")
            self.toast.emit(f"Ingest failed: {err.strip().splitlines()[-1]}", "error")

        run_async(work, on_done=done, on_error=fail)

    @Slot(str)
    def loadProgress(self, ref):
        """Reads the run log for epoch detail, metrics and GPU telemetry."""
        self._set_busy("Reading the run log from Kaggle …")

        def work():
            return ref, kaggle_bridge.resolve_job_outcome(ref)

        def done(res):
            key, prog = res
            self._set_busy("")
            self._progress[key] = self._shape_progress(prog)
            self.progressChanged.emit(key)
            self.refreshJobs()

        def fail(err):
            self._set_busy("")
            self.toast.emit(f"Could not read log: {err.strip().splitlines()[-1]}", "warn")

        run_async(work, on_done=done, on_error=fail)

    @staticmethod
    def _shape_progress(prog: Dict[str, Any]) -> Dict[str, Any]:
        m = prog.get("metrics") or {}
        series = prog.get("gpu_series") or []
        by_gpu: Dict[int, List[Dict[str, float]]] = {}
        for s in series:
            by_gpu.setdefault(int(s.get("index", 0)), []).append(
                {"x": round(float(s.get("t", 0)) / 60.0, 2),
                 "y": float(s.get("util") or 0)})
        return {
            "epoch": prog.get("epoch") or 0,
            "totalEpochs": prog.get("total_epochs") or 0,
            "pct": (prog.get("pct") or 0) / 100.0,
            "etaStr": prog.get("eta_str") or "",
            "secPerEpoch": prog.get("sec_per_epoch") or 0,
            "metrics": {k: float(m.get(k, 0) or 0)
                        for k in ("mAP50", "mAP50_95", "precision", "recall")} if m else {},
            "gpuLatest": prog.get("gpu_latest") or [],
            "gpuSummary": prog.get("gpu_summary") or {},
            "gpuSeries": [{"name": f"GPU {i} util %", "points": pts}
                          for i, pts in sorted(by_gpu.items()) if len(pts) > 2],
            "errorLine": prog.get("error_line") or "",
            "tail": prog.get("tail") or "(log empty)",
            "logAvailable": bool(prog.get("log_available")),
        }

    @Slot(str, result="QVariantMap")
    def progressFor(self, ref):
        return self._progress.get(ref, {})

    @Slot(str)
    def stopJob(self, ref):
        self._set_busy("Checking live status on Kaggle …")

        def work():
            return kaggle_bridge.request_kernel_stop(ref)

        def done(res):
            stopped, msg, url = res
            self._set_busy("")
            self.toast.emit(msg, "success" if stopped else "warn")
            if not stopped:
                self.stopInstructions.emit(f"{msg}\n\n{STOP_STEPS}", url)
            self.refreshJobs()

        def fail(err):
            self._set_busy("")
            self.toast.emit(err.strip().splitlines()[-1], "error")

        run_async(work, on_done=done, on_error=fail)

    @Slot(str)
    def removeJob(self, ref):
        kaggle_bridge.delete_job_from_history(ref)
        self._jobs = [j for j in self._jobs if j["ref"] != ref]
        self.jobsChanged.emit()
        self.toast.emit("Removed from local tracking.", "info")

    @Slot(str, result=str)
    def kernelUrl(self, ref):
        job = self._job(ref)
        return (job or {}).get("url") or f"https://www.kaggle.com/code/{ref}"

    @Slot(result=str)
    def notebookTemplate(self):
        p = core.PROJECT_ROOT / "kaggle_yolo_train_template.ipynb"
        return str(p) if p.exists() else ""

    def _job(self, ref):
        return next((j for j in self._jobs if j["ref"] == ref), None)
