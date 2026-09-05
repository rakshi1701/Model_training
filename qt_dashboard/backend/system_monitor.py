"""Live CPU / RAM / swap / disk / GPU telemetry for the training dashboard."""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional

import psutil
from PySide6.QtCore import QObject, Property, QTimer, Signal, Slot


class SystemMonitor(QObject):
    """Polls host utilisation every 2s, the same cadence as the web dashboard."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stats: Dict[str, Any] = {
            "cpu_percent": 0.0, "cpu_per_core": [], "cpu_count_logical": 0,
            "cpu_count_physical": 0, "cpu_freq": 0.0,
            "ram_total_gb": 0.0, "ram_used_gb": 0.0, "ram_available_gb": 0.0,
            "ram_percent": 0.0, "swap_total_gb": 0.0, "swap_used_gb": 0.0,
            "swap_percent": 0.0, "disk_total_gb": 0.0, "disk_used_gb": 0.0,
            "disk_percent": 0.0, "gpus": [], "train_proc": None,
        }
        self._tracked_pid: Optional[int] = None
        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    # -- exposed to QML ----------------------------------------------------
    @Property("QVariantMap", notify=changed)
    def stats(self):
        return self._stats

    @Property(bool, notify=changed)
    def hasGpu(self):
        return bool(self._stats["gpus"])

    def track_process(self, pid: Optional[int]):
        """Report per-process RSS/CPU/threads for the running training job."""
        self._tracked_pid = pid

    # -- polling -----------------------------------------------------------
    @Slot()
    def refresh(self):
        s = self._stats
        s["cpu_percent"] = psutil.cpu_percent(interval=0)
        s["cpu_per_core"] = psutil.cpu_percent(interval=0, percpu=True)
        s["cpu_count_logical"] = psutil.cpu_count(logical=True) or 0
        s["cpu_count_physical"] = psutil.cpu_count(logical=False) or 0
        try:
            s["cpu_freq"] = psutil.cpu_freq().current
        except Exception:
            s["cpu_freq"] = 0.0

        mem = psutil.virtual_memory()
        s["ram_total_gb"] = mem.total / 1024 ** 3
        s["ram_used_gb"] = mem.used / 1024 ** 3
        s["ram_available_gb"] = mem.available / 1024 ** 3
        s["ram_percent"] = mem.percent

        swap = psutil.swap_memory()
        s["swap_total_gb"] = swap.total / 1024 ** 3
        s["swap_used_gb"] = swap.used / 1024 ** 3
        s["swap_percent"] = swap.percent

        disk = psutil.disk_usage("/")
        s["disk_total_gb"] = disk.total / 1024 ** 3
        s["disk_used_gb"] = disk.used / 1024 ** 3
        s["disk_percent"] = disk.percent

        s["gpus"] = self._query_gpus()
        s["train_proc"] = self._query_train_proc()
        self.changed.emit()

    @staticmethod
    def _query_gpus() -> List[Dict[str, Any]]:
        gpus: List[Dict[str, Any]] = []
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                 "temperature.gpu,power.draw,power.limit",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=3).strip()
        except Exception:
            return gpus

        def num(v):
            try:
                return float(v)
            except Exception:
                return 0.0

        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            g = {
                "index": int(num(parts[0])), "name": parts[1],
                "util_percent": num(parts[2]),
                "mem_used_mb": num(parts[3]), "mem_total_mb": num(parts[4]),
                "temp_c": num(parts[5]),
                "power_w": num(parts[6]) if len(parts) > 6 else 0.0,
                "power_limit_w": num(parts[7]) if len(parts) > 7 else 0.0,
            }
            g["mem_percent"] = (g["mem_used_mb"] / g["mem_total_mb"] * 100
                                if g["mem_total_mb"] else 0.0)
            gpus.append(g)
        return gpus

    def _query_train_proc(self):
        if not self._tracked_pid:
            return None
        try:
            proc = psutil.Process(self._tracked_pid)
            children = proc.children(recursive=True)
            rss = proc.memory_info().rss
            cpu = proc.cpu_percent(interval=0)
            threads = proc.num_threads()
            for c in children:
                try:
                    rss += c.memory_info().rss
                    cpu += c.cpu_percent(interval=0)
                    threads += c.num_threads()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return {"pid": proc.pid, "rss_gb": rss / 1024 ** 3,
                    "cpu_percent": cpu, "num_threads": threads}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
