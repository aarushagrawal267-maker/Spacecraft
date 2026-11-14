"""Compute-device selection. CPU always works; CUDA is used only when available."""

from __future__ import annotations

import os
import platform


def resolve_device(preference: str = "auto") -> str:
    import torch

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return "cuda"
    if preference == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def system_info() -> dict:
    import torch

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
    }
    try:
        import psutil  # optional

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 2**30, 1)
        info["ram_available_gb"] = round(vm.available / 2**30, 1)
    except Exception:  # noqa: BLE001
        info.update(_ram_fallback())
    return info


def _ram_fallback() -> dict:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return {"ram_total_gb": round(stat.ullTotalPhys / 2**30, 1),
                    "ram_available_gb": round(stat.ullAvailPhys / 2**30, 1)}
        with open("/proc/meminfo") as f:
            vals = {line.split(":")[0]: int(line.split()[1]) for line in f}
        return {"ram_total_gb": round(vals["MemTotal"] / 2**20, 1),
                "ram_available_gb": round(vals.get("MemAvailable", 0) / 2**20, 1)}
    except Exception:  # noqa: BLE001
        return {"ram_total_gb": None, "ram_available_gb": None}
