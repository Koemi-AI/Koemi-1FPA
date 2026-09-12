from __future__ import annotations

import sys

import torch


CUDA_PEAK_SOURCE = "cuda_max_memory_allocated"
PROCESS_PEAK_SOURCE = "process_peak_resident_set"


def measure_peak_memory(device: str) -> tuple[int, str]:
    if device.startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated()), CUDA_PEAK_SOURCE
    return process_peak_resident_bytes(), PROCESS_PEAK_SOURCE


def process_peak_resident_bytes() -> int:
    if sys.platform == "win32":
        return windows_peak_working_set_bytes()
    import resource

    maximum_resident = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return int(maximum_resident) * scale


def windows_peak_working_set_bytes() -> int:
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    current_process = ctypes.windll.kernel32.GetCurrentProcess
    current_process.restype = wintypes.HANDLE
    query_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    query_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    query_memory.restype = wintypes.BOOL
    if not query_memory(current_process(), ctypes.byref(counters), counters.cb):
        raise RuntimeError("GetProcessMemoryInfo failed while measuring peak working set")
    return int(counters.PeakWorkingSetSize)
