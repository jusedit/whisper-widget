"""Performance tracking for Whisper Widget.

Provides PerfTimer context manager for structured timing logs and
log_memory() for process memory snapshots. All output uses the
'whisper' logger with PERF prefix for easy grep/parsing.

Usage:
    from perf_logger import PerfTimer, log_memory

    with PerfTimer("model.encoder"):
        session.run(...)
    # Logs: PERF model.encoder: 150ms

    log_memory("after_model_load")
    # Logs: PERF memory [after_model_load]: rss=450MB peak=500MB
"""

import time
import logging

log = logging.getLogger("whisper")


class PerfTimer:
    """Context manager that logs elapsed time for a named operation."""

    def __init__(self, name: str):
        self.name = name
        self.elapsed_ms = 0.0
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        log.info("PERF %s: %.0fms", self.name, self.elapsed_ms)
        return False


def log_memory(label: str = ""):
    """Log current process memory usage via Windows API."""
    try:
        import ctypes
        from ctypes import c_size_t, Structure, sizeof, byref, c_ulong

        class PMC(Structure):
            _fields_ = [
                ("cb", c_ulong), ("PageFaultCount", c_ulong),
                ("PeakWorkingSetSize", c_size_t), ("WorkingSetSize", c_size_t),
                ("QuotaPeakPagedPoolUsage", c_size_t), ("QuotaPagedPoolUsage", c_size_t),
                ("QuotaPeakNonPagedPoolUsage", c_size_t), ("QuotaNonPagedPoolUsage", c_size_t),
                ("PagefileUsage", c_size_t), ("PeakPagefileUsage", c_size_t),
            ]

        pmc = PMC()
        pmc.cb = sizeof(PMC)
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), byref(pmc), pmc.cb
        ):
            tag = f" [{label}]" if label else ""
            log.info(
                "PERF memory%s: rss=%.0fMB peak=%.0fMB",
                tag, pmc.WorkingSetSize / 1048576, pmc.PeakWorkingSetSize / 1048576,
            )
    except Exception:
        pass
