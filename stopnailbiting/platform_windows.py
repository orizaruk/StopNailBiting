"""Windows-specific helpers."""

import ctypes
import sys


def set_windows_dpi_awareness():
    """Enable DPI awareness on Windows so geometry matches real monitor pixels."""
    if sys.platform != "win32":
        return

    try:
        # Per-monitor v1 awareness (Windows 8.1+).
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        print("[Display] DPI awareness set (per-monitor)")
        return
    except Exception:
        pass

    try:
        # Legacy fallback for older systems.
        ctypes.windll.user32.SetProcessDPIAware()
        print("[Display] DPI awareness set (system)")
    except Exception as e:
        print(f"[Display] Failed to set DPI awareness: {e}")
