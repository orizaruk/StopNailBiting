"""Fullscreen red flash alert windows across all monitors (tkinter)."""

import tkinter as tk

from screeninfo import get_monitors


class RedFlashAlert:
    """Multi-monitor red screen flash for nail biting alerts."""

    def __init__(self):
        """Initialize alert windows for all connected monitors.

        Creates hidden fullscreen red Tkinter windows and keeps enough state
        to rebuild them if the display topology changes while the app is running.
        """
        self.windows = []
        self.is_showing = False
        self._monitor_signature = ()
        self._needs_rebuild = False
        self._build_windows_for_current_monitors()

    def _capture_monitor_layout(self):
        """Return monitor list and a stable signature for change detection."""
        monitors = get_monitors()
        signature = tuple(
            sorted(
                (monitor.x, monitor.y, monitor.width, monitor.height)
                for monitor in monitors
            )
        )
        return monitors, signature

    def _destroy_windows(self):
        """Destroy all existing Tk windows and reset window state."""
        for window in self.windows:
            try:
                window.destroy()
            except Exception:
                pass
        self.windows = []
        self.is_showing = False

    def _build_windows(self, monitors, signature):
        """Create a fullscreen window for each monitor in the provided layout."""
        print(f"[RedFlash] Detected {len(monitors)} monitor(s)")

        for i, monitor in enumerate(monitors):
            # First window is Tk root, subsequent are Toplevel
            if not self.windows:
                window = tk.Tk()
            else:
                window = tk.Toplevel(self.windows[0])

            window.title("Alert")
            window.configure(background="red")
            window.overrideredirect(True)
            window.attributes("-topmost", True)

            # Position and size to cover this monitor exactly
            window.geometry(f"{monitor.width}x{monitor.height}+{monitor.x}+{monitor.y}")
            print(
                f"[RedFlash] Monitor {i + 1}: {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})"
            )

            label = tk.Label(
                window,
                text="⚠️ STOP NAIL BITING ⚠️",
                font=("Arial", 48, "bold"),
                fg="white",
                bg="red",
            )
            label.place(relx=0.5, rely=0.5, anchor="center")

            window.withdraw()
            self.windows.append(window)

        self._monitor_signature = signature
        if self.windows:
            self.windows[0].update()

    def _build_windows_for_current_monitors(self):
        """Build windows from the currently connected monitor layout."""
        monitors, signature = self._capture_monitor_layout()
        self._build_windows(monitors, signature)
        self._needs_rebuild = False

    def _rebuild_windows(self):
        """Recreate alert windows based on the latest monitor configuration."""
        print("[RedFlash] Rebuilding alert windows for current monitor layout")
        self._destroy_windows()
        try:
            self._build_windows_for_current_monitors()
        except Exception as e:
            print(f"[RedFlash] Rebuild failed: {e}")
            self._needs_rebuild = True

    def flash(self):
        """Show alert on all monitors"""
        if self.is_showing:
            return

        try:
            _, current_signature = self._capture_monitor_layout()
            monitor_layout_changed = current_signature != self._monitor_signature
        except Exception as e:
            print(f"[RedFlash] Failed to read monitor layout: {e}")
            self._needs_rebuild = True
            monitor_layout_changed = False

        if self._needs_rebuild or monitor_layout_changed or not self.windows:
            self._rebuild_windows()

        if not self.windows:
            return

        for window in self.windows:
            window.deiconify()
            window.lift()
            window.attributes("-topmost", True)
        self.is_showing = True
        print("Red flash activated")

    def update(self):
        """Process tkinter events"""
        if self.windows:
            try:
                self.windows[0].update()
            except Exception as e:
                print(f"[RedFlash] Update error: {e}")
                self._needs_rebuild = True

    def hide(self):
        """Hide alert on all monitors"""
        if self.is_showing:
            try:
                for window in self.windows:
                    window.withdraw()
                if self.windows:
                    self.windows[0].update()
            except Exception as e:
                print(f"[RedFlash] Hide error: {e}")
                self._needs_rebuild = True
            self.is_showing = False
            print("Red flash deactivated")

    def cleanup(self):
        """Destroy all windows"""
        self._destroy_windows()

