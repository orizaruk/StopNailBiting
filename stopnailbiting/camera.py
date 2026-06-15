"""Webcam enumeration (ffmpeg/WMI on Windows) and capture management."""

import re
import subprocess
import sys

import cv2


class CameraManager:
    """Handles camera enumeration and opening logic."""

    INDEX_PREFIX = "__index__:"
    MAX_INDEX_PROBE = 10

    def __init__(self):
        self._label_by_selection = {}
        self._selection_by_label = {}

    def _make_index_selection(self, index):
        return f"{self.INDEX_PREFIX}{index}"

    def _parse_index_selection(self, selection):
        if not isinstance(selection, str):
            return None
        if not selection.startswith(self.INDEX_PREFIX):
            return None
        try:
            return int(selection[len(self.INDEX_PREFIX) :])
        except ValueError:
            return None

    def selection_to_label(self, selection):
        if selection is None:
            return "Auto / Default"
        if selection in self._label_by_selection:
            return self._label_by_selection[selection]
        index = self._parse_index_selection(selection)
        if index is not None:
            return f"Camera {index}"
        return selection

    def selection_to_log_label(self, selection):
        label = self.selection_to_label(selection)
        return str(label).encode("ascii", errors="backslashreplace").decode("ascii")

    def _list_windows_camera_names_ffmpeg(self):
        """List Windows camera names in DirectShow order via ffmpeg."""
        if sys.platform != "win32":
            return []

        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-list_devices",
                    "true",
                    "-f",
                    "dshow",
                    "-i",
                    "dummy",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
                check=False,
            )
            output = f"{result.stdout}\n{result.stderr}"
            output = re.sub(r"\x1b\[[0-9;]*m", "", output)
        except Exception as e:
            print(f"[Camera] ffmpeg DirectShow enumeration failed: {e}")
            return []

        pattern = re.compile(r'"([^"]+)"\s+\((video|none)\)')
        names = []
        seen = set()
        for match in pattern.finditer(output):
            name = match.group(1).strip()
            if not name:
                continue
            normalized = name.casefold()
            if normalized in seen:
                continue
            names.append(name)
            seen.add(normalized)
        return names

    def _list_windows_camera_names_wmi(self):
        if sys.platform != "win32":
            return []

        ps_script = (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' -or $_.Service -eq 'usbvideo' } | "
            "Select-Object -ExpandProperty Name"
        )

        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            names = [
                line.strip() for line in result.stdout.splitlines() if line.strip()
            ]
        except Exception as e:
            print(f"[Camera] Failed to enumerate Windows camera names: {e}")
            return []

        unique_names = []
        seen = set()
        for name in names:
            normalized = name.casefold()
            if normalized in seen:
                continue
            unique_names.append(name)
            seen.add(normalized)
        return unique_names

    def _list_windows_camera_names(self):
        """List Windows camera names with best-effort stable ordering."""
        names = self._list_windows_camera_names_ffmpeg()
        if names:
            return names
        return self._list_windows_camera_names_wmi()

    def _probe_camera_indices(self):
        indices = []
        for index in range(self.MAX_INDEX_PROBE):
            cap = self._open_camera_by_index(index)
            if cap is None:
                continue
            cap.release()
            indices.append(index)
        return indices

    def list_camera_choices(self):
        """Return a list of (selection_id, label) tuples."""
        if sys.platform == "win32":
            names = self._list_windows_camera_names()
            choices = []
            for index, name in enumerate(names):
                choices.append(
                    (self._make_index_selection(index), f"{name} (Camera {index})")
                )
            self._label_by_selection = {
                selection: label for selection, label in choices
            }
            self._selection_by_label = {
                label.casefold(): selection for selection, label in choices
            }
            for index, name in enumerate(names):
                self._selection_by_label.setdefault(
                    name.casefold(),
                    self._make_index_selection(index),
                )
            return choices

        indices = self._probe_camera_indices()
        choices = [
            (self._make_index_selection(index), f"Camera {index}") for index in indices
        ]
        self._label_by_selection = {selection: label for selection, label in choices}
        self._selection_by_label = {
            label.casefold(): selection for selection, label in choices
        }
        return choices

    def _open_camera_by_name(self, name):
        # This OpenCV build cannot open Windows cameras by name.
        return None

    def _open_camera_by_index(self, index):
        if sys.platform == "win32":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap
            if cap is not None:
                cap.release()

        cap = cv2.VideoCapture(index)

        if cap is not None and cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        if cap is not None:
            cap.release()
        return None

    def _open_camera_by_selection(self, selection):
        index = self._parse_index_selection(selection)
        if index is not None:
            return self._open_camera_by_index(index)
        if sys.platform == "win32":
            mapped = self._selection_by_label.get(str(selection).casefold())
            if mapped:
                mapped_index = self._parse_index_selection(mapped)
                if mapped_index is not None:
                    return self._open_camera_by_index(mapped_index)
        return self._open_camera_by_name(selection)

    def open_camera(self, preferred_selection=None, fallback=True):
        """Open preferred camera and optionally fallback to another working camera."""
        choices = self.list_camera_choices()
        resolved_preferred = preferred_selection

        if isinstance(preferred_selection, str):
            mapped = self._selection_by_label.get(preferred_selection.casefold())
            if mapped:
                resolved_preferred = mapped

        if resolved_preferred:
            cap = self._open_camera_by_selection(resolved_preferred)
            if cap is not None:
                return cap, resolved_preferred
            if not fallback:
                return None, resolved_preferred

        if not fallback:
            return None, None

        if sys.platform == "win32":
            for selection, _label in choices:
                if selection == resolved_preferred:
                    continue
                cap = self._open_camera_by_selection(selection)
                if cap is not None:
                    return cap, selection
            return None, None

        for index in range(self.MAX_INDEX_PROBE):
            selection = self._make_index_selection(index)
            if selection == resolved_preferred:
                continue
            cap = self._open_camera_by_index(index)
            if cap is not None:
                return cap, selection

        return None, None

