"""
Stop Nail Biting - Real-time nail biting detection and alert system.

This application uses computer vision (MediaPipe) to detect when fingers are near
the mouth, indicating potential nail-biting behavior. When detected, it triggers
visual (red screen flash) and audio alerts to help break the habit.

The app runs silently in the system tray and supports:
- Multi-monitor visual alerts
- Configurable audio alerts with volume control
- Drinking detection to reduce false positives (cups, bottles, glasses)
- Persistent settings saved to user config directory
- Windows startup integration
"""

import cv2
import mediapipe as mp
from shapely import Point, Polygon
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import pygame
import tkinter as tk
import time
import os
import sys
import threading
import queue
import json
import re
from datetime import datetime, timedelta
import subprocess
from PIL import Image, ImageDraw
import pystray
from screeninfo import get_monitors

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    winsound = None
    WINSOUND_AVAILABLE = False

# Optional WinRT import for media control (graceful degradation if unavailable)
GlobalSystemMediaTransportControlsSessionManager = None
GlobalSystemMediaTransportControlsSessionPlaybackStatus = None
try:
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager,
        GlobalSystemMediaTransportControlsSessionPlaybackStatus,
    )
    WINRT_AVAILABLE = True
except ImportError:
    WINRT_AVAILABLE = False

# Optional CoreAudio (pycaw) import for browser audio muting fallback
comtypes = None
AudioUtilities = None
try:
    # comtypes generates python modules on demand. When frozen (PyInstaller), the
    # default generation directory may be non-writable, so we pin it to AppData.
    if sys.platform == "win32":
        _comtypes_base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        _comtypes_base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    os.environ.setdefault(
        "COMTYPES_GEN_DIR",
        os.path.join(_comtypes_base, "StopNailBiting", "comtypes_gen"),
    )

    import comtypes
    from pycaw.pycaw import AudioUtilities
    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False


class ConfigManager:
    """Manages persistent configuration settings"""

    DEFAULT_CONFIG = {
        "flash_enabled": True,
        "sound_enabled": True,
        "start_with_windows": False,
        "volume": 0.75,
        "drinking_detection_enabled": True,
        "pause_media_on_alert": True,
        "camera_name": None,
    }

    def __init__(self):
        """Initialize config manager and load settings from disk.

        Loads existing config from %APPDATA%/StopNailBiting/config.json on Windows,
        or creates a new config file with defaults if none exists.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.config_dir = self._get_config_dir()
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.load()

    def _get_config_dir(self):
        """Get the config directory path (platform-specific)"""
        if sys.platform == "win32":
            base = os.environ.get("APPDATA", os.path.expanduser("~"))
        else:
            base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        return os.path.join(base, "StopNailBiting")

    def load(self):
        """Load config from file, create default if doesn't exist"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    loaded = json.load(f)
                    # Merge with defaults (in case new settings are added)
                    self.config = {**self.DEFAULT_CONFIG, **loaded}
                print(f"[Config] Loaded from {self.config_file}")
            else:
                # Create default config
                self.save()
                print(f"[Config] Created default config at {self.config_file}")
        except Exception as e:
            print(f"[Config] Error loading config: {e}, using defaults")
            self.config = self.DEFAULT_CONFIG.copy()

    def save(self):
        """Save current config to file"""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"[Config] Error saving config: {e}")

    def get(self, key):
        """Get a config value"""
        return self.config.get(key, self.DEFAULT_CONFIG.get(key))

    def set(self, key, value):
        """Set a config value and save"""
        self.config[key] = value
        self.save()


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
            return int(selection[len(self.INDEX_PREFIX):])
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
                ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
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
            names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
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
                choices.append((self._make_index_selection(index), f"{name} (Camera {index})"))
            self._label_by_selection = {selection: label for selection, label in choices}
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
        choices = [(self._make_index_selection(index), f"Camera {index}") for index in indices]
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

    def open_camera(self, preferred_selection=None):
        """Open preferred camera if possible, else fallback to first working one."""
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


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller."""
    if hasattr(sys, "_MEIPASS"):
        # Running as PyInstaller bundle
        return os.path.join(sys._MEIPASS, relative_path)
    # Running in development
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def find_sound_file(base_name="noise"):
    """Find a sound file with supported extension in assets folder.

    Searches for base_name with .mp3, .wav, .ogg extensions.
    Returns the path if found, None otherwise.
    """
    assets_dir = resource_path("assets")
    supported_extensions = [".mp3", ".wav", ".ogg"]

    for ext in supported_extensions:
        path = os.path.join(assets_dir, base_name + ext)
        if os.path.exists(path):
            return path
    return None


# Configuration
SENSITIVITY = 0.01  # Lip polygon buffer size
Z_DEPTH_THRESHOLD = 0.1  # Max z-difference between finger and lips for valid detection
FRAMES_REQUIRED = 3  # Consecutive frames needed before triggering alert
TARGET_FPS = 15  # Target frame rate to reduce CPU usage
COOLDOWN_PERIOD = 1.5  # Time in seconds to keep alert visible after biting stops
SOUND_FILE = find_sound_file("noise")  # Auto-detects .mp3, .wav, or .ogg in assets/

hand_model_path = resource_path(os.path.join("models", "hand_landmarker.task"))
face_model_path = resource_path(os.path.join("models", "face_landmarker.task"))
object_model_path = resource_path(os.path.join("models", "efficientdet_lite0.tflite"))

# Drinking detection constants (to reduce false positives when drinking)
DRINKING_DETECTION_INTERVAL = 3  # Run object detection every N frames
DRINKING_CONFIDENCE_THRESHOLD = 0.35  # Minimum confidence for drinking detection
DRINKING_PERSISTENCE_FRAMES = 30  # Frames to persist drinking detection state (~2 seconds at 15 FPS)
DRINKING_CLASS_LABELS = {"cup", "bottle", "wine glass"}

# Detection confidence thresholds (reduce false positives in low light)
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_PRESENCE_CONFIDENCE = 0.5

# Initializations of options to configure the model
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions

# Create a hand landmarker instance with video mode for temporal smoothing:
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
    min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
)

object_options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=object_model_path),
    running_mode=VisionRunningMode.VIDEO,
    max_results=5,
    score_threshold=DRINKING_CONFIDENCE_THRESHOLD,
)

# MediaPipe FaceLandmarker indices that form the outer lip contour polygon.
# These 21 points trace around the lips to create a closed shape for collision detection.
LIP_INDICES = [
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
]

# MediaPipe HandLandmarker indices for fingertips and finger joints to check.
# Pairs: (tip, joint below) for thumb(4,3), index(8,7), middle(12,11), ring(16,15), pinky(20,19)
HAND_INDICES = [4, 3, 8, 7, 12, 11, 16, 15, 20, 19]


class RedFlashAlert:
    """Multi-monitor red screen flash for nail biting alerts."""

    def __init__(self):
        """Initialize alert windows for all connected monitors.

        Creates a hidden fullscreen red Tkinter window for each detected monitor.
        The first window is the Tk root; subsequent windows are Toplevel children.
        """
        self.windows = []
        self.is_showing = False
        self._init_windows()

    def _init_windows(self):
        """Create a fullscreen window for each monitor"""
        monitors = get_monitors()
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
            print(f"[RedFlash] Monitor {i+1}: {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})")

            # Add warning label
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

        # Process events
        if self.windows:
            self.windows[0].update()

    def flash(self):
        """Show alert on all monitors"""
        if not self.is_showing:
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
                print(f"Update error: {e}")

    def hide(self):
        """Hide alert on all monitors"""
        if self.is_showing:
            for window in self.windows:
                window.withdraw()
            if self.windows:
                self.windows[0].update()
            self.is_showing = False
            print("Red flash deactivated")

    def cleanup(self):
        """Destroy all windows"""
        for window in self.windows:
            try:
                window.destroy()
            except Exception:
                pass  # Window may already be destroyed
        self.windows = []


class SoundManager:
    """Manages sound alerts with anti-flicker protection and graceful degradation."""

    def __init__(self, sound_file, volume=0.75):
        """Initialize sound manager with optional audio file.

        Args:
            sound_file: Path to the alert sound file (.mp3, .wav, or .ogg).
                       If None or file doesn't exist, runs in visual-only mode.
            volume: Initial volume level from 0.0 (silent) to 1.0 (full).
        """
        self.enabled = False
        self.sound_playing = False
        self.alert_sound = None
        self.volume = volume
        self.backend = "none"
        self._beep_stop_event = threading.Event()
        self._beep_thread = None

        if sound_file is not None and os.path.exists(sound_file):
            try:
                pygame.mixer.init()
                self.alert_sound = pygame.mixer.Sound(sound_file)
                self.alert_sound.set_volume(self.volume)
                self.enabled = True
                self.backend = "pygame"
                print(f"[Info] Sound loaded successfully: {sound_file} (volume: {int(volume * 100)}%)")
                return
            except pygame.error as e:
                print(f"[Warning] Failed to initialize pygame audio: {e}")
            except Exception as e:
                print(f"[Warning] Unexpected error loading sound file: {e}")

        if sound_file is None:
            print("[Warning] No sound file found (expected assets/noise.mp3|.wav|.ogg)")
        else:
            print(f"[Warning] Sound file not found: {sound_file}")

        if WINSOUND_AVAILABLE and sys.platform == "win32":
            self.enabled = True
            self.backend = "winsound"
            print("[Info] Using Windows beep fallback for alert sound")
        else:
            print("[Warning] No audio backend available - running in visual-only mode")

    def _beep_loop(self):
        while not self._beep_stop_event.is_set():
            try:
                winsound.Beep(1400, 180)
            except Exception:
                break
            if self._beep_stop_event.wait(0.12):
                break

    def start_sound(self):
        """Start the alert sound if not already playing"""
        if not self.enabled or self.sound_playing:
            return

        if self.backend == "pygame":
            self.alert_sound.play(loops=-1)
            self.sound_playing = True
            print("Alert sound started")
            return

        if self.backend == "winsound":
            self._beep_stop_event.clear()
            self._beep_thread = threading.Thread(target=self._beep_loop, daemon=True)
            self._beep_thread.start()
            self.sound_playing = True
            print("Alert sound started")

    def stop_sound(self):
        """Stop the alert sound"""
        if not self.enabled or not self.sound_playing:
            return

        if self.backend == "pygame":
            self.alert_sound.stop()
        elif self.backend == "winsound":
            self._beep_stop_event.set()
            if self._beep_thread is not None:
                self._beep_thread.join(timeout=0.5)
                self._beep_thread = None

        self.sound_playing = False
        print("Alert sound stopped")

    def set_volume(self, volume):
        """Set the alert sound volume (0.0 to 1.0)"""
        self.volume = volume
        if self.backend == "pygame" and self.alert_sound:
            self.alert_sound.set_volume(volume)
            print(f"[Sound] Volume set to {int(volume * 100)}%")
        elif self.backend == "winsound":
            print("[Sound] Volume changes are not supported for Windows beep fallback")

    def cleanup(self):
        """Release pygame mixer resources"""
        self.stop_sound()
        if self.backend == "pygame":
            pygame.mixer.quit()


class MediaController:
    """Pauses and resumes system media during alerts using Windows SMTC.

    Uses the Windows System Media Transport Controls API to pause any currently
    playing media (Spotify, YouTube, etc.) when an alert triggers, and resume
    it when the alert ends. Gracefully degrades if WinRT is unavailable.

    Also includes an optional CoreAudio fallback (pycaw) to temporarily mute
    browser audio during alerts. This is necessary because Chromium browsers
    typically expose only a single SMTC session even when multiple tabs play
    audio.

    Note: WinRT async operations and CoreAudio COM calls are executed on a
    dedicated worker thread to avoid blocking the main detection loop.
    """

    def __init__(self):
        """Initialize the media controller."""
        self.paused_session_ids = set()
        self.enabled = WINRT_AVAILABLE
        self._muted_browser_session_states = {}
        self._muted_browser_pids = set()
        self._browser_process_names = {
            "brave.exe",
            "chrome.exe",
            "msedge.exe",
            "firefox.exe",
            "opera.exe",
            "opera_gx.exe",
            "vivaldi.exe",
        }

        self._queue = queue.Queue()
        self._worker_thread = None
        if self.enabled or PYCAW_AVAILABLE:
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

        if self.enabled and PYCAW_AVAILABLE:
            print("[MediaController] SMTC + CoreAudio fallback enabled")
        elif self.enabled:
            print("[MediaController] SMTC enabled (CoreAudio fallback unavailable)")
        elif PYCAW_AVAILABLE:
            print("[MediaController] CoreAudio fallback enabled (SMTC unavailable)")
        else:
            print("[MediaController] Media control unavailable (WinRT + pycaw missing)")

    def _worker_loop(self):
        """Run queued media operations on a dedicated worker thread."""
        if PYCAW_AVAILABLE and comtypes is not None:
            try:
                # WinRT blocking waits (op.get()) require an MTA thread.
                # CoreAudio (pycaw) is also fine in MTA.
                comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
                print("[MediaController] Worker initialized (MTA)")
            except Exception:
                # Fall back to STA if MTA init fails; CoreAudio may still work,
                # but WinRT SMTC blocking waits may fail in this mode.
                try:
                    comtypes.CoInitialize()
                    print("[MediaController] Worker initialized (STA fallback)")
                except Exception:
                    pass

        try:
            while True:
                fn = self._queue.get()
                if fn is None:
                    break
                try:
                    fn()
                except Exception as e:
                    print(f"[MediaController] Worker error: {e}")
                finally:
                    self._queue.task_done()
        finally:
            if PYCAW_AVAILABLE and comtypes is not None:
                try:
                    comtypes.CoUninitialize()
                except Exception:
                    pass

    def _submit(self, fn, wait=False, timeout=2.0):
        """Submit a media operation to the worker thread."""
        if self._worker_thread is None:
            return

        done = threading.Event()

        def _wrapped():
            try:
                fn()
            finally:
                done.set()

        self._queue.put(_wrapped)
        if wait:
            done.wait(timeout=timeout)

    def _get_manager(self):
        """Get the SMTC session manager (must be called from worker thread)."""
        return GlobalSystemMediaTransportControlsSessionManager.request_async().get()

    def _audio_session_key(self, audio_session):
        """Generate a stable-ish key for an audio session.

        Uses (pid, session instance id) when available, and falls back to
        (pid, display name) when necessary.
        """
        pid = None
        try:
            if getattr(audio_session, "Process", None) is not None:
                pid = audio_session.Process.pid
        except Exception:
            pid = None

        if pid is None:
            try:
                pid = audio_session._ctl.GetProcessId()
            except Exception:
                pid = -1

        instance_id = None
        try:
            instance_id = audio_session._ctl.GetSessionInstanceIdentifier()
        except Exception:
            instance_id = None

        if instance_id:
            return (pid, instance_id)

        session_id = None
        try:
            session_id = audio_session._ctl.GetSessionIdentifier()
        except Exception:
            session_id = None

        if session_id:
            return (pid, session_id)

        display_name = ""
        try:
            display_name = getattr(audio_session, "DisplayName", "") or ""
        except Exception:
            display_name = ""

        return (pid, display_name)

    def _mute_browser_audio_sessions(self):
        """Mute browser CoreAudio sessions and record previous mute state."""
        if not PYCAW_AVAILABLE:
            return

        try:
            sessions = AudioUtilities.GetAllSessions()
        except Exception as e:
            print(f"[MediaController] Error enumerating audio sessions: {e}")
            return

        muted_count = 0
        for s in sessions:
            try:
                proc = getattr(s, "Process", None)
                if proc is None:
                    continue
                name = (proc.name() or "").lower()
                if name not in self._browser_process_names:
                    continue

                vol = s.SimpleAudioVolume
                prev_muted = bool(vol.GetMute())
                if prev_muted:
                    continue

                key = self._audio_session_key(s)
                if key in self._muted_browser_session_states:
                    continue

                vol.SetMute(True, None)
                self._muted_browser_session_states[key] = prev_muted
                self._muted_browser_pids.add(proc.pid)
                muted_count += 1
            except Exception:
                continue

        if muted_count:
            print(f"[MediaController] Muted {muted_count} browser audio session(s)")

    def _restore_browser_audio_sessions(self):
        """Restore browser CoreAudio sessions muted by this controller."""
        if not PYCAW_AVAILABLE or not self._muted_browser_session_states:
            return

        states = self._muted_browser_session_states.copy()
        muted_pids = self._muted_browser_pids.copy()

        try:
            sessions = AudioUtilities.GetAllSessions()
        except Exception as e:
            print(f"[MediaController] Error enumerating audio sessions: {e}")
            return

        restored = 0
        remaining = set(states.keys())
        for s in sessions:
            try:
                key = self._audio_session_key(s)
                if key not in states:
                    continue
                s.SimpleAudioVolume.SetMute(states[key], None)
                restored += 1
                remaining.discard(key)
            except Exception:
                continue

        # If no sessions matched (keys can be unstable for some drivers/apps),
        # fall back to unmuting browser sessions for pids we muted.
        if restored == 0 and muted_pids:
            for s in sessions:
                try:
                    proc = getattr(s, "Process", None)
                    if proc is None or proc.pid not in muted_pids:
                        continue
                    name = (proc.name() or "").lower()
                    if name not in self._browser_process_names:
                        continue
                    vol = s.SimpleAudioVolume
                    if bool(vol.GetMute()):
                        vol.SetMute(False, None)
                except Exception:
                    continue

        # Clear state after restore attempt to avoid unmuting later unrelated sessions.
        self._muted_browser_session_states.clear()
        self._muted_browser_pids.clear()

        if restored:
            print(f"[MediaController] Restored {restored} browser audio session(s)")

    def pause_all(self):
        """Pause all currently playing media sessions and store their IDs."""
        if not self.enabled and not PYCAW_AVAILABLE:
            return

        def _do_pause():
            if self.enabled:
                self.paused_session_ids.clear()
                try:
                    manager = self._get_manager()
                    for session in manager.get_sessions():
                        info = session.get_playback_info()
                        if info.playback_status == GlobalSystemMediaTransportControlsSessionPlaybackStatus.PLAYING:
                            app_id = session.source_app_user_model_id
                            if session.try_pause_async().get():
                                self.paused_session_ids.add(app_id)
                                print(f"[MediaController] Paused: {app_id}")
                except Exception as e:
                    print(f"[MediaController] Error pausing media: {e}")

            # Browser CoreAudio fallback: silences additional tabs that SMTC can't pause.
            self._mute_browser_audio_sessions()

        self._submit(_do_pause)

    def resume_all(self, wait=False):
        """Resume only the sessions that were paused by this controller."""
        if not self.enabled and not PYCAW_AVAILABLE:
            return

        if not self.paused_session_ids and not self._muted_browser_session_states:
            return

        def _do_resume():
            if self.enabled and self.paused_session_ids:
                session_ids_to_resume = self.paused_session_ids.copy()
                self.paused_session_ids.clear()
                try:
                    manager = self._get_manager()
                    for session in manager.get_sessions():
                        if session.source_app_user_model_id in session_ids_to_resume:
                            info = session.get_playback_info()
                            if info.playback_status == GlobalSystemMediaTransportControlsSessionPlaybackStatus.PAUSED:
                                session.try_play_async().get()
                                print(f"[MediaController] Resumed: {session.source_app_user_model_id}")
                except Exception as e:
                    print(f"[MediaController] Error resuming media: {e}")

            self._restore_browser_audio_sessions()

        self._submit(_do_resume, wait=wait)


class AppController:
    """Controls app state and system tray integration."""

    def __init__(self, config, sound_manager, camera_manager):
        """Initialize app controller with config and integration managers."""
        self.config = config
        self.sound_manager = sound_manager
        self.camera_manager = camera_manager
        self.running = True
        self.paused = False
        self.pause_until = None
        self.pause_timer = None
        self.icon = None
        self.icon_active = self._create_icon_image(active=True)
        self.icon_paused = self._create_icon_image(active=False)
        self.camera_unavailable = False
        self.camera_choices = self.camera_manager.list_camera_choices()
        self.active_camera_selection = self.config.get("camera_name")
        self._pending_camera_switch = None
        self._has_pending_camera_switch = False
        self._camera_switch_lock = threading.Lock()

    def _create_icon_image(self, active=True):
        """Create a 64x64 tray icon image."""
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        fill_color = (220, 53, 69) if active else (128, 128, 128)
        margin = 4
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=fill_color,
            outline=(255, 255, 255),
            width=2,
        )
        center = size // 2
        draw.rectangle(
            [center - 8, center - 12, center + 8, center + 12],
            fill=(255, 255, 255),
        )
        return img

    def _current_status_text(self):
        if self.camera_unavailable:
            return "No camera"
        if self.paused:
            return "Paused"
        return "Monitoring"

    def _update_icon_and_title(self):
        if not self.icon:
            return

        status = self._current_status_text()
        is_paused_visual = self.paused or self.camera_unavailable
        self.icon.icon = self.icon_paused if is_paused_visual else self.icon_active
        camera_label = self.camera_manager.selection_to_label(self.active_camera_selection)
        self.icon.title = f"Stop Nail Biting - {status} ({camera_label})"

    def _update_menu(self):
        if not self.icon:
            return
        self.icon.menu = self._build_main_menu()
        try:
            self.icon.update_menu()
        except Exception:
            pass

    def set_camera_unavailable(self, unavailable):
        self.camera_unavailable = unavailable
        self._update_icon_and_title()
        self._update_menu()

    def set_active_camera(self, selection):
        self.active_camera_selection = selection
        self._update_icon_and_title()
        self._update_menu()

    def refresh_camera_choices(self, icon=None, item=None):
        self.camera_choices = self.camera_manager.list_camera_choices()
        print(f"[Tray] Cameras refreshed ({len(self.camera_choices)} found)")
        self._update_menu()

    def request_camera_switch(self, selection):
        with self._camera_switch_lock:
            self._pending_camera_switch = selection
            self._has_pending_camera_switch = True
        self.config.set("camera_name", selection)
        print(f"[Tray] Camera switch requested: {self.camera_manager.selection_to_log_label(selection)}")

    def consume_camera_switch_request(self):
        with self._camera_switch_lock:
            if not self._has_pending_camera_switch:
                return False, None
            selection = self._pending_camera_switch
            self._pending_camera_switch = None
            self._has_pending_camera_switch = False
        return True, selection

    def toggle_pause(self, icon, item):
        """Toggle pause state and update icon."""
        if self.paused:
            self._cancel_timed_pause()
        self.paused = not self.paused
        status = "Paused" if self.paused else "Monitoring"
        print(f"[Tray] {status}")
        self._update_icon_and_title()

    def pause_for_interval(self, minutes):
        """Pause detection for a specific duration."""
        self._cancel_timed_pause()
        self.paused = True
        self.pause_until = datetime.now() + timedelta(minutes=minutes)
        self.pause_timer = threading.Timer(minutes * 60, self._resume_from_timer)
        self.pause_timer.daemon = True
        self.pause_timer.start()
        print(f"[Tray] Paused for {minutes} minutes (until {self.pause_until.strftime('%H:%M')})")
        self._update_icon_and_title()

    def _resume_from_timer(self):
        """Called when timed pause expires to resume detection."""
        self.paused = False
        self.pause_until = None
        self.pause_timer = None
        print("[Tray] Timed pause expired - Monitoring")
        self._update_icon_and_title()

    def _cancel_timed_pause(self):
        """Cancel any active timed pause timer."""
        if self.pause_timer is not None:
            self.pause_timer.cancel()
            self.pause_timer = None
        self.pause_until = None

    def quit_app(self, icon, item):
        """Signal the app to quit."""
        print("[Tray] Quit requested")
        self._cancel_timed_pause()
        self.running = False
        if self.icon:
            self.icon.stop()

    def get_pause_text(self, item):
        """Dynamic menu item text for pause/resume."""
        if not self.paused:
            return "Pause"

        if self.pause_until is not None:
            remaining = self.pause_until - datetime.now()
            if remaining.total_seconds() > 0:
                total_minutes = int(remaining.total_seconds() / 60)
                hours = total_minutes // 60
                minutes = total_minutes % 60
                if hours > 0:
                    return f"Resume ({hours}h {minutes}m remaining)"
                return f"Resume ({minutes}m remaining)"

        return "Resume"

    def is_flash_enabled(self, item):
        return self.config.get("flash_enabled")

    def is_sound_enabled(self, item):
        return self.config.get("sound_enabled")

    def toggle_flash(self, icon, item):
        new_value = not self.config.get("flash_enabled")
        self.config.set("flash_enabled", new_value)
        print(f"[Tray] Flash {'enabled' if new_value else 'disabled'}")

    def toggle_sound(self, icon, item):
        new_value = not self.config.get("sound_enabled")
        self.config.set("sound_enabled", new_value)
        if not new_value:
            self.sound_manager.stop_sound()
        print(f"[Tray] Sound {'enabled' if new_value else 'disabled'}")

    def is_drinking_detection_enabled(self, item):
        return self.config.get("drinking_detection_enabled")

    def toggle_drinking_detection(self, icon, item):
        new_value = not self.config.get("drinking_detection_enabled")
        self.config.set("drinking_detection_enabled", new_value)
        print(f"[Tray] Drinking detection {'enabled' if new_value else 'disabled'}")

    def is_pause_media_enabled(self, item):
        return self.config.get("pause_media_on_alert")

    def toggle_pause_media(self, icon, item):
        new_value = not self.config.get("pause_media_on_alert")
        self.config.set("pause_media_on_alert", new_value)
        print(f"[Tray] Pause media on alert {'enabled' if new_value else 'disabled'}")

    def is_start_with_windows(self, item):
        return self.config.get("start_with_windows")

    def toggle_start_with_windows(self, icon, item):
        new_value = not self.config.get("start_with_windows")
        if new_value:
            success = self._create_startup_shortcut()
        else:
            success = self._remove_startup_shortcut()
        if success:
            self.config.set("start_with_windows", new_value)
            print(f"[Tray] Start with Windows {'enabled' if new_value else 'disabled'}")
        else:
            print("[Tray] Failed to update startup setting")

    def _get_startup_folder(self):
        if sys.platform != "win32":
            return None
        return os.path.join(
            os.environ.get("APPDATA", ""),
            "Microsoft", "Windows", "Start Menu", "Programs", "Startup"
        )

    def _get_shortcut_path(self):
        startup_folder = self._get_startup_folder()
        if startup_folder:
            return os.path.join(startup_folder, "StopNailBiting.lnk")
        return None

    def _get_exe_path(self):
        if getattr(sys, 'frozen', False):
            return sys.executable
        return f'"{sys.executable}" "{os.path.abspath(__file__)}"'

    def _create_startup_shortcut(self):
        if sys.platform != "win32":
            print("[Startup] Not on Windows, skipping")
            return False

        shortcut_path = self._get_shortcut_path()
        exe_path = self._get_exe_path()
        if not shortcut_path:
            return False

        try:
            ps_script = f'''
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{exe_path}"
$Shortcut.WorkingDirectory = "{os.path.dirname(exe_path)}"
$Shortcut.Description = "Stop Nail Biting Detection"
$Shortcut.Save()
'''
            subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                check=True
            )
            print(f"[Startup] Created shortcut at {shortcut_path}")
            return True
        except Exception as e:
            print(f"[Startup] Failed to create shortcut: {e}")
            return False

    def _remove_startup_shortcut(self):
        shortcut_path = self._get_shortcut_path()
        if not shortcut_path:
            return False

        try:
            if os.path.exists(shortcut_path):
                os.remove(shortcut_path)
                print(f"[Startup] Removed shortcut from {shortcut_path}")
            return True
        except Exception as e:
            print(f"[Startup] Failed to remove shortcut: {e}")
            return False

    def _volume_menu_item(self, label, level):
        def is_checked(item):
            return abs(self.config.get("volume") - level) < 0.01

        def set_level(icon, item):
            self.config.set("volume", level)
            self.sound_manager.set_volume(level)

        return pystray.MenuItem(label, set_level, checked=is_checked)

    def _camera_menu_item(self, selection, label):
        def set_camera(icon, item):
            self.request_camera_switch(selection)

        def is_selected(item):
            return self.active_camera_selection == selection

        return pystray.MenuItem(label, set_camera, checked=is_selected)

    def _build_camera_menu(self):
        items = [
            pystray.MenuItem("Refresh Cameras", self.refresh_camera_choices),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Auto / Default",
                lambda icon, item: self.request_camera_switch(None),
                checked=lambda item: self.active_camera_selection is None,
            ),
        ]

        if self.camera_choices:
            for selection, label in self.camera_choices:
                items.append(self._camera_menu_item(selection, label))
        else:
            items.append(
                pystray.MenuItem(
                    "No cameras found",
                    lambda icon, item: None,
                    enabled=False,
                )
            )
        return pystray.Menu(*items)

    def _build_main_menu(self):
        volume_menu = pystray.Menu(
            self._volume_menu_item("100%", 1.0),
            self._volume_menu_item("75%", 0.75),
            self._volume_menu_item("50%", 0.50),
            self._volume_menu_item("25%", 0.25),
        )

        pause_interval_menu = pystray.Menu(
            pystray.MenuItem(
                "30 minutes",
                lambda icon, item: self.pause_for_interval(30),
            ),
            pystray.MenuItem(
                "1 hour",
                lambda icon, item: self.pause_for_interval(60),
            ),
            pystray.MenuItem(
                "2 hours",
                lambda icon, item: self.pause_for_interval(120),
            ),
        )

        alert_settings_menu = pystray.Menu(
            pystray.MenuItem(
                "Enable Flash",
                self.toggle_flash,
                checked=self.is_flash_enabled,
            ),
            pystray.MenuItem(
                "Enable Sound",
                self.toggle_sound,
                checked=self.is_sound_enabled,
            ),
            pystray.MenuItem(
                "Pause Media on Alert",
                self.toggle_pause_media,
                checked=self.is_pause_media_enabled,
            ),
            pystray.MenuItem(
                "Drinking Detection",
                self.toggle_drinking_detection,
                checked=self.is_drinking_detection_enabled,
            ),
        )

        return pystray.Menu(
            pystray.MenuItem("Choose Camera", self._build_camera_menu()),
            pystray.MenuItem("Alert Settings", alert_settings_menu),
            pystray.MenuItem("Volume", volume_menu),
            pystray.MenuItem(
                "Start with Windows",
                self.toggle_start_with_windows,
                checked=self.is_start_with_windows,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Pause for...", pause_interval_menu),
            pystray.MenuItem(
                self.get_pause_text,
                self.toggle_pause,
            ),
            pystray.MenuItem("Quit", self.quit_app),
        )

    def setup_tray(self):
        """Create and configure the system tray icon with menu."""
        self.icon = pystray.Icon(
            "StopNailBiting",
            self.icon_active,
            "Stop Nail Biting - Monitoring",
            self._build_main_menu(),
        )
        self._update_icon_and_title()
        return self.icon

    def run_tray(self):
        """Run the tray icon (call from a separate thread)."""
        icon = self.setup_tray()
        icon.run()


# Initialize config first (needed by other components)
config_manager = ConfigManager()
camera_manager = CameraManager()

# Initialize sound, alert, and media managers
sound_manager = SoundManager(SOUND_FILE, volume=config_manager.get("volume"))
red_flash = RedFlashAlert()
media_controller = MediaController()
app_controller = AppController(config_manager, sound_manager, camera_manager)
cap = None

alert_active = False
last_biting_time = 0  # Track when biting was last detected
consecutive_detections = 0  # Track consecutive frames with detection

# Drinking detection state
drinking_frame_counter = 0
drinking_detected = False
drinking_persistence_counter = 0


def _stop_active_alert():
    """Reset active alert state before camera soft-reset or downtime."""
    global alert_active
    global consecutive_detections
    global last_biting_time
    global drinking_frame_counter
    global drinking_detected
    global drinking_persistence_counter

    if alert_active:
        sound_manager.stop_sound()
        red_flash.hide()
        media_controller.resume_all()
        alert_active = False

    consecutive_detections = 0
    last_biting_time = 0
    drinking_frame_counter = 0
    drinking_detected = False
    drinking_persistence_counter = 0


def _warmup_camera(capture, frames=2):
    """Read a couple of frames after opening to avoid transient failures."""
    for _ in range(frames):
        if capture is None:
            return
        capture.read()
        time.sleep(0.02)


def _open_camera_and_update_state(preferred_selection, reason):
    """Open preferred camera with fallback and synchronize tray + config state."""
    opened_cap, active_selection = camera_manager.open_camera(preferred_selection)

    if active_selection is not None and active_selection != config_manager.get("camera_name"):
        config_manager.set("camera_name", active_selection)

    app_controller.set_active_camera(active_selection)
    app_controller.set_camera_unavailable(opened_cap is None)

    if opened_cap is None:
        print(f"[Camera] No available camera ({reason})")
    else:
        print(
            f"[Camera] Active camera: {camera_manager.selection_to_log_label(active_selection)} [{active_selection}] ({reason})"
        )
        _warmup_camera(opened_cap)
    return opened_cap


def cleanup():
    """Release all resources on application exit.

    Releases webcam, stops audio, destroys alert windows, resumes any paused media,
    and stops tray icon. Called from the finally block to ensure cleanup on normal
    exit or exception.
    """
    print("\nCleaning up...")
    if cap is not None:
        cap.release()
    media_controller.resume_all(wait=True)
    sound_manager.cleanup()
    red_flash.cleanup()
    if app_controller.icon:
        app_controller.icon.stop()
    print("Shutdown complete.")


try:
    cap = _open_camera_and_update_state(
        config_manager.get("camera_name"),
        reason="startup",
    )

    # Start the system tray icon in a background thread
    tray_thread = threading.Thread(target=app_controller.run_tray, daemon=True)
    tray_thread.start()
    print("[Info] System tray icon started")

    with HandLandmarker.create_from_options(
        hand_options
    ) as hand_landmarker, FaceLandmarker.create_from_options(
        face_options
    ) as face_landmarker, ObjectDetector.create_from_options(
        object_options
    ) as object_detector:

        biting_detected = False

        while app_controller.running:
            # UPDATE THE RED FLASH (must be called even when paused for tkinter)
            red_flash.update()

            has_switch_request, requested_selection = app_controller.consume_camera_switch_request()
            if has_switch_request:
                _stop_active_alert()
                if cap is not None:
                    cap.release()
                cap = _open_camera_and_update_state(
                    requested_selection,
                    reason="tray switch",
                )

            if cap is None:
                time.sleep(0.2)
                continue

            # Skip detection if paused
            if app_controller.paused:
                # Make sure alerts are off when paused
                _stop_active_alert()
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                print("[Camera] Frame read failed - attempting reopen")
                cap.release()
                cap = _open_camera_and_update_state(
                    app_controller.active_camera_selection,
                    reason="read failure recovery",
                )
                _stop_active_alert()
                time.sleep(0.2)
                continue

            # Get current timestamp in milliseconds (must be monotonically increasing)
            timestamp_ms = int(time.time() * 1000)

            # Prepare the mediapipe image object
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Reset detection state at the start of each frame
            biting_detected = False

            # Run hand detection
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Only check face if hands detected
            if hand_result.hand_landmarks:
                face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

                if face_result.face_landmarks:
                    # Extract lip landmarks
                    face_landmarks_array = face_result.face_landmarks[0]
                    lip_points_coords_list = []
                    for index in LIP_INDICES:
                        landmark_info = face_landmarks_array[index]
                        lip_points_coords_list.append(
                            (landmark_info.x, landmark_info.y)
                        )

                    # Create polygon
                    precise_polygon = Polygon(lip_points_coords_list)
                    buffered_polygon = precise_polygon.buffer(SENSITIVITY)

                    # Calculate average lip z-depth for comparison
                    lip_avg_z = sum(
                        face_landmarks_array[i].z for i in LIP_INDICES
                    ) / len(LIP_INDICES)

                    # Drinking detection (run every N frames to save resources)
                    if config_manager.get("drinking_detection_enabled"):
                        drinking_frame_counter += 1

                        if drinking_frame_counter >= DRINKING_DETECTION_INTERVAL:
                            drinking_frame_counter = 0

                            # Run object detection
                            object_result = object_detector.detect_for_video(mp_image, timestamp_ms)

                            # Check if any drinking object is detected
                            drinking_this_frame = False
                            for detection in object_result.detections:
                                category_name = detection.categories[0].category_name
                                if category_name in DRINKING_CLASS_LABELS:
                                    drinking_this_frame = True
                                    print(f"[Drinking] {category_name} detected - suppressing alerts")
                                    break

                            if drinking_this_frame:
                                drinking_detected = True
                                drinking_persistence_counter = DRINKING_PERSISTENCE_FRAMES
                            elif drinking_persistence_counter > 0:
                                drinking_persistence_counter -= 1
                                if drinking_persistence_counter == 0:
                                    drinking_detected = False
                        else:
                            # Between detection frames, decrement persistence counter
                            if drinking_persistence_counter > 0:
                                drinking_persistence_counter -= 1
                                if drinking_persistence_counter == 0:
                                    drinking_detected = False

                    # Skip fingertip check if drinking object detected
                    if drinking_detected and config_manager.get("drinking_detection_enabled"):
                        biting_detected = False
                    else:
                        # Check if hand landmarks intersect with mouth polygon
                        for hand in hand_result.hand_landmarks:
                            for hand_landmark_index in HAND_INDICES:
                                landmark_info = hand[hand_landmark_index]
                                point = Point(landmark_info.x, landmark_info.y)
                                if buffered_polygon.contains(point):
                                    # Z-depth check: finger must be at similar depth to lips
                                    # This filters out fingers passing in front of the face
                                    if abs(landmark_info.z - lip_avg_z) < Z_DEPTH_THRESHOLD:
                                        biting_detected = True
                                        break
                            if biting_detected:
                                break

            current_time = time.time()

            # Temporal smoothing: require multiple consecutive positive frames
            # to reduce false positives from brief hand movements near face
            if biting_detected:
                consecutive_detections += 1
            else:
                consecutive_detections = 0

            # Alert state machine: activate after FRAMES_REQUIRED consecutive detections,
            # then maintain alert until COOLDOWN_PERIOD after detection stops
            if consecutive_detections >= FRAMES_REQUIRED:
                # Sustained detection confirmed - update timestamp and activate alert
                last_biting_time = current_time

                if not alert_active:
                    # Check config before triggering each alert type
                    if config_manager.get("pause_media_on_alert"):
                        media_controller.pause_all()
                    if config_manager.get("sound_enabled"):
                        sound_manager.start_sound()
                    if config_manager.get("flash_enabled"):
                        red_flash.flash()
                    alert_active = True
            else:
                # Not enough consecutive detections - check if cooldown period has expired
                time_since_last_bite = current_time - last_biting_time
                if alert_active and time_since_last_bite > COOLDOWN_PERIOD:
                    sound_manager.stop_sound()
                    red_flash.hide()
                    if config_manager.get("pause_media_on_alert"):
                        media_controller.resume_all()
                    alert_active = False

            # Limit frame rate to reduce CPU usage
            time.sleep(1 / TARGET_FPS)

except KeyboardInterrupt:
    pass  # Normal exit via Ctrl+C
finally:
    cleanup()
