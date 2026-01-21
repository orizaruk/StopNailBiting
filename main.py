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
import json
import subprocess
from PIL import Image, ImageDraw
import pystray
from screeninfo import get_monitors


class ConfigManager:
    """Manages persistent configuration settings"""

    DEFAULT_CONFIG = {
        "flash_enabled": True,
        "sound_enabled": True,
        "start_with_windows": False,
        "volume": 0.75,
        "drinking_detection_enabled": True,
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
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    running_mode=VisionRunningMode.VIDEO,
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
            window.destroy()
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

        if sound_file is None:
            print("[Info] Sound disabled by configuration")
            return

        try:
            if not os.path.exists(sound_file):
                print(f"[Warning] Sound file not found: {sound_file}")
                print(f"          Expected location: {os.path.abspath(sound_file)}")
                print("          Running in visual-only mode (no audio alerts)")
                return

            pygame.mixer.init()
            self.alert_sound = pygame.mixer.Sound(sound_file)
            self.alert_sound.set_volume(self.volume)
            self.enabled = True
            print(f"[Info] Sound loaded successfully: {sound_file} (volume: {int(volume * 100)}%)")

        except pygame.error as e:
            print(f"[Warning] Failed to initialize audio: {e}")
            print("          Running in visual-only mode (no audio alerts)")
        except Exception as e:
            print(f"[Warning] Unexpected error loading sound: {e}")
            print("          Running in visual-only mode (no audio alerts)")

    def start_sound(self):
        """Start the alert sound if not already playing"""
        if self.enabled and not self.sound_playing:
            self.alert_sound.play(loops=-1)
            self.sound_playing = True
            print("Alert sound started")

    def stop_sound(self):
        """Stop the alert sound"""
        if self.enabled and self.sound_playing:
            self.alert_sound.stop()
            self.sound_playing = False
            print("Alert sound stopped")

    def set_volume(self, volume):
        """Set the alert sound volume (0.0 to 1.0)"""
        self.volume = volume
        if self.enabled and self.alert_sound:
            self.alert_sound.set_volume(volume)
            print(f"[Sound] Volume set to {int(volume * 100)}%")

    def cleanup(self):
        """Release pygame mixer resources"""
        self.stop_sound()
        if self.enabled:
            pygame.mixer.quit()


class AppController:
    """Controls app state and system tray integration.

    Manages the pystray system tray icon, menu actions, and shared state
    (running/paused) that coordinates between the main detection loop
    and the tray UI thread.
    """

    def __init__(self, config, sound_manager):
        """Initialize app controller with config and sound manager.

        Args:
            config: ConfigManager instance for persisting settings.
            sound_manager: SoundManager instance for volume control integration.
        """
        self.config = config
        self.sound_manager = sound_manager
        self.running = True
        self.paused = False
        self.icon = None
        self.icon_active = self._create_icon_image(active=True)
        self.icon_paused = self._create_icon_image(active=False)

    def _create_icon_image(self, active=True):
        """Create a 64x64 tray icon image.

        Args:
            active: If True, creates a red icon (monitoring). If False, gray (paused).

        Returns:
            PIL.Image.Image: RGBA image suitable for pystray icon.
        """
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Red when monitoring, gray when paused
        fill_color = (220, 53, 69) if active else (128, 128, 128)

        margin = 4
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=fill_color,
            outline=(255, 255, 255),
            width=2,
        )

        # Center symbol
        center = size // 2
        draw.rectangle(
            [center - 8, center - 12, center + 8, center + 12],
            fill=(255, 255, 255),
        )

        return img

    def toggle_pause(self, icon, item):
        """Toggle pause state and update icon"""
        self.paused = not self.paused
        status = "Paused" if self.paused else "Monitoring"
        print(f"[Tray] {status}")

        # Update icon and tooltip
        if self.icon:
            self.icon.icon = self.icon_paused if self.paused else self.icon_active
            self.icon.title = f"Stop Nail Biting - {status}"

    def quit_app(self, icon, item):
        """Signal the app to quit"""
        print("[Tray] Quit requested")
        self.running = False
        if self.icon:
            self.icon.stop()

    def get_pause_text(self, item):
        """Dynamic menu item text for pause/resume"""
        return "Resume" if self.paused else "Pause"

    def is_flash_enabled(self, item):
        """Check if flash is enabled (for checkbox state)"""
        return self.config.get("flash_enabled")

    def is_sound_enabled(self, item):
        """Check if sound is enabled (for checkbox state)"""
        return self.config.get("sound_enabled")

    def toggle_flash(self, icon, item):
        """Toggle flash enabled state"""
        new_value = not self.config.get("flash_enabled")
        self.config.set("flash_enabled", new_value)
        print(f"[Tray] Flash {'enabled' if new_value else 'disabled'}")

    def toggle_sound(self, icon, item):
        """Toggle sound enabled state"""
        new_value = not self.config.get("sound_enabled")
        self.config.set("sound_enabled", new_value)
        print(f"[Tray] Sound {'enabled' if new_value else 'disabled'}")

    def is_drinking_detection_enabled(self, item):
        """Check if drinking detection is enabled (for checkbox state)"""
        return self.config.get("drinking_detection_enabled")

    def toggle_drinking_detection(self, icon, item):
        """Toggle drinking detection enabled state"""
        new_value = not self.config.get("drinking_detection_enabled")
        self.config.set("drinking_detection_enabled", new_value)
        print(f"[Tray] Drinking detection {'enabled' if new_value else 'disabled'}")

    def is_start_with_windows(self, item):
        """Check if start with Windows is enabled (for checkbox state)"""
        return self.config.get("start_with_windows")

    def toggle_start_with_windows(self, icon, item):
        """Toggle start with Windows"""
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
        """Get the Windows Startup folder path"""
        if sys.platform != "win32":
            return None
        return os.path.join(
            os.environ.get("APPDATA", ""),
            "Microsoft", "Windows", "Start Menu", "Programs", "Startup"
        )

    def _get_shortcut_path(self):
        """Get the path for the startup shortcut"""
        startup_folder = self._get_startup_folder()
        if startup_folder:
            return os.path.join(startup_folder, "StopNailBiting.lnk")
        return None

    def _get_exe_path(self):
        """Get the path to the current executable"""
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller bundle
            return sys.executable
        else:
            # Running as script - return python + script path
            return f'"{sys.executable}" "{os.path.abspath(__file__)}"'

    def _create_startup_shortcut(self):
        """Create a shortcut in the Windows Startup folder"""
        if sys.platform != "win32":
            print("[Startup] Not on Windows, skipping")
            return False

        shortcut_path = self._get_shortcut_path()
        exe_path = self._get_exe_path()

        if not shortcut_path:
            return False

        try:
            # Use PowerShell to create the shortcut
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
        """Remove the shortcut from the Windows Startup folder"""
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
        """Create a volume menu item with radio-button behavior.

        Args:
            label: Display text for the menu item (e.g., "75%").
            level: Volume level from 0.0 to 1.0.

        Returns:
            pystray.MenuItem: Menu item that shows checked state and updates volume.
        """
        def is_checked(item):
            return abs(self.config.get("volume") - level) < 0.01

        def set_level(icon, item):
            self.config.set("volume", level)
            self.sound_manager.set_volume(level)

        return pystray.MenuItem(label, set_level, checked=is_checked)

    def setup_tray(self):
        """Create and configure the system tray icon with menu.

        Returns:
            pystray.Icon: Configured tray icon ready to run.
        """
        # Volume submenu
        volume_menu = pystray.Menu(
            self._volume_menu_item("25%", 0.25),
            self._volume_menu_item("50%", 0.50),
            self._volume_menu_item("75%", 0.75),
            self._volume_menu_item("100%", 1.0),
        )

        menu = pystray.Menu(
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
            pystray.MenuItem("Volume", volume_menu),
            pystray.MenuItem(
                "Drinking Detection (Cups/Bottles)",
                self.toggle_drinking_detection,
                checked=self.is_drinking_detection_enabled,
            ),
            pystray.MenuItem(
                "Start with Windows",
                self.toggle_start_with_windows,
                checked=self.is_start_with_windows,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                self.get_pause_text,
                self.toggle_pause,
            ),
            pystray.MenuItem("Quit", self.quit_app),
        )

        self.icon = pystray.Icon(
            "StopNailBiting",
            self.icon_active,
            "Stop Nail Biting - Monitoring",
            menu,
        )
        return self.icon

    def run_tray(self):
        """Run the tray icon (call from a separate thread)"""
        icon = self.setup_tray()
        icon.run()


# Initialize config first (needed by other components)
config_manager = ConfigManager()

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Initialize sound and alert managers
sound_manager = SoundManager(SOUND_FILE, volume=config_manager.get("volume"))
red_flash = RedFlashAlert()
app_controller = AppController(config_manager, sound_manager)

alert_active = False
last_biting_time = 0  # Track when biting was last detected
consecutive_detections = 0  # Track consecutive frames with detection

# Drinking detection state
drinking_frame_counter = 0
drinking_detected = False
drinking_persistence_counter = 0


def cleanup():
    """Release all resources on application exit.

    Releases webcam, stops audio, destroys alert windows, and stops tray icon.
    Called from the finally block to ensure cleanup on normal exit or exception.
    """
    print("\nCleaning up...")
    cap.release()
    sound_manager.cleanup()
    red_flash.cleanup()
    if app_controller.icon:
        app_controller.icon.stop()
    print("Shutdown complete.")


try:
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

            # Skip detection if paused
            if app_controller.paused:
                # Make sure alerts are off when paused
                if alert_active:
                    sound_manager.stop_sound()
                    red_flash.hide()
                    alert_active = False
                    consecutive_detections = 0
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

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
                    alert_active = False

            # Limit frame rate to reduce CPU usage
            time.sleep(1 / TARGET_FPS)

except KeyboardInterrupt:
    pass  # Normal exit via Ctrl+C
finally:
    cleanup()
