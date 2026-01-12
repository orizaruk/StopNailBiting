import cv2
import mediapipe as mp
from shapely import Point, Polygon
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


class ConfigManager:
    """Manages persistent configuration settings"""

    DEFAULT_CONFIG = {
        "flash_enabled": True,
        "sound_enabled": True,
        "start_with_windows": False,
        "volume": 0.75,
    }

    def __init__(self):
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

# Configuration
SENSITIVITY = 0.01  # Lip polygon buffer size
Z_DEPTH_THRESHOLD = 0.1  # Max z-difference between finger and lips for valid detection
FRAMES_REQUIRED = 3  # Consecutive frames needed before triggering alert
TARGET_FPS = 15  # Target frame rate to reduce CPU usage
COOLDOWN_PERIOD = 1.5  # Time in seconds to keep alert visible after biting stops
SOUND_FILE = resource_path(
    "assets/noise.wav"
)  # Path to alert sound (.mp3, .wav, .ogg) or None to disable

hand_model_path = resource_path("models/hand_landmarker.task")
face_model_path = resource_path("models/face_landmarker.task")

# Initializations of options to configure the model
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

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

# LIST OF THE RELEVANT LANDMARK INDICES
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
HAND_INDICES = [4, 3, 8, 7, 12, 11, 16, 15, 20, 19]


class RedFlashAlert:
    """Simple red screen flash for nail biting alerts"""

    def __init__(self):
        self.root = None
        self.is_showing = False
        self._init_window()

    def _init_window(self):
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.configure(background="red")
        self.root.overrideredirect(True)

        label = tk.Label(
            self.root,
            text="⚠️ STOP NAIL BITING ⚠️",
            font=("Arial", 48, "bold"),
            bg="red",
            fg="white",
        )
        label.pack(expand=True)

        # Start hidden
        self.root.withdraw()
        self.root.update()

    def flash(self):
        """Show red overlay"""
        if not self.is_showing:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.is_showing = True
            print("Red flash activated")

    def update(self):
        """Call this in your main loop to update the window"""
        try:
            self.root.update()
        except Exception as e:
            print(f"Update error: {e}")

    def hide(self):
        """Manually hide the overlay"""
        if self.is_showing:
            self.root.withdraw()
            self.root.update()
            self.is_showing = False
            print("Red flash deactivated")

    def cleanup(self):
        """Destroy the tkinter window"""
        self.root.destroy()


class SoundManager:
    """Manages sound alerts with anti-flicker protection and graceful degradation"""

    def __init__(self, sound_file, volume=0.75):
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
    """Controls app state and system tray integration"""

    def __init__(self, config, sound_manager):
        self.config = config
        self.sound_manager = sound_manager
        self.running = True
        self.paused = False
        self.icon = None
        # Pre-create both icon images
        self.icon_active = self._create_icon_image(active=True)
        self.icon_paused = self._create_icon_image(active=False)

    def _create_icon_image(self, active=True):
        """Create icon image - red when active, gray when paused"""
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

    # Volume control methods
    def _volume_menu_item(self, label, level):
        """Create a volume menu item with proper checked state and action"""
        def is_checked(item):
            return abs(self.config.get("volume") - level) < 0.01

        def set_level(icon, item):
            self.config.set("volume", level)
            self.sound_manager.set_volume(level)

        return pystray.MenuItem(label, set_level, checked=is_checked)

    def setup_tray(self):
        """Create and return the system tray icon"""
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


def cleanup():
    """Clean up all resources"""
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
    ) as face_landmarker:

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

            # Update consecutive detection counter
            if biting_detected:
                consecutive_detections += 1
            else:
                consecutive_detections = 0

            # Handle biting detection with temporal consistency and cooldown
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
