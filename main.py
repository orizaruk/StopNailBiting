import cv2
import mediapipe as mp
from shapely import Point, Polygon
import pygame
import tkinter as tk
import time
import os

# Configuration
SENSITIVITY = 0.01  # Lip polygon buffer size
Z_DEPTH_THRESHOLD = 0.1  # Max z-difference between finger and lips for valid detection
FRAMES_REQUIRED = 3  # Consecutive frames needed before triggering alert
TARGET_FPS = 15  # Target frame rate to reduce CPU usage
COOLDOWN_PERIOD = 1.5  # Time in seconds to keep alert visible after biting stops
SOUND_FILE = (
    "assets/noise.wav"  # Path to alert sound (.mp3, .wav, .ogg) or None to disable
)

hand_model_path = "models/hand_landmarker.task"
face_model_path = "models/face_landmarker.task"

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

    def __init__(self, sound_file):
        self.enabled = False
        self.sound_playing = False
        self.alert_sound = None

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
            self.enabled = True
            print(f"[Info] Sound loaded successfully: {sound_file}")

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

    def cleanup(self):
        """Release pygame mixer resources"""
        self.stop_sound()
        if self.enabled:
            pygame.mixer.quit()


# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Initialize sound and alert managers
sound_manager = SoundManager(SOUND_FILE)
red_flash = RedFlashAlert()

alert_active = False
last_biting_time = 0  # Track when biting was last detected
consecutive_detections = 0  # Track consecutive frames with detection


def cleanup():
    """Clean up all resources"""
    print("\nCleaning up...")
    cap.release()
    sound_manager.cleanup()
    red_flash.cleanup()
    print("Shutdown complete.")


try:
    with HandLandmarker.create_from_options(
        hand_options
    ) as hand_landmarker, FaceLandmarker.create_from_options(
        face_options
    ) as face_landmarker:

        biting_detected = False

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Get current timestamp in milliseconds (must be monotonically increasing)
            timestamp_ms = int(time.time() * 1000)

            # UPDATE THE RED FLASH
            red_flash.update()

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
                    sound_manager.start_sound()
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
