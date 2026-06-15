"""Detection tuning constants, model paths, landmark indices, and MediaPipe options."""

import os

import mediapipe as mp

from .resources import resource_path

# Detection tuning
SENSITIVITY = 0.01  # Lip polygon buffer size
Z_DEPTH_THRESHOLD = 0.1  # Max z-difference between finger and lips for valid detection
FRAMES_REQUIRED = 3  # Consecutive frames needed before triggering alert
TARGET_FPS = 15  # Target frame rate to reduce CPU usage
COOLDOWN_PERIOD = 1.5  # Time in seconds to keep alert visible after biting stops

# Model file paths (resolved for dev or PyInstaller bundle)
HAND_MODEL_PATH = resource_path(os.path.join("models", "hand_landmarker.task"))
FACE_MODEL_PATH = resource_path(os.path.join("models", "face_landmarker.task"))
OBJECT_MODEL_PATH = resource_path(os.path.join("models", "efficientdet_lite0.tflite"))

# Drinking detection constants (to reduce false positives when drinking)
DRINKING_DETECTION_INTERVAL = 3  # Run object detection every N frames
DRINKING_CONFIDENCE_THRESHOLD = 0.35  # Minimum confidence for drinking detection
DRINKING_PERSISTENCE_FRAMES = 30  # Frames to persist drinking state (~2s at 15 FPS)
DRINKING_CLASS_LABELS = {"cup", "bottle", "wine glass"}

# Detection confidence thresholds (reduce false positives in low light)
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_PRESENCE_CONFIDENCE = 0.5

# MediaPipe task API aliases
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions

# MediaPipe FaceLandmarker indices that form the outer lip contour polygon.
# These 21 points trace around the lips to create a closed shape for collision detection.
LIP_INDICES = [
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
    191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
]

# MediaPipe HandLandmarker indices for fingertips and finger joints to check.
# Pairs: (tip, joint below) thumb(4,3), index(8,7), middle(12,11), ring(16,15), pinky(20,19)
HAND_INDICES = [4, 3, 8, 7, 12, 11, 16, 15, 20, 19]


def create_detector_options():
    """Build the MediaPipe option objects for the hand, face, and object detectors.

    Deferred into a function so that importing this module has no heavy side
    effects; the detection engine calls this once when it starts.
    """
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
    )
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
        min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
    )
    object_options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=OBJECT_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        max_results=5,
        score_threshold=DRINKING_CONFIDENCE_THRESHOLD,
    )
    return hand_options, face_options, object_options
