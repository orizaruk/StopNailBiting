import cv2
import mediapipe as mp
import numpy as np

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

# --- Model Options ---
# <<< CHANGE 1: CONFIGURE THE MODEL FOR TWO HANDS >>>
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,  # We are now looking for up to two hands
)
face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
)

# --- Image and File Paths ---
image_path = "assets/img9.jpg"
output_image_path = "combined_landmark_map_two_hands.jpg"

# --- YOUR CHOSEN LANDMARK INDICES ---
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

print("Loading image...")
image = cv2.imread(image_path)
output_image = image.copy()
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
image_height, image_width, _ = image.shape

print("Running Hand and Face Landmark detection...")
with HandLandmarker.create_from_options(
    hand_options
) as hand_landmarker, FaceLandmarker.create_from_options(
    face_options
) as face_landmarker:

    hand_result = hand_landmarker.detect(mp_image)
    face_result = face_landmarker.detect(mp_image)

    # --- Draw Face Landmarks (This part remains the same) ---
    if face_result.face_landmarks:
        # ... (face drawing logic is unchanged)
        print("Face detected. Drawing lip polygon...")
        face_landmarks = face_result.face_landmarks[0]
        lip_pixel_coords = []
        for index in LIP_INDICES:
            landmark = face_landmarks[index]
            x_px, y_px = int(landmark.x * image_width), int(landmark.y * image_height)
            lip_pixel_coords.append((x_px, y_px))
            cv2.circle(output_image, (x_px, y_px), 2, (0, 255, 0), -1)
            cv2.putText(
                output_image,
                str(index),
                (x_px + 4, y_px),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )
        if len(lip_pixel_coords) > 1:
            pts = np.array(lip_pixel_coords, np.int32)
            cv2.polylines(
                output_image, [pts], isClosed=False, color=(255, 255, 0), thickness=1
            )

    # --- Draw Hand Landmarks ---
    if hand_result.hand_landmarks:
        # <<< CHANGE 2: LOOP THROUGH EACH DETECTED HAND >>>
        print(
            f"Found {len(hand_result.hand_landmarks)} hand(s). Drawing finger landmarks..."
        )
        for hand_landmarks in hand_result.hand_landmarks:
            # The rest of the drawing logic is now inside this loop
            for index in HAND_INDICES:
                landmark = hand_landmarks[index]
                x_px = int(landmark.x * image_width)
                y_px = int(landmark.y * image_height)
                cv2.circle(
                    output_image, (x_px, y_px), 2, (0, 255, 255), -1
                )  # Yellow dots
                cv2.putText(
                    output_image,
                    str(index),
                    (x_px - 15, y_px + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 100, 0),
                    1,
                )  # Blue text
    else:
        print("No hand was detected.")

# --- Save the Final Image ---
if face_result.face_landmarks or hand_result.hand_landmarks:
    print(f"Saving combined map to {output_image_path}...")
    cv2.imwrite(output_image_path, output_image)
    print("Done.")
