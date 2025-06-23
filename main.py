import mediapipe as mp
from shapely import Point, Polygon
from pprint import pp

IMAGE_PATH = "assets/img7.jpg"

hand_model_path = "models/hand_landmarker.task"
face_model_path = "models/face_landmarker.task"

# Initializations of options to configure the model
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

# Create a hand landmarker instance with the image mode:
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
)

# Load the image, need to modify for live stream in the future
mp_image = mp.Image.create_from_file(IMAGE_PATH)

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

with HandLandmarker.create_from_options(
    hand_options
) as hand_landmarker, FaceLandmarker.create_from_options(
    face_options
) as face_landmarker:
    # Run the inference process for the hand landmark model
    hand_result = hand_landmarker.detect(mp_image)

    # Only if we detected hands, let us continue with the face inference (for efficiency, if no hands, no need.)
    if hand_result.hand_landmarks:
        # Run the Face Inference
        face_result = face_landmarker.detect(mp_image)
        # Continue the checks only if a face has been detected
        if face_result.face_landmarks:
            # Extract only the relevant lip landmarks from the face scan, and store them in an array in the format of [(x1,y1), (x2,y2),...]
            face_landmarks_array = face_result.face_landmarks[0]
            lip_points_coords_list = []
            for index in LIP_INDICES:
                landmark_info = face_landmarks_array[index]
                lip_points_coords_list.append((landmark_info.x, landmark_info.y))

            # Create the Polygon object from the sequence of points
            precise_polygon = Polygon(lip_points_coords_list)
            # Create a buffered polygon to smooth it out and allow increasing the size of it for fine-tuning
            SENSITIVITY = 0.01
            buffered_polygon = precise_polygon.buffer(SENSITIVITY)

            # Now we need to check for each of the relevant hand landmarks if it's inside the polygon, for all the hands
            for hand in hand_result.hand_landmarks:
                # For each relevant landmark, let us check if it's within the polygon
                for hand_landmark_index in HAND_INDICES:
                    landmark_info = hand[hand_landmark_index]
                    # Create a Point object using the (x,y) coords
                    point = Point(landmark_info.x, landmark_info.y)
                    # CHECK IF THE POINT IS WITHIN THE POLYGON, IF IT IS - NAIL BITING!
                    if buffered_polygon.contains(point):
                        print("BITING DETECTED!")
