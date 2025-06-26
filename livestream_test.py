import mediapipe as mp
from shapely import Point, Polygon
from pprint import pp

face_model_path = "models/face_landmarker.task"

# Initializations of options to configure the model
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


def process_result(result, output_image, timestamp_ms):
    print(f"Result received for timestamp {timestamp_ms}")


face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=process_result,
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

with FaceLandmarker.create_from_options(face_options) as face_landmarker:
    face_result = face_landmarker.detect()
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
