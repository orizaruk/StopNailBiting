import mediapipe as mp
import pprint

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
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
)

# Load the image, need to modify for live stream in the future
mp_image = mp.Image.create_from_file("assets/img2.jpg")

with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
    hand_landmarker_result = hand_landmarker.detect(mp_image)
    for index, keypoint in enumerate(hand_landmarker_result.hand_landmarks[0]):
        key_point_coords = (keypoint.x, keypoint.y, keypoint.z)
        print(f"Hand Landmark {index}: {key_point_coords}")

with FaceLandmarker.create_from_options(face_options) as face_landmarker:
    face_landmarker_result = face_landmarker.detect(mp_image)
    # face_landmarker_result.face_landmarks[0] is the array of normalized_landmarks objects
    # print result to file
    result_landmarks_array = face_landmarker_result.face_landmarks[0]
    lip_landmarks = [
        0,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
    ]  # the landmarks we care about (the ones of the lip in this case)
    output_filename = "face_landmarker_result_output.txt"
    output_string = ""
    for landmark_num in lip_landmarks:
        landmark_object = result_landmarks_array[landmark_num]
        landmark_coords = (landmark_object.x, landmark_object.y, landmark_object.z)
        print(f"Face Landmark {landmark_num}: {landmark_coords}")
