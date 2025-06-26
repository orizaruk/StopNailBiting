import cv2 as cv
import time

cap = cv.VideoCapture()
# Check if the camera has opened succesfully
if not cap.isOpened():
    print("Cannot open camera.")
    exit()

while True:
    retrive_status, frame = cap.read()

    # Couldn't read a frame, stream ended?
    if not retrive_status:
        print("Can't receive frame. Exiting.")
        break

    cv.imshow("Capture", frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
