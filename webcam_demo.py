import imutils
import time
import dlib
import cv2
import numpy as np
from makeup import makeup

detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")

print("[INFO] camera sensor warming up...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 800)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detected_faces = detector(gray, 0)

    m = makeup(frame)

    for i, face_rect in enumerate(detected_faces):
        pose_landmarks = face_pose_predictor(gray, face_rect)
        landmark = np.empty([68, 2], dtype=int)
        for i in range(68):
            landmark[i][0] = pose_landmarks.part(i).x
            landmark[i][1] = pose_landmarks.part(i).y

        frame = m.apply_makeup(landmark)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
