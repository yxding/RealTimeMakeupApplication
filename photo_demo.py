from matplotlib import pyplot as plt
from skimage import io
from makeup import makeup
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")

img = io.imread('./data/Input.jpg')
detected_faces = detector(img, 0)
pose_landmarks = face_pose_predictor(img, detected_faces[0])

landmark = np.empty([68, 2], dtype=int)
for i in range(68):
    landmark[i][0] = pose_landmarks.part(i).x
    landmark[i][1] = pose_landmarks.part(i).y

m = makeup(img)
im = m.apply_makeup_all(landmark)

plt.figure()
plt.imshow(im)
plt.show()

import cv2
im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
cv2.imwrite('foundation.jpg',im)
