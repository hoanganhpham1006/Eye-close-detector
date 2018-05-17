from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import pickle
from keras.models import model_from_json

#Load model from colab
json_file = open('./trained_model/model_json_1526538438.1061006.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('./trained_model/weight_1526538438.1061006.h5')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

#Prediction
image = cv2.imread("./close.jpg")
if image.shape[1] > 500:
    image = imutils.resize(image, width=500, inter=cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 3)
count = 0
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the face parts individually
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        if name == "left_eye" or name == "right_eye":
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            #to square
            #get center
            x = x + w/2
            y = y + h/2
            #make square
            if w != h:
                if w > h:
                    h = w
                else:
                    w = h
            x = x - w/2
            y = y - h/2
            roi = gray[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=24, inter=cv2.INTER_CUBIC)
            roi = np.reshape(roi, newshape=(1, roi.shape[0], roi.shape[1], 1))
            if model.predict_classes(roi) == 0:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                count += 1
cv2.putText(image, "CLOSED EYES: " + str(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imshow("Closed_eyes", image)
cv2.waitKey(0)
