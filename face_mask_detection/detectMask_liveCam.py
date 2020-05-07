# importing required packages

from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# argument parsing

ap = argparse.ArgumentParser()

ap.add_argument("-f", "--face", type=str, required=True, help="path to face detector model...")
ap.add_argument("-m", "--mask", type=str, required=True, help="path to mask detector model...")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum confidence value to discard weak detections...")

args = vars(ap.parse_args())

# loading face detector model from disk

print("[INFO] Loading Face_Detector_Model from disk...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
print("\n[INFO] Successfully Loaded Face_Detector_Model from disk...")

# initializing our face detector model
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# loading our mask detector model as well
print("[INFO] Loading Mask_Detector Model from disk...")
model = load_model(args["mask"])

# lets start our video_camera and provide a warm_up of 2 secs
print("[INFO] We are Going Live...")
vs = VideoStream(src=0).start()
time.sleep(2)

# lets loop over the frames
while True:

    # reading frames one by one
    frame = vs.read()
    # resizing frames and storing their shapes
    frame = imutils.resize(frame, height=650, width=700)
    (h, w) = frame.shape[:2]

    # lets start blobbing
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # passing the blob through face_detector model
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # looping over each detected face
    for i in range(detections.shape[2]):

        # accessing confidence value of each face
        confidence = detections[0, 0, i, 2]

        # filtering weak detections
        if confidence > args["confidence"]:

            # grabbing the boxes dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # lets make sure that box don't fall outside the frame

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # extracting ROIs and pre_processing each face

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # lets pass the face now to make predictions about mask
            (mask, withoutMask) = model.predict(face)[0]

            # labelling the face
            label = "Mask" if mask > withoutMask else "No Mask"
            label = "{}: {:.2f} %".format(label, max(mask, withoutMask) * 100)

            # deciding color value
            color = (0, 255, 0) if mask > withoutMask else (0, 0, 255)

            # writing label and drawing rectangle onto the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # lets display the frames
    cv2.imshow("Mask_Detection_Live:", frame)
    key = cv2.waitKey(1) & 0xFF

    # option for ending the session
    if key == ord("q"):
        break

# doing clean Up
cv2.destroyAllWindows()
vs.stop()