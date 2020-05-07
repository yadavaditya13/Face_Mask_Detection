# importing required packages

from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import numpy as np
import argparse
import imutils
import cv2
import os

# argument parsing

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", type=str, required=True, help="path to input image....")
ap.add_argument("-f", "--face", type=str, required=True, help="path to face detector directory...")
ap.add_argument("-m", "--mask", type=str, required=True, help="path to mask detector model file...")
ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum confidence value required for filtering weak detections...")

args = vars(ap.parse_args())

# lets load face detector model from disk

print("[INFO] Loading Face_Detector_Model from disk...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
print("\n[INFO] Successfully Loaded Face_Detector_Model from disk...")

# Initializing the model
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# lets load face detector model from disk

print("[INFO] loading Mask_Detector from disk...")
model = load_model(args["mask"])

# lets load the input image
print("[INFO] Loading Input Image...")
image = cv2.imread(args["image"])
orig = imutils.resize(image, height=650, width=700)
(h, w) = orig.shape[:2]

# resizing the image
image = cv2.resize(orig, (300, 300))

# lets begin blobbing
print("[INFO] Passing Image for Blobbing...")
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# passing blobs to face detector model
print("[INFO] Passing Blobs through Face_Detector_Model...")
faceNet.setInput(blob)
detections = faceNet.forward()

# looping over each face detected
print("[INFO] Looping over each detected face...")
for i in range(detections.shape[2]):

    # grabbing the confidence value for detected face
    confidence = detections[0, 0, i, 2]

    # filtering weak detections
    if confidence > args["confidence"]:

        # grabbing the box coordinates
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # make sure the face does not falls outside frame
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w - 1, endX)
        endY = min(h - 1, endY)

        # lets extract ROIs from image and pre_process it
        face = orig[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # passing the ROIs to mask_detector
        (mask, withoutMask) = model.predict(face)[0]

        # labelling the face
        label = "Mask" if mask > withoutMask else "No Mask"
        label = "{}: {:.2f} %".format(label, max(mask, withoutMask) * 100)

        # deciding color value
        color = (0, 255, 0) if mask > withoutMask else (0, 0, 255)

        # writing the box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), color, 2)

        # lets put_text on the image
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# displaying the output image
cv2.imshow("Mask_Detection:", orig)
key = cv2.waitKey(0) & 0xFF

if key == ord("q"):
    cv2.destroyAllWindows()