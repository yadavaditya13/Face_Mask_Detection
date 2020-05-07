# importing required packages

from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import numpy as np
import argparse
import imutils
import cv2
import os

# parsing arguments

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", type=str, required=True, help="path to input video file...")
ap.add_argument("-o", "--output", type=str, required=True, help="path to output video file...")
ap.add_argument("-f", "--face", type=str, required=True, help="path to face detector model file...")
ap.add_argument("-m", "--mask", type=str, required=True, help="path to mask detector model file...")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum confidence value for filtering faces...")

args = vars(ap.parse_args())

# loading our face-detector model
print("[INFO] Loading Face_Detector_Model from disk...")

prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])

# initializing our face-detector model
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# loading mask-detector model
print("[INFO] Loading Mask_Detector_Model from disk...")
model = load_model(args["mask"])

# Loading Input Video file from disk
print("[INFO] Loading the Input Video_File...")

vs = cv2.VideoCapture(args["input"])
writer = None
# lets count the number of frames in our videoFile

try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] Total: {} frames in videoFile!".format(total))

except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# lets begin reading frames in video
print("[INFO] Reading Frames in Video_File one by one...")

while vs.isOpened():

    # read frames
    grabbed, frame = vs.read()
    # break the loop if frames end
    if not grabbed:
        break

    # resizing our frame
    frame = imutils.resize(frame, height=650, width=700)
    (h, w) = frame.shape[:2]

    # lets load the frame for blobbing
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # now lets detect faces in the frames
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # lets begin looping over our detected faces
    for i in range(detections.shape[2]):

        # grabbing the confidence value
        confidence = detections[0, 0, i, 2]

        # filtering weak detections
        if confidence > args["confidence"]:

            # grabbing box dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # lets make sure that faces don't fall outside frames

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # lets extract ROIs and pre_process it

            face = frame[startY:endY, startX:endX]
            # if the video is black n white
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # lets make predictions
            (mask, withoutMask) = model.predict(face)[0]

            # labelling the face
            label = "Mask" if mask > withoutMask else "No Mask"
            label = "{}: {:.2f} %".format(label, max(mask, withoutMask) * 100)

            # deciding color value
            color = (0, 255, 0) if mask > withoutMask else (0, 0, 255)

            # writing label and drawing box onto the frame

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # lets initialize the writer
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 40, (frame.shape[1], frame.shape[0]), True)

    # writing the frame to disk
    writer.write(frame)

print("[INFO] Your OutPut Video_FIle is read...Jump to desired OutPut directory to have a look!...")
# lets clean up

vs.release()
writer.release()
cv2.destroyAllWindows()