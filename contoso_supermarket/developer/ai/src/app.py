#!/usr/bin/env python
from importlib import import_module
import os
import cv2
from flask import Flask, render_template, Response, request, url_for, redirect, session
import socket
import secrets
from datetime import datetime
import json
import time
import numpy as np
import imutils
from imutils.video import FPS
from imutils.video import VideoStream

app = Flask(__name__)

model_cfg = "./models/yolov3-tiny.cfg"
model_w = "./models/yolov3-tiny.weights"
INPUT_FILE = "https://agoravideos.blob.core.windows.net/videos/supermarket.mp4"
CONFIDENCE_THRESHOLD = 0.3

H = None
W = None
fps = FPS().start()

net = cv2.dnn.readNetFromDarknet(model_cfg, model_w)
vs = cv2.VideoCapture(INPUT_FILE)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/people_count')
def people_count():
    global person_count
    return str(person_count)

def get_frame():

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    cnt = 0
    while True:
        cnt += 1
        try:
            (grabbed, image) = vs.read()
        except:
            break
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        if W is None or H is None:
            (H, W) = image.shape[:2]
        layerOutputs = net.forward(ln)

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONFIDENCE_THRESHOLD:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(
                boxes, confidences, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD
            )

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # show the output image
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        fps.update()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
