import math
import os
import requests
from ultralytics import YOLO
import cv2
import cvzone
from sort import *

def sort_update(resultTracker):
    for res in resultTracker:
        x1, y1, x2, y2, id = res
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w,h = x2 - x1, y2 - y1

        cvzone.putTextRect(img, f'ID: {id}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))

url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/'

fileNames = ['yolov8n.pt', 'yolov8x.pt']
for fileName in fileNames:
    if not os.path.exists(fileName):
        r = requests.get(url + fileName, allow_redirects=True)
        open(fileName, 'wb').write(r.content)

model = YOLO('best.pt')

source = 'short_out.mp4'
cap = cv2.VideoCapture(source)
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_height, frame_width)

out = cv2.VideoWriter("output_sort.avi",
                         cv2.VideoWriter_fourcc(*"MJPG"),
                         60,
                         (3440, 1440))

tracker = Sort(max_age=120, min_hits=3, iou_threshold=0.3)
while True:
    success, img = cap.read()
    if not success:
        break
    detections = np.empty((0, 5))

    result = model(img, stream=True)
    for r in result:
        for box in r.boxes:
            conf = math.ceil(box.conf[0]*100)/100
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox = x1, y1, x2, y2
            w, h = x2 - x1, y2 - y1

            class_name = model.names[int(box.cls[0])]

            if class_name in ['rocket_car', 'rocket_ball'] and conf > 0.25:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    resultTracker = tracker.update(detections)
    sort_update(resultTracker)
    cv2.imshow('Image', img)
    out.write(img)
    if cv2.waitKey(1) == ord('q'):
        break

out.release()

