import math
import os
import requests
from ultralytics import YOLO
import cv2
import cvzone

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

out = cv2.VideoWriter("output_yolo.avi",
                         cv2.VideoWriter_fourcc(*"MJPG"),
                         60,
                         (3440, 1440))

while True:
    success, img = cap.read()
    if not success:
        break

    result = model(img, stream=True)
    for r in result:
        for box in r.boxes:
            confidence = math.ceil(box.conf[0]*100)/100
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox = x1, y1, x2, y2
            w, h = x2 - x1, y2 - y1

            class_name = model.names[int(box.cls[0])]
            if class_name == 'rocket_car' and confidence > 0.05:
                cvzone.putTextRect(img, f'{class_name} {confidence}', (max(
                    0, x1), max(35, y1)), scale=0.75, thickness=1, offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
            if class_name == 'rocket_ball' and confidence > 0.05:
                cvzone.putTextRect(img, f'{class_name} {confidence}', (max(
                    0, x1), max(35, y1)), scale=0.75, thickness=1, offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
    cv2.imshow('Image', img)
    out.write(img)
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
