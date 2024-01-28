import cv2

cap = cv2.VideoCapture('data_generation/model_in.mp4')
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imwrite('data_generation/frames/frame{:d}.jpg'.format(count), frame)
        count += 60 # i.e. at 60 fps, this advances one second
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    else:
        cap.release()
        break
