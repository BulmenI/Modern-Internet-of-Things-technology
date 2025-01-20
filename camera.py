import cv2
import numpy as np

# Загрузка каскадного классификатора для лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка HOG дескриптора для туловища
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

video_capture = cv2.VideoCapture(0)

human_count = 0
show_count = False 
display_count = 0 
danger_activated = False 

while True:
    # Чтение кадра и преобразование в оттенки серого
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Обнаружение туловища
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    detections = []
    if len(faces) > 0:
        detections.extend(faces)
    if len(rects) > 0:
        detections.extend(np.array([[x, y, w, h] for (x, y, w, h) in rects]))
