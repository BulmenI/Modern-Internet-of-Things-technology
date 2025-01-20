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
# Обработка обнаружений
    if len(detections) > 0:
        is_human = False 
        for (x, y, w, h) in detections:
            aspect_ratio = float(w) / h
            if 0.2 < aspect_ratio < 0.8: 
                is_human = True
                break 

        if is_human: # Рисуем прямоугольник и увеличиваем счетчики, только если это человек
            if not show_count:
                human_count += 1
                show_count = True
                print(f"human count: {human_count}")
            display_count += 1 
            if display_count >= 1000: 
                danger_activated = True
            for (x, y, w, h) in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


            # Отображение счетчика на кадре
            cv2.putText(frame, f"Display Count: {display_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        



        else:
            print("not a human")

    else:
        show_count = False
        print("not a human")
        
    if danger_activated:
        cv2.putText(frame, "ОПАСНО", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)



    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
