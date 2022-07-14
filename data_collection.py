import cv2
from gaze_tracking import GazeTracking
import csv
import random
import mouse
from screeninfo import get_monitors

m = get_monitors()[0]

w = m.width
h = m.height

gaze = GazeTracking()
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

_, frame = webcam.read()
gaze.refresh(frame)

new_frame = gaze.annotated_frame()

y = random.randint(0, h)
x = random.randint(0, w)
mouse.move(x, y)

while True:
    mouse.move(x, y)
    _, frame = webcam.read()
    gaze.refresh(frame)

    frame = gaze.annotated_frame()

    cv2.imshow("Data Collector", frame)

    e = cv2.waitKey(1)

    if e == ord('q'):
        with open('data/data.csv', 'a', newline='') as file:
            if gaze.horizontal_ratio() is not None and gaze.vertical_ratio() is not None:
                writer = csv.writer(file)
                writer.writerow([str(x), str(y), str(gaze.horizontal_ratio()), str(gaze.vertical_ratio())])
            y = random.randint(0, h)
            x = random.randint(0, w)
            mouse.move(x, y)

    if e == 27:
        break

cv2.destroyAllWindows()
webcam.release()