import cv2
from gaze_tracking import GazeTracking
import torch
import torch.nn as nn
import mouse

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

gaze = GazeTracking()
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
model = Net()
model = torch.load("model")

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)
    new_frame = gaze.annotated_frame()

    hr = gaze.horizontal_ratio()
    vr = gaze.vertical_ratio()
    if hr is not None and vr is not None:
        out = model(torch.tensor([hr, vr]))

        mouse.move(int(out[0]), int(out[1]))

    cv2.imshow("Data Collector", new_frame)

    e = cv2.waitKey(1)

    if e == 27:
        break

cv2.destroyAllWindows()
webcam.release()