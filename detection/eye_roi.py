import cv2
import numpy as np


def extract_eye_roi(frame, eye_points):

    x = [p[0] for p in eye_points]
    y = [p[1] for p in eye_points]

    x1 = max(min(x) - 20, 0)
    y1 = max(min(y) - 20, 0)

    x2 = min(max(x) + 20, frame.shape[1])
    y2 = min(max(y) + 20, frame.shape[0])

    roi = frame[y1:y2, x1:x2]

    return roi