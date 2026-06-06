import numpy as np


def euclidean(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


def calculate_ear(eye):

    A = euclidean(eye[1], eye[5])

    B = euclidean(eye[2], eye[4])

    C = euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear