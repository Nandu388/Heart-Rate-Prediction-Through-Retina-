import cv2
import numpy as np

from detection.facemesh_detector import FaceMeshDetector

from cnn.efficientphys import EfficientPhys

from processing.blink_detector import calculate_ear
from processing.filter import bandpass_filter
from processing.bpm import calculate_bpm


# --------------------------------
# INITIALIZE
# --------------------------------

detector = FaceMeshDetector()

model = EfficientPhys()

cap = cv2.VideoCapture(0)

stable_bpm = 72

closed_frames = 0

signal = []


# --------------------------------
# PROFESSIONAL GRAPH
# --------------------------------

def create_graph(signal, title="Signal"):

    graph = np.zeros((100, 220, 3), dtype=np.uint8)

    # BORDER

    cv2.rectangle(
        graph,
        (0, 0),
        (219, 99),
        (120, 120, 120),
        1
    )

    # AXES

    cv2.line(
        graph,
        (25, 10),
        (25, 85),
        (255, 255, 255),
        1
    )

    cv2.line(
        graph,
        (25, 85),
        (210, 85),
        (255, 255, 255),
        1
    )

    # TITLE

    cv2.putText(
        graph,
        title,
        (80, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

    # Y LABELS

    cv2.putText(
        graph,
        "1",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (200, 200, 200),
        1
    )

    cv2.putText(
        graph,
        "0",
        (5, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (200, 200, 200),
        1
    )

    # X LABELS

    for i in range(0, 181, 60):

        x = 25 + i

        cv2.line(
            graph,
            (x, 82),
            (x, 88),
            (255, 255, 255),
            1
        )

        cv2.putText(
            graph,
            str(i),
            (x - 5, 97),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            (180, 180, 180),
            1
        )

    # SIGNAL

    if signal is None:
        return graph

    if len(signal) < 5:
        return graph

    signal = np.array(signal[-180:])

    signal = signal - np.min(signal)

    maximum = np.max(signal)

    if maximum != 0:

        signal = signal / maximum

    signal = (signal * 65).astype(np.int32)

    for i in range(1, len(signal)):

        x1 = 25 + i - 1
        y1 = 85 - signal[i - 1]

        x2 = 25 + i
        y2 = 85 - signal[i]

        cv2.line(
            graph,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return graph


# --------------------------------
# MAIN LOOP
# --------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame = cv2.resize(
        frame,
        (430, 320)
    )

    # WHITE DASHBOARD

    canvas = np.ones(
        (360, 760, 3),
        dtype=np.uint8
    ) * 240

    bpm = stable_bpm

    frequency = round(
        bpm / 60,
        2
    )

    preview = np.zeros(
        (80, 80, 3),
        dtype=np.uint8
    )

    # --------------------------------
    # LANDMARKS
    # --------------------------------

    landmarks = detector.detect_landmarks(frame)

    if landmarks:

        left_eye, right_eye = detector.get_eye_points(
            landmarks,
            frame.shape
        )

        # COMBINED EYE ROI

        all_points = left_eye + right_eye

        x = [p[0] for p in all_points]
        y = [p[1] for p in all_points]

        x1 = max(min(x) - 30, 0)
        y1 = max(min(y) - 20, 0)

        x2 = min(max(x) + 30, frame.shape[1])
        y2 = min(max(y) + 25, frame.shape[0])

        eye_roi = frame[y1:y2, x1:x2]

        # BLUE BOX

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # PREVIEW

        try:

            preview = cv2.resize(
                eye_roi,
                (80, 80)
            )

        except:
            pass

        # EAR

        ear = calculate_ear(left_eye)

        # EYES CLOSED

        if ear < 0.18:

            closed_frames += 1

        else:

            closed_frames = 0

        # IF CLOSED

        if closed_frames > 10:

            bpm = 0

            stable_bpm = 0

            frequency = 0

            signal = []

            model.signal_buffer.clear()

        # EYES OPEN

        else:

            signal = model.predict_signal(
                eye_roi
            )

            filtered = bandpass_filter(
                signal
            )

            new_bpm = calculate_bpm(
                filtered
            )

            # STABLE BPM

            if new_bpm != 0:

                stable_bpm = int(
                    (
                        stable_bpm * 0.94
                    ) +
                    (
                        new_bpm * 0.06
                    )
                )

                # LIMITS

                if stable_bpm > 110:
                    stable_bpm = 110

                if stable_bpm < 55:
                    stable_bpm = 55

            bpm = stable_bpm

            frequency = round(
                bpm / 60,
                2
            )

            signal = filtered

    # --------------------------------
    # CAMERA
    # --------------------------------

    canvas[10:330, 10:440] = frame

    # --------------------------------
    # PREVIEW
    # --------------------------------

    canvas[20:100, 500:580] = preview

    # --------------------------------
    # TEXT
    # --------------------------------

    cv2.putText(
        canvas,
        f"Freq: {frequency}",
        (610, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (40, 40, 40),
        1
    )

    cv2.putText(
        canvas,
        f"Heart rate: {bpm} bpm",
        (610, 70),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (40, 40, 40),
        1
    )

    # --------------------------------
    # SIGNAL GRAPH
    # --------------------------------

    graph1 = create_graph(
        signal,
        "Signal"
    )

    canvas[120:220, 500:720] = graph1

    # --------------------------------
    # ACCURACY GRAPH
    # --------------------------------

    smooth_signal = []

    if len(signal) > 20:

        smooth_signal = np.convolve(
            signal,
            np.ones(15) / 15,
            mode='same'
        )

    graph2 = create_graph(
        smooth_signal,
        "Accuracy"
    )

    canvas[230:330, 500:720] = graph2

    # --------------------------------
    # EXIT TEXT
    # --------------------------------

    cv2.putText(
        canvas,
        "Press ESC to Exit",
        (20, 350),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (80, 80, 80),
        1
    )

    # --------------------------------
    # SHOW WINDOW
    # --------------------------------

    cv2.imshow(
        "RETINA HEART RATE AI",
        canvas
    )

    key = cv2.waitKey(1)

    if key == 27:
        break


# --------------------------------
# RELEASE
# --------------------------------

cap.release()

cv2.destroyAllWindows()