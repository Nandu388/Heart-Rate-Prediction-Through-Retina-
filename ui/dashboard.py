import cv2
import numpy as np

GRAPH_WIDTH = 300
GRAPH_HEIGHT = 150


def draw_graph(signal):

    graph = np.zeros((GRAPH_HEIGHT, GRAPH_WIDTH, 3), dtype=np.uint8)

    if len(signal) < 2:
        return graph

    signal = np.array(signal)

    signal = signal[-GRAPH_WIDTH:]

    signal = signal - np.min(signal)

    if np.max(signal) != 0:
        signal = signal / np.max(signal)

    signal = (signal * GRAPH_HEIGHT).astype(np.int32)

    for i in range(1, len(signal)):

        cv2.line(
            graph,
            (i - 1, GRAPH_HEIGHT - signal[i - 1]),
            (i, GRAPH_HEIGHT - signal[i]),
            (0, 255, 0),
            2
        )

    return graph


def draw_dashboard(frame, bpm, blink, signal):

    h, w, _ = frame.shape

    dashboard = np.zeros((h, 400, 3), dtype=np.uint8)

    cv2.putText(
        dashboard,
        f"Heart Rate: {bpm} BPM",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        dashboard,
        f"Blink: {blink}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    frequency = round(bpm / 60, 2)

    cv2.putText(
        dashboard,
        f"Freq: {frequency} Hz",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    graph = draw_graph(signal)

    dashboard[180:330, 50:350] = graph

    combined = np.hstack((frame, dashboard))

    return combined