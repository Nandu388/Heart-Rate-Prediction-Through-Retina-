import numpy as np
import cv2


class EfficientPhys:

    def __init__(self):

        self.signal_buffer = []

        self.prev_value = None

    def predict_signal(self, roi):

        if roi is None:
            return []

        if roi.size == 0:
            return []

        # RESIZE ROI

        roi = cv2.resize(
            roi,
            (160, 80)
        )

        # GREEN CHANNEL

        green = roi[:, :, 1]

        # CLAHE ENHANCEMENT

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        green = clahe.apply(green)

        # BLUR

        green = cv2.GaussianBlur(
            green,
            (5, 5),
            0
        )

        # FLOAT

        green = green.astype(np.float32)

        # AVERAGE INTENSITY

        value = np.mean(green)

        if self.prev_value is None:

            self.prev_value = value

        # DIFFERENTIAL PULSE

        pulse = value - self.prev_value

        self.prev_value = value

        # SIGNAL BOOST

        pulse = pulse * 10

        self.signal_buffer.append(pulse)

        # KEEP BUFFER

        if len(self.signal_buffer) > 300:
            self.signal_buffer.pop(0)

        return self.signal_buffer