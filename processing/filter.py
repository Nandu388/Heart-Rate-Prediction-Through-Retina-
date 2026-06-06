from scipy.signal import butter, filtfilt
import numpy as np


def bandpass_filter(signal):

    signal = np.array(signal)

    if len(signal) < 60:
        return signal

    fs = 30

    low = 0.8 / (fs / 2)

    high = 2.0 / (fs / 2)

    b, a = butter(
        3,
        [low, high],
        btype='band'
    )

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered