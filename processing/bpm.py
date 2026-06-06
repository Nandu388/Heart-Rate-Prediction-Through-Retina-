import numpy as np


def calculate_bpm(signal):

    signal = np.array(signal)

    if len(signal) < 150:
        return 0

    signal = signal - np.mean(signal)

    fft = np.fft.rfft(signal)

    freqs = np.fft.rfftfreq(
        len(signal),
        d=1/30
    )

    magnitudes = np.abs(fft)

    idx = np.where(
        (freqs >= 0.8) &
        (freqs <= 1.8)
    )

    freqs = freqs[idx]

    magnitudes = magnitudes[idx]

    if len(magnitudes) == 0:
        return 0

    peak = freqs[
        np.argmax(magnitudes)
    ]

    bpm = int(peak * 60)

    # LIMITS

    if bpm < 55:
        bpm = 55

    if bpm > 110:
        bpm = 110

    return bpm