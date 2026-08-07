#  Heart Rate Prediction Through Retina 👁️❤️

A professional AI-based real-time heart rate prediction system using:

* OpenCV
* MediaPipe FaceMesh
* Retina/Eye ROI Analysis
* Signal Processing
* Webcam-based Pulse Detection

This project predicts heart rate through eye-retina motion and pulse signal extraction using a webcam without any external sensors.

---

# Features

✅ Real-time webcam monitoring
✅ Retina/Eye-based pulse extraction
✅ MediaPipe FaceMesh eye tracking
✅ Heart rate prediction (BPM)
✅ Blink/Eye-close detection
✅ Signal waveform graph
✅ Accuracy/filtered graph
✅ Professional medical dashboard UI
✅ Stable BPM smoothing
✅ Auto reset BPM to 0 when eyes closed

---

# Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* SciPy
* PyTorch
* Signal Processing

---

# Project Structure

```text
retina_hr_ai/
│
├── cnn/
│   └── efficientphys.py
│
├── detection/
│   ├── facemesh_detector.py
│   └── eye_roi.py
│
├── processing/
│   ├── blink_detector.py
│   ├── bpm.py
│   └── filter.py
│
├── ui/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/Nandu388/Heart-Rate-Prediction-Through-Retina-.git
```

---

## 2. Open Project Folder

```bash
cd Heart-Rate-Prediction-Through-Retina-
```

---

## 3. Create Virtual Environment

```bash
python -m venv venv
```

---

## 4. Activate Environment

### Windows

```bash
venv\Scripts\activate
```

---

## 5. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Run Project

```bash
python main.py
```

---

# Working Principle

1. Webcam captures face in real time
2. MediaPipe detects eye landmarks
3. Eye ROI (retina region) extracted
4. Pulse signal estimated from retina variations
5. Signal filtered using bandpass filtering
6. BPM calculated from processed signal
7. Dashboard displays:

   * Heart Rate
   * Frequency
   * Signal Graph
   * Accuracy Graph

---

# Output Dashboard

The dashboard includes:

* Live webcam feed
* Retina preview
* Pulse signal graph
* Accuracy graph
* Real-time BPM prediction

---



# Important Note

This project is for:

* Educational Purpose
* Research/Demo Purpose
* AI Healthcare Visualization

This is not a medically certified diagnostic system.

---

# Future Improvements

* Deep learning trained retina model
* Infrared camera support
* ECG comparison
* Medical-grade signal processing
* Cloud dashboard
* Patient history tracking

---

# Author

Nandini

---

# License

This project is open-source for educational purposes.
