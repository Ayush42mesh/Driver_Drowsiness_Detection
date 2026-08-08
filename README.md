# DDD - Driver Drowsiness Detection

DDD is a professional, modern, and highly responsive desktop application built with Python, CustomTkinter, and TensorFlow. It detects driver drowsiness in real-time using MediaPipe Face Detection and a Keras-trained MobileNetV2 classification model.
This is Live URL of The project - https://ddd-backend-ft7x.onrender.com
---

## Features

- **Modern Dark/Light Themes**: Beautiful visual layouts built with CustomTkinter cards, progress bars, and animated icons.
- **Multi-Threaded Video Streaming**: Smooth webcam feed capture running in a separate worker thread to prevent UI freezing.
- **MediaPipe Face Detection**: Instantly locates faces, crop frames, and executes prediction solely on the driver's face.
- **Averaged Classification Filtering**: A moving average of the last 10 frames filters predictions to prevent flashing labels or false alerts.
- **Smart Alarm Workflow**: 
  - Triggers an audible loop via Pygame when drowsiness is detected for 15 consecutive frames.
  - Flashes a high-visibility warning screen.
  - Automatically captures screenshots and records data logs.
  - Requires manual intervention (`STOP ALARM`) to disable.
- **CSV Data Logger**: Automatically logs timestamps, classification statuses, and confidence percentages.
- **Settings Dashboard**: Allows dynamically updating camera index, confidence threshold, consecutive trigger frame counts, alarm volume, and visual theme.
- **Interactive Statistics Panel**: Monitors total frames, total alarms, session duration, and logs.

---

## Folder Structure

```text
DDD/
├── app.py               # Application entry point
├── gui.py               # Main CustomTkinter UI frame, pages, and Pygame sound players
├── webcam.py            # Multi-threaded cv2 webcam stream read and auto-reconnection
├── predictor.py         # Smoothed classification moving average pipeline
├── model_loader.py      # Thread-safe TensorFlow/Keras .keras model importer
├── face_detector.py     # MediaPipe Face Detection crop interface
├── utils.py             # Common helper scripts: sound synthesizers, screenshot savers, loggers
├── config.py            # Local settings JSON parser and path builders
│
├── model/
│   └── DDD.keras        # Place your trained classification model here
│
├── alarm/
│   └── alarm.wav        # Loop alarm track (auto-generated if missing)
│
├── logs/
│   └── drowsiness_log.csv  # Auto-generated CSV recording event classifications
│
├── screenshots/         # Directory containing captured alert screenshots
├── assets/              # Standard assets folder
├── requirements.txt     # Python dependency lists
└── README.md            # System documentation
```

---

## Installation & Setup

1. **Clone or Download** this directory to your machine.
2. **Install Python 3.9 - 3.11** (recommended version range for stable TensorFlow and MediaPipe execution).
3. Open a terminal in the project directory and run the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. **Model Setup**:
   Place your pre-trained `DDD.keras` model file into the `model/` folder.
   *Note: If the application cannot find `model/DDD.keras` at startup, it will run in a **demo simulation mode** with placeholder predictions. This allows testing the GUI, camera frames, face detection, screenshot triggers, logging, and alarms without needing a model pre-loaded.*

---

## Running the Application

Launch the desktop client using the Python interpreter:
```bash
python app.py
```

---

## Configuration Details

You can change configurations on the **Settings Page** within the application, or edit `settings.json` manually:

- **Camera Index**: Set to `0` for default built-in cameras, or `1`, `2` for external USB webcams.
- **Drowsiness Confidence Threshold**: Adjust between `0.1` and `0.9` (default is `0.5`). Higher limits reduce false alarms but may delay genuine triggers.
- **Consecutive Alarm Trigger Frames**: The number of consecutive frames indicating drowsiness required before triggering the alarm sound (default is `15`).
- **Alarm Volume Level**: Adjust sliding bar from `0%` to `100%`.
- **UI Color Theme**: Toggle between `Dark` and `Light` themes instantly.

---
*Deployed on Vercel (Frontend) and Render (Backend).*
