import os
import csv
import cv2
import wave
import struct
import math
from datetime import datetime
import shutil

def generate_default_alarm(filepath):
    """
    Generates a default alarm sound file (a 1-second 880Hz beep) 
    using the standard library wave and struct modules.
    """
    if os.path.exists(filepath):
        return
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    sample_rate = 44100
    duration = 1.0  # seconds
    freq = 880.0    # Hz (high pitched beep)
    num_samples = int(sample_rate * duration)
    
    try:
        with wave.open(filepath, 'w') as wav_file:
            # params: (nchannels, sampwidth, framerate, nframes, comptype, compname)
            wav_file.setparams((1, 2, sample_rate, num_samples, 'NONE', 'not compressed'))
            for i in range(num_samples):
                # Simple sine wave, multiplying by 0.5 to keep volume moderate
                value = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * i / sample_rate))
                data = struct.pack('<h', value)
                wav_file.writeframes(data)
    except Exception as e:
        print(f"Error generating default alarm file: {e}")

def save_screenshot(frame, screenshots_dir, filename_prefix="alarm"):
    """
    Saves the given cv2 frame as a JPEG screenshot in the screenshots directory.
    Filename format: YYYY-MM-DD_HH-MM-SS.jpg
    """
    os.makedirs(screenshots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(screenshots_dir, filename)
    try:
        cv2.imwrite(filepath, frame)
        return filepath
    except Exception as e:
        print(f"Error saving screenshot: {e}")
        return None

def log_prediction(logs_dir, prediction, confidence):
    """
    Appends a new prediction entry to the CSV log.
    Fields: Date, Time, Prediction, Confidence
    """
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "drowsiness_log.csv")
    
    file_exists = os.path.exists(log_file)
    
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    try:
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Date", "Time", "Prediction", "Confidence"])
            writer.writerow([date_str, time_str, prediction, f"{confidence * 100:.1f}%"])
    except Exception as e:
        print(f"Error writing to log CSV: {e}")

def export_logs(logs_dir, destination_path):
    """
    Exports/copies the current drowsiness log file to a destination path.
    """
    source_file = os.path.join(logs_dir, "drowsiness_log.csv")
    if not os.path.exists(source_file):
        raise FileNotFoundError("No log data available to export.")
    
    try:
        shutil.copy2(source_file, destination_path)
        return True
    except Exception as e:
        print(f"Error exporting logs: {e}")
        raise e
