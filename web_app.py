import os
import base64
import cv2
import numpy as np
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory

# Import project files
from config import Config
from utils import generate_default_alarm, save_screenshot, log_prediction
from model_loader import ModelLoader
from face_detector import FaceDetector
from predictor import DrowsinessPredictor

app = Flask(__name__)

# Initialize configurations and sub-systems
config = Config()
config.create_dirs()
generate_default_alarm(config.alarm_path)

model_loader = ModelLoader()
# Preload the default Keras model
print(f"Preloading model from: {config.model_path}...")
success, msg = model_loader.load_model(config.model_path)
print(f"Model Preload Status: {success} ({msg})")

face_detector = FaceDetector(min_detection_confidence=0.5)
predictor = DrowsinessPredictor(model_loader, config)

def decode_base64_image(base64_str):
    """Decodes a base64 encoded data-uri image string into a BGR opencv frame."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def encode_image_to_base64(frame):
    """Encodes a BGR opencv frame into a base64 image data-uri string."""
    try:
        _, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{jpg_as_text}"
    except Exception as e:
        print(f"Error encoding image: {e}")
        return ""

@app.before_request
def handle_options_preflight():
    """Intercepts and resolves preflight OPTIONS requests for CORS compliance."""
    if request.method == 'OPTIONS':
        return '', 204

@app.after_request
def add_cors_headers(response):
    """Appends CORS headers to all outgoing responses."""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def index():
    """Serves the main application page."""
    return render_template('index.html')

@app.route('/alarm/alarm.wav')
def serve_alarm():
    """Serves the alarm sound file to the web browser."""
    return send_from_directory(config.alarm_dir, 'alarm.wav')

@app.route('/detect', methods=['POST'])
def detect():
    """
    POST route that receives a webcam frame from the browser,
    runs the face detector and drowsiness classifier, and
    returns real-time telemetry along with the annotated frame.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "No image payload found"}), 400

    frame = decode_base64_image(data['image'])
    if frame is None:
        return jsonify({"success": False, "error": "Failed to decode frame"}), 400

    h, w, c = frame.shape
    face_crop, bbox, annotated_frame = face_detector.detect_and_crop(frame)

    prediction_label = "No Face"
    confidence_val = 0.0
    consecutive_drowsy_count = 0
    should_alarm = False

    is_fallback = False
    if face_crop is None:
        face_crop = frame
        is_fallback = True

    # Draw bounding box
    if bbox and not is_fallback:
        xmin, ymin, width, height = bbox
        # Green outline by default, Red if alarm is active
        box_color = (0, 0, 255) if predictor.alarm_active else (0, 255, 0)
        cv2.rectangle(annotated_frame, (xmin, ymin), (xmin + width, ymin + height), box_color, 3)

    # Run inference
    label, conf, count, should_alarm = predictor.predict_and_smooth(face_crop)
    prediction_label = label
    confidence_val = conf
    consecutive_drowsy_count = count

    # Render status texts on the annotated frame
    text_color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)
    status_text = "DROWSY" if label == "Drowsy" else "ACTIVE"
    display_text = f"{status_text} ({confidence_val * 100:.1f}%)"
    if is_fallback:
        display_text += " [No Face Locked]"

    cv2.putText(
        annotated_frame,
        display_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2,
        cv2.LINE_AA
    )

    # Handle alert trigger logging & screenshot
    if should_alarm:
        threading.Thread(
            target=save_screenshot,
            args=(annotated_frame.copy(), config.screenshots_dir),
            daemon=True
        ).start()
        threading.Thread(
            target=log_prediction,
            args=(config.logs_dir, label, conf),
            daemon=True
        ).start()

    # Encode annotated frame back to base64
    annotated_base64 = encode_image_to_base64(annotated_frame)

    return jsonify({
        "success": True,
        "image": annotated_base64,
        "bbox": [int(x) for x in bbox] if bbox else None,
        "label": prediction_label,
        "confidence": confidence_val,
        "count": consecutive_drowsy_count,
        "consecutive_frames_threshold": config.consecutive_frames,
        "should_alarm": should_alarm,
        "alarm_active": predictor.alarm_active,
        "model_loaded": model_loader.is_loaded,
        "is_demo_mode": model_loader.is_demo_mode
    })

@app.route('/reset_alarm', methods=['POST'])
def reset_alarm():
    """Resets the alarm and counter state in the drowsiness predictor."""
    predictor.reset_alarm_state()
    return jsonify({"success": True, "message": "Alarm states reset."})

@app.route('/get_settings', methods=['GET'])
def get_settings():
    """Returns current settings to the UI."""
    return jsonify({
        "confidence_threshold": config.confidence_threshold,
        "consecutive_frames": config.consecutive_frames,
        "alarm_volume": config.alarm_volume,
        "theme": config.theme,
        "model_path": config.model_path,
        "model_loaded": model_loader.is_loaded,
        "is_demo_mode": model_loader.is_demo_mode
    })

@app.route('/save_settings', methods=['POST'])
def save_settings():
    """Updates and saves user-configured settings."""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Invalid data payload"}), 400

    try:
        config.confidence_threshold = float(data.get("confidence_threshold", config.confidence_threshold))
        config.consecutive_frames = int(data.get("consecutive_frames", config.consecutive_frames))
        config.alarm_volume = float(data.get("alarm_volume", config.alarm_volume))
        
        # Check if the model path changed, reload if so
        new_model_path = data.get("model_path", config.model_path)
        reload_needed = new_model_path != config.model_path
        config.model_path = new_model_path
        
        config.save()
        
        reload_success = True
        reload_msg = ""
        if reload_needed:
            print(f"Reloading model from new path: {config.model_path}...")
            reload_success, reload_msg = model_loader.load_model(config.model_path)
            predictor.reset_alarm_state()

        return jsonify({
            "success": True,
            "message": "Settings saved successfully.",
            "reload_success": reload_success,
            "reload_msg": reload_msg,
            "model_loaded": model_loader.is_loaded,
            "is_demo_mode": model_loader.is_demo_mode
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get_logs', methods=['GET'])
def get_logs():
    """Reads and returns the drowsiness detection history log as JSON."""
    log_file = os.path.join(config.logs_dir, "drowsiness_log.csv")
    if not os.path.exists(log_file):
        return jsonify([])

    import csv
    logs = []
    try:
        with open(log_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                logs.append(row)
        return jsonify(logs[-100:])  # Return last 100 entries
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the application
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
