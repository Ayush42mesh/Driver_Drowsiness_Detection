import cv2
import numpy as np

class FaceDetector:
    """
    Wrapper around MediaPipe Face Detection.
    Detects faces, extracts the bounding box, adds padding, 
    and crops the face region from a frame.
    """
    def __init__(self, min_detection_confidence=0.5):
        self.min_confidence = min_detection_confidence
        self.face_detection = None
        self.mp_face_detection = None
        self.mp_drawing = None
        self.use_opencv = False
        self.face_cascade = None
        self.initialize_detector()

    def initialize_detector(self):
        """Initializes the MediaPipe Face Detection modules, falling back to OpenCV Cascade if unavailable."""
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_drawing = mp.solutions.drawing_utils
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=0, 
                    min_detection_confidence=self.min_confidence
                )
                print("MediaPipe Face Detection initialized successfully.")
                return
        except Exception as e:
            print(f"MediaPipe initialization bypassed: {e}")

        # Fallback to OpenCV Haar Cascade
        try:
            import os
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                # Try local file
                local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
                os.makedirs(local_dir, exist_ok=True)
                local_path = os.path.join(local_dir, "haarcascade_frontalface_default.xml")
                
                if not os.path.exists(local_path):
                    print("Haar Cascade XML not found. Downloading fallback from OpenCV repository...")
                    import urllib.request
                    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                    urllib.request.urlretrieve(url, local_path)
                    print("Download complete.")
                
                self.face_cascade = cv2.CascadeClassifier(local_path)
                
            if not self.face_cascade.empty():
                self.use_opencv = True
                print("OpenCV Haar Cascade Face Detector initialized as fallback.")
            else:
                print("Warning: Haar Cascade XML file could not be loaded. Face detection will not function.")
        except Exception as e:
            print(f"Error initializing OpenCV Face Detector: {e}")

    def detect_and_crop(self, frame, padding_ratio=0.15):
        """
        Detects faces in the BGR frame.
        Crops and returns the primary face with padding.
        Returns:
            cropped_face: np.ndarray or None
            bbox_coords: tuple (xmin, ymin, width, height) or None
            annotated_frame: np.ndarray (frame with detection box drawn)
        """
        if self.use_opencv:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                if len(faces) == 0:
                    return None, None, frame
                
                # Sort by area size to get the largest face
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                (x, y, w, h) = faces[0]
                
                # Apply padding
                pad_w = int(w * padding_ratio)
                pad_h = int(h * padding_ratio)
                
                height, width, _ = frame.shape
                xmin_pad = max(0, x - pad_w)
                ymin_pad = max(0, y - pad_h)
                xmax_pad = min(width, x + w + pad_w)
                ymax_pad = min(height, y + h + pad_h)
                
                cropped_face = frame[ymin_pad:ymax_pad, xmin_pad:xmax_pad]
                bbox_coords = (x, y, w, h)
                return cropped_face, bbox_coords, frame.copy()
            except Exception as e:
                print(f"Error during OpenCV face detection: {e}")
                return None, None, frame

        if self.face_detection is None:
            return None, None, frame

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        
        try:
            results = self.face_detection.process(rgb_frame)
        except Exception as e:
            print(f"Error running face detection: {e}")
            return None, None, frame

        annotated_frame = frame.copy()
        
        if not results.detections:
            return None, None, annotated_frame

        best_detection = results.detections[0]
        bbox = best_detection.location_data.relative_bounding_box
        xmin = int(bbox.xmin * w)
        ymin = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        pad_w = int(width * padding_ratio)
        pad_h = int(height * padding_ratio)
        
        xmin_pad = max(0, xmin - pad_w)
        ymin_pad = max(0, ymin - pad_h)
        xmax_pad = min(w, xmin + width + pad_w)
        ymax_pad = min(h, ymin + height + pad_h)
        
        cropped_face = frame[ymin_pad:ymax_pad, xmin_pad:xmax_pad]
        
        if cropped_face.size == 0:
            return None, None, annotated_frame
            
        bbox_coords = (xmin, ymin, width, height)
        return cropped_face, bbox_coords, annotated_frame
        
    def close(self):
        """Releases the face detection resource."""
        if self.face_detection:
            self.face_detection.close()
            self.face_detection = None
