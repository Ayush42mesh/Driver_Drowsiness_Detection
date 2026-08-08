import os
import numpy as np

# Suppress TensorFlow logging to keep terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ModelLoader:
    """
    Handles loading the model (both Keras .keras and ONNX .onnx formats)
    in a thread-safe, error-resilient manner.
    Supports a demo fallback mode if the model is not present or invalid.
    """
    def __init__(self):
        self.model = None
        self.input_shape = (224, 224, 3) # Default for MobileNetV2
        self.is_loaded = False
        self.is_demo_mode = False
        self.is_onnx = False
        self.has_lambda = False

    def load_model(self, model_path):
        """
        Attempts to load the model file from model_path.
        Returns (success, message).
        """
        # Check if tensorflow is available
        try:
            import tensorflow as tf
            tf_available = True
        except ImportError:
            tf_available = False

        # If TF is not available, try to redirect to ONNX model if possible
        if not tf_available and model_path.endswith(".keras"):
            onnx_path = model_path.replace(".keras", ".onnx")
            if os.path.exists(onnx_path):
                print(f"TensorFlow not available. Redirecting to ONNX model at {onnx_path}")
                model_path = onnx_path
            else:
                print("TensorFlow not available and ONNX model not found. Fallback to demo mode.")

        # If the file does not exist, check if there's an .onnx version instead
        if not os.path.exists(model_path):
            onnx_path = model_path.replace(".keras", ".onnx")
            if os.path.exists(onnx_path):
                model_path = onnx_path
            else:
                self.is_loaded = False
                self.is_demo_mode = True
                self.model = None
                return False, f"Model file not found at {model_path}. Running in demo simulation mode."

        self.is_onnx = model_path.endswith(".onnx")

        # Load ONNX Model
        if self.is_onnx:
            try:
                import onnxruntime as ort
                # Load the model session
                self.model = ort.InferenceSession(model_path)
                self.is_loaded = True
                self.is_demo_mode = False
                
                # Retrieve input shape dynamically
                shape = self.model.get_inputs()[0].shape
                if len(shape) == 4:
                    self.input_shape = (shape[1] or 224, shape[2] or 224, shape[3] or 3)
                elif len(shape) == 3:
                    self.input_shape = shape
                
                # MobileNetV2 uses preprocess_input, which is compiled in the graph. We keep has_lambda = True.
                self.has_lambda = True
                
                # Perform a warm-up prediction
                dummy_input = np.zeros((1, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
                input_name = self.model.get_inputs()[0].name
                _ = self.model.run(None, {input_name: dummy_input})
                
                return True, "ONNX Model Loaded Successfully."
                
            except ImportError:
                self.is_loaded = False
                self.is_demo_mode = True
                self.model = None
                return False, "onnxruntime is not installed. Running in demo simulation mode."
            except Exception as e:
                self.is_loaded = False
                self.is_demo_mode = True
                self.model = None
                return False, f"Failed to load ONNX model: {str(e)}. Running in demo simulation mode."

        # Load Keras Model
        try:
            import tensorflow as tf
            
            # Register common custom functions (like preprocess_input from MobileNetV2)
            custom_objects = {}
            try:
                custom_objects["preprocess_input"] = tf.keras.applications.mobilenet_v2.preprocess_input
            except Exception:
                pass

            # Load the Keras model
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            self.is_loaded = True
            self.is_demo_mode = False
            
            # Inspect input shape
            try:
                shape = self.model.input_shape
                if isinstance(shape, list):
                    shape = shape[0]
                if len(shape) == 4:
                    h, w = shape[1], shape[2]
                    self.input_shape = (h, w, shape[3])
                elif len(shape) == 3:
                    self.input_shape = shape
            except Exception as shape_err:
                print(f"Could not determine model input shape: {shape_err}. Using default (224, 224, 3)")
                self.input_shape = (224, 224, 3)
                
            # Check for built-in preprocessing layers
            try:
                self.has_lambda = any(
                    isinstance(l, tf.keras.layers.Lambda) or 'lambda' in l.name.lower() 
                    for l in self.model.layers
                )
            except Exception:
                self.has_lambda = False
                
            # Perform a warm-up prediction
            dummy_input = np.zeros((1, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            
            return True, "Model Loaded Successfully."
            
        except ImportError:
            self.is_loaded = False
            self.is_demo_mode = True
            self.model = None
            return False, "TensorFlow is not installed. Running in demo simulation mode."
        except Exception as e:
            self.is_loaded = False
            self.is_demo_mode = True
            self.model = None
            return False, f"Failed to load Keras model: {str(e)}. Running in demo simulation mode."

    def predict(self, face_image):
        """
        Runs inference on the preprocessed face image.
        If in demo mode, returns a simulated classification.
        Returns (prediction_class, confidence_score).
        """
        if self.is_demo_mode or self.model is None:
            # Demo Mode: Simulate prediction.
            gray = cv2_gray_if_color(face_image)
            brightness = np.mean(gray) if gray is not None else 127
            prob = np.random.uniform(0.01, 0.25)
            if np.random.rand() < 0.05:
                prob = np.random.uniform(0.75, 0.98)
                
            if prob >= 0.5:
                return "Drowsy", prob
            else:
                return "Non Drowsy", 1.0 - prob

        # Predict using ONNX Runtime
        if self.is_onnx:
            try:
                h, w, c = self.input_shape
                resized = cv2_resize_safe(face_image, w, h)
                if c == 3:
                    import cv2
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                if self.has_lambda:
                    img_array = resized.astype(np.float32)
                else:
                    img_array = resized.astype(np.float32) / 255.0
                
                batch = np.expand_dims(img_array, axis=0)
                
                input_name = self.model.get_inputs()[0].name
                preds = self.model.run(None, {input_name: batch})[0]
                return self.decode_predictions(preds)
            except Exception as e:
                print(f"Error during ONNX inference: {e}")
                return "Non Drowsy", 0.99

        # Predict using TensorFlow/Keras
        try:
            import tensorflow as tf
            h, w, c = self.input_shape
            resized = cv2_resize_safe(face_image, w, h)
            if c == 3:
                import cv2
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            if self.has_lambda:
                img_array = resized.astype(np.float32)
            else:
                img_array = resized.astype(np.float32) / 255.0
            
            batch = np.expand_dims(img_array, axis=0)
            preds = self.model.predict(batch, verbose=0)
            return self.decode_predictions(preds)
            
        except Exception as e:
            print(f"Error during TensorFlow inference: {e}")
            return "Non Drowsy", 0.99

    def decode_predictions(self, preds):
        """Decodes raw probability predictions to class label and confidence score."""
        pred_shape = preds.shape
        if len(pred_shape) == 2 and pred_shape[1] == 1:
            prob = float(preds[0][0])
            if prob < 0.5:
                return "Drowsy", 1.0 - prob
            else:
                return "Non Drowsy", prob
        elif len(pred_shape) == 2 and pred_shape[1] == 2:
            prob_drowsy = float(preds[0][0])
            prob_non_drowsy = float(preds[0][1])
            if prob_drowsy >= prob_non_drowsy:
                return "Drowsy", prob_drowsy
            else:
                return "Non Drowsy", prob_non_drowsy
        else:
            max_idx = int(np.argmax(preds[0]))
            prob = float(preds[0][max_idx])
            label = "Non Drowsy" if max_idx == 1 else "Drowsy"
            return label, prob

# Helpers for prediction to avoid cv2 circular imports or dependencies
def cv2_gray_if_color(img):
    if img is None:
        return None
    if len(img.shape) == 3:
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    return img

def cv2_resize_safe(img, w, h):
    import cv2
    return cv2.resize(img, (w, h))
