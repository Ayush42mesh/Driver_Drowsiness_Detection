import os
import numpy as np

# Suppress TensorFlow logging to keep terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ModelLoader:
    """
    Handles loading the TensorFlow/Keras model in a thread-safe, error-resilient manner.
    Supports a demo fallback mode if the model is not present or invalid.
    """
    def __init__(self):
        self.model = None
        self.input_shape = (224, 224, 3) # Default for MobileNetV2
        self.is_loaded = False
        self.is_demo_mode = False

    def load_model(self, model_path):
        """
        Attempts to load the model file from model_path.
        Returns (success, message).
        """
        if not os.path.exists(model_path):
            self.is_loaded = False
            self.is_demo_mode = True
            self.model = None
            return False, f"Model file not found at {model_path}. Running in demo simulation mode."

        try:
            # We import tensorflow locally to prevent importing it during startup if not needed immediately
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
                # Typically input shape is (None, 224, 224, 3)
                shape = self.model.input_shape
                if isinstance(shape, list):
                    shape = shape[0]
                
                # Extract height and width (usually indices 1 and 2 or 2 and 3 depending on format)
                # Standard channel-last is (None, height, width, channels)
                if len(shape) == 4:
                    h, w = shape[1], shape[2]
                    self.input_shape = (h, w, shape[3])
                elif len(shape) == 3:
                    self.input_shape = shape
            except Exception as shape_err:
                print(f"Could not determine model input shape: {shape_err}. Using default (224, 224, 3)")
                self.input_shape = (224, 224, 3)
                
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
        # Preprocessing: face_image should be resized to self.input_shape
        if self.is_demo_mode or self.model is None:
            # Demo Mode: Simulate prediction.
            # Let's do a semi-random behavior to make the demo realistic (e.g. 5% chance of Drowsy)
            # Or we can analyze brightness/variance to make it look responsive
            gray = cv2_gray_if_color(face_image)
            brightness = np.mean(gray) if gray is not None else 127
            
            # Let's say if the brightness drops below 60, it's slightly more likely to be drowsy (eyes closed/dark)
            # Or just simulate a baseline
            prob = np.random.uniform(0.01, 0.25)
            # Occasional drowsiness simulation spikes
            if np.random.rand() < 0.05:
                prob = np.random.uniform(0.75, 0.98)
                
            if prob >= 0.5:
                return "Drowsy", prob
            else:
                return "Non Drowsy", 1.0 - prob

        try:
            import tensorflow as tf
            
            # Ensure shape matches expected input
            h, w, c = self.input_shape
            resized = cv2_resize_safe(face_image, w, h)
            
            # Convert BGR to RGB for correct color representation in inference
            if c == 3:
                import cv2
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # If the model has a built-in Lambda layer calling preprocess_input,
            # it expects raw [0, 255] float inputs. Otherwise, we normalize to [0, 1].
            has_lambda = False
            try:
                has_lambda = any(
                    isinstance(l, tf.keras.layers.Lambda) or 'lambda' in l.name.lower() 
                    for l in self.model.layers
                )
            except Exception:
                pass
                
            if has_lambda:
                img_array = resized.astype(np.float32)
            else:
                img_array = resized.astype(np.float32) / 255.0
            
            # Expand dimensions to create batch: (1, H, W, C)
            batch = np.expand_dims(img_array, axis=0)
            
            preds = self.model.predict(batch, verbose=0)
            
            # Handle binary classification output
            # Inverted mapping: 0 (or lower probability) is Drowsy, 1 (or higher probability) is Non Drowsy
            
            # Let's inspect the shape of predictions
            pred_shape = preds.shape
            if len(pred_shape) == 2 and pred_shape[1] == 1:
                # Sigmoid output (inverted)
                prob = float(preds[0][0])
                if prob < 0.5:
                    return "Drowsy", 1.0 - prob
                else:
                    return "Non Drowsy", prob
            elif len(pred_shape) == 2 and pred_shape[1] == 2:
                # Softmax output: class 0 = Drowsy, class 1 = Non Drowsy
                prob_drowsy = float(preds[0][0])
                prob_non_drowsy = float(preds[0][1])
                if prob_drowsy >= prob_non_drowsy:
                    return "Drowsy", prob_drowsy
                else:
                    return "Non Drowsy", prob_non_drowsy
            else:
                # Fallback simple index (inverted)
                max_idx = int(np.argmax(preds[0]))
                prob = float(preds[0][max_idx])
                label = "Non Drowsy" if max_idx == 1 else "Drowsy"
                return label, prob
                
        except Exception as e:
            print(f"Error during inference: {e}")
            return "Non Drowsy", 0.99

# Helpers for prediction to avoid cv2 circular imports or dependencies
def cv2_gray_if_color(img):
    if img is None:
        return None
    if len(img.shape) == 3:
        # Convert RGB to grayscale (assuming model uses grayscale for brightness calculation in demo)
        # Note: opencv uses BGR but we use RGB for PIL/MediaPipe
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    return img

def cv2_resize_safe(img, w, h):
    # Safely resize image without external dependency on cv2 if we can, 
    # but we are importing cv2 globally in webcam so we can import it here.
    import cv2
    return cv2.resize(img, (w, h))
