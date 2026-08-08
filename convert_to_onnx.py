import os
import sys
import shutil

# Ensure dependencies are installed
try:
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow is not installed. Please run: pip install tensorflow")
    sys.exit(1)

try:
    import tf2onnx
except ImportError:
    print("Installing tf2onnx...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tf2onnx"])
    import tf2onnx

model_path = os.path.join("model", "DDD.keras")
onnx_path = os.path.join("model", "DDD.onnx")
export_path = "model_export"

if not os.path.exists(model_path):
    print(f"Error: Keras model not found at {model_path}")
    sys.exit(1)

custom_objects = {}
try:
    custom_objects["preprocess_input"] = tf.keras.applications.mobilenet_v2.preprocess_input
except Exception:
    pass

print(f"Loading Keras model from {model_path}...")
try:
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
except Exception as e:
    print(f"Error loading Keras model: {e}")
    sys.exit(1)

# Export the Keras model as a SavedModel directory (Keras 3 compliant)
print(f"Exporting model to temporary directory: {export_path}...")
if os.path.exists(export_path):
    shutil.rmtree(export_path)

try:
    if hasattr(model, 'export'):
        model.export(export_path)
    else:
        # Fallback for Keras 2
        tf.saved_model.save(model, export_path)
    print("Model exported successfully.")
except Exception as e:
    print(f"Error exporting Keras model to SavedModel: {e}")
    sys.exit(1)

print("Converting SavedModel to ONNX...")
try:
    # Run tf2onnx CLI command to convert saved model
    import subprocess
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", export_path,
        "--output", onnx_path,
        "--opset", "13"
    ]
    subprocess.check_call(cmd)
    
    # Cleanup temporary SavedModel directory
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
        
    print(f"Successfully converted and saved ONNX model to {onnx_path} 🎉")
except Exception as e:
    print(f"Error converting SavedModel to ONNX: {e}")
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    sys.exit(1)
