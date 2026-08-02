import os
import json

class Config:
    """
    Manages application configuration settings, allowing saving and loading
    from a local JSON settings file.
    """
    DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

    def __init__(self, config_path=None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        
        # Default Settings
        self.confidence_threshold = 0.5
        self.camera_index = 0
        self.alarm_volume = 0.8
        self.consecutive_frames = 15
        self.theme = "Dark"
        
        # Paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.base_dir, "model")
        self.alarm_dir = os.path.join(self.base_dir, "alarm")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.screenshots_dir = os.path.join(self.base_dir, "screenshots")
        
        # Model path
        self.model_path = os.path.join(self.model_dir, "DDD.keras")
        self.alarm_path = os.path.join(self.alarm_dir, "alarm.wav")
        
        self.load()

    def load(self):
        """Loads configuration from settings.json if it exists."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.confidence_threshold = float(data.get("confidence_threshold", self.confidence_threshold))
                    self.camera_index = int(data.get("camera_index", self.camera_index))
                    self.alarm_volume = float(data.get("alarm_volume", self.alarm_volume))
                    self.consecutive_frames = int(data.get("consecutive_frames", self.consecutive_frames))
                    self.theme = str(data.get("theme", self.theme))
                    self.model_path = str(data.get("model_path", self.model_path))
            except Exception as e:
                print(f"Error loading configuration: {e}. Using defaults.")

    def save(self):
        """Saves current configuration parameters to settings.json."""
        data = {
            "confidence_threshold": self.confidence_threshold,
            "camera_index": self.camera_index,
            "alarm_volume": self.alarm_volume,
            "consecutive_frames": self.consecutive_frames,
            "theme": self.theme,
            "model_path": self.model_path
        }
        try:
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def create_dirs(self):
        """Ensures all project subdirectories exist."""
        for d in [self.model_dir, self.alarm_dir, self.logs_dir, self.screenshots_dir]:
            os.makedirs(d, exist_ok=True)
