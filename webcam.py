import cv2
import threading
import time

class WebcamStream:
    """
    Manages the webcam feed in a separate thread. Calculates real-time FPS,
    handles thread-safe frame access, and supports automatic reconnection
    and manual resetting of camera devices.
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.stopped = True
        self.is_connected = False
        self.lock = threading.Lock()
        
        # FPS Calculation
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = 0.0
        
        self.thread = None

    def start(self):
        """Starts the background thread to capture frames."""
        if not self.stopped:
            return
            
        self.stopped = False
        self.is_connected = False
        self.frame = None
        
        self.fps_start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        
        self.thread = threading.Thread(target=self._update, name="WebcamStreamThread", daemon=True)
        self.thread.start()

    def _update(self):
        """Internal thread loop to capture frames and compute FPS."""
        # Try initializing
        self._connect_camera()
        
        while not self.stopped:
            if not self.is_connected:
                # Wait before trying to reconnect automatically
                time.sleep(2.0)
                self._connect_camera()
                continue
                
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Failed to grab frame. Camera disconnected?")
                self.is_connected = False
                if self.cap:
                    self.cap.release()
                continue

            # Update frame safely
            with self.lock:
                self.frame = frame.copy()

            # FPS calculation
            self.frame_count += 1
            elapsed_time = time.time() - self.fps_start_time
            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.fps_start_time = time.time()
                
            # Sleep slightly to avoid hogging CPU (target ~30-60 FPS)
            time.sleep(0.01)

        # Cleanup when stopped
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False

    def _connect_camera(self):
        """Attempts to open the camera device."""
        try:
            if self.cap:
                self.cap.release()
            
            # On Windows, using DSHOW often speeds up initialization and limits errors
            # We'll try cv2.CAP_DSHOW and fallback to standard if it fails
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.camera_index)
                
            if self.cap.isOpened():
                # Set buffer size to 1 to ensure we get the latest frame
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.is_connected = True
                print(f"Successfully connected to camera index {self.camera_index}")
            else:
                self.is_connected = False
                print(f"Failed to open camera index {self.camera_index}")
        except Exception as e:
            self.is_connected = False
            print(f"Error opening camera: {e}")

    def read(self):
        """Returns the latest frame and current FPS thread-safely."""
        with self.lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
        return frame_copy, self.fps

    def stop(self):
        """Stops the camera stream thread."""
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

    def change_camera(self, new_index):
        """Changes the camera index and restarts the connection."""
        running = not self.stopped
        self.stop()
        self.camera_index = new_index
        if running:
            self.start()
