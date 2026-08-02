from collections import deque
import numpy as np

class DrowsinessPredictor:
    """
    Manages predictions, applies a moving average smoothing filter over the last
    10 prediction scores to prevent flickering, and tracks consecutive drowsy
    detections to determine if the alarm should trigger.
    """
    def __init__(self, model_loader, config):
        self.model_loader = model_loader
        self.config = config
        
        # History queue for moving average (maxlen=10)
        self.history = deque(maxlen=10)
        
        # Consecutive frame counter for alarm triggering
        self.consecutive_drowsy_count = 0
        self.alarm_active = False

    def predict_and_smooth(self, face_image):
        """
        Runs prediction on the face image, updates history, smooths confidence,
        and tracks consecutive frames of drowsiness.
        Returns:
            smoothed_label: "Drowsy" or "Non Drowsy"
            smoothed_confidence: float (0.0 to 1.0)
            consecutive_count: int (current consecutive drowsy frames)
            should_alarm: bool (True if consecutive_count exceeds threshold and not already active)
        """
        # Run inference
        label, confidence = self.model_loader.predict(face_image)
        
        # Convert prediction to probability of "Drowsy"
        if label == "Drowsy":
            drowsy_prob = confidence
        else:
            drowsy_prob = 1.0 - confidence
            
        # Append to moving average history
        self.history.append(drowsy_prob)
        
        # Calculate moving average
        avg_drowsy_prob = sum(self.history) / len(self.history)
        
        # Classify based on smoothed average and threshold
        threshold = self.config.confidence_threshold
        if avg_drowsy_prob >= threshold:
            smoothed_label = "Drowsy"
            smoothed_confidence = avg_drowsy_prob
        else:
            smoothed_label = "Non Drowsy"
            smoothed_confidence = 1.0 - avg_drowsy_prob
            
        # Drowsiness logic: Check consecutive frames
        should_alarm = False
        if smoothed_label == "Drowsy":
            # If alarm is already active, we don't need to trigger it again, 
            # but we continue counting or maintain the state.
            if not self.alarm_active:
                self.consecutive_drowsy_count += 1
                if self.consecutive_drowsy_count >= self.config.consecutive_frames:
                    should_alarm = True
                    self.alarm_active = True
        else:
            # Non-drowsy frame resets the counter unless the alarm is already active.
            # Once the alarm is active, the counter is not reset until the user clicks STOP ALARM.
            if not self.alarm_active:
                self.consecutive_drowsy_count = 0

        return smoothed_label, smoothed_confidence, self.consecutive_drowsy_count, should_alarm

    def reset_alarm_state(self):
        """Resets the alarm and the consecutive frames counter."""
        self.consecutive_drowsy_count = 0
        self.alarm_active = False
        self.history.clear()
