import numpy as np
from collections import deque

class PostureDetector:
    def __init__(self, aspect_ratio_threshold=0.8, height_threshold=0.6, min_frames=10):
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.height_threshold = height_threshold
        self.min_frames = min_frames
        self.posture_history = {}
        
    def detect_posture(self, detection, frame_height):
        # Ensure detection is a dictionary with the expected keys
        if not isinstance(detection, dict) or 'bbox' not in detection:
            return 'unknown'
            
        # Extract bbox coordinates safely
        bbox = detection['bbox']
        if len(bbox) < 4:
            return 'unknown'
            
        x1, y1, x2, y2 = bbox
        
        width = x2 - x1
        height = y2 - y1
        
        # Calculate metrics
        aspect_ratio = width / height if height > 0 else 0
        center_y = (y1 + y2) / 2
        vertical_position = center_y / frame_height
        
        # Determine posture
        is_sitting = (aspect_ratio > self.aspect_ratio_threshold and 
                     vertical_position > self.height_threshold and
                     height > 50)  # Minimum height threshold
        
        # Get track_id safely
        track_id = detection.get('track_id', 'unknown')
        
        # Update posture history
        if track_id not in self.posture_history:
            self.posture_history[track_id] = deque(maxlen=self.min_frames)
        
        self.posture_history[track_id].append(is_sitting)
        
        # Determine stable posture
        if len(self.posture_history[track_id]) >= self.min_frames:
            sitting_count = sum(self.posture_history[track_id])
            sitting_ratio = sitting_count / len(self.posture_history[track_id])
            
            if sitting_ratio > 0.7:
                return 'working'
            elif sitting_ratio < 0.3:
                return 'standing'
        
        return 'unknown'