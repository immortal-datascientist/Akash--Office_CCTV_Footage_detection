from ultralytics import YOLO
import cv2 # Not directly used for detection, but good to have for potential image ops
import numpy as np
from config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, TARGET_CLASSES

class YOLODetector:
   
    def __init__(self):
   
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            print(f"YOLOv8 model loaded successfully from: '{YOLO_MODEL_PATH}'")
            print(f"Detection Confidence Threshold: {CONFIDENCE_THRESHOLD}")
            print(f"NMS (IOU) Threshold: {NMS_THRESHOLD}")
            print(f"Target classes for detection: {TARGET_CLASSES} (0 usually means 'person' in COCO dataset)")
        except Exception as e:
            # Raise a RuntimeError to indicate a critical failure in loading the model
            raise RuntimeError(f"Error: Failed to load YOLO model from '{YOLO_MODEL_PATH}'. "
                               f"Please ensure the path is correct and the file exists. Details: {e}")

    def detect(self, frame):
        """
        Performs object detection on the given video frame to find persons.

        Args:
            frame (numpy.ndarray): The input video frame (H, W, 3 BGR image).

        Returns:
            list: A list of dictionaries, where each dictionary represents a detected person
                  and contains their 'bbox' (x1, y1, x2, y2 coordinates), 'confidence' score,
                  and 'class_id'. Returns an empty list if no persons are detected or if an error occurs.
        """
        detections = []
        try:
       
            results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD)[0]
            
            if results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    
                    # Only include detections that belong to our `TARGET_CLASSES` (i.e., 'person')
                    if class_id in TARGET_CLASSES:
                        # Extract bounding box coordinates (x1, y1, x2, y2)
                        # `xyxy` returns [x1, y1, x2, y2] format
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) 
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id
                        })
        except Exception as e:
            print(f"Warning: Error during YOLO detection on a frame: {e}")
            # Continue processing even if detection fails for a frame
        
        return detections