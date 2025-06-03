# models/car_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class CarDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize car detector using YOLOv8
        """
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)
        
        # Vehicle class IDs in COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
    
    def detect_vehicles(self, image):
        """
        Detect vehicles in the image
        Returns: List of vehicle bounding boxes with confidence scores
        """
        results = self.model(image, verbose=False)
        vehicles = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if it's a vehicle and meets confidence threshold
                    if class_id in self.vehicle_classes and confidence >= self.confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        vehicles.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class': self.vehicle_classes[class_id],
                            'class_id': class_id
                        })
        
        return vehicles
    
    def draw_vehicle_boxes(self, image, vehicles):
        """
        Draw bounding boxes around detected vehicles
        """
        annotated_image = image.copy()
        
        for i, vehicle in enumerate(vehicles):
            x1, y1, x2, y2 = vehicle['bbox']
            confidence = vehicle['confidence']
            class_name = vehicle['class']
            
            # Draw rectangle
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Add vehicle number
            cv2.putText(annotated_image, f"Vehicle {i+1}", (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated_image