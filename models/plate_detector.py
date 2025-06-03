# models/plate_detector.py
from ultralytics import YOLO

# --- NEW: ML-based Plate Detector ---
class MLPlateDetector:
    def __init__(self, model_path='models/license_plate_yolov8_model.pt', confidence_threshold=0.3): # Placeholder model_path
        """
        Initialize ML-based plate detector.
        Args:
            model_path (str): Path to the trained plate detection model (e.g., YOLO .pt file).
                              YOU WILL NEED TO PROVIDE A VALID MODEL PATH HERE.
            confidence_threshold (float): Minimum confidence to consider a detection.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        try:
            if model_path and model_path != 'models/plate_yolov8_model.pt': # Attempt to load only if a real path is given
                self.model = YOLO(self.model_path)
                print(f"Successfully loaded ML plate detector model from: {self.model_path}")
            else:
                print(f"WARNING: Placeholder or no ML plate detector model path provided ('{model_path}').")
                print("ML plate detection will not function. Please provide a trained model.")
        except Exception as e:
            print(f"ERROR: Could not load ML plate detector model from {self.model_path}: {e}")
            print("ML Plate detection will likely fail. Check model path and integrity.")

    def detect(self, image_roi_bgr):
        """
        Detects license plates in the given image ROI using the ML model.
        Returns: List of dictionaries, each with 'bbox' (x1, y1, x2, y2) and 'confidence'.
        """
        detected_plates = []
        if self.model is None:
            # print("WARN: ML Plate Detector model not loaded. Cannot perform ML detection.")
            return detected_plates

        try:
            # Adjust input size if your model expects a specific size, e.g., results = self.model(image_roi_bgr, imgsz=640, ...)
            results = self.model(image_roi_bgr, verbose=False, conf=self.confidence_threshold)

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        
                        if confidence >= self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            detected_plates.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence
                            })
            
            detected_plates.sort(key=lambda x: x['confidence'], reverse=True)

        except Exception as e:
            print(f"Error during ML plate detection: {e}")
        
        return detected_plates


# --- Main LicensePlateDetector class ---
class LicensePlateDetector:
    def __init__(self, ml_model_path='models/license_plate_yolov8_model.pt', ml_confidence_threshold=0.25): # Placeholder
        """
        Initialize the license plate detector.
        It will try to use an ML model.

        Args:
            ml_model_path (str): Path to the ML plate detection model.
                                 **IMPORTANT**: Replace the default placeholder with an actual model path.
            ml_confidence_threshold (float): Confidence for ML plate detector.
        """
        self.ml_plate_detector_instance = MLPlateDetector(
            model_path=ml_model_path,
            confidence_threshold=ml_confidence_threshold
        )

    def detect_plates_in_roi(self, image_bgr, vehicle_bbox):
        """
        Detect license plates within a vehicle's ROI.
        Primarily uses ML model if available and loaded.
        """
        x1_v, y1_v, x2_v, y2_v = vehicle_bbox
        
        # Define ROI from vehicle bbox, possibly with slight padding
        # Padding can help if plate is near the edge of car detection
        padding_x = int((x2_v - x1_v) * 0.05) # 5% width padding
        padding_y = int((y2_v - y1_v) * 0.05) # 5% height padding

        roi_x1 = max(0, x1_v - padding_x)
        roi_y1 = max(0, y1_v - padding_y)
        roi_x2 = min(image_bgr.shape[1], x2_v + padding_x)
        roi_y2 = min(image_bgr.shape[0], y2_v + padding_y)
        
        vehicle_roi_bgr = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

        if vehicle_roi_bgr.size == 0:
            return []

        plates_abs_coords = []

        # --- Attempt ML-based detection ---
        if self.ml_plate_detector_instance.model is not None:
            ml_detected_plates_relative = self.ml_plate_detector_instance.detect(vehicle_roi_bgr)
            
            for plate_rel in ml_detected_plates_relative:
                bbox_rel = plate_rel['bbox']
                # Convert relative ROI bbox to absolute image coordinates
                abs_bbox = [
                    bbox_rel[0] + roi_x1, bbox_rel[1] + roi_y1,
                    bbox_rel[2] + roi_x1, bbox_rel[3] + roi_y1
                ]
                plates_abs_coords.append({
                    'bbox': abs_bbox,
                    'confidence': plate_rel['confidence']
                })
            
            # Apply Non-Maximum Suppression if ML model outputs multiple overlapping boxes
            if plates_abs_coords:
                plates_abs_coords = self._non_max_suppression(plates_abs_coords, iou_threshold=0.3)
            
            return plates_abs_coords[:1] # Return only the top 1 (or few) detection(s)

        else:
            print("WARN: ML Plate model not available. Plate detection will not be performed effectively.")
            return []

    def _non_max_suppression(self, boxes_data, iou_threshold):
        """
        Apply Non-Maximum Suppression.
        boxes_data: list of dicts, each with 'bbox' and 'confidence'.
        """
        if not boxes_data:
            return []

        # Sort by confidence score in descending order
        boxes_data = sorted(boxes_data, key=lambda x: x['confidence'], reverse=True)
        
        selected_boxes = []
        while boxes_data:
            current_box_data = boxes_data.pop(0)
            selected_boxes.append(current_box_data)
            
            remaining_boxes = []
            for box_data in boxes_data:
                iou = self._calculate_iou(current_box_data['bbox'], box_data['bbox'])
                if iou < iou_threshold:
                    remaining_boxes.append(box_data)
            boxes_data = remaining_boxes
            
        return selected_boxes

    def _calculate_iou(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0
        return inter_area / union_area