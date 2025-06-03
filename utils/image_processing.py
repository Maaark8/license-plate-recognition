# utils/image_processing.py
import cv2
import numpy as np

def enhance_image_quality(image):
    """
    Enhance image quality for better detection
    """
    # Convert to LAB color space for better enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def preprocess_for_detection(image):
    """
    Preprocess image for better vehicle and plate detection
    """
    # Enhance contrast and brightness
    enhanced = enhance_image_quality(image)
    
    # Reduce noise while preserving edges
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return denoised

def extract_plate_roi(image, bbox, padding=10):
    """
    Extract license plate ROI with padding
    """
    x1, y1, x2, y2 = bbox
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    
    return image[y1:y2, x1:x2]