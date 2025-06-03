# utils/visualization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_detection_results(image, vehicles, plates, plate_texts):
    """
    Draw all detection results on the image
    """
    result_image = image.copy()
    
    # Draw vehicle bounding boxes
    for i, vehicle in enumerate(vehicles):
        x1, y1, x2, y2 = vehicle['bbox']
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Vehicle label
        label = f"Vehicle {i+1}: {vehicle['class']} ({vehicle['confidence']:.2f})"
        cv2.putText(result_image, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw plate bounding boxes and text
    for i, (plate, text_result) in enumerate(zip(plates, plate_texts)):
        x1, y1, x2, y2 = plate['bbox']
        
        # Choose color based on text confidence
        confidence = text_result['confidence']
        if confidence > 70:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 40:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Draw plate rectangle
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Plate text
        if text_result['text']:
            text_label = f"{text_result['text']} ({confidence:.1f}%)"
        else:
            text_label = "No text detected"
            
        # Create background for text
        text_size = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(result_image, (x1, y2 + 5), 
                     (x1 + text_size[0], y2 + text_size[1] + 15), color, -1)
        
        # Draw text
        cv2.putText(result_image, text_label, (x1 + 2, y2 + text_size[1] + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return result_image

def create_detection_summary_plot(vehicles, plates, plate_texts):
    """
    Create a summary plot showing detection statistics
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Vehicle Types', 'Plate Confidence Distribution', 
                       'OCR Method Performance', 'Character Count Distribution'),
        specs=[[{'type': 'pie'}, {'type': 'histogram'}],
               [{'type': 'bar'}, {'type': 'histogram'}]]
    )
    
    # Vehicle types pie chart
    if vehicles:
        vehicle_types = [v['class'] for v in vehicles]
        type_counts = {}
        for vtype in vehicle_types:
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(type_counts.keys()), 
                   values=list(type_counts.values()),
                   name="Vehicle Types"),
            row=1, col=1
        )
    
    # Plate confidence distribution
    if plate_texts:
        confidences = [pt['confidence'] for pt in plate_texts if pt['confidence'] > 0]
        if confidences:
            fig.add_trace(
                go.Histogram(x=confidences, nbinsx=10, name="Confidence"),
                row=1, col=2
            )
    
    # OCR method performance
    if plate_texts:
        methods = [pt['method'] for pt in plate_texts if pt['text']]
        if methods:
            method_counts = {}
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            
            fig.add_trace(
                go.Bar(x=list(method_counts.keys()), 
                       y=list(method_counts.values()),
                       name="OCR Methods"),
                row=2, col=1
            )
    
    # Character count distribution
    if plate_texts:
        char_counts = [len(pt['text']) for pt in plate_texts if pt['text']]
        if char_counts:
            fig.add_trace(
                go.Histogram(x=char_counts, nbinsx=8, name="Character Count"),
                row=2, col=2
            )
    
    fig.update_layout(height=600, showlegend=False, 
                     title_text="Detection Analysis Summary")
    
    return fig