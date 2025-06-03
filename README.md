# 🚗 License Plate Recognition System

This project is a Streamlit web application for detecting vehicles in an image, locating their license plates, and performing Optical Character Recognition (OCR) to extract the plate number.

## Features

*   **Vehicle Detection:** Utilizes a YOLOv8 model to detect cars, motorcycles, buses, and trucks in uploaded images.
*   **License Plate Detection:** Employs a dedicated YOLOv8 model (or similar ML model) to find license plates within the detected vehicle regions.
*   **License Plate OCR:** Primarily uses the `fast-plate-ocr` library for efficient and specialized license plate character recognition.
*   **Image Upload:** Users can upload common image formats (JPG, PNG, BMP).
*   **Interactive Controls:** Sidebar controls allow adjustment of detection confidence thresholds and image enhancement options.
*   **Results Visualization:**
    *   Displays the original image.
    *   Shows an annotated image with bounding boxes for detected vehicles and plates, along with the recognized plate text.
    *   Provides detailed information for each recognized plate, including the text and confidence score.
*   **Performance Metrics:** Shows summary statistics like the number of vehicles/plates detected and OCR success rates.
*   **Export Options:** Allows downloading of processing results in JSON format and the annotated image.
*   **Processing History:** Keeps a session-based history of processed images and key results.
*   **OCR Debugging:** Includes an option to display intermediate steps from the OCR process for troubleshooting and tuning (though with `fast-plate-ocr`, internal steps are minimal).

## Project Structure
car_license_detection/
│
├── app.py # Main Streamlit web application
├── models/
│ ├── car_detector.py # YOLO car detection logic
│ ├── plate_detector.py # ML-based license plate detection
│ ├── ocr_model.py # OCR engine (integrates fast-plate-ocr)
│ └── license_plate_yolov8_model.pt
├── utils/
│ ├── image_processing.py # Image enhancement and ROI extraction
│ └── visualization.py # Drawing detection results
├── requirements.txt # Python dependencies
└── README.md # This file

## Setup and Installation

1.  **Clone the Repository (Example):**
    ```bash
    git clone https://github.com/Maaark8/license-plate-recognition.git
    cd car_license_detection
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download/Place Necessary ML Models:**

    *   **Car Detection Model (YOLOv8):**
        The `ultralytics` library (used by `CarDetector`) typically downloads `yolov8n.pt` (or other specified YOLOv8 variants) automatically on first use if it's not found. If you wish to use a specific version, place it in the project's root directory or ensure `models/car_detector.py` points to it.

    *   **License Plate Detection Model (ML-based):**
        You need to provide your own trained model for license plate detection.
        - Place your plate detection model file (e.g., a `.pt` file if using YOLO, or `.onnx` if using another ONNX model) into the `models/` directory.
        - Update the path in `models/plate_detector.py` (inside the `LicensePlateDetector` or `MLPlateDetector` class `__init__`) to point to your model file.

    *   **`fast-plate-ocr` Model:**
        The `fast-plate-ocr` library might download its required ONNX model automatically based on the identifier you use when initializing `ONNXPlateRecognizer` in `models/ocr_model.py`.
        ```python
        # In models/ocr_model.py, inside OCREngine.__init__
        # model_id_to_use = 'global-plates-mobile-vit-v2-model' # Example ID
        # self.fast_plate_ocr_instance = ONNXPlateRecognizer(model_id_to_use)
        ```
        Consult the `fast-plate-ocr` documentation.

## Running the Application

Once the setup is complete and all dependencies and models are in place:

1.  Navigate to the project's root directory (e.g., `car_license_detection/`).
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  The application should open in your web browser.

## Key Libraries Used

*   **Streamlit:** For creating the web application interface.
*   **OpenCV (`opencv-python`):** For image processing tasks.
*   **Pillow (PIL):** For image manipulation.
*   **NumPy:** For numerical operations, especially with images.
*   **Pandas:** For displaying data in tables (e.g., processing history).
*   **Ultralytics YOLO:** For vehicle and license plate detection.
*   **`fast-plate-ocr`:** For specialized license plate character recognition.
*   **ONNX Runtime (`onnxruntime` / `onnxruntime-gpu`):** Inference engine for `fast-plate-ocr`.
