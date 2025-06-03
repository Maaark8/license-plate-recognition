# models/ocr_model.py
import cv2
import numpy as np
import os

# --- Fast Plate OCR ---
try:
    from fast_plate_ocr import ONNXPlateRecognizer
    FAST_PLATE_OCR_MODEL_IDENTIFIER = 'global-plates-mobile-vit-v2-model'

    FAST_PLATE_OCR_INITIALIZED = True
except ImportError:
    print("WARNING: `fast-plate-ocr` library not found. Please install it: pip install fast-plate-ocr")
    FAST_PLATE_OCR_INITIALIZED = False
except Exception as e_fp_ocr_init:
    print(f"WARNING: Error importing FastPlateOCR components: {e_fp_ocr_init}. FastPlateOCR will not be available.")
    FAST_PLATE_OCR_INITIALIZED = False
# --- End Fast Plate OCR ---

class OCREngine:
    def __init__(self):
        self.fast_plate_ocr_instance = None
        if FAST_PLATE_OCR_INITIALIZED:
            try:
                model_id_to_use = 'global-plates-mobile-vit-v2-model' 

                self.fast_plate_ocr_instance = ONNXPlateRecognizer(model_id_to_use, device="cpu")
                print(f"FastPlateOCR instance initialized with model/ID: '{model_id_to_use}'.")
            except Exception as e:
                print(f"ERROR initializing ONNXPlateRecognizer instance with '{model_id_to_use}': {e}. It will not be available.")
                print("Ensure the model identifier is correct or the model file exists if providing a path.")
                self.fast_plate_ocr_instance = None
        
        self.enable_debug_flags = False 
        self.debug_data = {} 

    def _run_fast_plate_ocr(self, plate_roi_bgr_image):
        if not self.fast_plate_ocr_instance or plate_roi_bgr_image is None or plate_roi_bgr_image.size == 0:
            if self.enable_debug_flags: self.debug_data['S_FastPlateOCR_Error'] = "Instance not ready or no image."
            return None

        try:
            # --- CONVERT TO GRAYSCALE ---
            if len(plate_roi_bgr_image.shape) == 3 and plate_roi_bgr_image.shape[2] == 3:
                gray_plate_roi = cv2.cvtColor(plate_roi_bgr_image, cv2.COLOR_BGR2GRAY)
            elif len(plate_roi_bgr_image.shape) == 2: # Already grayscale
                gray_plate_roi = plate_roi_bgr_image
            else: # Unexpected shape
                if self.enable_debug_flags: self.debug_data['S_FastPlateOCR_Error'] = f"Unexpected image shape for FastPlateOCR: {plate_roi_bgr_image.shape}"
                return None
            
            if self.enable_debug_flags: self.debug_data['S_FastPlateOCR_InputGray'] = gray_plate_roi.copy()
            # --- END CONVERSION ---

            # Pass the grayscale image to the run method
            predictions = self.fast_plate_ocr_instance.run(gray_plate_roi) 

            if self.enable_debug_flags: self.debug_data['S_FastPlateOCR_RawResult'] = predictions

            if predictions:
                if isinstance(predictions, list) and len(predictions) > 0:
                    best_result = predictions[0] 
                    plate_text_raw, confidence_raw = "", 0.0
                    if isinstance(best_result, dict):
                        plate_text_raw = best_result.get('text', '')
                        confidence_raw = best_result.get('score', 0.0)
                    elif isinstance(best_result, tuple) and len(best_result) == 2:
                        plate_text_raw, confidence_raw = best_result
                    elif isinstance(best_result, str):
                        plate_text_raw = best_result
                        confidence_raw = 0.9 
                    else:
                        if self.enable_debug_flags: self.debug_data['S_FastPlateOCR_Error'] = "Unexpected prediction item format."
                        return None
                    plate_text = "".join(filter(str.isalnum, plate_text_raw.upper()))
                    confidence = float(confidence_raw) * 100 if confidence_raw <= 1.0 and confidence_raw !=0 else float(confidence_raw) # Handle if already 0-100
                    if confidence_raw > 1.0: # If it's already 0-100 scale
                        confidence = float(confidence_raw)

                    if plate_text:
                        return {'text': plate_text, 'confidence': confidence, 'method': 'fast_plate_ocr'}
                else:
                     if self.enable_debug_flags: self.debug_data['S_FastPlateOCR_Error'] = "Result is not a non-empty list."
                     return None
            if self.enable_debug_flags: self.debug_data['S_FastPlateOCR_Error'] = "No results returned."
            
        except Exception as e:
            print(f"Error during FastPlateOCR (run method) processing: {e}")
            if self.enable_debug_flags: self.debug_data['S_FastPlateOCR_Error'] = str(e)
        return None

    def recognize_plate_text(self, original_plate_roi_bgr): # Orchestrator
        self.debug_data = {} 
        if self.enable_debug_flags: self.debug_data['S0_InputOriginalPlateROI_BGR'] = original_plate_roi_bgr.copy()

        final_text, final_conf, final_method = "OCR_UNAVAILABLE", 0.0, "none"
        all_res_dict = {} 

        if self.fast_plate_ocr_instance:
            fp_ocr_result = self._run_fast_plate_ocr(original_plate_roi_bgr)
            all_res_dict['fast_plate_ocr'] = fp_ocr_result 
            
            if fp_ocr_result and fp_ocr_result.get('text'):
                final_text, final_conf, final_method = fp_ocr_result['text'], fp_ocr_result['confidence'], 'fast_plate_ocr'
            elif fp_ocr_result: 
                final_text, final_conf, final_method = "", 0.0, 'fast_plate_ocr (no text)'
            else: 
                final_text, final_conf, final_method = "FAST_OCR_FAIL", 0.0, 'fast_plate_ocr (error)'
        else:
            if self.enable_debug_flags: print("FastPlateOCR instance not available.")
            final_text, final_conf, final_method = "FAST_OCR_NA", 0.0, 'fast_plate_ocr (N/A)'
            all_res_dict['fast_plate_ocr'] = {'text':final_text, 'confidence':final_conf, 'method':final_method}
        
        return {'text':final_text, 'confidence':final_conf, 'method':final_method, 'all_results':all_res_dict}