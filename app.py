# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
import json
from datetime import datetime
import time
import io

# Import our custom modules
from models.car_detector import CarDetector
from models.plate_detector import LicensePlateDetector 
from models.ocr_model import OCREngine
from utils.image_processing import enhance_image_quality, extract_plate_roi
from utils.visualization import draw_detection_results

# Page configuration
st.set_page_config(
    page_title="🚗 License Plate Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .detection-card { border: 2px solid #e0e0e0; border-radius: 10px; padding: 1rem; margin: 1rem 0; background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .high-confidence { border-color: #28a745; background: linear-gradient(135deg, #d4edda 0%, #f8fff9 100%); }
    .medium-confidence { border-color: #ffc107; background: linear-gradient(135deg, #fff3cd 0%, #fffef7 100%); }
    .low-confidence { border-color: #dc3545; background: linear-gradient(135deg, #f8d7da 0%, #fff7f7 100%); }
    .metric-card { background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .step-card { background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #007bff; }
    .debug-segmentation-container img { border: 1px solid #ccc; margin: 2px; background-color: #333; }
    .easyocr-debug-img { border: 2px solid lightblue; margin-top: 5px; } 
</style>
""", unsafe_allow_html=True)

class LicensePlateRecognitionApp:
    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        if 'processing_history' not in st.session_state: st.session_state.processing_history = []
        if 'current_results' not in st.session_state: st.session_state.current_results = None
        if 'ocr_debug_output_for_display' not in st.session_state:
            st.session_state.ocr_debug_output_for_display = []

    @st.cache_resource 
    def load_models(_self): 
        try:
            with st.spinner("🔄 Loading AI models..."):
                car_detector = CarDetector()
                actual_plate_model_path = "models/license_plate_yolov8_model.pt" # Your ML plate detector model
                if not os.path.exists(actual_plate_model_path):
                    st.error(f"CRITICAL: Plate detection model not found at '{actual_plate_model_path}'.")
                plate_detector = LicensePlateDetector(ml_model_path=actual_plate_model_path, ml_confidence_threshold=0.25)
                
                ocr_engine = OCREngine() # This will initialize FastPlateOCR
                
            st.success("✅ AI Models Loaded!")
            return car_detector, plate_detector, ocr_engine
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}"); st.exception(e); st.stop()
    
    def sidebar_controls(self):
        st.sidebar.markdown("## 🎛️ Settings")
        st.sidebar.markdown("### 🚗 Vehicle Detection")
        vehicle_confidence = st.sidebar.slider("Vehicle Conf. Threshold", 0.1, 1.0, 0.4, 0.05, key="app_focr_vc")
        
        st.sidebar.markdown("### 🏷️ Plate Detection (ML)")
        plate_confidence_ml = st.sidebar.slider("ML Plate Conf. Threshold", 0.05, 1.0, 0.30, 0.05, key="app_focr_pc")
        
        st.sidebar.markdown("### 🖼️ Image Processing")
        enhance_image = st.sidebar.checkbox("Enhance Image Quality (for Car/Plate Detection)", True, key="app_focr_ei")
        
        st.sidebar.markdown("---<h3>🔧 OCR Debug</h3>", unsafe_allow_html=True)
        debug_ocr_data = st.sidebar.checkbox("Show OCR Debug Info", True, key="app_focr_dos") # Default True
        
        return {
            'vehicle_confidence': vehicle_confidence,
            'plate_confidence_ml': plate_confidence_ml,
            'enhance_image': enhance_image,
            'debug_ocr_data': debug_ocr_data 
        }
    
    def process_image(self, image_bgr, settings):
        car_detector, plate_detector, ocr_engine = self.load_models()
        car_detector.confidence_threshold = settings['vehicle_confidence']
        if hasattr(plate_detector, 'ml_plate_detector_instance'):
            plate_detector.ml_plate_detector_instance.confidence_threshold = settings['plate_confidence_ml']
        
        results = {'vehicles':[], 'plates':[], 'plate_texts':[], 'total_processing_time':0.0}
        st.session_state.ocr_debug_output_for_display = [] 
        start_time = time.time()

        img_detect = enhance_image_quality(image_bgr) if settings['enhance_image'] else image_bgr.copy()
        results['vehicles'] = car_detector.detect_vehicles(img_detect)
        
        all_plates = []
        for v_idx, v in enumerate(results['vehicles']):
            for p_data in plate_detector.detect_plates_in_roi(image_bgr, v['bbox']):
                p_data['vehicle_id'] = v_idx; all_plates.append(p_data)
        results['plates'] = all_plates
        
        ocr_texts_list = []
        for plate_data_from_detector in all_plates:
            original_roi_for_ocr_bgr = extract_plate_roi(image_bgr, plate_data_from_detector['bbox'])
            current_ocr_output_dict = {'text':'ROI_ERR','confidence':0.0,'method':'error'}
            if original_roi_for_ocr_bgr.size > 0:
                ocr_engine.enable_debug_flags = settings['debug_ocr_data'] # Set flag on OCREngine
                current_ocr_output_dict = ocr_engine.recognize_plate_text(original_roi_for_ocr_bgr)
                
                if settings['debug_ocr_data']:
                    debug_entry = {
                        'original_plate_roi_rgb': cv2.cvtColor(original_roi_for_ocr_bgr.copy(),cv2.COLOR_BGR2RGB),
                        'ocr_result_dict': current_ocr_output_dict.copy(),
                        'ocr_engine_internal_debug_stages': ocr_engine.debug_data.copy() 
                    }
                    st.session_state.ocr_debug_output_for_display.append(debug_entry)
            
            current_ocr_output_dict['plate_image_rgb_for_display'] = cv2.cvtColor(original_roi_for_ocr_bgr, cv2.COLOR_BGR2RGB) if original_roi_for_ocr_bgr.size > 0 else None
            ocr_texts_list.append(current_ocr_output_dict)
        results['plate_texts'] = ocr_texts_list
        
        results['total_processing_time'] = time.time() - start_time
        results['annotated_image_bgr'] = draw_detection_results(image_bgr, results['vehicles'], results['plates'], results['plate_texts'])
        return results
    
    def display_results(self, results, settings):
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("### 📊 Detections")
            if results.get('annotated_image_bgr') is not None:
                st.image(cv2.cvtColor(results['annotated_image_bgr'],cv2.COLOR_BGR2RGB), caption="Annotated Image", use_container_width=True)
        with col2:
            st.markdown("### 📈 Summary"); self.display_metrics(results)
            st.metric("Total Processing Time", f"{results.get('total_processing_time',0):.3f}s")

        # --- OCR DEBUG DISPLAY for FastPlateOCR ---
        if settings.get('debug_ocr_data', False) and st.session_state.ocr_debug_output_for_display:
            st.markdown("---<h3>🔧 OCR Debug Information</h3>", unsafe_allow_html=True)
            for i, debug_entry_data in enumerate(st.session_state.ocr_debug_output_for_display):
                ocr_result_dict = debug_entry_data.get('ocr_result_dict', {})
                method_used = ocr_result_dict.get('method', 'N/A')
                st.markdown(f"#### Plate {i+1} (OCR Method: {method_used})")

                with st.expander(f"Debug Details for Plate {i+1}", expanded= i==0):
                    st.image(debug_entry_data['original_plate_roi_rgb'], caption="S0. Original Plate ROI (Input to OCREngine)", width=300)
                    st.write(f"Final Text: `{ocr_result_dict.get('text', 'N/A')}` (Conf: {ocr_result_dict.get('confidence', 0.0):.1f}%)")

                    ocr_engine_internals = debug_entry_data.get('ocr_engine_internal_debug_stages', {})

                    if method_used == 'fast_plate_ocr':
                        st.markdown("##### FastPlateOCR Debug:")
                        if 'S_FastPlateOCR_RawPrediction' in ocr_engine_internals:
                            st.write("Raw Prediction from FastPlateOCR library:")
                            st.json(ocr_engine_internals['S_FastPlateOCR_RawPrediction'], expanded=False)
                        if 'S_FastPlateOCR_Error' in ocr_engine_internals:
                            st.error(f"FastPlateOCR Error: {ocr_engine_internals['S_FastPlateOCR_Error']}")
            st.markdown("---")
        # --- END OCR DEBUG ---

        if results.get('plates') and results.get('plate_texts'):
            st.markdown("### 🔍 Recognized Plates")
            for i, (plate, text_res) in enumerate(zip(results.get('plates',[]), results.get('plate_texts',[]))):
                conf, ocr_method = text_res.get('confidence',0.0), text_res.get('method','N/A')
                card_class, icon_char = ("low","🔴")
                if conf > 70: card_class, icon_char = ("high","🟢")
                elif conf > 40: card_class, icon_char = ("medium","🟡")
                st.markdown(f"""<div class="detection-card {card_class}-confidence">
                                <h4>{icon_char} Plate {i+1} (Vehicle {plate.get('vehicle_id',-1)+1})</h4>
                                <p><b>Recognized Text:</b> {text_res.get('text','N/A')}</p>
                                <p><b>Confidence:</b> {conf:.1f}%</p>
                                <p><b>OCR Method:</b> {ocr_method}</p>
                             </div>""", unsafe_allow_html=True)
                if text_res.get('plate_image_rgb_for_display') is not None:
                    st.image(text_res['plate_image_rgb_for_display'], caption=f"Plate ROI {i+1}", width=200)

    def display_metrics(self, results):
        vc,pc = len(results.get('vehicles',[])), len(results.get('plates',[]))
        valid_ocr_texts = ['ROI_ERR', 'OCR_FAIL_ALL', 'OCR_UNAVAILABLE', 'INPUT_ERR_CNN', 'BIN_FAIL_CNN', 'ROI_FAIL_S1','FAST_OCR_FAIL','FAST_OCR_NA', 'FAST_OCR_EMPTY']
        sr = sum(1 for tr in results.get('plate_texts',[]) if tr.get('text') and tr.get('text') not in valid_ocr_texts and tr.get('confidence',0) > 30)
        ac = np.mean([tr['confidence'] for tr in results.get('plate_texts',[]) if tr.get('text') and tr.get('text') not in valid_ocr_texts and tr.get('confidence',0) > 0]) if sr > 0 else 0.0
        m1,m2,m3,m4=st.columns(4); m1.metric("Vehicles",vc); m2.metric("Plates",pc); m3.metric("OCR OK",sr); m4.metric("Avg. OCR Conf",f"{ac:.1f}%")

    def export_results(self, results):
        if not results: return None
        exp_data = {'ts':datetime.now().isoformat(), 'v':results.get('vehicles',[]), 'pt':[], 'time':results.get('total_processing_time',0)}
        for i, (p,t) in enumerate(zip(results.get('plates',[]), results.get('plate_texts',[]))):
            exp_data['pt'].append({'id':i+1, 'v_id':p.get('vehicle_id',-1)+1, 'p_box':p.get('bbox'), 'p_conf':p.get('confidence'),
                                   'txt':t.get('text'), 'txt_conf':t.get('confidence'), 'method':t.get('method')})
        try: return json.dumps(exp_data, indent=2, default=lambda o: "<obj_not_serializable>")
        except TypeError: return json.dumps({"error":"export_failed_serialization"},indent=2)

    def run(self):
        st.markdown('<h1 class="main-header">🚗 License Plate Recognition</h1>', unsafe_allow_html=True)
        st.markdown("<div class='step-card'><h3>🎯 Pipeline:</h3><p>Upload Image ➔ Detect Vehicles ➔ Detect Plates ➔ OCR Text (FastPlateOCR)</p></div>", unsafe_allow_html=True)
        settings = self.sidebar_controls()
        uploaded_file = st.file_uploader("📁 Choose an image file", type=['jpg','jpeg','png','bmp'], key="app_uploader_focr_final")
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file)
            bgr_numpy_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)
            if st.button("🚀 Process Image", type="primary", key="app_proc_btn_focr_final"):
                with st.spinner("🔄 Processing... (FastPlateOCR)"):
                    try:
                        results_dict = self.process_image(bgr_numpy_image, settings)
                        st.session_state.current_results = results_dict
                        history_entry = {'Time':datetime.now().strftime('%H:%M:%S'), 'Vehicles':len(results_dict.get('vehicles',[])),
                                         'Plates':len(results_dict.get('plates',[])), 'Proc. Time (s)':f"{results_dict.get('total_processing_time',0):.2f}"}
                        if results_dict.get('plate_texts') and results_dict['plate_texts']:
                            history_entry['First OCR'] = results_dict['plate_texts'][0].get('text','N/A')
                        st.session_state.processing_history.insert(0, history_entry)
                        st.success("✅ Processing Complete!")
                    except Exception as e: st.error(f"❌ Processing Error: {str(e)}"); st.exception(e)
            if st.session_state.current_results:
                self.display_results(st.session_state.current_results, settings)
                st.markdown("### 💾 Export Options")
                exp_col1, exp_col2 = st.columns(2)
                json_data_to_export = self.export_results(st.session_state.current_results)
                if json_data_to_export:
                    exp_col1.download_button(label="📄 Download JSON", data=json_data_to_export,
                        file_name=f"lpr_results_{time.strftime('%Y%m%d-%H%M%S')}.json", mime="application/json", key="app_dl_json_focr_final")
                if st.session_state.current_results.get('annotated_image_bgr') is not None:
                    annotated_rgb_array = cv2.cvtColor(st.session_state.current_results['annotated_image_bgr'], cv2.COLOR_BGR2RGB)
                    annotated_pil_image = Image.fromarray(annotated_rgb_array)
                    img_byte_buffer = io.BytesIO()
                    annotated_pil_image.save(img_byte_buffer, format='PNG')
                    exp_col2.download_button(label="🖼️ Download Image", data=img_byte_buffer.getvalue(),
                        file_name=f"lpr_annotated_img_{time.strftime('%Y%m%d-%H%M%S')}.png", mime="image/png", key="app_dl_img_focr_final")
        if st.session_state.processing_history:
            st.markdown("### 📊 Processing History (Last 10)")
            history_df_display = pd.DataFrame(st.session_state.processing_history[:10])
            cols_for_df = [col for col in ['Time','Vehicles','Plates','First OCR','Proc. Time (s)'] if col in history_df_display.columns]
            if not history_df_display.empty: st.dataframe(history_df_display[cols_for_df],use_container_width=True)
            if st.button("🗑️ Clear History", key="app_clear_hist_focr_final"):
                st.session_state.processing_history=[]; st.session_state.current_results=None
                st.session_state.ocr_debug_output_for_display=[]; st.rerun()

if __name__ == "__main__":
    LicensePlateRecognitionApp().run()