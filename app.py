import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import pipeline
import base64
from io import BytesIO

# Set page config
st.set_page_config(page_title="BRB Hair Color Classifier", layout="wide", initial_sidebar_state="expanded")

# Load the trained Keras model
@st.cache_resource
def load_classification_model():
    return load_model('color_final_h5.h5')

model = load_classification_model()

class_names = ['Azure', 'Blue_spruce', 'Blush', 'Denim', 'Emeral', 'Flamingo', 'Fuchsia',
           'Garnet', 'Ginger', 'Lavender', 'Lemon', 'Magenta', 'Mauva_gauva',
           'Peach', 'Purple', 'Radiant_orchid', 'Rose_gold', 'Rose_petal',
           'Salmon', 'Saphire', 'Spring_cactus', 'Tanzanite', 'Titanium', 'Wintergreen']

@st.cache_resource
def load_segmentation_pipeline():
    return pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

segmentation_pipeline = load_segmentation_pipeline()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Functions for face detection and image classification
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

def classify_image(image):
    image = Image.fromarray(image).convert("RGB")
    
    segmented_image = segmentation_pipeline(image)
    
    image = np.array(segmented_image)
    image = Image.fromarray(image).convert("RGB")
    image = image.resize((224, 224))  # Adjust size according to your model's input shape
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image

    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions[0])]
    return segmented_image, predicted_class

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f0f4f8;
        color: #333;
    }
    .main {
        padding: 1rem;
    }
    h1, h2, h3 {
        color: #1a5f7a;
    }
    .stButton > button {
        background-color: #1a5f7a;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 30px;
        padding: 0.6rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2e8bc0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .webcam-container {
        border: 2px solid #1a5f7a;
        border-radius: 10px;
        overflow: hidden;
        margin: 0 auto;
    }
    .result-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .result-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a5f7a;
        margin-bottom: 0.5rem;
    }
    .result-text {
        font-size: 1rem;
        color: #34495e;
    }
    .hair-type {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2e8bc0;
        background-color: #e6f3f8;
        padding: 0.5rem;
        border-radius: 5px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .app-description {
        background-color: #e6f3f8;
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 1rem;
        color: #1a5f7a;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Header and Description
st.markdown("<h1 style='text-align: center; color: #1a5f7a; font-size: 2rem;'>BRB Hair Classifier</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="app-description">
    <p>Discover your unique hair type with our AI-powered classifier. Capture a selfie for personalized styling tips!</p>
</div>
""", unsafe_allow_html=True)

# Create a column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h3 style='text-align: center; font-size: 1.2rem;'>Capture Your Style</h3>", unsafe_allow_html=True)
    st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
    
    # Initialize session state for the captured frame
    if 'frame' not in st.session_state:
        st.session_state.frame = None

    # Placeholder for the video feed
    frame_placeholder = st.empty()

    # Debug information
    debug_info = st.empty()

    # Capture button
    capture_button = st.button("📸 Capture & Analyze")

    # Video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not open camera. Please check your camera connection and permissions.")
    else:
        debug_info.text("Camera opened successfully. Attempting to capture frames...")
        
        frame_count = 0
        max_frames = 1000  # Limit the number of frames to prevent infinite loop

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                debug_info.error(f"Failed to capture frame. Attempts: {frame_count + 1}")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", use_column_width=True)
            debug_info.text(f"Displaying frame {frame_count + 1}")
            
            if capture_button:
                st.session_state.frame = frame
                debug_info.text("Frame captured successfully!")
                break

            frame_count += 1
            # Use st.empty() instead of st.rerun() to update the UI
            placeholder = st.empty()
            placeholder.text(f"Frame {frame_count}")
            placeholder.empty()

        if frame_count >= max_frames:
            debug_info.warning("Reached maximum frame limit. Please refresh the page if the camera feed is not visible.")

    cap.release()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='text-align: center; font-size: 1.2rem;'>Your Hair Analysis</h3>", unsafe_allow_html=True)
    result_placeholder = st.empty()

    if capture_button and st.session_state.frame is not None:
        if detect_face(st.session_state.frame):
            try:
                background_removed_img, predicted_class = classify_image(st.session_state.frame)
                
                result_content = f"""
                <div class="result-container">
                    <img src="data:image/png;base64,{image_to_base64(background_removed_img)}" style="width:100%; border-radius:10px; margin-bottom:10px;">
                    <p class="result-header">Hair Analysis Results</p>
                    <p class="result-text">Your Hair Type:</p>
                    <p class="hair-type">{predicted_class}</p>
                    <p class="result-text">Our stylists recommend personalized treatments for your hair type. Book a consultation for expert advice!</p>
                </div>
                """
                result_placeholder.markdown(result_content, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during image classification: {str(e)}")
        else:
            st.warning("No face detected. Please ensure your face is clearly visible and try again.")

# Call-to-action section
st.markdown("""
<div style="text-align: center; margin-top: 1rem; padding: 1rem; background-color: #1a5f7a; border-radius: 10px; color: white;">
    <h3 style="font-size: 1.2rem;">Ready for a Style Revolution?</h3>
    <p style="font-size: 0.9rem;">Book an appointment with our expert stylists!</p>
    <a href="#" style="background-color: white; color: #1a5f7a; padding: 8px 16px; text-decoration: none; border-radius: 30px; display: inline-block; margin-top: 10px; font-weight: 600; font-size: 0.9rem;">Book Now</a>
</div>
""", unsafe_allow_html=True)