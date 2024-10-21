from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import pipeline
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the trained Keras model
model = load_model('color_final_h5.h5')

class_names = ['Azure', 'Blue_spruce', 'Blush', 'Denim', 'Emeral', 'Flamingo', 'Fuchsia',
           'Garnet', 'Ginger', 'Lavender', 'Lemon', 'Magenta', 'Mauva_gauva',
           'Peach', 'Purple', 'Radiant_orchid', 'Rose_gold', 'Rose_petal',
           'Salmon', 'Saphire', 'Spring_cactus', 'Tanzanite', 'Titanium', 'Wintergreen']

segmentation_pipeline = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    return predicted_class

class ClassificationResult(BaseModel):
    hair_type: str

class ErrorResponse(BaseModel):
    error: str

@app.post("/classify_hair_color", response_model=ClassificationResult, responses={400: {"model": ErrorResponse}})
async def classify_hair(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    if not detect_face(img):
        return ErrorResponse(error="No face detected in the image")
    
    try:
        predicted_class = classify_image(img)
        return ClassificationResult(hair_type=predicted_class)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during classification: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Hair Color Classifier API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)