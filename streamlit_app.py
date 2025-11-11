import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("emotion_model.h5")
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("😊 Emotion Detection Web App")

st.subheader("Upload an image or use your webcam")

# Option 1: Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Option 2: Webcam input
webcam_image = st.camera_input("Or take a photo using your webcam")

# Determine which input to use
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
elif webcam_image is not None:
    image = Image.open(webcam_image).convert('RGB')
else:
    image = None

if image is not None:
    st.image(image, caption='Input Image', use_column_width=True)

    # Convert to numpy array
    img_array = np.array(image)

    # Use OpenCV Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected! Try a different image or adjust your webcam.")
    else:
        # Take the first detected face
        x, y, w, h = faces[0]
        face = img_array[y:y+h, x:x+w]  # Keep RGB channels

        # Resize face to match model input
        face = cv2.resize(face, (224, 224))
        face = face / 255.0  # Normalize
        face = np.expand_dims(face, axis=0)  # Shape: (1, 224, 224, 3)

        # Predict emotion
        preds = model.predict(face)
        emotion = class_labels[np.argmax(preds)]

        st.success(f"Predicted Emotion: **{emotion}**")
