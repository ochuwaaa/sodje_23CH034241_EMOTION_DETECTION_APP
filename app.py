from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model once
model = load_model("emotion_model.h5")
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img, target_size=(48,48), grayscale=True):
    face = cv2.resize(face_img, target_size)
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    if grayscale:
        face = np.expand_dims(face, axis=-1)
    return face

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = None

    # --- Handle file upload ---
    if 'file' in request.files:
        file = request.files['file']
        img = Image.open(file).convert('RGB')

    # --- Handle webcam/base64 image ---
    elif 'webcam' in request.json:
        img_data = request.json['webcam']
        img_bytes = base64.b64decode(img_data.split(',')[1])  # remove data:image/jpeg;base64,
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    if img is None:
        return jsonify({'error': 'No image provided'})

    # Convert to OpenCV format
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Optional: resize very large images for faster detection
    max_width = 800
    if img_bgr.shape[1] > max_width:
        scale = max_width / img_bgr.shape[1]
        img_bgr = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # enhance contrast

    # Detect faces (more sensitive parameters)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))

    if len(faces) == 0:
        return jsonify({'error': 'No face detected'})

    # Use the first detected face
    x, y, w, h = faces[0]
    face_bgr = img_bgr[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess for MobileNetV2 model
    face_input = cv2.resize(face_rgb, (224, 224)) / 255.0
    face_input = np.expand_dims(face_input, axis=0)

    # Predict emotion
    preds = model.predict(face_input)
    emotion = class_labels[np.argmax(preds)]

    # Draw rectangle and label
    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (255,0,0), 2)
    cv2.putText(img_bgr, emotion, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)

    # Encode image to base64 for response
    _, buffer = cv2.imencode('.jpg', img_bgr)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'emotion': emotion, 'image': img_b64})


if __name__ == '__main__':
    app.run(debug=True)
