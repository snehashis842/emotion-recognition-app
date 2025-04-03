# app.py - Main Flask Application

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the trained emotion recognition model
try:
    model = load_model("emotion_recognition_model.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Face detector using Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def preprocess_image(image_path=None, image_array=None):
    """
    Preprocess image for emotion detection
    Accept either file path or numpy array
    """
    if image_path:
        # Read image from file
        image = cv2.imread(image_path)
    elif image_array is not None:
        # Use provided image array
        image = image_array
    else:
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None, None

    # Process the first face found
    x, y, w, h = faces[0]

    # Extract face ROI
    face_roi = gray[y : y + h, x : x + w]

    # Resize to match model input size (48x48)
    resized_face = cv2.resize(face_roi, (48, 48))

    # Normalize pixel values
    normalized_face = resized_face / 255.0

    # Reshape for model input
    input_face = normalized_face.reshape(1, 48, 48, 1)

    return input_face, (x, y, w, h)


def predict_emotion(face_input):
    """
    Predict emotion from preprocessed face input
    """
    if model is None:
        return "Model not loaded"

    # Get prediction
    prediction = model.predict(face_input)

    # Get label with highest probability
    emotion_idx = np.argmax(prediction)
    emotion = emotion_labels[emotion_idx]

    return emotion


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process-image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    if file and allowed_file(file.filename):
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Convert file to OpenCV format
        image_array = cv2.imdecode(
            np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        # Preprocess image
        face_input, face_coords = preprocess_image(image_array=image_array)

        # Remove temporary file
        os.remove(filepath)

        if face_input is None:
            return jsonify({"error": "No face detected in image"}), 400

        # Predict emotion
        emotion = predict_emotion(face_input)

        return jsonify(
            {
                "emotion": emotion,
                "face_coordinates": (
                    {
                        "x": int(face_coords[0]),
                        "y": int(face_coords[1]),
                        "width": int(face_coords[2]),
                        "height": int(face_coords[3]),
                    }
                    if face_coords
                    else None
                ),
            }
        )

    return jsonify({"error": "Invalid file type"}), 400


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)
