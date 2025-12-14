import os
import sys
import traceback
import tensorflow as tf
import numpy as np
from keras import preprocessing
from PIL import Image
import cv2
from keras import models
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = None
MODEL_PATH = 'BrainTumor10EpochsCategorical.h5'
INPUT_SIZE = 160

# Try loading the model with clear error output
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print('Model loaded. Check http://127.0.0.1:5000/')
except Exception as e:
    print('ERROR loading model:', file=sys.stderr)
    traceback.print_exc()
    print(f"Model not loaded. Ensure {MODEL_PATH} exists and is compatible.", file=sys.stderr)

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor Detected"
    elif classNo == 1:
        return "Brain Tumor Detected"
    return "Unknown"

def getResult(img_path):
    if model is None:
        raise RuntimeError("Model not loaded")
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Failed to read image at " + img_path)
    # Convert BGR (OpenCV) -> RGB (training pipeline) and resize to model input size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = np.array(image).astype('float32')
    input_img = np.expand_dims(image, axis=0)
    input_img = input_img / 255.0
    predictions = model.predict(input_img)
    print("Predictions:", predictions)
    result = np.argmax(predictions, axis=1)
    return int(result[0])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    })

@app.route('/', methods=['GET'])
def index():
    # If you have an index.html in templates/, Flask will serve it.
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    f = request.files['file']
    if f.filename == '':
        return "No selected file", 400

    basepath = os.path.dirname(__file__)
    upload_dir = os.path.join(basepath, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, secure_filename(f.filename))
    f.save(file_path)

    try:
        value = getResult(file_path)
        result = get_className(value)
        return result
    except Exception as e:
        traceback.print_exc()
        return str(e), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


#fixed using flask and render
