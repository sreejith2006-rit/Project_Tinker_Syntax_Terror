from flask import Flask, request, redirect, render_template
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_palm():
    if 'palmImage' not in request.files:
        return "No image part", 400

    file = request.files['palmImage']
    if file.filename == '':
        return "No image selected", 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    analysis_result = analyze_palm(filepath)
    return render_template("result.html", result_path=analysis_result)

# 🖐️ Helper function: palm line analysis
def analyze_palm(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    result_filename = os.path.basename(image_path)
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, edges)

    return f"results/{result_filename}"
app.run(debug=True)

