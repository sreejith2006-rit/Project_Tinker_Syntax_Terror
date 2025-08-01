from flask import Flask, request, render_template
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)
# Configurable Canny thresholds
CANNY_LOW = 20
CANNY_HIGH = 100

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

    # Generate unique filename
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    # Analyze and get result path
    result_path = analyze_palm(filepath)
    return render_template("result.html", result_path=result_path)

def analyze_palm(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Image not loaded", 400

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original, contours, -1, (0, 0, 255), 1)

    result_filename = os.path.basename(image_path)
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, original)

    return f"results/{result_filename}"




if __name__ == "__main__":
    app.run(debug=True)



