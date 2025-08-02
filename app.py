from flask import Flask, request, render_template
import json
from datetime import datetime
import cv2
import numpy as np
import os
import uuid
import google.generativeai as genai

# Initialize Gemini with your API key
genai.configure(api_key="AIzaSyB5mVSzSyXXoe4338nQ48mLOpOQG1UCSsI")
def generate_malayalam_prediction(features):
    model = genai.GenerativeModel(model_name='gemini-2.5-flash')

    prompt = f"""You are a humorous yet culturally grounded palm reader who crafts predictions in Malayalam based on palm features. Your tone is lightly sarcastic, playful, and sprinkled with intelligent wit — never offensive, always entertaining. Each prediction should:

- Use clever exaggeration without being crude
- Be short (1–3 paragraphs), crisp, and neatly worded
- Feel relatable and mischievously logical
-his should sound like a horoscope gone wrong — filled with absurd logic, impossible twists, and predictions that are so ridiculous they're almost believable.
-keep sarcasm and logic at 60:40 ratio
- Incorporate and cover all themes traditional themes: love, luck, health, or career one by one
- Respect the user while still sounding mock-serious and delightfully absurd
-please remove all the scietific terms like :contour related to palmstry just give the prediction
-Always print just for fun in a sarcastic way at the end


Malayalam Prediction:


:
{features}

Malayalam prediction:"""

    response = model.generate_content(prompt)
    return response.text.strip()

app = Flask(__name__)
HISTORY_FILE = "prediction_history.json"

def save_prediction_history(image_path, prediction):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "image_filename": image_path,
        "prediction": prediction
    }

    # Load existing history or start fresh
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

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
    result = analyze_palm(filepath)
    save_prediction_history(result["image_path"], result["prediction"])
    return render_template(
                "result.html",
                result_path=result["image_path"],
                prediction=result["prediction"]
            )

def analyze_palm(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Image not loaded", 400

    # Step 1: Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: Edge detection
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # Step 3: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Dim the original image slightly
    dimmed = cv2.addWeighted(img, 0.5, np.zeros_like(img), 0.5, 0)  # 50% brightness

    # Step 5: Draw contours in yellow
    cv2.drawContours(dimmed, contours, -1, (0, 255, 255), 1)  # Yellow lines

    # Step 6: Save and return result path
    result_filename = os.path.basename(image_path)
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, dimmed)
    feature_summary = f"""
    Number of contour clusters: {len(contours)}
    Image shape: {img.shape}
    Gray tone variance: {np.var(gray):.2f}
    Presence of strong edges: {'Yes' if np.mean(edges) > 50 else 'No'}
    """
    prediction = generate_malayalam_prediction(feature_summary)

    return {
        "image_path": f"results/{result_filename}",
        "prediction": prediction
    }
@app.route('/history')
def view_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    return render_template("history.html", entries=history)
@app.route('/about')
def ab():
    return render_template('About.html')






if __name__ == "__main__":
    app.run(debug=True)



