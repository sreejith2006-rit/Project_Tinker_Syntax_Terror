from flask import Flask, request, redirect, render_template
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    return render_template("result.html", result=analysis_result)
