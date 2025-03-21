from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import requests
import json
import time
import re
import tensorflow as tf
from PIL import Image
import os
import cv2
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

app = Flask(__name__)

# Model URLs
CKD_MODEL_URL = "https://drive.google.com/uc?export=download&id=1B4suh6dP70mpWaSou2hsthJaU8UH-GeB"
SCALER_URL = "https://drive.google.com/uc?export=download&id=1W085sZvdFn2-cvh96L7e94Q-k63sCb0O"
KIDNEY_STONE_MODEL_URL = "https://drive.google.com/uc?export=download&id=1cs5fbkyksCkMmcB0Xm5Qnqyroq6scY1k"
KIDNEY_TUMOR_MODEL_URL = "https://drive.google.com/uc?export=download&id=1-5Dj32fZ--a3muF_VYvFataF-r5Gj6rY"
PROCESSED_FEATURES_URL = "https://drive.google.com/uc?export=download&id=YOUR_ID"  # Add your ID here

# Download function
def download_file(url, path):
    if not os.path.exists(path):
        os.makedirs("saved_model", exist_ok=True)
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)

# CKD feature names
ckd_feature_names = ['age', 'bp', 'sg', 'al', 'bgr', 'bu', 'sc', 'hemo', 'pcv', 'wc']

# CKD Prediction function
def predict_ckd(input_data):
    download_file(CKD_MODEL_URL, 'saved_model/ckd_model.pkl')
    download_file(SCALER_URL, 'saved_model/scaler.pkl')
    with open('saved_model/ckd_model.pkl', 'rb') as f:
        ckd_model = pickle.load(f)
    with open('saved_model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    input_df = pd.DataFrame([input_data], columns=ckd_feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = ckd_model.predict(input_scaled)[0]
    probability = ckd_model.predict_proba(input_scaled)[0][prediction]
    return prediction, probability

# Kidney Stone Prediction function
def predict_kidney_stone(image_path):
    download_file(KIDNEY_STONE_MODEL_URL, 'saved_model/kidney_stone_model.h5')
    kidney_stone_model = tf.keras.models.load_model('saved_model/kidney_stone_model.h5')
    img = Image.open(image_path).convert('L')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array_rgb = np.repeat(img_array[..., np.newaxis], 3, axis=-1)
    img_array_rgb = np.expand_dims(img_array_rgb, axis=0)
    prediction = kidney_stone_model.predict(img_array_rgb)
    result = "Kidney Stone Detected" if prediction[0][0] > 0.5 else "No Kidney Stone"
    confidence = prediction[0][0] * 100 if prediction[0][0] > 0.5 else (1 - prediction[0][0]) * 100
    return result, confidence

# Kidney Tumor Prediction function
def predict_kidney_tumor(image_path):
    download_file(KIDNEY_TUMOR_MODEL_URL, 'saved_model/kidney_tumor_model.h5')
    kidney_tumor_model = tf.keras.models.load_model('saved_model/kidney_tumor_model.h5')
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image could not be loaded.")
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = kidney_tumor_model.predict(img)
    class_names = ['Normal', 'Tumor']
    result = class_names[int(prediction[0][0] > 0.5)]
    confidence = prediction[0][0] * 100 if prediction[0][0] > 0.5 else (1 - prediction[0][0]) * 100
    return result, confidence

# [Rest of your code remains unchanged: get_recommendations, get_medicine_details, generate_pdf_report, routes]

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)