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

# Model URLs (replace with your actual direct download links, e.g., from Google Drive)
CKD_MODEL_URL = "https://drive.google.com/uc?export=download&id=1B4suh6dP70mpWaSou2hsthJaU8UH-GeB"
SCALER_URL = "https://drive.google.com/uc?export=download&id=1W085sZvdFn2-cvh96L7e94Q-k63sCb0O"
KIDNEY_STONE_MODEL_URL = "https://drive.google.com/uc?export=download&id=1cs5fbkyksCkMmcB0Xm5Qnqyroq6scY1k"
KIDNEY_TUMOR_MODEL_URL = "https://drive.google.com/uc?export=download&id=1-5Dj32fZ--a3muF_VYvFataF-r5Gj6rY"

# Load CKD model and scaler
def load_model(url, path, is_pickle=True):
    if not os.path.exists(path):
        os.makedirs("saved_model", exist_ok=True)
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)
    return pickle.load(open(path, "rb")) if is_pickle else tf.keras.models.load_model(path)

try:
    ckd_model = load_model(CKD_MODEL_URL, 'saved_model/ckd_model.pkl', is_pickle=True)
    scaler = load_model(SCALER_URL, 'saved_model/scaler.pkl', is_pickle=True)
    kidney_stone_model = load_model(KIDNEY_STONE_MODEL_URL, 'saved_model/kidney_stone_model.h5', is_pickle=False)
    kidney_tumor_model = load_model(KIDNEY_TUMOR_MODEL_URL, 'saved_model/kidney_tumor_model.h5', is_pickle=False)
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# CKD feature names
ckd_feature_names = ['age', 'bp', 'sg', 'al', 'bgr', 'bu', 'sc', 'hemo', 'pcv', 'wc']

# CKD Prediction function
def predict_ckd(input_data):
    input_df = pd.DataFrame([input_data], columns=ckd_feature_names)
    input_df = input_df[ckd_feature_names]
    input_scaled = scaler.transform(input_df)
    prediction = ckd_model.predict(input_scaled)[0]
    probability = ckd_model.predict_proba(input_scaled)[0][prediction]
    return prediction, probability

# CKD Recommendation function
def get_recommendations(prediction):
    if prediction == 0:
        return "No CKD detected. Preventive Measures: Maintain hydration, monitor blood pressure regularly, follow a balanced diet low in salt and sugar."
    else:
        return "CKD detected. Recommendations: Consult a nephrologist immediately. Possible treatments include ACE inhibitors for blood pressure control and glucose management (under medical supervision)."

# Kidney Stone Prediction function
def predict_kidney_stone(image_path):
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

# OpenRouter API configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# OpenRouter API function for medicine details
def get_medicine_details(medicine_name, max_retries=3):
    if not API_KEY:
        return {"status": "Error", "message": "OpenRouter API key not configured."}
    prompt = (
        f"Provide a concise description of the medicine '{medicine_name}' including its common uses, "
        f"side effects, and relevance to Chronic Kidney Disease (CKD) management. Keep it brief and clear."
    )
    payload = {
        "model": "qwen/qwq-32b-preview",
        "messages": [
            {"role": "system", "content": "You are a medical assistant providing accurate drug information related to kidney health and CKD."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
            if response.status_code == 200:
                data = response.json()
                message = data["choices"][0]["message"]["content"].strip()
                return {"status": "Success", "name": medicine_name, "message": message}
            elif response.status_code == 429:
                wait_time = 2 ** attempt
                print(f"Rate limit hit (429). Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                return {"status": "Error", "message": f"API request failed with status {response.status_code}."}
        except Exception as e:
            return {"status": "Error", "message": f"Error: {str(e)}"}
    return {"status": "Error", "message": "Max retries exceeded due to rate limiting (429)."}

# Function to generate PDF report
def generate_pdf_report(prediction, probability, advice):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Chronic Kidney Disease Prediction Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Confidence:</b> {probability}%", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Recommendations:</b>", styles['Normal']))
    story.append(Paragraph(advice, styles['Normal']))
    story.append(Spacer(1, 12))
    current_date = datetime.now().strftime('%B %d, %Y')
    story.append(Paragraph(f"Generated on: {current_date}", styles['Normal']))
    story.append(Paragraph("Note: Consult a healthcare professional for proper diagnosis and treatment.", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Routes (unchanged)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def ckd_prediction():
    if request.method == 'POST':
        input_data = {}
        for feature in ckd_feature_names:
            value = request.form.get(feature, '')
            input_data[feature] = np.nan if value in ['', '?'] else value
        input_df = pd.DataFrame([input_data], columns=ckd_feature_names)
        input_df = input_df.apply(pd.to_numeric, errors='coerce')
        processed_features = pd.read_csv('saved_model/processed_features.csv')
        input_df = input_df.fillna(processed_features.median())

        prediction, probability = predict_ckd(input_df.iloc[0].tolist())
        result = "Chronic Kidney Disease Detected" if prediction == 1 else "No Chronic Kidney Disease"
        advice = get_recommendations(prediction)
        probability_percentage = round(probability * 100, 2)

        return render_template('index.html', prediction=result, probability=probability_percentage, advice=advice)
    return render_template('index.html', prediction=None)

@app.route('/download_report')
def download_report():
    prediction = request.args.get('prediction')
    probability = request.args.get('probability')
    advice = request.args.get('advice')
    
    if not prediction or "Chronic Kidney Disease Detected" not in prediction:
        return "Report download is only available for CKD detection cases.", 400

    pdf_buffer = generate_pdf_report(prediction, probability, advice)
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name="CKD_Prediction_Report.pdf",
        mimetype='application/pdf'
    )

@app.route('/kidney_stone', methods=['GET', 'POST'])
def kidney_stone_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('kidney_stone.html', message='No file uploaded', image=None)
        file = request.files['file']
        if file.filename == '':
            return render_template('kidney_stone.html', message='No file selected', image=None)
        
        upload_folder = 'static/uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        result, confidence = predict_kidney_stone(file_path)
        message = f"{result} (Confidence: {confidence:.2f}%)"
        return render_template('kidney_stone.html', message=message, image=file.filename)
    return render_template('kidney_stone.html', message=None, image=None)

@app.route('/kidney_tumor', methods=['GET', 'POST'])
def kidney_tumor_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('kidney_tumor.html', message='No file uploaded', image=None)
        file = request.files['file']
        if file.filename == '':
            return render_template('kidney_tumor.html', message='No file selected', image=None)
        
        upload_folder = 'static/uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        try:
            result, confidence = predict_kidney_tumor(file_path)
            message = f"Predicted: {result} (Confidence: {confidence:.2f}%)"
        except ValueError as e:
            message = str(e)
        return render_template('kidney_tumor.html', message=message, image=file.filename)
    return render_template('kidney_tumor.html', message=None, image=None)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/chatbot_message', methods=['POST'])
def chatbot_message():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'response': "Please ask about kidney health, CKD, kidney stones, treatments, or prevention."})

        medicine_match = re.search(r"(?:tell me about|what is|describe)\s+([a-zA-Z\s]+)", user_message, re.IGNORECASE)
        if medicine_match:
            medicine_name = medicine_match.group(1).strip()
            result = get_medicine_details(medicine_name)
            return jsonify({'response': result['message'] if result['status'] == "Success" else result['message']})

        payload = {
            "model": "qwen/qwq-32b-preview",
            "messages": [
                {"role": "system", "content": "You are a CKD Assistant, expert in kidney health. Provide concise, complete responses (within 150 tokens) on CKD stages, kidney stones (causes, symptoms, prevention), dialysis, transplants, treatments, prevention, and symptoms."},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        result = response.json()
        bot_response = result['choices'][0]['message']['content'].strip()

        if bot_response and not bot_response.endswith(('.', '!', '?')) and len(bot_response.split()) > 5:
            payload["messages"][1]["content"] = f"Briefly answer: {user_message} with causes, symptoms, or prevention (within 140 tokens)."
            response = requests.post(API_URL, json=payload, headers=HEADERS)
            response.raise_for_status()
            result = response.json()
            bot_response = result['choices'][0]['message']['content'].strip()

        return jsonify({'response': bot_response})
    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}. Ask about kidney health, CKD, or related topics!"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)