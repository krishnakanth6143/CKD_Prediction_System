from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import requests
import json
import re
import tensorflow as tf
from PIL import Image
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

app = Flask(__name__)

# Load models at startup from saved_model/ directory
try:
    with open('saved_model/ckd_model.pkl', 'rb') as f:
        ckd_model = pickle.load(f)
    with open('saved_model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    kidney_stone_model = tf.keras.models.load_model('saved_model/kidney_stone_model.h5')
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# CKD feature names
ckd_feature_names = ['age', 'bp', 'sg', 'al', 'bgr', 'bu', 'sc', 'hemo', 'pcv', 'wc']

# CKD Prediction function
def predict_ckd(input_data):
    input_df = pd.DataFrame([input_data], columns=ckd_feature_names)
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
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
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
    story = [
        Paragraph("Chronic Kidney Disease Prediction Report", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"<b>Prediction:</b> {prediction}", styles['Normal']),
        Spacer(1, 12),
        Paragraph(f"<b>Confidence:</b> {probability}%", styles['Normal']),
        Spacer(1, 12),
        Paragraph("<b>Recommendations:</b>", styles['Normal']),
        Paragraph(advice, styles['Normal']),
        Spacer(1, 12),
        Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']),
        Paragraph("Note: Consult a healthcare professional for proper diagnosis and treatment.", styles['Normal'])
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def ckd_prediction():
    if request.method == 'POST':
        input_data = {feature: np.nan if request.form.get(feature, '') in ['', '?'] else request.form.get(feature, '') 
                      for feature in ckd_feature_names}
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
        if 'file' not in request.files or not request.files['file'].filename:
            return render_template('kidney_stone.html', message='No file uploaded or selected', image=None)
        
        file = request.files['file']
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        result, confidence = predict_kidney_stone(file_path)
        message = f"{result} (Confidence: {confidence:.2f}%)"
        return render_template('kidney_stone.html', message=message, image=file.filename)
    return render_template('kidney_stone.html', message=None, image=None)

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
            return jsonify({'response': result['message']})

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
            bot_response = response.json()['choices'][0]['message']['content'].strip()

        return jsonify({'response': bot_response})
    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}. Ask about kidney health, CKD, or related topics!"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)