import pandas as pd
import numpy as np
import pickle
from datetime import datetime  # Added for dynamic date
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Load the trained model and scaler
try:
    with open('saved_model/ckd_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('saved_model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("Error: Model or scaler file not found in 'saved_model/' directory.")
    exit(1)

# Define feature names
feature_names = ['age', 'bp', 'sg', 'al', 'bgr', 'bu', 'sc', 'hemo', 'pcv', 'wc']

# Prediction function
def predict_ckd(input_data):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]
    return prediction, probability

# Recommendation function
def get_recommendations(prediction):
    if prediction == 0:
        return "No CKD detected. Preventive Measures: Maintain hydration, monitor blood pressure regularly, follow a balanced diet low in salt and sugar."
    else:
        return "CKD detected. Recommendations: Consult a nephrologist immediately. Possible treatments include ACE inhibitors for blood pressure control and glucose management (under medical supervision)."

# Function to generate PDF report
def generate_pdf_report(prediction, probability, advice, filename="CKD_Prediction_Report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Chronic Kidney Disease Prediction Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Prediction
    story.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Confidence
    probability_percentage = round(probability * 100, 2)
    story.append(Paragraph(f"<b>Confidence:</b> {probability_percentage}%", styles['Normal']))
    story.append(Spacer(1, 12))

    # Recommendations
    story.append(Paragraph("<b>Recommendations:</b>", styles['Normal']))
    story.append(Paragraph(advice, styles['Normal']))
    story.append(Spacer(1, 12))

    # Footer with dynamic date
    current_date = datetime.now().strftime('%B %d, %Y')  # e.g., "October 25, 2023"
    story.append(Paragraph(f"Generated on: {current_date}", styles['Normal']))
    story.append(Paragraph("Note: Consult a healthcare professional for proper diagnosis and treatment.", styles['Normal']))

    doc.build(story)
    print(f"\nPDF report saved as '{filename}'")

# Example usage (interactive input)
def collect_user_input():
    print("Enter patient data (use '?' for unknown values):")
    input_data = []
    for feature in feature_names:
        value = input(f"{feature}: ")
        input_data.append(value if value != '?' else np.nan)

    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_df = input_df.apply(pd.to_numeric, errors='coerce')
    input_df = input_df.fillna(pd.read_csv('saved_model/processed_features.csv').median())

    return input_df.iloc[0].tolist()

if __name__ == "__main__":
    user_input = collect_user_input()
    prediction, probability = predict_ckd(user_input)
    result = "Chronic Kidney Disease Detected" if prediction == 1 else "No Chronic Kidney Disease"
    advice = get_recommendations(prediction)

    print("\nPrediction Result:")
    print(f"{result} (Confidence: {probability:.2%})")
    print("\nRecommendations:")
    print(advice)

    # Generate and save PDF report if CKD is detected
    if "Chronic Kidney Disease Detected" in result:
        generate_pdf_report(result, probability, advice)
    else:
        print("\nNo PDF report generated as CKD was not detected.")