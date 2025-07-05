import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client

# ------------------------- CONFIG -------------------------
# Email alert configuration
EMAIL_ADDRESS = ""
EMAIL_PASSWORD = ""  # Use App Password (Gmail)

# SMS alert configuration (Twilio)
TWILIO_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_PHONE = ""
LIFEGUARD_PHONE = ""

# ------------------------- Streamlit UI -------------------------
st.set_page_config(page_title="AI-Based Drowning Detection System", layout="wide")
st.title("🌊 AI-Based Drowning Detection System")
st.markdown("Upload an image or frame and select a model to classify behavior.")

uploaded_file = st.sidebar.file_uploader("📁 Upload Image", type=["jpg", "png"])
model_choice = st.sidebar.selectbox("🔍 Select Model", ["Logistic Regression", "SVM", "Random Forest", "KNN"])

# ------------------------- Load Models -------------------------
lr_model = joblib.load("logistic_regression_model.pkl")
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
knn_model = joblib.load("knn_model.pkl")

class_labels = ['Drowning', 'Swimming', 'Person Out of Water']

# ------------------------- Preprocessing -------------------------
def preprocess_image(img, target_size=(64, 64)):
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return img

# ------------------------- Alert Functions -------------------------
def send_email_alert():
    msg = MIMEText("⚠️ DROWNING DETECTED! Immediate attention required!")
    msg["Subject"] = "Drowning Alert 🚨"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = ""

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        st.success("Email alert sent to lifeguards!")
    except Exception as e:
        st.error(f"Email alert failed: {e}")

def send_sms_alert():
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body="⚠️ DROWNING DETECTED! Immediate attention required!",
            from_=TWILIO_PHONE,
            to=LIFEGUARD_PHONE
        )
        st.success("📱 SMS alert sent to lifeguards!")
    except Exception as e:
        st.error(f"SMS alert failed: {e}")

# ------------------------- Main Prediction Logic -------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    img_array = np.array(image)
    img_processed = preprocess_image(img_array)
    X_input = img_processed.reshape(1, -1)

    if st.button("Predict"):
        if model_choice == "Logistic Regression":
            prediction = lr_model.predict(X_input)
        elif model_choice == "SVM":
            prediction = svm_model.predict(X_input)
        elif model_choice == "Random Forest":
            prediction = rf_model.predict(X_input)
        elif model_choice == "KNN":
            prediction = knn_model.predict(X_input)

        pred_label = prediction[0]
        st.success(f"Prediction: **{pred_label}**")

        if pred_label == "Drowning":
            send_email_alert()
            send_sms_alert()
