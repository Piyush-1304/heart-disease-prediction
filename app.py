import streamlit as st
import numpy as np
import joblib

# Load trained pipeline model
model = joblib.load("heart_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details to predict heart disease risk.")

# Input fields (must match training order)
age = st.number_input("Age", 20, 100, 40)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, thalach, exang]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Heart Disease Detected (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ No Heart Disease Detected (Risk: {probability*100:.2f}%)")

st.caption("⚠️ For educational purposes only. Not a medical diagnosis.")
