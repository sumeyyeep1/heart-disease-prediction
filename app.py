import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Model ve normalizasyon değerleri
model = joblib.load("best_model.pkl")
x_train_mean = np.load("x_train_mean.npy")
x_train_std = np.load("x_train_std.npy")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("Heart Disease Risk Prediction")
st.write("Enter patient information:")

# --- Inputs ---
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0, 1])
restecg = st.number_input("Resting ECG (0-2)", 0, 2, 0)
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.number_input("Slope (0-2)", 0, 2, 1)
ca = st.number_input("Number of Major Vessels (0-3)", 0, 3, 0)
thal = st.number_input("Thal (0-3)", 0, 3, 2)

# --- Predict ---
if st.button("Predict"):
    data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])

    # Normalize
    data_scaled = (data.values - x_train_mean) / x_train_std

    prediction = model.predict(data_scaled)[0]

    low_risk_proba = None
    high_risk_proba = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(data_scaled)[0]
        high_risk_proba = probs[0]
        low_risk_proba = probs[1]

    # Bu dataset için:
    # prediction == 1  -> Low Risk
    # prediction == 0  -> High Risk
    if prediction == 1:
        st.success("Low Risk of Heart Disease")
    else:
        st.error("High Risk of Heart Disease")

    if low_risk_proba is not None and high_risk_proba is not None:
        st.write(f"Low Risk Probability: {low_risk_proba * 100:.2f}%")
        st.write(f"High Risk Probability: {high_risk_proba * 100:.2f}%")