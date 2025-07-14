import streamlit as st
import joblib
import numpy as np
import gdown
import os

# Download model if not already present
model_url = "https://colab.research.google.com/drive/1AZ8UkkEXTKrNjAZvKluXkof56R3g5KRv?usp=sharing"
scaler_url = "https://colab.research.google.com/drive/1AZ8UkkEXTKrNjAZvKluXkof56R3g5KRv?usp=sharing"

if not os.path.exists("rf_model.pkl"):
    gdown.download(model_url, "rf_model.pkl", quiet=False)

if not os.path.exists("scaler.pkl"):
    gdown.download(scaler_url, "scaler.pkl", quiet=False)

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")  # Remove if not using

st.title("🏢 Smart Building Energy Predictor")
st.markdown("Predict daily energy usage using key variables.")

# Input features
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=50.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.0)
dew_point = st.number_input("Dew Point (°C)", value=15.0)

# Predict
if st.button("Predict"):
    X_input = np.array([[temperature, humidity, wind_speed, dew_point]])
    X_scaled = scaler.transform(X_input)  # Skip if not used
    prediction = model.predict(X_scaled)
    st.success(f"🔋 Estimated Energy Consumption: **{prediction[0]:.2f} kWh**")
