import streamlit as st
import joblib
import numpy as np
import os
import gdown

# Google Drive file ID
model_file_id = "1RHoTunAI4AMWcd2FImjNEAilnygBni26"
model_url = f"https://drive.google.com/uc?id={model_file_id}"

# Download model if not present
if not os.path.exists("rf_model.pkl"):
    gdown.download(model_url, "rf_model.pkl", quiet=False)

# Load the model
model = joblib.load("rf_model.pkl")

st.title("🏢 Smart Building Energy Predictor")
st.markdown("Predict daily energy usage using key variables.")

# Inputs
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=50.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.0)
dew_point = st.number_input("Dew Point (°C)", value=15.0)

# Predict
if st.button("Predict"):
    X_input = np.array([[temperature, humidity, wind_speed, dew_point]])
    prediction = model.predict(X_input)
    st.success(f"🔋 Estimated Energy Consumption: **{prediction[0]:.2f} kWh**")
