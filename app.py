import streamlit as st
import joblib
import numpy as np

# Load your model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")  # Remove if you didn’t scale

st.title("🏢 Smart Building Energy Predictor")
st.markdown("Predict daily energy usage using key variables.")

# Example input fields (update as per your features)
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=50.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.0)
dew_point = st.number_input("Dew Point (°C)", value=15.0)

# Prediction
if st.button("Predict"):
    X_input = np.array([[temperature, humidity, wind_speed, dew_point]])
    X_scaled = scaler.transform(X_input)  # Remove if not using scaler
    prediction = model.predict(X_scaled)
    st.success(f"🔋 Estimated Energy Consumption: **{prediction[0]:.2f} kWh**")
