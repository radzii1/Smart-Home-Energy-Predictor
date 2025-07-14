import streamlit as st
import pickle
import numpy as np

# Load model
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🏢 Smart Building Energy Predictor")

temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=50.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.0)
dew_point = st.number_input("Dew Point (°C)", value=15.0)

if st.button("Predict"):
    X_input = np.array([[temperature, humidity, wind_speed, dew_point]])
    prediction = model.predict(X_input)
    st.success(f"🔋 Estimated Energy Consumption: **{prediction[0]:.2f} kWh**")
