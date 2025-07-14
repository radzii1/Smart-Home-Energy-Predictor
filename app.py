import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Smart Building Energy Predictor", layout="centered")

st.title("🏢 Smart Building Energy Predictor")
st.markdown("Predict daily energy usage using temperature, humidity, wind speed, and dew point.")

# 1. Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("energydata_complete.csv")
    df = df[["T2", "RH_2", "Windspeed", "Tdewpoint", "Appliances"]]  # Rename columns if needed
    df.columns = ["temperature", "humidity", "wind_speed", "dew_point", "energy_usage"]
    return df

df = load_data()

# 2. Train RF Model
X = df[["temperature", "humidity", "wind_speed", "dew_point"]]
y = df["energy_usage"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. User Inputs
temperature = st.number_input("Temperature (°C)", value=22.0)
humidity = st.number_input("Humidity (%)", value=40.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.5)
dew_point = st.number_input("Dew Point (°C)", value=12.0)

# 4. Prediction
if st.button("Predict Energy Usage"):
    input_data = np.array([[temperature, humidity, wind_speed, dew_point]])
    prediction = model.predict(input_data)[0]
    st.success(f"🔋 Estimated Energy Consumption: **{prediction:.2f} kWh**")
 value=22.0)
humidity = st.number_input("Humidity (%)", value=40.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.5)
dew_point = st.number_input("Dew Point (°C)", value=12.0)

# 4. Prediction
if st.button("Predict Energy Usage"):
    input_data = np.array([[temperature, humidity, wind_speed, dew_point]])
    prediction = model.predict(input_data)[0]
    st.success(f"🔋 Estimated Energy Consumption: **{prediction:.2f} kWh**")
