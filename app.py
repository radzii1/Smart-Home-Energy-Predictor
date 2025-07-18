import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Smart Building Energy Predictor", layout="centered")

st.title("🏢 Smart Building Energy Predictor")
st.markdown("Predict daily energy usage using temperature, humidity, wind speed, and dew point.")

# 1. Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("energydata_complete.csv")
    df = df[["T2", "RH_2", "Windspeed", "Tdewpoint", "Appliances"]]
    df.columns = ["temperature", "humidity", "wind_speed", "dew_point", "energy_usage"]
    return df

df = load_data()

# 2. Train RF Model
X = df[["temperature", "humidity", "wind_speed", "dew_point"]]
y = df["energy_usage"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. User Inputs
temperature = st.number_input("Temperature (°C)", value=22.0, key="temp_input")
humidity = st.number_input("Humidity (%)", value=40.0, key="humidity_input")
wind_speed = st.number_input("Wind Speed (m/s)", value=3.5, key="wind_input")
dew_point = st.number_input("Dew Point (°C)", value=12.0, key="dew_input")

# 4. Prediction
if st.button("Predict Energy Usage"):
    input_data = np.array([[temperature, humidity, wind_speed, dew_point]])
    prediction = model.predict(input_data)[0]
    st.success(f"🔋 Estimated Energy Consumption: **{prediction:.2f} kWh**")

# 5. Visualization Section
st.header("📈 Visual Insights from Data")

# A. Scatter Plot: Temperature vs Energy
st.subheader("Energy Usage vs. Temperature")
fig1, ax1 = plt.subplots()
ax1.scatter(df["temperature"], df["energy_usage"], alpha=0.5, color='orange')
ax1.set_xlabel("Temperature (°C)")
ax1.set_ylabel("Energy Usage (kWh)")
st.pyplot(fig1)

# B. Line Plot: Feature vs Average Energy Usage
st.subheader("Average Energy Usage by Feature (Binned)")
features = ["temperature", "humidity", "wind_speed", "dew_point"]
fig2, ax2 = plt.subplots()
for feature in features:
    binned = df[[feature, "energy_usage"]].copy()
    binned[feature] = pd.cut(binned[feature], bins=10)
    avg_usage = binned.groupby(feature).mean()
    avg_usage.plot(ax=ax2, label=feature)
ax2.set_ylabel("Avg Energy Usage (kWh)")
plt.legend()
st.pyplot(fig2)

# C. Feature Importance from RF
st.subheader("🔍 Feature Importance")
importances = model.feature_importances_
feature_names = X.columns
fig3, ax3 = plt.subplots()
ax3.barh(feature_names, importances, color='teal')
ax3.set_xlabel("Importance Score")
ax3.set_title("Random Forest Feature Importance")
st.pyplot(fig3)
