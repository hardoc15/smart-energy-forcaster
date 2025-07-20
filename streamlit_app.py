import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from datetime import timedelta
from src.data_preprocessing import preprocess_data
from src.predict import predict_and_plot, forecast_next_24_hours


st.set_page_config(page_title="Smart Energy Dashboard", layout="wide")
st.title("Smart Energy Consumption Dashboard")

st.sidebar.header("Upload Your Energy CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
else:
    st.sidebar.info("Using default dataset.")
    df = pd.read_csv("data/energy_consumption.csv", parse_dates=['timestamp'])

if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Raw Data")
    st.dataframe(df.head())

st.subheader("Energy Consumption Over Time")
df['energy_kwh_smooth'] = df['energy_kwh'].rolling(window=24).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['timestamp'], df['energy_kwh_smooth'], label="Smoothed (24h avg)", color='tab:blue')
ax.set_title("Smoothed Energy Consumption")
ax.set_xlabel("Time")
ax.set_ylabel("kWh")
ax.grid(True)
ax.legend()
st.pyplot(fig)

df_processed = preprocess_data(df)

if st.sidebar.checkbox("Show Last 7 Days vs Prediction", True):
    st.subheader("📊 Model Performance (Last 7 Days)")
    predict_and_plot(df_processed)

st.subheader("Forecast: Next 24 Hours")
forecast_df = forecast_next_24_hours(df_processed)

csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Forecast as CSV", csv, "next_24hr_forecast.csv", "text/csv")
