# 🔋 Smart Energy Forecaster

A machine learning-powered forecasting tool that predicts hourly household energy consumption using historical usage patterns and time-based features. Built with CatBoost and visualized through an interactive Streamlit dashboard.

---

## 📈 Project Overview

The Smart Energy Forecaster helps monitor and predict residential electricity demand. It uses a high-performance regression model to:

- Forecast hourly energy consumption 24 hours into the future
- Analyze trends and anomalies in past energy usage
- Assist in energy efficiency planning for homeowners or smart grid systems

---

## 🚀 Key Features

- 📊 **Forecast 24-Hour Energy Demand**: Predict future energy usage based on time, lag, and statistical window features.
- ⚡ **CatBoost Model**: Trained with high accuracy (R² = 0.9840) and low RMSE using a gradient boosting algorithm.
- 🧠 **Advanced Feature Engineering**:
  - Time-based features (hour of day, day of week, weekend)
  - Lag features (`lag_1`, `lag_24`)
  - Rolling statistics (`rolling_mean_3`, `rolling_std_3`)
- 📉 **Visualization Dashboard**: Interactive Streamlit app to:
  - View model predictions vs actuals
  - Forecast and plot future energy demand
- 💾 **Model Persistence**: Trained model is saved as `.cbm` and reused without retraining.

---

## 🛠️ Tech Stack

- **Python 3.12**
- **CatBoost** – for gradient boosting regression
- **Scikit-learn** – for evaluation metrics
- **Pandas / NumPy** – for data manipulation
- **Matplotlib / Seaborn** – for plotting
- **Streamlit** – for the dashboard interface

---

## 📁 Project Structure

