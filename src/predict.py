import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def predict_and_plot(df, model_path='models/catboost_model.cbm'):
    # Load trained model
    model = CatBoostRegressor()
    model.load_model(model_path)

    # Split into features and target
    X = df.drop(columns=['timestamp', 'energy_kwh'])
    y = df['energy_kwh']

    # Predict
    y_pred = model.predict(X)

    # Metrics
    # Metrics
    mse = mean_squared_error(y, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y, y_pred)

    st.subheader("Model Performance on Entire Dataset")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"R² Score: {r2:.4f}")

    # Add predictions to DataFrame
    df['predicted_kwh'] = y_pred

    # Plot last 7 days of actual vs. predicted with smoothing
    df_recent = df.copy()
    df_recent['energy_kwh_smooth'] = df_recent['energy_kwh'].rolling(window=3, min_periods=1).mean()
    df_recent['predicted_kwh_smooth'] = df_recent['predicted_kwh'].rolling(window=3, min_periods=1).mean()
    last_7_days = df_recent[df_recent['timestamp'] >= df_recent['timestamp'].max() - pd.Timedelta(days=7)]

    plt.figure(figsize=(14, 6))
    plt.plot(last_7_days['timestamp'], last_7_days['energy_kwh_smooth'], label='Actual (7-day avg)', linewidth=2)
    plt.plot(last_7_days['timestamp'], last_7_days['predicted_kwh_smooth'], label='Predicted (7-day avg)', linestyle='--', linewidth=2)
    plt.xlabel('Timestamp')
    plt.ylabel('Energy (kWh)')
    plt.title('Energy Consumption: Actual vs. Predicted (Last 7 Days)')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())


def forecast_next_24_hours(df: pd.DataFrame, model_path="models/catboost_model.cbm"):
    model = CatBoostRegressor()
    model.load_model(model_path)

    df = df.copy()
    last_known = df.iloc[-1].copy()
    future_predictions = []

    for i in range(24):
        new_time = last_known['timestamp'] + timedelta(hours=1)
        hour = new_time.hour
        day_of_week = new_time.dayofweek
        is_weekend = int(day_of_week in [5, 6])
        lag_1 = last_known['energy_kwh']
        lag_24 = lag_1
        recent_preds = [lag_1] + [p for p in reversed(future_predictions[-2:])]
        rolling_mean_3 = sum(recent_preds) / len(recent_preds)
        rolling_std_3 = 0

        input_features = pd.DataFrame([{
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'lag_1': lag_1,
            'lag_24': lag_24,
            'rolling_mean_3': rolling_mean_3,
            'rolling_std_3': rolling_std_3
        }])

        pred = model.predict(input_features)[0]
        future_predictions.append(pred)

        last_known['timestamp'] = new_time
        last_known['energy_kwh'] = pred

    # Create forecast DataFrame
    future_timestamps = [df['timestamp'].iloc[-1] + timedelta(hours=i+1) for i in range(24)]
    forecast_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'predicted_energy_kwh': future_predictions
    })

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_df['timestamp'], forecast_df['predicted_energy_kwh'], marker='o', label='Forecast')
    ax.set_title("Next 24-Hour Energy Consumption Forecast")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Predicted Energy (kWh)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    return forecast_df
