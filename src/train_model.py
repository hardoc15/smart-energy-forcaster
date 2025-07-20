import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_model(df: pd.DataFrame, save_path="models/catboost_model.cbm"):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Prepare data
    X = df.drop(columns=['timestamp', 'energy_kwh'])
    y = df['energy_kwh']

    # Split chronologically
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Debug
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("First 5 y_train values:\n", y_train.head())

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data is empty. Check preprocessing.")

    # Train
    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        random_seed=42,
        verbose=0
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R² Score: {r2:.4f}")

    # Save
    model.save_model(save_path)
    print(f"✅ Model saved to {save_path}")
