import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Make sure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by time just in case
    df = df.sort_values('timestamp')

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Lag features
    df['lag_1'] = df['energy_kwh'].shift(1)
    df['lag_24'] = df['energy_kwh'].shift(24)

    # Rolling window features
    df['rolling_mean_3'] = df['energy_kwh'].rolling(window=3).mean()
    df['rolling_std_3'] = df['energy_kwh'].rolling(window=3).std()

    # Drop rows with any NaN values from lag/rolling
    df = df.dropna()

    return df
