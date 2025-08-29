import pandas as pd
import os
import sys

# Paths
PROCESSED_DATA_PATH = os.path.join("data", "processed", "walmart_clean.csv")
FEATURES_DATA_PATH = os.path.join("data", "processed", "walmart_features.csv")

def load_data(path=PROCESSED_DATA_PATH):
    """Load cleaned dataset"""
    return pd.read_csv(path, parse_dates=["Date"])

def get_sales_column(df):
    """Detect sales column name"""
    possible_names = ["Sales", "sales", "Weekly_Sales", "weekly_sales"]
    for name in possible_names:
        if name in df.columns:
            return name
    print("No valid sales column found. Available columns:", df.columns.tolist())
    sys.exit(1)

def create_features(df, lags=[1, 7], windows=[7, 30]):
    """Create lag, rolling, and date features"""
    df = df.sort_values("Date")

    sales_col = get_sales_column(df)

    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df[sales_col].shift(lag)

    # Rolling averages
    for window in windows:
        df[f"rolling_mean_{window}"] = df[sales_col].shift(1).rolling(window=window).mean()

    # Date features
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["dayofweek"] = df["Date"].dt.dayofweek

    return df

def save_data(df, path=FEATURES_DATA_PATH):
    """Save dataset with features"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Feature engineered data saved at: {path}")

if __name__ == "__main__":
    print("Loading processed data...")
    df = load_data()

    print("Creating features...")
    df_features = create_features(df)

    print("Saving feature engineered data...")
    save_data(df_features)
