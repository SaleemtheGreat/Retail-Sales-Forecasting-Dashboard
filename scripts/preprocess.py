import pandas as pd
import os
import sys

# Paths
RAW_DATA_PATH = os.path.join("data", "raw", "walmart.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "walmart_clean.csv")

def load_data(path=RAW_DATA_PATH):
    """Load raw sales dataset with safety checks"""
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    if os.path.getsize(path) == 0:
        print(f"File is empty: {path}")
        sys.exit(1)

    try:
        df = pd.read_csv(path)
        print("Raw data loaded successfully.")
        print(df.head())  # preview first rows
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

def clean_data(df):
    """Basic preprocessing steps"""
    # Convert Date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Handle missing values
    if 'Sales' in df.columns:
        df['Sales'] = df['Sales'].fillna(0)

    df = df.ffill().bfill()
    return df

def save_data(df, path=PROCESSED_DATA_PATH):
    """Save cleaned data"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned data saved at: {path}")

if __name__ == "__main__":
    print("Loading raw data...")
    df = load_data()

    print("Cleaning data...")
    df_clean = clean_data(df)

    print("Saving processed data...")
    save_data(df_clean)
