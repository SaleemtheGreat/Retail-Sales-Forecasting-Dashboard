import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "models/rf_sales_pipeline.pkl"
DATA_PATH = "data/processed/walmart_with_features.csv"

# -------------------------------
# LOAD MODEL & DATA
# -------------------------------
@st.cache_resource
def load_model():
    """Loads the trained machine learning pipeline."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure it exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_data
def load_data():
    """Loads and preprocesses the data for the dashboard."""
    try:
        df_loaded = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        
        # Ensure 'Dept' column exists
        if 'Dept' not in df_loaded.columns:
            df_loaded['Dept'] = 1
        
        # Ensure all required columns are present. If not, create a placeholder.
        required_cols = [
            "Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price",
            "CPI", "Unemployment", "Year", "Month", "Week", "Day", "Dept",
            "sales_lag_1", "sales_lag_2", "sales_lag_4", # Updated
            "rolling_mean_4w", "rolling_mean_12w", "rolling_std_4w" # Updated
        ]
        
        for col in required_cols:
            if col not in df_loaded.columns:
                st.warning(f"Missing column in data: '{col}'. Attempting to create a placeholder.")
                df_loaded[col] = 0.0

        # Ensure 'Week' column is integer type
        if 'Week' in df_loaded.columns:
            df_loaded['Week'] = df_loaded['Week'].astype(int)

        return df_loaded.sort_values(by=["Store", "Date"])
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_PATH}. Please ensure it exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


model = load_model()
df = load_data()


# -------------------------------
# APP TITLE
# -------------------------------
st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")
st.title("🛒 Retail Sales Forecasting Dashboard")
st.markdown(
    """
    Retailers need to know **how much stock to keep in advance**.
    - Understock &rarr; Lost sales
    - Overstock &rarr; Wasted money
    This dashboard forecasts **future weekly sales** for products and stores.
    """
)

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("Filters")
stores = sorted(df["Store"].unique().tolist())
depts = sorted(df["Dept"].unique().tolist())

store_selected = st.sidebar.selectbox("Select Store", stores)
dept_selected = st.sidebar.selectbox("Select Department", depts)

# Filter dataset
filtered_df = df.copy()
filtered_df = filtered_df[(filtered_df["Store"] == store_selected) & (filtered_df["Dept"] == dept_selected)]

if filtered_df.empty:
    st.warning(f"No data found for Store {store_selected} and Department {dept_selected}.")
    st.stop()

st.write(f"📊 Showing data for **Store {store_selected} | Dept {dept_selected}**")

# Prepare features (drop target) - ensure only model features are passed
model_features = [
    'Store', 'Dept', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'Year', 'Month', 'Week', 'Day', 'day_of_week', 'is_weekend',
    'sales_lag_1', 'sales_lag_2', 'sales_lag_4', # Updated
    'rolling_mean_4w', 'rolling_mean_12w', 'rolling_std_4w' # Updated
]

# Ensure filtered_df has all model_features. Fill missing with 0 or a sensible default.
for feature in model_features:
    if feature not in filtered_df.columns:
        st.warning(f"Feature '{feature}' is missing in the filtered data. Filling with 0 for prediction.")
        filtered_df[feature] = 0.0

X_historic = filtered_df[model_features]

# Predictions for historical data
try:
    preds_historic = model.predict(X_historic)
    filtered_df = filtered_df.copy()
    filtered_df["Predicted_Sales"] = preds_historic
except Exception as e:
    st.error(f"Error during historical prediction: {e}")
    st.write(f"Traceback: {e}")
    filtered_df["Predicted_Sales"] = np.nan

# -------------------------------
# FORECASTING
# -------------------------------
st.subheader("🔮 Forecast Future Sales")
st.write("Last 10 rows (Actual vs Predicted):")
st.dataframe(filtered_df[["Date", "Weekly_Sales", "Predicted_Sales"]].tail(10))

# -------------------------------
# VISUALIZATION - Historical Sales
# -------------------------------
st.subheader("📈 Historical Sales (Actual vs. Predicted)")
fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
ax_hist.plot(filtered_df["Date"], filtered_df["Weekly_Sales"], label="Actual Sales", marker="o", markersize=4)
ax_hist.plot(filtered_df["Date"], filtered_df["Predicted_Sales"], label="Predicted Sales", marker="x", markersize=4, linestyle="--")
ax_hist.set_xlabel("Date")
ax_hist.set_ylabel("Weekly Sales")
ax_hist.set_title(f"Historical Sales - Store {store_selected}, Dept {dept_selected}")
ax_hist.legend()
plt.xticks(rotation=45)
st.pyplot(fig_hist)


# -------------------------------
# FUTURE FORECASTING (True Time Series Iterative Prediction)
# -------------------------------
st.subheader("🗓️ Future Forecast (Iterative)")
n_weeks = st.slider("Weeks to forecast into the future", 1, 52, 4)

future_records = []
if not filtered_df.empty:
    df_temp = filtered_df.copy()
    historical_sales = df_temp["Weekly_Sales"].tolist()

    for i in range(n_weeks):
        last_known_record = df_temp.iloc[-1]
        next_date = last_known_record["Date"] + pd.Timedelta(weeks=1)

        future_row = last_known_record.copy()
        future_row["Date"] = next_date

        future_row["Year"] = next_date.year
        future_row["Month"] = next_date.month
        future_row["Week"] = next_date.isocalendar().week
        future_row["Day"] = next_date.day
        future_row["day_of_week"] = next_date.dayofweek
        future_row["is_weekend"] = 1 if next_date.dayofweek in [5, 6] else 0
        
        future_row["sales_lag_1"] = historical_sales[-1]
        future_row["sales_lag_2"] = historical_sales[-2] if len(historical_sales) >= 2 else 0
        future_row["sales_lag_4"] = historical_sales[-4] if len(historical_sales) >= 4 else 0 # New Feature
        
        future_row["rolling_mean_4w"] = np.mean(historical_sales[-4:]) if len(historical_sales) >= 4 else np.mean(historical_sales)
        future_row["rolling_mean_12w"] = np.mean(historical_sales[-12:]) if len(historical_sales) >= 12 else np.mean(historical_sales)
        future_row["rolling_std_4w"] = np.std(historical_sales[-4:]) if len(historical_sales) >= 4 else 0 # New Feature
        
        X_future_predict = pd.DataFrame([future_row[model_features]])
        pred_future = model.predict(X_future_predict)[0]
        future_row["Predicted_Sales"] = pred_future
        
        historical_sales.append(pred_future)
        future_records.append(future_row)
        df_temp.loc[len(df_temp)] = future_row

future_df = pd.DataFrame(future_records)

st.write("📅 Future Forecasted Sales")
if not future_df.empty:
    st.dataframe(future_df[["Date", "Predicted_Sales"]])
else:
    st.info("No future forecast generated. Please ensure historical data is available.")

# plot full forecast (history + future)
st.subheader("Overall Sales Forecast")
fig_full, ax_full = plt.subplots(figsize=(12, 6))

if not filtered_df.empty:
    ax_full.plot(filtered_df["Date"], filtered_df["Weekly_Sales"], label="Actual Sales", marker="o", markersize=4)
    ax_full.plot(filtered_df["Date"], filtered_df["Predicted_Sales"], label="Predicted (Historical)", marker="x", markersize=4, linestyle="--")

if not future_df.empty:
    ax_full.plot(future_df["Date"], future_df["Predicted_Sales"], label="Forecast (Future)", marker="s", markersize=4, linestyle=":")

ax_full.set_xlabel("Date")
ax_full.set_ylabel("Weekly Sales")
ax_full.set_title(f"Overall Sales Forecast - Store {store_selected}, Dept {dept_selected}")
ax_full.legend()
plt.xticks(rotation=45)
st.pyplot(fig_full)