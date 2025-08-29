# Retail Sales Forecasting Dashboard

An interactive dashboard for forecasting future weekly sales using a machine learning model built with **Python**, **Scikit-learn**, and **Streamlit**.  

This project demonstrates how a trained model can be used to predict future sales, helping businesses with inventory and supply chain management.

---

## Project Description

Retailers lose money every year due to understocking (lost sales) or overstocking (wasted inventory).  
This project aims to forecast weekly sales to help optimize inventory levels.

**Key Features:**
- Forecast sales using a **Random Forest Regressor** (or other ML models).  
- Interactive **Streamlit app** for filtering by store and department.  
- **Historical analysis** with visualizations showing actual vs. predicted sales.  
- **Iterative forecasting** to predict multi-step future sales trends.  
- Handles various features including time-based data, lags, and rolling averages.

---

## Dataset

**Source:** Public dataset of historical weekly sales for 45 Walmart stores over a period of time.

**Features:**
- `Store`: Store ID.  
- `Date`: Week of the sale.  
- `Weekly_Sales`: Total sales for that week (**target variable**).  
- Other features: `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, and engineered features like `sales_lag_1`, `rolling_mean_4w`, etc.

---

## Model Accuracy

The project uses a **Random Forest Regressor** trained on the Walmart sales dataset.

**Evaluation Metrics (Regression):**
- **Mean Absolute Error (MAE):** Average magnitude of the prediction errors.  
- **Root Mean Squared Error (RMSE):** Square root of the average squared errors.  
- **R-squared (R²):** Proportion of variance explained by the model.

---

## Installation & Setup

Clone the repository to get started:

```bash
git clone https://github.com/SaleemtheGreat/Retail-Sales-Forecasting-Dashboard.git
cd Retail-Sales-Forecasting-Dashboard
