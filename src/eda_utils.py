import pandas as pd
from statsmodels.tsa.stattools import adfuller

def run_adf_test(series: pd.Series, name: str = "Series"):
    """Runs the Augmented Dickey-Fuller test to check for stationarity."""
    result = adfuller(series.dropna())
    print(f"📊 --- STATISTICAL STATIONARITY (ADF) TEST FOR {name} ---")
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"p-value       : {result[1]:.4f}")
    status = "Stationary (Ready)" if result[1] <= 0.05 else "Non-Stationary (Has Trend)"
    print(f"Status        : {status}\n")
    return result[1]