import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Loads, sorts chronologically, and cleans financial data."""
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
    df = df.dropna()
    return df

def split_and_scale(df: pd.DataFrame, target_col: str = 'Close', split_date: str = '2024-12-31'):
    """Splits data sequentially to avoid leakage and scales target variance."""
    train_df = df.loc[:split_date]
    test_df = df.loc[split_date:]
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[[target_col]])
    test_scaled = scaler.transform(test_df[[target_col]])
    
    return train_scaled, test_scaled, scaler