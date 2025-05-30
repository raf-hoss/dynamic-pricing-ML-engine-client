import pandas as pd
import numpy as np
import os

# Add temporal features and lagged demand
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month
    return df

# Add lag and rolling features (for demand pattern detection)
def add_lag_features(df: pd.DataFrame, lags=[1, 7], rolling_windows=[7]) -> pd.DataFrame:
    df = df.sort_values(["sku_id", "date"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("sku_id")["units_sold"].shift(lag)
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = (
            df.groupby("sku_id")["units_sold"]
            .shift(1)
            .rolling(window=window)
            .mean()
            .reset_index(0, drop=True)
        )
    return df

# Add price elasticity proxy features
def add_price_elasticity_features(df: pd.DataFrame) -> pd.DataFrame:
    df["price_change"] = df.groupby("sku_id")["price"].pct_change()
    df["demand_change"] = df.groupby("sku_id")["units_sold"].pct_change()
    df["elasticity"] = df["demand_change"] / df["price_change"]
    df["elasticity"] = df["elasticity"].clip(-10, 10)  # handle outliers
    return df

# Full pipeline
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_price_elasticity_features(df)
    return df

# Save engineered dataset
def save_featured_data(df: pd.DataFrame, output_path: str = "data/processed/featured_sales_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    from src.data.preprocess import preprocess_pandas
    from src.data.load_data import load_data_pandas

    _, sales_df = load_data_pandas()
    cleaned_df = preprocess_pandas(sales_df)
    featured_df = build_features(cleaned_df)
    save_featured_data(featured_df)
    print("✅ Feature engineering done. File was saved to data/processed/featured_sales_data.csv")