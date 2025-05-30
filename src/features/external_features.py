import pandas as pd
import numpy as np

# Inject synthetic external signals: weather forecast and competitor trend
def add_external_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Simulate moving competitor pricing over time using noise
    df["competitor_trend"] = (
        df.groupby("sku_id")["competitor_price"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    )

    # Create temperature anomalies or volatility (simulate weather shocks)
    df["temp_volatility"] = (
        df.groupby("sku_id")["temperature"]
        .transform(lambda x: x.diff().abs().rolling(window=3, min_periods=1).mean())
    )

    return df

# Save version with external signals
def save_external_featured(df: pd.DataFrame, path: str = "data/processed/featured_with_external.csv"):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    from src.features.build_features import build_features
    from src.data.load_data import load_data_pandas
    from src.data.preprocess import preprocess_pandas

    # Load → clean → feature → enrich
    _, sales_df = load_data_pandas()
    cleaned_df = preprocess_pandas(sales_df)
    base_features = build_features(cleaned_df)
    enriched = add_external_signals(base_features)

    save_external_featured(enriched)
    print("✅ External features added. Saved to data/processed/featured_with_external.csv")