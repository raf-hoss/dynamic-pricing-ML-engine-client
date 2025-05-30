import pandas as pd
import numpy as np
import os

# Rule-based pricing optimization logic
def optimize_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strategy:
    # - If demand is high → increase price (but cap at 10% above competitor)
    # - If demand is low → decrease price (but floor at 10% above cost)
    # - Else → keep price stable

    df["adjustment"] = 0.0
    high_demand = df["predicted_units_sold"] > df["rolling_mean_7"] * 1.1
    low_demand = df["predicted_units_sold"] < df["rolling_mean_7"] * 0.9

    df.loc[high_demand, "adjustment"] = 0.05  # +5% price bump
    df.loc[low_demand, "adjustment"] = -0.05  # -5% price cut

    # Apply new price
    df["optimized_price"] = df["price"] * (1 + df["adjustment"])

    # Boundaries: not lower than cost * 1.1, not more than competitor * 1.1
    df["optimized_price"] = df[["optimized_price", "cost"]].apply(
        lambda x: max(x["optimized_price"], x["cost"] * 1.1), axis=1
    )
    df["optimized_price"] = df[["optimized_price", "competitor_price"]].apply(
        lambda x: min(x["optimized_price"], x["competitor_price"] * 1.1), axis=1
    )

    return df

# Save optimized pricing outputs
def save_optimized_prices(df: pd.DataFrame, path: str = "data/processed/optimized_prices.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

if __name__ == "__main__":
    from src.forecasting.forecaster import train_forecast_model
    from src.features.build_features import build_features
    from src.data.load_data import load_data_pandas
    from src.data.preprocess import preprocess_pandas

    # Load & prepare data
    _, sales_df = load_data_pandas()
    cleaned_df = preprocess_pandas(sales_df)
    featured_df = build_features(cleaned_df)
    _, predicted_df = train_forecast_model(featured_df)

    # Run pricing engine
    optimized_df = optimize_prices(predicted_df)
    save_optimized_prices(optimized_df)
    print("✅ Pricing optimization complete. Saved to data/processed/optimized_prices.csv")