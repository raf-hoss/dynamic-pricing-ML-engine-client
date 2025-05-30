import pandas as pd
from src.pricing.pricing_engine import optimize_prices, save_optimized_prices
from src.forecasting.forecaster import train_forecast_model
from src.features.build_features import build_features
from src.data.load_data import load_data_pandas
from src.data.preprocess import preprocess_pandas
import os

# End-to-end pricing pipeline
def run_pricing_pipeline():
    # Step 1: Load raw sales data
    _, sales_df = load_data_pandas()

    # Step 2: Preprocess
    cleaned_df = preprocess_pandas(sales_df)

    # Step 3: Feature engineering
    featured_df = build_features(cleaned_df)

    # Step 4: Forecast demand
    _, forecast_df = train_forecast_model(featured_df)

    # Step 5: Optimize prices
    optimized_df = optimize_prices(forecast_df)

    # Step 6: Save
    save_optimized_prices(optimized_df)
    print("✅ Pricing pipeline complete. Output saved to data/processed/optimized_prices.csv")

if __name__ == "__main__":
    run_pricing_pipeline()