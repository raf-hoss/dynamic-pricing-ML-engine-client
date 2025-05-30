import pandas as pd
from src.data.load_data import load_data_pandas
from src.data.preprocess import preprocess_pandas
from src.features.build_features import build_features
from src.forecasting.forecaster import train_forecast_model
import os

# Orchestrate end-to-end forecasting pipeline
def run_forecasting_pipeline():
    # Step 1: Load raw data
    _, sales_df = load_data_pandas()

    # Step 2: Clean + validate
    cleaned_df = preprocess_pandas(sales_df)

    # Step 3: Engineer features
    featured_df = build_features(cleaned_df)

    # Step 4: Train model and forecast
    model, forecast_df = train_forecast_model(featured_df)

    # Step 5: Save predictions
    output_path = "data/processed/predicted_demand.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecast_df.to_csv(output_path, index=False)

    print("✅ Forecasting pipeline complete. Output saved to:", output_path)

if __name__ == "__main__":
    run_forecasting_pipeline()