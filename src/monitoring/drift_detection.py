import pandas as pd
import nannyml as nml
import os

# Run univariate feature drift detection using NannyML
def detect_feature_drift(reference_df: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "price", "promo_discount", "competitor_price", "temperature",
        "price_margin", "price_vs_competitor", "lag_1", "rolling_mean_7", "elasticity"
    ]

    # Configure drift calculator
    calculator = nml.UnivariateDriftCalculator(
        column_names=features,
        timestamp_column_name="date",
        chunk_size="30d"
    )

    # Fit on reference data
    calculator.fit(reference_df)

    # Calculate drift
    results = calculator.calculate(analysis_df)

    return results

# Save drift results to CSV
def save_drift_output(results, output_path="data/processed/feature_drift_results.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_pandas().to_csv(output_path, index=False)

if __name__ == "__main__":
    from src.forecasting.forecaster import train_forecast_model
    from src.features.build_features import build_features
    from src.data.load_data import load_data_pandas
    from src.data.preprocess import preprocess_pandas
    from src.monitoring.post_deploy_monitor import simulate_post_deploy_data

    # Load and prepare data
    _, sales_df = load_data_pandas()
    cleaned_df = preprocess_pandas(sales_df)
    featured_df = build_features(cleaned_df)
    _, predicted_df = train_forecast_model(featured_df)
    post_df = simulate_post_deploy_data(predicted_df)

    # Detect drift
    drift_results = detect_feature_drift(predicted_df, post_df)
    save_drift_output(drift_results)

    print("✅ Drift detection complete. Results saved to data/processed/feature_drift_results.csv")