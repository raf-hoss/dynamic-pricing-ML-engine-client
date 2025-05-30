import pandas as pd
import numpy as np
import nannyml as nml
import os

# Simulate post-deployment data drift (to mimic production environment)
def simulate_post_deploy_data(df: pd.DataFrame) -> pd.DataFrame:
    post_df = df.copy()

    # Introduce slight drift in competitor prices and promo discounts
    post_df["competitor_price"] *= np.random.uniform(0.97, 1.05, size=len(post_df))
    post_df["promo_discount"] = np.clip(post_df["promo_discount"] + np.random.normal(0, 0.01, len(post_df)), 0, 0.5)

    # Simulate true units sold in production (with small noise)
    post_df["actual_units_sold"] = (
        post_df["predicted_units_sold"] * np.random.uniform(0.8, 1.2, size=len(post_df))
    ).astype(int)

    return post_df

# Run NannyML CBPE monitoring
def run_monitoring(reference_df: pd.DataFrame, analysis_df: pd.DataFrame):
    # Define columns
    features = [
        "price", "promo_discount", "competitor_price", "temperature",
        "price_margin", "price_vs_competitor", "lag_1", "rolling_mean_7", "elasticity"
    ]
    target = "actual_units_sold"
    prediction = "predicted_units_sold"

    # Initialize CBPE
    cbpe = nml.CBPE(
        problem_type='regression',
        y_pred=prediction,
        y_true=target,
        metrics=['rmse'],
        timestamp_column_name="date",
        chunk_size="30d"
    )

    # Fit on reference (training) data
    cbpe.fit(reference_df)

    # Run on post-deploy (production) data
    results = cbpe.estimate(analysis_df)
    return results

# Save NannyML output
def save_monitoring_output(results, output_path="data/processed/monitoring_results.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_pandas().to_csv(output_path, index=False)

if __name__ == "__main__":
    from src.forecasting.forecaster import train_forecast_model
    from src.features.build_features import build_features
    from src.data.load_data import load_data_pandas
    from src.data.preprocess import preprocess_pandas

    # Load and build pipeline
    _, sales_df = load_data_pandas()
    cleaned_df = preprocess_pandas(sales_df)
    featured_df = build_features(cleaned_df)
    _, predicted_df = train_forecast_model(featured_df)

    # Simulate post-deploy data
    post_df = simulate_post_deploy_data(predicted_df)

    # Monitor
    results = run_monitoring(predicted_df, post_df)
    save_monitoring_output(results)

    print("✅ Post-deployment monitoring is now complete. file was saved monitoring results to data/processed/")