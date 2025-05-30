from src.data.load_data import load_data_pandas
from src.data.preprocess import preprocess_pandas
from src.features.build_features import build_features
from src.forecasting.forecaster import train_forecast_model

def test_forecasting_pipeline_runs():
    _, sales_df = load_data_pandas()
    clean_df = preprocess_pandas(sales_df)
    featured_df = build_features(clean_df)
    _, forecast_df = train_forecast_model(featured_df)

    assert "predicted_units_sold" in forecast_df.columns
    assert forecast_df["predicted_units_sold"].notnull().all(), "Missing predictions"