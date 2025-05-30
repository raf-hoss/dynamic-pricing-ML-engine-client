from src.pricing.pricing_engine import optimize_prices
from src.data.load_data import load_data_pandas
from src.data.preprocess import preprocess_pandas
from src.features.build_features import build_features
from src.forecasting.forecaster import train_forecast_model

def test_price_optimization():
    _, sales_df = load_data_pandas()
    clean_df = preprocess_pandas(sales_df)
    featured_df = build_features(clean_df)
    _, forecast_df = train_forecast_model(featured_df)
    optimized_df = optimize_prices(forecast_df)

    assert "optimized_price" in optimized_df.columns
    assert all(optimized_df["optimized_price"] > 0)