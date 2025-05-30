from src.data.load_data import load_data_pandas
from src.data.preprocess import preprocess_pandas
from src.features.build_features import build_features

def test_feature_engineering():
    _, sales_df = load_data_pandas()
    clean_df = preprocess_pandas(sales_df)
    featured_df = build_features(clean_df)

    expected_cols = ["lag_1", "rolling_mean_7", "elasticity", "day_of_week"]
    for col in expected_cols:
        assert col in featured_df.columns, f"Missing feature: {col}"