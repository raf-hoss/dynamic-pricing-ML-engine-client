import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Train a segment-level elasticity model using Ridge regression
def train_elasticity_model(df: pd.DataFrame, segment_col: str = "category") -> dict:
    df = df.dropna(subset=[
        "price", "promo_discount", "competitor_price", "temperature",
        "units_sold", "price_margin", "price_vs_competitor"
    ])

    feature_cols = [
        "price", "promo_discount", "competitor_price", "temperature",
        "price_margin", "price_vs_competitor"
    ]
    target_col = "units_sold"

    segment_models = {}

    for segment, group in df.groupby(segment_col):
        if len(group) < 30:
            continue  # skip small groups

        X = group[feature_cols]
        y = group[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        print(f"📦 Segment '{segment}' → R²: {r2:.2f}, RMSE: {rmse:.1f}")
        segment_models[segment] = model

    return segment_models

# Predict demand for a single row based on learned segment elasticity
def predict_demand_with_elasticity(row: pd.Series, model: Ridge, feature_cols: list) -> float:
    input_data = row[feature_cols].values.reshape(1, -1)
    return model.predict(input_data)[0]

if __name__ == "__main__":
    from src.data.load_data import load_data_pandas
    from src.data.preprocess import preprocess_pandas
    from src.features.build_features import build_features

    _, sales_df = load_data_pandas()
    cleaned_df = preprocess_pandas(sales_df)
    featured_df = build_features(cleaned_df)

    models = train_elasticity_model(featured_df)