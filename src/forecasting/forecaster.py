import pandas as pd
import lightgbm as lgb
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Train model on feature-rich dataframe
def train_forecast_model(df: pd.DataFrame):
    df = df.dropna(subset=["lag_1", "rolling_mean_7", "elasticity"])  # Drop rows with NA lag features

    # Define features and target
    features = [
        "price", "promo_discount", "competitor_price", "temperature",
        "price_margin", "price_vs_competitor",
        "lag_1", "rolling_mean_7", "elasticity",
        "day_of_week", "is_weekend", "month"
    ]
    target = "units_sold"

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # LightGBM dataset
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

    # Model parameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "random_state": 22
    }

    # Train
    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_test], num_boost_round=100, early_stopping_rounds=10)

    # Predict and evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"✅ Forecast RMSE: {rmse:.2f}")

    return model, test_df.assign(predicted_units_sold=preds)

# Save predictions to CSV
def save_predictions(df: pd.DataFrame, path: str = "data/processed/predicted_demand.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

if __name__ == "__main__":
    from src.features.build_features import build_features
    from src.data.load_data import load_data_pandas
    from src.data.preprocess import preprocess_pandas

    _, sales_df = load_data_pandas()
    cleaned_df = preprocess_pandas(sales_df)
    featured_df = build_features(cleaned_df)

    model, forecast_df = train_forecast_model(featured_df)
    save_predictions(forecast_df)