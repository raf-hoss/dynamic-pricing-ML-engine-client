import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import numpy as np
import mlflow
import mlflow.lightgbm

# General-purpose training function using LightGBM + MLflow tracking
def train_lightgbm_regressor(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    params: dict = None,
    test_size: float = 0.2,
    random_state: int = 22,
    run_name: str = "lightgbm_forecast"
):
    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "random_state": random_state
        }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

    # Set tracking directory
    mlflow.set_tracking_uri("experiments/tracking_with_mlflow")
    mlflow.set_experiment("demand_forecasting")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            num_boost_round=100,
            early_stopping_rounds=10,
            verbose_eval=False
        )

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_metrics({"rmse": rmse, "r2": r2})
        mlflow.lightgbm.log_model(model, artifact_path="model")

        print(f"✅ Model trained. RMSE: {rmse:.2f}, R²: {r2:.2f}")

        return model, X_test, y_test, preds

# save to disk as well but this is optional
def save_model(model, path="models/lightgbm_model.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)

# For standalone execution
if __name__ == "__main__":
    from src.data.load_data import load_data_pandas
    from src.data.preprocess import preprocess_pandas
    from src.features.build_features import build_features

    _, sales_df = load_data_pandas()
    cleaned_df = preprocess_pandas(sales_df)
    featured_df = build_features(cleaned_df)

    features = [
        "price", "promo_discount", "competitor_price", "temperature",
        "price_margin", "price_vs_competitor",
        "lag_1", "rolling_mean_7", "elasticity",
        "day_of_week", "is_weekend", "month"
    ]

    model, X_test, y_test, preds = train_lightgbm_regressor(featured_df, features, "units_sold")
    save_model(model)