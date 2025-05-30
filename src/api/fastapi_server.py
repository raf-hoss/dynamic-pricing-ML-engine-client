from fastapi import FastAPI
from pydantic import BaseModel
import lightgbm as lgb
import numpy as np
import pandas as pd
import uvicorn
import os

# Load trained model
MODEL_PATH = "models/lightgbm_model.txt"
model = lgb.Booster(model_file=MODEL_PATH)

# Initialize API
app = FastAPI(title="Dynamic Pricing API", version="1.0")

# Input schema
class PricingRequest(BaseModel):
    sku_id: str
    price: float
    cost: float
    promo_discount: float
    competitor_price: float
    temperature: float
    lag_1: float
    rolling_mean_7: float
    elasticity: float
    day_of_week: int
    is_weekend: int
    month: int

# Helper: Apply pricing logic
def apply_pricing_rules(predicted_demand, price, cost, competitor_price, rolling_mean_7):
    adjustment = 0.0
    if predicted_demand > rolling_mean_7 * 1.1:
        adjustment = 0.05
    elif predicted_demand < rolling_mean_7 * 0.9:
        adjustment = -0.05

    new_price = price * (1 + adjustment)
    new_price = max(new_price, cost * 1.1)
    new_price = min(new_price, competitor_price * 1.1)
    return round(new_price, 2)

# API endpoint
@app.post("/predict-price/")
def predict_price(request: PricingRequest):
    input_data = pd.DataFrame([request.dict()])

    feature_cols = [
        "price", "promo_discount", "competitor_price", "temperature",
        "price_margin", "price_vs_competitor",
        "lag_1", "rolling_mean_7", "elasticity",
        "day_of_week", "is_weekend", "month"
    ]

    # Derived features
    input_data["price_margin"] = input_data["price"] - input_data["cost"]
    input_data["price_vs_competitor"] = input_data["price"] / input_data["competitor_price"]

    # Predict demand
    pred = model.predict(input_data[feature_cols])[0]

    # Optimize price
    optimized_price = apply_pricing_rules(
        predicted_demand=pred,
        price=request.price,
        cost=request.cost,
        competitor_price=request.competitor_price,
        rolling_mean_7=request.rolling_mean_7
    )

    return {
        "sku_id": request.sku_id,
        "predicted_demand": round(pred, 2),
        "optimized_price": optimized_price
    }

# Run the server (for local testing)
if __name__ == "__main__":
    uvicorn.run("src.api.fastapi_server:app", host="0.0.0.0", port=8000, reload=True)