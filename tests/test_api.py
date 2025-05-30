from fastapi.testclient import TestClient
from src.api.fastapi_server import app

client = TestClient(app)

def test_predict_price_endpoint():
    sample_input = {
        "sku_id": "WM001",
        "price": 100.0,
        "cost": 60.0,
        "promo_discount": 0.1,
        "competitor_price": 105.0,
        "temperature": 28.0,
        "lag_1": 7,
        "rolling_mean_7": 6.5,
        "elasticity": -1.2,
        "day_of_week": 2,
        "is_weekend": 0,
        "month": 5
    }

    response = client.post("/predict-price/", json=sample_input)
    assert response.status_code == 200
    data = response.json()

    assert "sku_id" in data
    assert "predicted_demand" in data
    assert "optimized_price" in data
    assert data["optimized_price"] > 0