from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] == True

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "E10 Static Causal CatBoost Ensemble"
    assert data["test_f1"] == "62.25%"

def test_valid_prediction():
    payload = {
        "TransactionAmt": 250.00,
        "ProductCD": "W",
        "card1": 10409,
        "P_emaildomain": "gmail.com"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "prediction" in data
    assert "risk_level" in data
    assert "signals" in data
    assert data["threshold"] == 0.404

def test_invalid_input():
    payload = {
        "TransactionAmt": "INVALID_AMOUNT",
        "card1": 10409
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Pydantic validation error
