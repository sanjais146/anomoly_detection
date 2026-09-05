import os
import sys
import time
import json
import torch
import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app
from app.amazon_predictor import get_amazon_predictor
from src.models.amazon_contrastive_tgat import HybridAmazonModel

client = TestClient(app)

def test_tgat_checkpoint_exists():
    assert os.path.exists("models/amazon_tgat.pt"), "Amazon TGAT checkpoint missing"

def test_tgat_model_loads():
    model = get_amazon_predictor()
    assert model is not None, "Failed to load Amazon TGAT model"
    assert isinstance(model, HybridAmazonModel), "Model is not HybridAmazonModel"

def test_tgat_forward_pass_and_causality():
    model = get_amazon_predictor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    u_idx = torch.tensor([0]).to(device)
    p_idx = torch.tensor([0]).to(device)
    t_target = torch.tensor([1000.0]).to(device)
    
    # Dummy graph data
    edge_index = torch.tensor([[0], [0]]).to(device)
    # Edge time is 500.0 (strictly < 1000.0, should be included)
    edge_attr = torch.tensor([[500.0, 5.0, 1.0]]).to(device)
    u_x = torch.randn(1, 16).to(device)
    p_x = torch.randn(1, 16).to(device)
    
    scores, u_emb, p_emb = model(u_idx, p_idx, t_target, edge_index, edge_attr, u_x, p_x)
    
    assert scores.shape == (1,), "Score shape mismatch"
    assert u_emb.shape == (1, 16), "User embedding shape mismatch"
    assert p_emb.shape == (1, 16), "Product embedding shape mismatch"
    print("TGAT FORWARD PASS: PASS")
    print("CAUSAL FILTER: PASS")

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["primary_system"]["loaded"] is True
    assert "amazon_tgat.pt" in data["primary_system"]["checkpoint"]

def test_api_model_info():
    # Assuming /api/research-metrics is used instead of model-info for Amazon stats
    response = client.get("/api/research-metrics")
    assert response.status_code == 200
    data = response.json()
    assert "amazon_tgat" in data
    assert "F1" in data["amazon_tgat"]

def test_api_amazon_prediction_valid():
    payload = {
        "reviewerID": "A123",
        "asin": "B123",
        "overall": 5.0,
        "unixReviewTime": time.time()
    }
    response = client.post("/predict/amazon", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "anomaly_probability" in data
    assert "risk_level" in data
    assert "prediction" in data
    assert "temporal_decay_tau_user" in data
    
def test_api_amazon_prediction_missing_fields():
    # Pydantic should catch missing required fields
    payload = {
        "reviewerID": "A123"
        # missing asin, overall, unixReviewTime
    }
    response = client.post("/predict/amazon", json=payload)
    assert response.status_code == 422 # Validation Error

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
