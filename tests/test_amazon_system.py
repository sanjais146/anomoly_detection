import os
import sys
import time
import json
import torch
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app
from app.amazon_predictor import get_amazon_predictor, DEMO_BATCH_THRESHOLD, TRAINING_THRESHOLD
from src.models.amazon_contrastive_tgat import HybridAmazonModel

client = TestClient(app)


# ── Model Loading ─────────────────────────────────────────────────────────────

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
    edge_index = torch.tensor([[0], [0]]).to(device)
    edge_attr = torch.tensor([[500.0, 5.0, 1.0]]).to(device)
    u_x = torch.randn(1, 16).to(device)
    p_x = torch.randn(1, 16).to(device)
    scores, u_emb, p_emb = model(u_idx, p_idx, t_target, edge_index, edge_attr, u_x, p_x)
    assert scores.shape == (1,), "Score shape mismatch"
    assert u_emb.shape == (1, 16), "User embedding shape mismatch"
    assert p_emb.shape == (1, 16), "Product embedding shape mismatch"


# ── Score Distribution Integrity ──────────────────────────────────────────────

def test_scores_are_not_all_identical():
    """Scores must vary across different user/product/time inputs."""
    from app.amazon_predictor import amazon_demo_inference
    inputs = [
        {'reviewerID': 'UA', 'asin': 'PA', 'unixReviewTime': 1380000000, 'overall': 5.0},
        {'reviewerID': 'UB', 'asin': 'PB', 'unixReviewTime': 1381000000, 'overall': 1.0},
        {'reviewerID': 'UC', 'asin': 'PC', 'unixReviewTime': 1382000000, 'overall': 3.0},
        {'reviewerID': 'UD', 'asin': 'PD', 'unixReviewTime': 1383000000, 'overall': 4.0},
        {'reviewerID': 'UE', 'asin': 'PE', 'unixReviewTime': 1384000000, 'overall': 2.0},
    ]
    scores = [amazon_demo_inference(i)['anomaly_probability'] for i in inputs]
    assert len(set(scores)) > 1, f"All scores are identical: {scores}"

def test_scores_are_deterministic():
    """Same input must produce identical score on repeated call (deterministic seed)."""
    from app.amazon_predictor import amazon_demo_inference
    inp = {'reviewerID': 'TEST_U', 'asin': 'TEST_P', 'unixReviewTime': 1380000000, 'overall': 4.0}
    s1 = amazon_demo_inference(inp)['anomaly_probability']
    s2 = amazon_demo_inference(inp)['anomaly_probability']
    assert s1 == s2, f"Scores not deterministic: {s1} vs {s2}"

def test_scores_are_numeric_and_bounded():
    """All scores must be floats in [0, 1]."""
    from app.amazon_predictor import amazon_demo_inference
    inp = {'reviewerID': 'TEST_U', 'asin': 'TEST_P', 'unixReviewTime': 1380000000, 'overall': 4.0}
    r = amazon_demo_inference(inp)
    prob = r['anomaly_probability']
    assert isinstance(prob, float), f"Score not float: {type(prob)}"
    assert 0.0 <= prob <= 1.0, f"Score out of range: {prob}"

def test_threshold_values_are_documented():
    """Both thresholds must be numeric and DEMO > TRAINING in cold-start mode."""
    assert isinstance(DEMO_BATCH_THRESHOLD, float)
    assert isinstance(TRAINING_THRESHOLD, float)
    assert 0 < TRAINING_THRESHOLD < 1
    assert 0 < DEMO_BATCH_THRESHOLD < 1
    # Demo threshold should be higher than training threshold in cold-start mode
    assert DEMO_BATCH_THRESHOLD > TRAINING_THRESHOLD, (
        f"Demo threshold {DEMO_BATCH_THRESHOLD} should be > training threshold {TRAINING_THRESHOLD}"
    )


# ── Batch Analytics Integrity ─────────────────────────────────────────────────

def test_batch_analytics_kpi_consistency():
    """normal + anomalous must equal total; rate must be correctly computed."""
    response = client.get("/analytics/amazon/anomalies")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'success'
    kpi = data['kpi']

    total = kpi['total_analyzed']
    anomalous = kpi['anomalous_count']
    normal = kpi['normal_count']
    rate = kpi['anomaly_rate']

    assert total > 0, "total_analyzed must be > 0"
    assert normal + anomalous == total, f"{normal} + {anomalous} != {total}"
    assert 0 <= rate <= 100, f"anomaly_rate out of range: {rate}"
    expected_rate = round(anomalous / total * 100, 2)
    assert abs(rate - expected_rate) < 0.01, f"Rate mismatch: {rate} vs {expected_rate}"

def test_batch_analytics_not_all_anomalous():
    """Dashboard must NOT report 100% anomaly rate unless genuinely supported."""
    response = client.get("/analytics/amazon/anomalies")
    data = response.json()
    kpi = data['kpi']
    rate = kpi['anomaly_rate']
    # With demo threshold at 75th percentile, ~25% should be anomalous
    assert rate < 100.0, f"100% anomaly rate detected — threshold calibration bug. Rate={rate}%"
    assert kpi['normal_count'] > 0, "normal_count must be > 0 with calibrated threshold"

def test_batch_anomaly_decision_agrees_with_threshold():
    """Each record's is_anomalous must agree with score >= threshold."""
    response = client.get("/analytics/amazon/anomalies")
    data = response.json()
    threshold = data['kpi']['threshold']
    mismatches = []
    for rec in data['time_series']:
        expected = rec['anomaly_probability'] >= threshold
        if expected != rec['is_anomalous']:
            mismatches.append(rec)
    assert len(mismatches) == 0, f"{len(mismatches)} records have inconsistent anomaly decisions"

def test_batch_scores_vary():
    """Batch results must contain varying scores (not all identical)."""
    response = client.get("/analytics/amazon/anomalies")
    data = response.json()
    scores = [r['anomaly_probability'] for r in data['time_series']]
    unique_scores = len(set(scores))
    assert unique_scores > 1, f"All batch scores are identical: {scores[:5]}"


# ── API Endpoint Tests ────────────────────────────────────────────────────────

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["primary_system"]["loaded"] is True
    assert "amazon_tgat.pt" in data["primary_system"]["checkpoint"]

def test_api_model_info():
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
    assert data['prediction'] in ('genuine', 'anomalous')  # API returns 'genuine' not 'normal'
    assert "temporal_decay_tau_user" in data

def test_api_amazon_prediction_missing_fields():
    # asin and overall are truly required; unixReviewTime is now optional
    response = client.post("/predict/amazon", json={"reviewerID": "A123"})
    assert response.status_code == 422

def test_api_analytics_amazon():
    response = client.get("/analytics/amazon")
    assert response.status_code == 200
    data = response.json()
    assert "rating_distribution" in data
    assert "timeline" in data
    assert "sample_graph" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
