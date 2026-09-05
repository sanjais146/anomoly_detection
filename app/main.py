"""
app/main.py — Amazon E-Commerce Anomaly Detection API

Architecture:
    Amazon E-Commerce Graph Input (User, Product, Interaction)
           ↓
    Amazon TGAT (Temporal Graph Attention Network)
       - Primary Implementation
       - Frozen checkpoint: models/amazon_tgat.pt
       - Contrastive Self-Supervised Link Prediction
       - Causal temporal attention with exp(-tau*delta_t) decay
           ↓
    Anomaly Probability (Reconstruction Error)

Parallel / Historical Baseline:
    IEEE-CIS Transaction Fraud Detection
    E10 CatBoost Ensemble (3 models)
    - Frozen: models/e10_base.cbm, e10_deep.cbm, e10_weight.cbm
"""

import os
import json
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Old schemas/predictors for baseline comparison
from app.schemas import TransactionInput
from app.predictor import get_predictor

# Primary Amazon TGAT predictor
from app.amazon_predictor import get_amazon_predictor, amazon_demo_inference
from app.analytics import get_amazon_analytics

app = FastAPI(
    title="Amazon E-Commerce Anomaly Detection API",
    description=(
        "Primary Implementation: Amazon TGAT (Temporal Graph Attention Network) via Self-Supervised Link Prediction. "
        "Baseline Implementation: IEEE-CIS E10 CatBoost Ensemble."
    ),
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

class AmazonInteractionInput(BaseModel):
    reviewerID: str
    asin: str
    overall: float
    unixReviewTime: float

@app.get("/")
def read_root():
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    return {"message": "Amazon E-Commerce Anomaly Detection API. Visit /docs."}

@app.get("/health")
def health_check():
    """System health: reports Amazon TGAT and IEEE-CIS E10 status."""
    e10 = get_predictor()
    amazon_tgat = get_amazon_predictor()
    
    return {
        "status": "healthy",
        "primary_system": {
            "name": "Amazon TGAT Anomaly Detection",
            "loaded": amazon_tgat is not None,
            "architecture": "Heterogeneous CausalTemporalAttention (Self-Supervised)",
            "checkpoint": "models/amazon_tgat.pt"
        },
        "baseline_system": {
            "name": "IEEE-CIS E10 CatBoost Ensemble",
            "loaded": e10 is not None,
            "test_f1": "62.25%"
        }
    }

@app.get("/api/research-metrics")
def research_metrics():
    """Frozen authoritative research metrics for the dashboard."""
    metrics_path = "frontend/research_metrics.json"
    amazon_path = "reports/amazon_tgat_results.json"
    
    data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data["ieee_cis_baseline"] = json.load(f)
            
    if os.path.exists(amazon_path):
        with open(amazon_path, "r") as f:
            data["amazon_tgat"] = json.load(f)
    else:
        # Fallback if training isn't finished
        data["amazon_tgat"] = {
            "F1": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "AUROC": 0.0,
            "Status": "Training in progress..."
        }
        
    return JSONResponse(content=data)

@app.get("/analytics/amazon")
def analytics_amazon():
    """Serves real Amazon dataset stats and sample graph interactions."""
    try:
        return get_amazon_analytics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/amazon")
def predict_amazon_anomaly(interaction: AmazonInteractionInput):
    """
    Primary Endpoint: Analyze an Amazon e-commerce interaction for anomalies.
    Uses Contrastive TGAT Link Reconstruction.
    """
    try:
        data = interaction.dict()
        if not data.get("unixReviewTime"):
            data["unixReviewTime"] = time.time()
            
        result = amazon_demo_inference(data)
        
        # Determine risk level based on probability
        prob = result.get("anomaly_probability", 0.0)
        if prob > 0.8:
            risk = "HIGH"
        elif prob > 0.5:
            risk = "ELEVATED"
        elif prob > 0.2:
            risk = "MODERATE"
        else:
            risk = "LOW"
            
        result["risk_level"] = risk
        result["prediction"] = "anomalous" if prob >= 0.5 else "genuine"
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/ieee")
def predict_ieee_transaction(tx: TransactionInput):
    """
    Baseline Endpoint: IEEE-CIS Transaction Fraud Detection.
    """
    try:
        e10 = get_predictor()
        if e10 is None:
            raise HTTPException(status_code=503, detail="E10 model not loaded")
        return e10.predict(tx.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
