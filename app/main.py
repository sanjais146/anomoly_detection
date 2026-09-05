"""
app/main.py — E-Commerce Anomaly Detection API

Architecture:
    E-Commerce Transaction Input
           ↓
    TGAT (Temporal Graph Attention Network)
       - Frozen checkpoint: src/models/best_tgat_final.pt
       - Causal card-history attention with temporal decay
       - 31-dim input → 32-dim embedding → standalone probability
           ↓
    E10 CatBoost Ensemble (3 models)
       - Frozen: models/e10_base.cbm, e10_deep.cbm, e10_weight.cbm
       - 508-dim causal feature space
       - Authoritative Test F1: 62.25%
           ↓
    Combined Anomaly Response
"""

import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.schemas import TransactionInput, PredictionResponse
from app.predictor import get_predictor
from app.tgat_predictor import get_tgat_predictor, tgat_demo_inference

app = FastAPI(
    title="E-Commerce Anomaly Detection API",
    description=(
        "Live inference for the E-Commerce Anomaly Detection system. "
        "Architecture: TGAT (temporal graph attention) + E10 CatBoost Ensemble. "
        "Causal evaluation protocol: 7-day label availability boundary."
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def read_root():
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    return {"message": "E-Commerce Anomaly Detection API. Visit /docs for OpenAPI UI."}


@app.get("/health")
def health_check():
    """System health: reports E10 and TGAT model status."""
    e10 = get_predictor()
    tgat = get_tgat_predictor()
    return {
        "status": "healthy",
        "components": {
            "e10_catboost_ensemble": {
                "loaded": e10 is not None,
                "models": 3,
                "test_f1": "62.25%",
                "checkpoint": "models/e10_base.cbm + e10_deep.cbm + e10_weight.cbm"
            },
            "tgat": {
                "loaded": tgat is not None,
                "checkpoint": "src/models/best_tgat_final.pt",
                "architecture": "TGATFinal (31-dim → 32-dim hidden, temporal attention)",
                "standalone_val_f1": "20.12%",
                "role": "Temporal graph representation; E10 is authoritative classifier"
            }
        },
        "system": {
            "causal_boundary": "7-day label delay",
            "inference_mode": "Causal Demonstration",
            "deployment": "Google Colab + ngrok"
        }
    }


@app.get("/model-info")
def model_info():
    """Detailed frozen model configuration."""
    tgat = get_tgat_predictor()
    return {
        "architecture": "TGAT + E10 CatBoost Ensemble",
        "components": {
            "tgat": {
                "name": "TGATFinal",
                "enabled": tgat is not None,
                "checkpoint": "src/models/best_tgat_final.pt",
                "input_dim": 31,
                "hidden_dim": 32,
                "entities": ["card_identity"],
                "temporal_mechanism": "CausalTemporalAttention with exp(-tau * delta_t_days) decay",
                "causal_mask": "t_hist < t_target (strict inequality)",
                "standalone_val_f1": 0.2012,
                "note": "TGAT produces temporal graph embeddings; E10 is the authoritative anomaly classifier"
            },
            "e10_ensemble": {
                "name": "E10 Static Causal CatBoost Ensemble",
                "constituents": [
                    {"name": "E10 Base", "depth": 6, "lr": 0.10, "iterations": 600},
                    {"name": "E10 Deep", "depth": 8, "lr": 0.08, "l2_leaf_reg": 5, "iterations": 600},
                    {"name": "E10 Class-Weighted", "depth": 6, "lr": 0.10, "scale_pos_weight": "auto"}
                ],
                "test_f1": 0.6225,
                "test_precision": 0.7225,
                "test_recall": 0.5469,
                "test_auroc": 0.9378,
                "test_auprc": 0.6455,
                "threshold": 0.4040,
                "features": 508,
                "evaluation": "Chronological inductive split, 7-day label availability boundary"
            }
        },
        "causal_protocol": {
            "label_delay_days": 7,
            "split": "Chronological: Day 0-120 Train, 120-150 Val, 150+ Test",
            "entity_features": ["card1+card2", "card1+addr1", "card1+DeviceInfo"],
            "feature_windows": ["1h", "24h", "3d", "7d", "30d"]
        }
    }


@app.get("/api/research-metrics")
def research_metrics():
    """Frozen authoritative research metrics — single source of truth for the dashboard."""
    metrics_path = "frontend/research_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(content={"error": "research_metrics.json not found"}, status_code=404)


@app.post("/predict")
def predict_transaction(tx: TransactionInput):
    """
    Analyze an e-commerce transaction for anomalous behavior.

    Pipeline:
    1. E10 CatBoost Ensemble → primary anomaly score (authoritative)
    2. TGAT → temporal graph context (supplementary)

    Returns combined anomaly assessment with both components.
    """
    try:
        tx_dict = tx.dict()

        # --- E10 CatBoost (primary, authoritative) ---
        e10 = get_predictor()
        if e10 is None:
            raise HTTPException(status_code=503, detail="E10 model not loaded")
        e10_result = e10.predict(tx_dict)

        # --- TGAT (supplementary temporal context) ---
        tgat_result = tgat_demo_inference(tx_dict)

        # Combine into unified response
        return {
            # Primary anomaly classification (E10)
            "fraud_probability": e10_result["fraud_probability"],
            "prediction": e10_result["prediction"],
            "risk_level": e10_result["risk_level"],
            "threshold": e10_result["threshold"],
            "model": e10_result["model"],
            "signals": e10_result["signals"],

            # TGAT temporal graph context
            "tgat": {
                "enabled": tgat_result.get("tgat_enabled", False),
                "probability": tgat_result.get("tgat_probability"),
                "temporal_decay_tau": tgat_result.get("temporal_decay_tau"),
                "temporal_half_life_days": tgat_result.get("temporal_half_life_days"),
                "card_context_norm": tgat_result.get("card_context_embedding_norm"),
                "temporal_context": tgat_result.get("temporal_context"),
                "note": tgat_result.get("note")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
