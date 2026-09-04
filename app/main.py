from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.schemas import TransactionInput, PredictionResponse
from app.predictor import get_predictor
import uvicorn
import os

app = FastAPI(title="E-commerce Fraud Detection API", description="Live inference endpoint for E10 Causal Ensemble")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend if it exists
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    return {"message": "Fraud Detection API Running. Visit /docs for OpenAPI UI."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": get_predictor() is not None}

@app.get("/model-info")
def model_info():
    return {
        "model": "E10 Static Causal CatBoost Ensemble",
        "evaluation": "Chronological / causal",
        "label_delay": "7 days",
        "test_f1": "62.25%",
        "auroc": "0.9378"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(tx: TransactionInput):
    try:
        pred = get_predictor()
        result = pred.predict(tx.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
