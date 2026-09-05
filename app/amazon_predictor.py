import os
import time
import torch
import numpy as np
from src.models.amazon_contrastive_tgat import HybridAmazonModel

_AMAZON_CHECKPOINT = "models/amazon_tgat.pt"
_amazon_model = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dummy node embeddings for demo consistency with training
# In production, these would be cached embeddings from the graph DB
_u_x = torch.randn(10000, 16).to(_device)
_p_x = torch.randn(10000, 16).to(_device)
_edge_index = torch.empty((2, 0), dtype=torch.long).to(_device)
_edge_attr = torch.empty((0, 3), dtype=torch.float).to(_device)

def get_amazon_predictor():
    global _amazon_model
    if _amazon_model is not None:
        return _amazon_model

    if not os.path.exists(_AMAZON_CHECKPOINT):
        print(f"[Amazon TGAT] WARNING: Checkpoint not found at {_AMAZON_CHECKPOINT}.")
        return None

    try:
        model = HybridAmazonModel(in_channels=16, hidden_channels=16, use_time=True, use_decay=True, alpha=0.5).to(_device)
        state_dict = torch.load(_AMAZON_CHECKPOINT, map_location=_device)
        model.load_state_dict(state_dict)
        model.eval()
        _amazon_model = model
        print("[Amazon TGAT] Loaded frozen checkpoint successfully.")
        return _amazon_model
    except Exception as e:
        print(f"[Amazon TGAT] ERROR loading checkpoint: {e}")
        return None

def amazon_demo_inference(tx_data: dict) -> dict:
    """
    Run Amazon TGAT inference in demo mode.
    Simulates a User -> Product review interaction anomaly score.
    """
    model = get_amazon_predictor()

    if model is None:
        return {
            "amazon_tgat_enabled": False,
            "anomaly_probability": None,
            "temporal_context": "Amazon TGAT checkpoint unavailable",
            "note": "Amazon TGAT module disabled."
        }

    # Demo hash IDs for the dummy embedding lookup
    u_raw = str(tx_data.get('reviewerID', 'A1234'))
    p_raw = str(tx_data.get('asin', 'B000123'))
    t_raw = float(tx_data.get('unixReviewTime', time.time()))

    u_idx = torch.tensor([hash(u_raw) % 10000], dtype=torch.long).to(_device)
    p_idx = torch.tensor([hash(p_raw) % 10000], dtype=torch.long).to(_device)
    t_target = torch.tensor([t_raw], dtype=torch.float).to(_device)

    with torch.no_grad():
        # Edge index and attr are empty in this demo pass (simulating a cold start interaction)
        # In a full deployment, we would pull the ego-graph for u_idx and p_idx
        scores, u_emb, p_emb = model(u_idx, p_idx, t_target, _edge_index, _edge_attr, _u_x, _p_x)
        
        # High dot product -> highly expected link -> Low anomaly probability
        # We apply sigmoid and invert
        similarity = torch.sigmoid(scores).item()
        anomaly_prob = 1.0 - similarity

        # Extract learned tau parameters
        tau_u = float(torch.nn.functional.softplus(model.encoder.raw_tau_user).item() + 1e-5)
        tau_p = float(torch.nn.functional.softplus(model.encoder.raw_tau_prod).item() + 1e-5)
        
        hl_u = float(np.log(2) / max(tau_u, 1e-6))
        hl_p = float(np.log(2) / max(tau_p, 1e-6))

    return {
        "amazon_tgat_enabled": True,
        "anomaly_probability": round(anomaly_prob, 4),
        "similarity_score": round(similarity, 4),
        "temporal_decay_tau_user": round(tau_u, 4),
        "temporal_decay_tau_product": round(tau_p, 4),
        "temporal_half_life_user_days": round(hl_u, 1),
        "temporal_half_life_product_days": round(hl_p, 1),
        "user_embedding_norm": round(float(u_emb.norm().item()), 4),
        "product_embedding_norm": round(float(p_emb.norm().item()), 4),
        "temporal_context": (
            f"User decay half-life: {hl_u:.1f}d | Prod decay half-life: {hl_p:.1f}d"
        ),
        "note": "DEMO MODE: Amazon Graph anomaly detection via link reconstruction."
    }

if __name__ == "__main__":
    import time
    res = amazon_demo_inference({"reviewerID": "UserA", "asin": "ProdB", "unixReviewTime": time.time()})
    print(res)
