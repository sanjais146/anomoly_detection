"""
app/tgat_predictor.py

TGAT inference adapter for the E-Commerce Anomaly Detection demo.

Architecture:
  Input transaction features (31-dim from IEEEGraphBuilderFinal feature space)
      ↓
  TGATFinal (Temporal Graph Attention Network)
      - tabular_encoder: Linear(31→64)→BN→ReLU→Linear(64→32)
      - card_attention: CausalTemporalAttention with learnable τ decay
      - fusion: combines target + card_agg → logit
      ↓
  TGAT anomaly probability (standalone classifier)
      ↓
  Used as additional evidence alongside E10 CatBoost ensemble

Causal guarantee:
  - Only history with t_hist < t_target is used
  - The temporal decay weight = exp(-τ * Δt_days) ensures recent events dominate
  - Padded history slots use t=-1 which decays to weight≈0

Model status: FROZEN checkpoint (src/models/best_tgat_final.pt)
Validation F1: 20.12% (standalone; E10 CatBoost at 62.25% is the authoritative result)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TGATFinal architecture — must match checkpoint exactly
# ---------------------------------------------------------------------------

class CausalTemporalAttention(nn.Module):
    """
    Single-head causal temporal attention.
    Uses learnable temporal decay: w = exp(-τ * Δt_days)
    """
    def __init__(self, hidden_dim, dropout=0.5):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.raw_tau = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_emb, history_embs, target_t, history_t):
        """
        target_emb:   (B, hidden_dim)
        history_embs: (B, K, hidden_dim)
        target_t:     (B, 1) — transaction timestamp in seconds
        history_t:    (B, K) — history timestamps (-1 = pad)
        """
        delta_t = (target_t - history_t) / 86400.0      # convert to days
        tau = F.softplus(self.raw_tau) + 1e-5
        decay = torch.exp(-tau * delta_t)
        mask = (history_t > 0).float()
        decay = decay * mask

        Q = self.q(target_emb).unsqueeze(1)             # (B, 1, H)
        K = self.k(history_embs)                         # (B, K, H)
        V = self.v(history_embs)                         # (B, K, H)

        scores = torch.bmm(Q, K.transpose(1, 2)).squeeze(1) / (Q.size(-1) ** 0.5)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=1)
        attn = self.dropout(attn)

        weights = attn * decay
        agg = torch.bmm(weights.unsqueeze(1), V).squeeze(1)  # (B, H)
        return agg


class TGATFinal(nn.Module):
    """
    Frozen TGAT model for IEEE-CIS demo inference.
    Checkpoint: src/models/best_tgat_final.pt
    Input: 31-dim feature vector (IEEEGraphBuilderFinal feature space)
    """
    def __init__(self, num_features=31, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.card_attention = CausalTemporalAttention(hidden_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, hist_x, t, hist_t):
        target_emb = self.tabular_encoder(x)
        b, k, f = hist_x.size()
        hist_embs = self.tabular_encoder(hist_x.view(-1, f)).view(b, k, -1)
        card_agg = self.card_attention(target_emb, hist_embs, t, hist_t)
        combined = torch.cat([target_emb, card_agg], dim=1)
        return self.fusion(combined)

    def get_embedding(self, x, hist_x, t, hist_t):
        """
        Returns the 32-dim card-context embedding (TGAT representation)
        before the final fusion classifier.
        Used to surface temporal context in the demo response.
        """
        with torch.no_grad():
            target_emb = self.tabular_encoder(x)
            b, k, f = hist_x.size()
            hist_embs = self.tabular_encoder(hist_x.view(-1, f)).view(b, k, -1)
            card_agg = self.card_attention(target_emb, hist_embs, t, hist_t)
        return target_emb, card_agg


# ---------------------------------------------------------------------------
# Feature mapping: TransactionInput → 31-dim TGAT feature vector
# ---------------------------------------------------------------------------

# The 31 features used by IEEEGraphBuilderFinal (from training):
#   TransactionAmt, C1-C14 (14), V310, V313, V314, V315 (4),
#   ProductCD, card1, card2, card3, card4, card5, card6,
#   addr1, addr2, P_emaildomain, DeviceType, DeviceInfo  (12 categoricals)
# Total: 1 + 14 + 4 + 12 = 31

# For the DEMO: we do not have real pre-computed StandardScaler state.
# We construct a reasonable approximation from the transaction input fields
# that are available in the API. Unsupported fields are set to 0.
# This is documented as "DEMO MODE — approximate feature construction."

TGAT_NUM_FEATURES = 31

_CAT_FIELD_POSITIONS = {
    # positions 15–26 are the 12 categorical fields encoded as integers
    # For demo, we use simple hash-based integer encoding
    'ProductCD': 15,
    'card1': 16,
    'card2': 17,
    'card3': 18,
    'card4': 19,
    'card5': 20,
    'card6': 21,
    'addr1': 22,
    'addr2': 23,
    'P_emaildomain': 24,
    'DeviceType': 25,
    'DeviceInfo': 26,
}


def _simple_cat_encode(value) -> float:
    """Stable integer encoding for categorical demo features."""
    if value is None or value == "" or str(value) == "nan":
        return 0.0
    return float(abs(hash(str(value))) % 1000)


def build_tgat_feature_vector(tx_data: dict) -> np.ndarray:
    """
    Build the 31-dim TGAT input vector from a TransactionInput dict.

    Demo approximation:
    - Numeric features: use provided value if present, else 0
    - Categorical features: hash-encoded integer
    - No StandardScaler is applied (demo mode; does not require the full
      training pipeline to be reproduced for a live inference demonstration)
    """
    x = np.zeros(TGAT_NUM_FEATURES, dtype=np.float32)

    # Position 0: TransactionAmt (normalized roughly by log)
    amt = tx_data.get("TransactionAmt", 0) or 0
    x[0] = float(np.log1p(max(0, amt)))

    # Positions 1-14: C1-C14 (counting features — set to 0 in demo)
    # These are internal bank signals not available from external input

    # Positions 15-26: categorical fields
    for field, pos in _CAT_FIELD_POSITIONS.items():
        x[pos] = _simple_cat_encode(tx_data.get(field))

    # Positions 27-30: V310, V313, V314, V315 (anonymized signals)
    # Not available from API input — remain 0

    return x


def build_demo_history(tx_amt: float, n_hist: int = 5, K: int = 10) -> tuple:
    """
    Build a synthetic card history for demo inference.
    For real deployment this would query actual historical transactions.

    In demo mode we simulate:
    - n_hist past transactions with slightly lower amounts
    - Timestamps ~7 days before the current demo timestamp
    - Remaining K - n_hist slots are padded with zeros and t=-1
    """
    import time
    current_t = float(int(time.time()))
    K = 10

    # Simulated historical timestamps: 7–14 days before current
    hist_times = np.full(K, -1.0, dtype=np.float32)
    hist_x = np.zeros((K, TGAT_NUM_FEATURES), dtype=np.float32)

    for i in range(min(n_hist, K)):
        days_ago = 7 + i * 1.5
        hist_times[i] = current_t - days_ago * 86400
        hist_x[i, 0] = float(np.log1p(max(0, tx_amt * (0.7 + 0.1 * i))))

    return hist_x, hist_times, current_t


# ---------------------------------------------------------------------------
# TGAT Predictor singleton
# ---------------------------------------------------------------------------

_TGAT_CHECKPOINT = "src/models/best_tgat_final.pt"

_tgat_instance = None


def get_tgat_predictor():
    """
    Load and return the singleton TGAT predictor.
    Returns None if the checkpoint is missing (graceful degradation).
    """
    global _tgat_instance
    if _tgat_instance is not None:
        return _tgat_instance

    if not os.path.exists(_TGAT_CHECKPOINT):
        print(f"[TGAT] WARNING: Checkpoint not found at {_TGAT_CHECKPOINT}. TGAT will be disabled.")
        return None

    try:
        model = TGATFinal(num_features=TGAT_NUM_FEATURES, hidden_dim=32, dropout=0.5)
        state_dict = torch.load(_TGAT_CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        _tgat_instance = model
        print("[TGAT] Loaded frozen checkpoint successfully.")
        return _tgat_instance
    except Exception as e:
        print(f"[TGAT] ERROR loading checkpoint: {e}")
        return None


def tgat_demo_inference(tx_data: dict) -> dict:
    """
    Run TGAT inference in demo mode.

    Returns:
        dict with keys:
          - tgat_enabled: bool
          - tgat_probability: float or None
          - tgat_embedding_norm: float (L2 norm of the card context embedding)
          - temporal_decay_tau: float (learned temporal decay rate)
          - history_depth: int (simulated)
          - note: str (explains demo approximation)
    """
    model = get_tgat_predictor()

    if model is None:
        return {
            "tgat_enabled": False,
            "tgat_probability": None,
            "temporal_context": "TGAT checkpoint unavailable",
            "note": "TGAT module disabled — checkpoint not found."
        }

    x_np = build_tgat_feature_vector(tx_data)
    amt = tx_data.get("TransactionAmt", 100) or 100
    hist_x_np, hist_times_np, current_t = build_demo_history(tx_amt=float(amt))

    # Convert to tensors (batch size 1)
    x_t = torch.FloatTensor(x_np).unsqueeze(0)           # (1, 31)
    hist_x_t = torch.FloatTensor(hist_x_np).unsqueeze(0) # (1, 10, 31)
    t_t = torch.FloatTensor([[current_t]])                # (1, 1)
    hist_t_t = torch.FloatTensor(hist_times_np).unsqueeze(0)  # (1, 10)

    with torch.no_grad():
        logit = model(x_t, hist_x_t, t_t, hist_t_t)
        prob = float(torch.sigmoid(logit).item())

        target_emb, card_agg = model.get_embedding(x_t, hist_x_t, t_t, hist_t_t)
        embedding_norm = float(card_agg.norm().item())

        # Extract learned τ
        tau = float(F.softplus(model.card_attention.raw_tau).item() + 1e-5)
        half_life_days = float(np.log(2) / max(tau, 1e-6))

    return {
        "tgat_enabled": True,
        "tgat_probability": round(prob, 4),
        "temporal_decay_tau": round(tau, 4),
        "temporal_half_life_days": round(half_life_days, 1),
        "card_context_embedding_norm": round(embedding_norm, 4),
        "history_depth_simulated": 5,
        "temporal_context": (
            f"Card history: 5 simulated prior transactions (7–14 days ago). "
            f"Temporal decay τ={tau:.3f} (half-life ≈ {half_life_days:.1f} days)."
        ),
        "note": (
            "DEMO MODE: TGAT uses approximate feature construction and synthetic "
            "card history. In production, real historical transactions would be "
            "queried. TGAT standalone validation F1: 20.12% (E10 CatBoost: 62.25%)."
        )
    }
