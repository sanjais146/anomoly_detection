# T-GAT Architecture Documentation (Phase 5B)

## 1. Causal T-GAT Architecture Diagram
```
Target Transaction T
       │
       ▼ (Causal Sequence Extraction)
┌─────────────────────────────────────────────────────┐
│ 1. Filter: Find historical Tx matching Target Card  │
│    (Strictly t_hist < t_target)                     │
│ 2. Sort: Chronological (oldest to newest)           │
│ 3. Slice: Keep last K events [h_1, h_2, ..., h_K]   │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼ 
┌─────────────────────────────────────────────────────┐
│  Temporal GAT (Message Calculation)                 │
│  m_j = alpha_j * exp(-tau * Delta_t_days) * W h_j   │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼ (Sequence of GAT messages)
┌─────────────────────────────────────────────────────┐
│                 GRU (batch_first=True)              │
│  Hidden State Output (Represents the Card Identity) │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼ 
┌─────────────────────────────────────────────────────┐
│   Concat: [Target_Tx, Card_State, Device_State]     │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              Fraud Classifier (MLP)                 │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
                Fraud Probability
```

## 2. Why the Previous Architecture Was Insufficient
*   **GRU Temporality:** The previous model used `GRUCell(h_context, tx_x)`, feeding the target's intrinsic features and a *single* pooled historical graph snapshot. This was merely gated fusion, not a multi-step sequence.
*   **Target Node Leakage:** Because it used a global graph forward pass without masking Hop 1 (`Tx -> Card`), the Card embedding implicitly aggregated *future* transactions and the *target's own* features before passing the state back to the target.

## 3. The New Temporal Sequence Construction
1.  **Extraction:** For a mini-batch of target transactions, the model looks up their associated `Card`. It then dynamically queries the bipartite edge list `(Tx -> Card)` to find all transactions belonging to that card where `t_hist < t_target`.
2.  **Sequence Building:** The historical transactions are sorted chronologically. The last $K$ (e.g., $K=10$) events form a true time-series sequence.
3.  **GRU Processing:** This chronologically ordered sequence of $K$ states is passed into an `nn.GRU`, authentically tracking the evolution of the entity's behavior over time.

## 4. Leakage Prevention (Causal Masking)
Leakage is strictly prevented by the boolean mask:
`mask = (dst_edges == query_entity) & (full_edge_time < query_time)`
*   `query_time` is the exact timestamp of the target transaction being predicted.
*   The strictly `<` operator ensures that **same-time** transactions and **future** transactions are permanently excluded from the entity's state computation. 
*   **Self-leakage** is physically impossible because the target transaction's own timestamp guarantees it fails the `<` check, meaning it can never enter its own sequence.

## 5. Temporal Decay Parameterization
*   **Formula:** $w = \exp(-\tau \cdot \Delta t_{days})$
*   **Time Scale:** $\Delta t$ is explicitly normalized to **days** ($\Delta t_{sec} / 86400$). A decay measured in seconds mathematically annihilated context within hours, making long-term fraud ring detection impossible.
*   **Tau Positivity:** $\tau$ is parameterized via `F.softplus(raw_tau) + 1e-6`. This enables smooth, unclipped gradients during backpropagation while strictly guaranteeing that $\tau > 0$ (decay never becomes growth). 
*   **Initial Scale:** `raw_tau` is initialized such that $\tau \approx 0.1$, allowing historical events from 10 days prior to retain significant weight ($\approx 0.36$).

## 6. Unknown Entity Handling
*   `nn.Embedding` was removed entirely. 
*   The state of a `Card` or `Device` is fundamentally defined *by the GRU output of its historical sequence*.
*   If a validation/test target queries a brand new Card, the causal mask yields an empty sequence (0 events). The GRU natively outputs a zero-vector. This cleanly maps to "unseen entity" behavior without requiring dynamic embedding expansion or crashing on out-of-bounds indices.

## 7. Static Baseline vs. T-GAT
*   **T-GAT:** Implements causal subgraph sequencing, temporal edge weights $\exp(-\tau \Delta t)$, and multi-step GRU modeling.
*   **Static Baseline:** Directly projects transaction features and aggregates adjacent Card/Device embeddings using a simple 1-hop static mapping. No temporal logic, no decay, no GRU. The architecture physically bypasses the `CausalSequenceExtractor`.
