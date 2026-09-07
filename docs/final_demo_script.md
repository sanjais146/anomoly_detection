# Final Viva Demo Script

Follow this script closely to ensure a flawless live demonstration of the E-Commerce Anomaly Detection system.

## Phase 1: Environment Startup
1. Open the **Google Colab Notebook** (`colab/run_demo.ipynb`).
2. Show the examiner the **Secrets** panel on the left to prove the `NGROK_AUTHTOKEN` is securely injected and not hardcoded.
3. Click **Runtime > Run all**.
4. **Narrate while it runs:** 
   * "The notebook is cloning our repository, installing PyTorch and FastAPI, and loading our frozen TGAT checkpoint (`amazon_tgat.pt`)."
   * "Because we are using Git LFS, the actual binary model is pulled automatically."
5. Scroll to the bottom cell. Point out the print statements:
   * "FastAPI: ONLINE"
   * "ngrok tunnel established."
6. Click the generated **ngrok URL** to open the Live Dashboard.

## Phase 2: Dashboard Overview
1. **The Command Center:** Once the UI loads, explain that this is a dense, Vercel-style analytics dashboard representing a production-ready interface.
2. **System Status:** Point to the top right to show "Dataset: Amazon Electronics" and "Causal Mode: ENABLED".
3. **Research Context (Overview Tab):**
   * Briefly mention the historical research: "You can see our historical CatBoost benchmark on the IEEE-CIS dataset achieved 62.25% F1."
   * Clarify immediately: "However, our final pipeline uses the Amazon Electronics dataset to power an unsupervised Temporal Graph Attention Network (TGAT)."

## Phase 3: Anomaly Analytics (The Core System)
1. Click the **Anomaly Analytics** tab.
2. **Provenance Banner:** Read the blue banner at the top. "We are executing the actual trained PyTorch TGAT checkpoint. No synthetic interactions, no fake data."
3. **Anomaly Definition Panel:** Explain how an anomaly is defined. "It is $1 - \sigma(\text{User} \cdot \text{Product})$. High score means a topologically unexpected link."
4. **KPI Cards:** Show the 100 analyzed interactions. Explain that the threshold (0.6668) is a calibrated cold-start threshold, clearly separating ~25% as anomalous.
5. **Score Chart:** Point out the varying anomaly probabilities over time, proving the model is dynamically scoring real data. Hover over a red dot to show the tooltip.
6. **Top Anomalies Table:** Scroll down to show the specific Reviewer IDs and ASINs that were flagged.

## Phase 4: Model Intelligence (Temporal Analysis)
1. Click the **Temporal Analysis** tab.
2. Explain the **Causal Timeline**: "We enforce strict causal masking. The model cannot look into the future."
3. Point to the **Temporal Decay Function** chart. 
   * "The model actually *learned* this parameter during training. For Amazon Electronics, the decay factor $\tau$ is 0.431, which corresponds to a half-life of 1.6 days. This proves that recent interactions are heavily weighted compared to older ones."

## Phase 5: Live Interaction Analysis
1. Click the **Analyze Interaction** tab.
2. The form is pre-filled with a sample Reviewer ID and ASIN. 
3. Click **Run TGAT Inference**.
4. The panel will update with a live score. 
5. Explain: "The FastAPI backend just passed this interaction through the TGAT PyTorch model. We see the final anomaly score, the exact threshold used, and the generated temporal node embeddings below."

## Phase 6: Conclusion
End with a strong summary:
"This system successfully demonstrates a scientifically defensible, live-executable E-Commerce Anomaly Detection pipeline. It leverages the topological and temporal power of TGAT while cleanly distinguishing unsupervised network anomalies from historical supervised fraud benchmarks."
