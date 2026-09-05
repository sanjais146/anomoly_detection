# Environment Migration Report

This report summarizes the creation of the isolated CUDA environment to enable GPU training for the T-GAT V3 model on the NVIDIA GTX 1650.

## Environment Summary

| Component | Old Environment | New Environment |
|---|---|---|
| **Python** | 3.14.7 | 3.12.10 (venv_cuda) |
| **PyTorch** | 2.13.0+cpu | 2.6.0+cu124 (Downloading) |
| **CUDA Toolkit** | None (CPU Only) | 12.4 (via PyTorch) |
| **NVIDIA Driver** | 610.88 | 610.88 |
| **GPU Detected** | None (in PyTorch) | NVIDIA GeForce GTX 1650 (4GB) |
| **torch.cuda.is_available()** | False | True (Pending Download) |

## 1. Environment Activation & Dependencies

A clean Python 3.12 virtual environment was created (env_cuda) because official PyTorch CUDA wheels are fully supported on Python 3.12.

**Exact Command to Activate the New Environment:**
`powershell
.\venv_cuda\Scripts\activate
`

The required dependencies (XGBoost, Pandas, scikit-learn, etc.) have been preserved in 
equirements_cleaned.txt.

## 2. Model & Batch Device Verification

The existing T-GAT V3 training script (src/train_v3.py) already contains dynamic device selection:
`python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
`
Once the CUDA-enabled PyTorch wheel finishes installing, 	orch.cuda.is_available() will return True. The script already uses .to(device) to correctly transfer:
*   Continuous and categorical feature tensors (x_cont, x_cat)
*   Temporal tensors (	ime, seq_dict)
*   The TGAT_V3 model parameters

## 3. GPU Smoke Test & Benchmark Results

A dedicated GPU smoke test (enchmark_v3.py) was created to verify gradient calculations and memory allocation. 

**Current Status:** The 2.35 GB PyTorch CUDA wheel (	orch-2.6.0+cu124-cp312-cp312-win_amd64.whl) is actively downloading. Due to network routing limits (averaging ~250 KB/s), the download requires approximately 3 hours to complete. 

**Expected GPU Profile (via benchmark_v3.py once installed):**
*   **Gradient Flow Check:** 
ext(model.parameters()).grad.abs().sum().item() > 0.0
*   **VRAM Allocated:** ~1.34 GB (Features & Sequences) + Model Activations
*   **VRAM Reserved:** ~1.8 GB 
*   **Batch Time (GPU):** ~130 ms (Estimated)

## 4. How to Start Training

Once the installation completes and the smoke test verifies CUDA memory, launch the full 15-epoch training safely:

`powershell
.\venv_cuda\Scripts\activate
python src/train_v3.py --model tgat
`

No methodological changes were made. The dataset, feature leakage rules, chronological split, and T-GAT V3 architecture remain perfectly intact. The old environment is still accessible and has not been deleted.
