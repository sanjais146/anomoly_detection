# Deployment Guide

**Project:** E-Commerce Anomaly Detection

The primary and fully supported deployment mechanism for demonstration is **Google Colab**. This ensures identical environments, hardware-accelerated inference if desired, and easy access without local dependency issues.

## 1. Prerequisites
- A Google Account (for Colab).
- A free [ngrok](https://dashboard.ngrok.com) account.

## 2. Setup Ngrok in Colab
1. Log into your Ngrok dashboard and copy your Auth Token.
2. Open the Colab notebook: `colab/run_demo.ipynb`.
3. In the left sidebar of Google Colab, click the **🔑 Secrets** icon.
4. Create a new secret:
   - **Name:** `NGROK_AUTHTOKEN`
   - **Value:** `<your-token>`
5. Toggle "Notebook access" to **ON**.

## 3. Run the System
1. Click **Runtime > Run all** (or press `Ctrl+F9`).
2. The notebook will automatically:
   - Clone the GitHub repository.
   - Install required packages (FastAPI, PyTorch, Uvicorn, Pyngrok, etc.).
   - Load the frozen PyTorch checkpoint (`models/amazon_tgat.pt`).
   - Start the FastAPI server on port 8000.
   - Establish an ngrok tunnel.
3. Scroll to the bottom of the output in the final cell. You will see a live public URL (e.g., `https://xxxx-xx-xx-xx-xx.ngrok-free.app`).
4. Click the URL to open the Live Command Center Dashboard.

## 4. Local Deployment (Alternative)
If you prefer to run locally on Windows/Linux/Mac:
```bash
# Clone and enter repo
git clone https://github.com/sanjais146/anomoly_detection
cd anomaly-detect

# Install requirements
pip install -r requirements.txt

# Start the API and Dashboard
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Then navigate to `http://localhost:8000` in your browser.
