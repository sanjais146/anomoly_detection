import os
import catboost as cb
import pandas as pd
import numpy as np

class E10EnsemblePredictor:
    def __init__(self, model_dir="models"):
        self.base_model = cb.CatBoostClassifier()
        self.base_model.load_model(os.path.join(model_dir, "e10_base.cbm"))
        
        self.deep_model = cb.CatBoostClassifier()
        self.deep_model.load_model(os.path.join(model_dir, "e10_deep.cbm"))
        
        self.weight_model = cb.CatBoostClassifier()
        self.weight_model.load_model(os.path.join(model_dir, "e10_weight.cbm"))
        
        self.feature_names = self.base_model.feature_names_
        self.cat_indices = self.base_model.get_cat_feature_indices()
        self.cat_names = [self.feature_names[i] for i in self.cat_indices]
        
        self.threshold = 0.4040
        
    def preprocess_demo(self, tx_data: dict):
        df = pd.DataFrame(columns=self.feature_names)
        df.loc[0] = 0.0 # Initialize all numeric to 0
        
        row = {}
        for f in self.feature_names:
            if f in self.cat_names:
                # Treat as categorical string
                if f in tx_data and tx_data[f] is not None:
                    row[f] = str(tx_data[f])
                else:
                    row[f] = "MISSING"
            else:
                # Treat as numeric
                if f in tx_data and tx_data[f] is not None:
                    try:
                        row[f] = float(tx_data[f])
                    except:
                        row[f] = 0.0
                elif f.endswith("_is_missing"):
                    row[f] = 1.0
                else:
                    row[f] = 0.0
        
        # Simulated behavioral signals for DEMO MODE:
        amt = tx_data.get("TransactionAmt", 0)
        if amt > 500:
            if "c1_count_7d_delayed" in row: row["c1_count_7d_delayed"] = 5.0
            if "c1_amt_sum_7d_delayed" in row: row["c1_amt_sum_7d_delayed"] = amt * 3
            
        return pd.DataFrame([row])

    def predict(self, tx_data: dict):
        X = self.preprocess_demo(tx_data)
        
        p_base = self.base_model.predict_proba(X)[0, 1]
        p_deep = self.deep_model.predict_proba(X)[0, 1]
        p_weight = self.weight_model.predict_proba(X)[0, 1]
        
        prob = float(np.mean([p_base, p_deep, p_weight]))
        is_fraud = prob >= self.threshold
        
        if prob > 0.8: risk = "HIGH"
        elif prob > self.threshold: risk = "ELEVATED"
        elif prob > 0.1: risk = "MODERATE"
        else: risk = "LOW"
            
        signals = []
        if tx_data.get("TransactionAmt", 0) > 300:
            signals.append("Unusual transaction amount")
        if tx_data.get("P_emaildomain") != tx_data.get("R_emaildomain") and tx_data.get("R_emaildomain"):
            signals.append("Email domain mismatch")
        if prob > self.threshold:
            signals.append("Simulated historical behavioral context - high risk pattern (demonstration only)")
        else:
            signals.append("Simulated historical behavioral context - normal pattern (demonstration only)")
            
        return {
            "fraud_probability": round(prob, 4),
            "prediction": "fraud" if is_fraud else "legitimate",
            "risk_level": risk,
            "threshold": self.threshold,
            "model": "E10 Static Causal CatBoost Ensemble",
            "signals": signals
        }

predictor_instance = None
def get_predictor():
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = E10EnsemblePredictor()
    return predictor_instance
