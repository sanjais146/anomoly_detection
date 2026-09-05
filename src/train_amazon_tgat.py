import os
import time
import json
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from pipeline.features.amazon_graph_builder import AmazonGraphBuilder
from src.models.amazon_contrastive_tgat import HybridAmazonModel

def sample_negatives_strict(u_idx, p_idx, t_target, edge_index, num_products):
    """Sample strictly unobserved edges as anomalies."""
    src, dst = edge_index
    neg_u = []
    neg_p = []
    
    # Pre-build user sets
    user_pos = {}
    for i in range(len(u_idx)):
        u = u_idx[i].item()
        if u not in user_pos:
            u_edges = (src == u)
            user_pos[u] = set(dst[u_edges].tolist())
            
    for i in range(len(u_idx)):
        u = u_idx[i].item()
        pos_set = user_pos[u]
        
        sampled = False
        while not sampled:
            rp = np.random.randint(0, num_products)
            if rp not in pos_set:
                neg_u.append(u)
                neg_p.append(rp)
                sampled = True
                
    return torch.tensor(neg_u, dtype=torch.long), torch.tensor(neg_p, dtype=torch.long)

def optimize_threshold(y_true, y_probs):
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.02):
        preds = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

def evaluate_anomalies(model, u_idx, p_idx, t_target, edge_index, edge_attr, u_x, p_x, num_products, device):
    """
    Evaluate anomaly detection using self-supervised link prediction.
    Positives = True edges (Class 0: Normal)
    Negatives = Sampled non-edges (Class 1: Anomaly)
    Model outputs similarity (high for normal, low for anomaly).
    Anomaly Probability = 1.0 - sigmoid(similarity)
    """
    model.eval()
    with torch.no_grad():
        # Genuine edges (Normal, y=0)
        pos_scores, _, _ = model(u_idx, p_idx, t_target, edge_index, edge_attr, u_x, p_x)
        
        # Synthetic anomalies (Anomalous, y=1)
        neg_u, neg_p = sample_negatives_strict(u_idx, p_idx, t_target, edge_index, num_products)
        neg_u, neg_p = neg_u.to(device), neg_p.to(device)
        
        neg_scores, _, _ = model(neg_u, neg_p, t_target, edge_index, edge_attr, u_x, p_x)
        
        # Convert similarities to anomaly probabilities
        # High similarity -> low anomaly prob. Low similarity -> high anomaly prob.
        pos_anom_prob = 1.0 - torch.sigmoid(pos_scores).cpu().numpy()
        neg_anom_prob = 1.0 - torch.sigmoid(neg_scores).cpu().numpy()
        
        y_true = np.concatenate([np.zeros(len(pos_scores)), np.ones(len(neg_scores))])
        y_probs = np.concatenate([pos_anom_prob, neg_anom_prob])
        
        thresh = optimize_threshold(y_true, y_probs)
        y_pred = (y_probs >= thresh).astype(int)
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        auroc = roc_auc_score(y_true, y_probs)
        auprc = average_precision_score(y_true, y_probs)
        cm = confusion_matrix(y_true, y_pred).tolist()
        
    return {
        "F1": float(f1),
        "Precision": float(prec),
        "Recall": float(rec),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Threshold": float(thresh),
        "ConfusionMatrix": cm
    }

def main():
    print("Loading Amazon Graph (Subset: 20,000 records for validation)...")
    builder = AmazonGraphBuilder(os.path.join('.', 'data', 'amazon', 'raw'), max_records=20000)
    data = builder.build()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    edge_index = data['user', 'reviews', 'product'].edge_index.to(device)
    edge_attr = data['user', 'reviews', 'product'].edge_attr.to(device)
    u_x = data['user'].x.to(device)
    p_x = data['product'].x.to(device)
    
    train_mask = data['user', 'reviews', 'product'].train_mask
    val_mask = data['user', 'reviews', 'product'].val_mask
    test_mask = data['user', 'reviews', 'product'].test_mask
    num_products = data['product'].num_nodes
    
    # We will train in small batches
    train_idx = torch.nonzero(train_mask).view(-1)
    val_idx = torch.nonzero(val_mask).view(-1)
    test_idx = torch.nonzero(test_mask).view(-1)
    
    print(f"Nodes: {data['user'].num_nodes} Users, {data['product'].num_nodes} Products")
    print(f"Edges: {len(train_idx)} Train, {len(val_idx)} Val, {len(test_idx)} Test")
    
    model = HybridAmazonModel(in_channels=16, hidden_channels=16, use_time=True, use_decay=True, alpha=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    batch_size = 512
    best_val_f1 = 0
    best_state = None
    
    print("\n--- Training Amazon TGAT ---")
    for epoch in range(10):
        model.train()
        total_loss = 0
        
        # Shuffle train
        perm = torch.randperm(len(train_idx))
        
        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[perm[i:i+batch_size]]
            u_idx = edge_index[0, batch_idx]
            p_idx = edge_index[1, batch_idx]
            t_target = edge_attr[batch_idx, 0]
            
            neg_u, neg_p = sample_negatives_strict(u_idx, p_idx, t_target, edge_index, num_products)
            neg_u, neg_p = neg_u.to(device), neg_p.to(device)
            
            optimizer.zero_grad()
            loss, _, _ = model.compute_loss(u_idx, p_idx, neg_u, neg_p, t_target, edge_index, edge_attr, u_x, p_x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        val_u_idx = edge_index[0, val_idx[:1000]] # Subsample val for speed
        val_p_idx = edge_index[1, val_idx[:1000]]
        val_t = edge_attr[val_idx[:1000], 0]
        
        metrics = evaluate_anomalies(model, val_u_idx, val_p_idx, val_t, edge_index, edge_attr, u_x, p_x, num_products, device)
        print(f"Epoch {epoch+1} | Loss: {total_loss/(len(train_idx)/batch_size):.4f} | Val Anomaly F1: {metrics['F1']:.4f} (AUROC: {metrics['AUROC']:.4f})")
        
        if metrics['F1'] > best_val_f1:
            best_val_f1 = metrics['F1']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    print("\n--- Final Test Evaluation ---")
    model.load_state_dict(best_state)
    
    test_u_idx = edge_index[0, test_idx[:2000]] # Subsample test for speed
    test_p_idx = edge_index[1, test_idx[:2000]]
    test_t = edge_attr[test_idx[:2000], 0]
    
    test_metrics = evaluate_anomalies(model, test_u_idx, test_p_idx, test_t, edge_index, edge_attr, u_x, p_x, num_products, device)
    print(f"Test F1: {test_metrics['F1']:.4f}")
    
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    with open('reports/amazon_tgat_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
        
    torch.save(best_state, 'models/amazon_tgat.pt')
    print("Saved models/amazon_tgat.pt and reports/amazon_tgat_results.json")
    
    # Save a CSV for the dashboard
    import csv
    with open('reports/amazon_tgat_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for k, v in test_metrics.items():
            if k != 'ConfusionMatrix':
                writer.writerow([k, v])

if __name__ == '__main__':
    main()
