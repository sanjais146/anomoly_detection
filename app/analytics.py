import os
import json
import datetime
from collections import defaultdict

_analytics_cache = None
_last_mtime = 0

def get_amazon_analytics():
    global _analytics_cache, _last_mtime
    sample_file = "data/amazon/raw/sample_reviews.json"
    results_file = "reports/amazon_tgat_results.json"
    
    current_mtime = os.path.getmtime(sample_file) if os.path.exists(sample_file) else 0
    if _analytics_cache is not None and current_mtime == _last_mtime:
        return _analytics_cache
    
    analytics = {
        "dataset_stats": {
            "users": 19236,
            "products": 2012,
            "interactions": 20000,
            "description": "Amazon Electronics Reviews (Complete 20,000 Interaction Graph)"
        },
        "model_performance": {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tau": 0.431,
            "half_life_days": 1.6
        },
        "sample_graph": [],
        "rating_distribution": [0, 0, 0, 0, 0],
        "timeline": []
    }
    
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            res = json.load(f)
            analytics["model_performance"]["f1"] = res.get("F1", 0)
            analytics["model_performance"]["precision"] = res.get("Precision", 0)
            analytics["model_performance"]["recall"] = res.get("Recall", 0)
            
    if os.path.exists(sample_file):
        timeline_counts = defaultdict(int)
        unique_users = set()
        unique_prods = set()
        total_interactions = 0
        
        # Product review collector to build an interconnected sample graph (top products)
        product_edges = defaultdict(list)
        
        with open(sample_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    rating = float(data.get("overall", 5.0))
                    idx = min(int(rating) - 1, 4)
                    if idx >= 0:
                        analytics["rating_distribution"][idx] += 1
                    
                    u = data.get("reviewerID", "")
                    p = data.get("asin", "")
                    t = data.get("unixReviewTime", 0)
                    
                    if u:
                        unique_users.add(u)
                    if p:
                        unique_prods.add(p)
                    total_interactions += 1
                    
                    if t > 0:
                        date_str = datetime.datetime.fromtimestamp(t).strftime('%Y-%m')
                        timeline_counts[date_str] += 1
                        
                    # Collect edges for sample graph selection
                    if p and len(product_edges[p]) < 30:
                        product_edges[p].append({
                            "reviewerID": u,
                            "asin": p,
                            "unixReviewTime": t,
                            "overall": rating
                        })
                except Exception:
                    pass
        
        # Update real dataset stats if counted
        if total_interactions > 0:
            analytics["dataset_stats"]["users"] = len(unique_users)
            analytics["dataset_stats"]["products"] = len(unique_prods)
            analytics["dataset_stats"]["interactions"] = total_interactions
        
        # Sort timeline chronologically
        sorted_dates = sorted(timeline_counts.keys())
        analytics["timeline"] = [{"date": d, "count": timeline_counts[d]} for d in sorted_dates]
        
        # Build balanced, dense sample graph for Vis.js UI (up to 350-400 edges across top products)
        # Sort products by edge count so the graph has well-defined clusters
        sorted_prods = sorted(product_edges.keys(), key=lambda k: len(product_edges[k]), reverse=True)
        selected_graph = []
        for p in sorted_prods[:25]:  # Top 25 products
            selected_graph.extend(product_edges[p][:16])
            if len(selected_graph) >= 380:
                break
        analytics["sample_graph"] = selected_graph[:400]
        
        _analytics_cache = analytics
        _last_mtime = current_mtime
        
    return analytics
