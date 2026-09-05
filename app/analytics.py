import os
import json
import ast

def get_amazon_analytics():
    sample_file = "data/amazon/raw/sample_reviews.json"
    results_file = "reports/amazon_tgat_results.json"
    
    analytics = {
        "dataset_stats": {
            "users": 19236,
            "products": 2012,
            "interactions": 20000,
            "description": "Amazon Electronics Reviews (Subsampled from 1.77 GB)"
        },
        "model_performance": {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tau": 0.431,
            "half_life_days": 1.6
        },
        "sample_graph": [],
        "rating_distribution": [0,0,0,0,0],
    }
    
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            res = json.load(f)
            analytics["model_performance"]["f1"] = res.get("F1", 0)
            analytics["model_performance"]["precision"] = res.get("Precision", 0)
            analytics["model_performance"]["recall"] = res.get("Recall", 0)
            
    if os.path.exists(sample_file):
        with open(sample_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = ast.literal_eval(line)
                    rating = float(data.get("overall", 5.0))
                    idx = min(int(rating) - 1, 4)
                    analytics["rating_distribution"][idx] += 1
                    
                    if len(analytics["sample_graph"]) < 100:
                        analytics["sample_graph"].append({
                            "reviewerID": data.get("reviewerID"),
                            "asin": data.get("asin"),
                            "unixReviewTime": data.get("unixReviewTime", 0),
                            "overall": rating
                        })
                except:
                    pass
                    
        # Calculate real timeline bins (Group by Year-Month or Day)
        import datetime
        timeline_counts = {}
        for edge in analytics["sample_graph"]:
            t = edge.get("unixReviewTime", 0)
            if t > 0:
                # Format to YYYY-MM
                date_str = datetime.datetime.fromtimestamp(t).strftime('%Y-%m')
                timeline_counts[date_str] = timeline_counts.get(date_str, 0) + 1
        
        # Sort chronologically
        sorted_dates = sorted(timeline_counts.keys())
        analytics["timeline"] = [{"date": d, "count": timeline_counts[d]} for d in sorted_dates]
        
    return analytics
