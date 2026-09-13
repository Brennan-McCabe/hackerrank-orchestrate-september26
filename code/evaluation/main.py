import pandas as pd
import os

def evaluate():
    print("--- Local Validation Engine ---")
    base_dir = os.path.dirname(__file__)
    gt_path = os.path.join(base_dir, "../../dataset/sample_requests.csv")
    pred_path = os.path.join(base_dir, "../../output.csv")
    
    if not os.path.exists(gt_path) or not os.path.exists(pred_path): return
        
    merged = pd.merge(pd.read_csv(gt_path), pd.read_csv(pred_path), on="request_id", suffixes=("_gt", "_pred"))
    if not merged.empty and 'affordability_status_gt' in merged.columns:
        acc = (merged['affordability_status_gt'].str.lower() == merged['affordability_status_pred'].str.lower()).mean()
        print(f"🎯 Affordability Status Accuracy on Samples: {acc * 100:.2f}%")

if __name__ == "__main__":
    evaluate()
