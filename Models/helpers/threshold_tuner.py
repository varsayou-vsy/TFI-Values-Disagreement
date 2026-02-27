
import numpy as np
import json
from sklearn.metrics import f1_score
from scipy.special import expit

def tune_thresholds(preds, labels, values, steps=100):
    thresholds = np.linspace(0.05, 0.95, steps)
    value_names = sorted(set(values))
    best_thresholds = {}
    per_value_f1 = {}

    for value in value_names:
        mask = np.array(values) == value
        y_true = np.array(labels)[mask]
        y_probs = expit(np.array(preds)[mask])
        best_f1 = 0
        best_thresh = 0.5
        for t in thresholds:
            y_pred = (y_probs > t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        best_thresholds[value] = round(best_thresh, 3)
        per_value_f1[value] = round(best_f1, 3)

    return best_thresholds, per_value_f1

if __name__ == "__main__":
    with open("PATH_TO_YOUR_RESULTS.json", "r") as f:
        data = json.load(f)
    preds = data["preds"]
    labels = data["true"]
    values = data["cli_args"].get("value_list")  # replace if needed

    thresholds, f1s = tune_thresholds(preds, labels, values)
    with open("thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print("Optimal thresholds saved to thresholds.json")
