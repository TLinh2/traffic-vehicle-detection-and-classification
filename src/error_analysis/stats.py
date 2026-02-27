# error_analysis/stats.py
import pandas as pd
from collections import Counter

def summarize(all_results, out_csv):
    counter = Counter()

    for res in all_results:
        counter["correct"] += len(res.get("correct", []))
        counter["wrong_class"] += len(res.get("wrong_class", []))
        counter["low_iou"] += len(res.get("low_iou", []))
        counter["false_positive"] += len(res.get("false_positive", []))
        counter["false_negative"] += len(res.get("false_negative", []))

    # Tổng số prediction & GT
    counter["total_pred"] = (
        counter["correct"]
        + counter["wrong_class"]
        + counter["low_iou"]
        + counter["false_positive"]
    )
    counter["total_gt"] = (
        counter["correct"]
        + counter["wrong_class"]
        + counter["false_negative"]
    )

    # Precision / Recall (optional nhưng rất nên có)
    counter["precision"] = (
        counter["correct"] / counter["total_pred"]
        if counter["total_pred"] > 0 else 0
    )
    counter["recall"] = (
        counter["correct"] / counter["total_gt"]
        if counter["total_gt"] > 0 else 0
    )

    df = pd.DataFrame([counter])
    df.to_csv(out_csv, index=False)
