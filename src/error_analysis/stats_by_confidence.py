# error_analysis/stats_by_confidence.py
import pandas as pd

def summarize_by_confidence(all_results, out_csv):
    rows = []

    for res in all_results:
        for pred in res["false_positive"]:
            rows.append({
                "type": "false_positive",
                "conf": pred["conf"]
            })

        for pred, gt, _ in res["wrong_class"]:
            rows.append({
                "type": "wrong_class",
                "conf": pred["conf"]
            })

        for pred, _ in res["low_iou"]:
            rows.append({
                "type": "low_iou",
                "conf": pred["conf"]
            })

        for pred, gt, _ in res["correct"]:
            rows.append({
                "type": "correct",
                "conf": pred["conf"]
            })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
