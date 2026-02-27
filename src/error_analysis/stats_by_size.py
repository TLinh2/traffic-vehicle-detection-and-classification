# error_analysis/stats_by_size.py
import pandas as pd
from collections import Counter

def summarize_by_size(all_results, out_csv):
    counter = Counter()

    for res in all_results:
        for pred, gt, _ in res["correct"]:
            counter[(gt["size"], "correct")] += 1

        for pred, gt, _ in res["wrong_class"]:
            counter[(gt["size"], "wrong_class")] += 1

        for pred, _ in res["low_iou"]:
            counter[(pred["size"], "low_iou")] += 1

        for gt in res["false_negative"]:
            counter[(gt["size"], "false_negative")] += 1

    rows = [
        {"size": k[0], "error_type": k[1], "count": v}
        for k, v in counter.items()
    ]

    pd.DataFrame(rows).to_csv(out_csv, index=False)
