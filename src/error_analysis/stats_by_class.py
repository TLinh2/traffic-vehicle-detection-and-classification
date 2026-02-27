# error_analysis/stats_by_class.py
import pandas as pd
from collections import Counter

def summarize_by_class(all_results, out_csv):
    wrong_class = Counter()
    false_negative = Counter()

    for res in all_results:
        for pred, gt, _ in res["wrong_class"]:
            wrong_class[(gt["cls"], pred["cls"])] += 1  # ✅ Đổi thành "cls"

        for gt in res["false_negative"]:
            false_negative[gt["cls"]] += 1  # ✅ Đổi thành "cls"

    rows = []
    for (gt_cls, pred_cls), cnt in wrong_class.items():
        rows.append({
            "gt_class": gt_cls,
            "pred_class": pred_cls,
            "count": cnt,
            "type": "wrong_class"
        })

    for cls, cnt in false_negative.items():
        rows.append({
            "gt_class": cls,
            "pred_class": None,
            "count": cnt,
            "type": "false_negative"
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)