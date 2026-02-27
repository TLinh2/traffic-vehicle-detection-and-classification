# error_analysis/match_gt_pred.py
from collections import defaultdict
from src.error_analysis.utils import iou

def match_predictions(  
    gt_boxes,
    pred_boxes,
    iou_thresh=0.5
):
    """
    Match GT và prediction cho 1 ảnh
    """
    matched_gt = set()
    results = {
        "correct": [],
        "wrong_class": [],
        "low_iou": [],
        "false_positive": [],
        "false_negative": []
    }

    for pred in sorted(pred_boxes, key=lambda x: x.get("conf", 1), reverse=True):
        best_iou = 0
        best_gt_idx = -1

        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou_val = iou(pred["bbox"], gt["bbox"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = i

        if best_iou >= iou_thresh:
            gt = gt_boxes[best_gt_idx]
            matched_gt.add(best_gt_idx)

            if pred["cls"] != gt["cls"]:
                results["wrong_class"].append((pred, gt, best_iou))
            else:
                results["correct"].append((pred, gt, best_iou))
        else:
            if best_iou > 0:
                results["low_iou"].append((pred, best_iou))
            else:
                results["false_positive"].append(pred)

    for i, gt in enumerate(gt_boxes):
        if i not in matched_gt:
            results["false_negative"].append(gt)

    return results
