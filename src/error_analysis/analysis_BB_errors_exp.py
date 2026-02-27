import argparse
from pathlib import Path
import json

import pandas as pd
import plotly.express as px


# ===== Class ID to Name mapping =====
CLASS_NAMES = {
    0: "bicycle",
    1: "bus",
    2: "car",
    3: "motorcycle",
}

# =========================================================
# Utils
# =========================================================
def yolo_to_xyxy(cx, cy, w, h):
    return (
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2,
    )


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)


def load_yolo_labels(path: Path, has_conf=False):
    boxes = []
    if not path.exists():
        return boxes

    with open(path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if has_conf and len(parts) < 6:
                continue
            if not has_conf and len(parts) < 5:
                continue

            cls = int(parts[0])
            cx, cy, w, h = parts[1:5]
            conf = parts[5] if has_conf else None

            boxes.append({
                "cls": cls,
                "w": w,
                "h": h,
                "bbox": yolo_to_xyxy(cx, cy, w, h),
                "conf": conf
            })
    return boxes


# =========================================================
# Matching
# =========================================================
def match_image(preds, gts, iou_thr):
    used_gt = set()

    fp, fn, wc, li = [], [], [], []

    preds = sorted(preds, key=lambda x: x.get("conf", 1), reverse=True)

    for p in preds:
        best_iou = 0
        best_j = None

        for j, g in enumerate(gts):
            if j in used_gt:
                continue
            i = iou(p["bbox"], g["bbox"])
            if i > best_iou:
                best_iou = i
                best_j = j

        if best_iou == 0:
            fp.append(p)
        elif best_iou < iou_thr:
            li.append(p)
            used_gt.add(best_j)
        else:
            g = gts[best_j]
            used_gt.add(best_j)
            if p["cls"] != g["cls"]:
                wc.append(p)

    for j, g in enumerate(gts):
        if j not in used_gt:
            fn.append(g)

    return fp, fn, wc, li


# =========================================================
# Plot (Interactive)
# =========================================================
def plot_interactive(df: pd.DataFrame, out_html: Path, title: str, bins=10000):
    if df.empty:
        return

    df = df[(df["area_ratio"] > 0) & (df["area_ratio"] <= 1)]

    fig = px.histogram(
        df,
        x="area_ratio",
        color="class",
        opacity=0.75,
        barmode="stack",
        histnorm=None
    )

    fig.update_traces(
        xbins=dict(start=0.0, end=1.0, size=1.0 / bins)
    )

    fig.update_layout(
        title=title,
        xaxis_title="bbox_area / image_area",
        yaxis_title="Number of bounding boxes",
        bargap=0,
        template="plotly_white",
    )

    fig.update_xaxes(rangeslider_visible=True)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html)
    print(f"[SAVED] {out_html}")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    errors = {
        "false_positive": [],
        "false_negative": [],
        "wrong_class": [],
        "low_iou": [],
    }

    for pred_file in pred_dir.glob("*.txt"):
        name = pred_file.stem
        gt_file = gt_dir / f"{name}.txt"

        preds = load_yolo_labels(pred_file, has_conf=True)
        gts = load_yolo_labels(gt_file, has_conf=False)

        fp, fn, wc, li = match_image(preds, gts, args.iou_thr)

        for p in fp:
            errors["false_positive"].append({
                "area_ratio": p["w"] * p["h"],
                "class": CLASS_NAMES.get(p["cls"], f"class_{p['cls']}")
            })

        for p in wc:
            errors["wrong_class"].append({
                "area_ratio": p["w"] * p["h"],
                "class": CLASS_NAMES.get(p["cls"], f"class_{p['cls']}")
            })

        for p in li:
            errors["low_iou"].append({
                "area_ratio": p["w"] * p["h"],
                "class": CLASS_NAMES.get(p["cls"], f"class_{p['cls']}")
            })

        for g in fn:
            errors["false_negative"].append({
                "area_ratio": g["w"] * g["h"],
                "class": CLASS_NAMES.get(g["cls"], f"class_{g['cls']}")
            })

    # Save JSON + Plot
    for k, items in errors.items():
        if not items:
            continue

        df = pd.DataFrame(items)

        json_path = out_dir / f"{k}.json"
        with open(json_path, "w") as f:
            json.dump(items, f, indent=2)

        plot_interactive(
            df,
            out_dir / f"{k}_area_ratio.html",
            title=f"{k.replace('_', ' ').title()} Area Ratio"
        )

    print("[DONE] Interactive error analysis completed")


if __name__ == "__main__":
    main()
