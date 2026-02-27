# error_analysis/run.py
from pathlib import Path

from src.error_analysis.utils import load_yolo_labels
from src.error_analysis.match_gt_pred import match_predictions
from src.error_analysis.visualize_errors import visualize
from src.error_analysis.stats import summarize
from src.error_analysis.stats_by_class import summarize_by_class
from src.error_analysis.stats_by_size import summarize_by_size
from src.error_analysis.stats_by_confidence import summarize_by_confidence


def run(gt_dir, pred_dir, img_dir, out_dir):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)

    all_results = []

    for gt_file in gt_dir.glob("*.txt"):
        image_id = gt_file.stem

        # --- load labels ---
        gt_boxes = load_yolo_labels(gt_file)

        pred_file = pred_dir / f"{image_id}.txt"
        pred_boxes = (
            load_yolo_labels(pred_file, with_conf=True)
            if pred_file.exists()
            else []
        )

        # --- match ---
        match_res = match_predictions(gt_boxes, pred_boxes)
        all_results.append(match_res)

        # --- image path ---
        img_path = img_dir / f"{image_id}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{image_id}.png"
        if not img_path.exists():
            continue

        # --- visualize each error type ---
        error_map = {
            "false_positive": "false_positive",
            "false_negative": "false_negative",
            "correct": "correct",
            "wrong_class": "wrong_class"  # ✅ Thêm dòng này
        }

        for err_type, key in error_map.items():
            items = match_res[key]

            if len(items) == 0:
                continue

            save_dir = out_dir / "visuals" / err_type
            save_dir.mkdir(parents=True, exist_ok=True)

            visualize(
                img_path=img_path,
                boxes=items,
                save_path=save_dir / f"{image_id}.jpg",
                err_type=err_type
            )

    # --- statistics ---
    summarize(all_results, out_dir / "summary.csv")
    summarize_by_class(all_results, out_dir / "by_class.csv")
    summarize_by_size(all_results, out_dir / "by_size.csv")
    summarize_by_confidence(all_results, out_dir / "by_confidence.csv")

    print(f"[DONE] Error analysis saved to {out_dir}")
