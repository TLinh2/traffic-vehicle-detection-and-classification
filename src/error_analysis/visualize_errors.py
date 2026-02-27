# error_analysis/visualize_errors.py
import cv2
from pathlib import Path

COLORS = {
    "correct": (0, 255, 0),        # Green
    "wrong_class": (0, 165, 255),  # Orange
    "low_iou": (0, 255, 255),      # Yellow
    "false_positive": (0, 0, 255), # Red
    "false_negative": (255, 0, 0)  # Blue
}

def draw_box(img, bbox, color, label):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    x1, y1 = int(x1 * w), int(y1 * h)
    x2, y2 = int(x2 * w), int(y2 * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

def visualize(img_path, boxes, save_path, err_type):
    img = cv2.imread(str(img_path))
    if img is None:
        return

    color = COLORS.get(err_type, (255, 255, 255))
    
    # ✅ Xử lý từng loại lỗi
    if err_type == "false_positive":
        for pred in boxes:
            label = f"FP cls={pred['cls']}"
            draw_box(img, pred["bbox"], color, label)
            
    elif err_type == "false_negative":
        for gt in boxes:
            label = f"FN cls={gt['cls']}"
            draw_box(img, gt["bbox"], color, label)
            
    elif err_type == "correct":
        for pred, gt, iou in boxes:
            label = f"OK cls={pred['cls']} iou={iou:.2f}"
            draw_box(img, pred["bbox"], color, label)
    
    elif err_type == "wrong_class":  # ✅ Thêm xử lý wrong_class
        for pred, gt, iou in boxes:
            label = f"WC p={pred['cls']} gt={gt['cls']}"
            draw_box(img, pred["bbox"], color, label)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)