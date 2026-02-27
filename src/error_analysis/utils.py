# error_analysis/utils.py
from pathlib import Path

def load_yolo_labels(label_path, with_conf=False):
    """
    Return list of dict:
    {
        "cls": int,
        "bbox": [x1, y1, x2, y2],   # normalized
        "conf": float (optional),
        "area": float,
        "size": str   # small | medium | large
    }
    """
    labels = []
    label_path = Path(label_path)

    if not label_path.exists():
        return labels

    with open(label_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5:
                continue

            cls = int(parts[0])
            x, y, w, h = parts[1:5]

            # YOLO xywh -> xyxy (normalized)
            x1, y1 = x - w / 2, y - h / 2
            x2, y2 = x + w / 2, y + h / 2

            area = w * h
            if area < 0.02:
                size = "small"
            elif area < 0.08:
                size = "medium"
            else:
                size = "large"

            item = {
                "cls": cls,                    
                "bbox": [x1, y1, x2, y2],
                "area": area,
                "size": size
            }

            if with_conf and len(parts) >= 6:
                item["conf"] = parts[5]

            labels.append(item)

    return labels


def iou(box1, box2):
    """
    box format: [x1, y1, x2, y2] (normalized)
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    ix1, iy1 = max(x1, x1g), max(y1, y1g)
    ix2, iy2 = min(x2, x2g), min(y2, y2g)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (
        (x2 - x1) * (y2 - y1)
        + (x2g - x1g) * (y2g - y1g)
        - inter
    )

    return inter / union if union > 0 else 0.0
