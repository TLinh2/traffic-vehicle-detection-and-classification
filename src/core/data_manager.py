# src/core/data_manager.py
import os
import cv2
from tqdm import tqdm
import yaml
from collections import Counter, defaultdict
from pathlib import Path
import shutil
from PIL import Image
import numpy as np

# @@@ Phân tích tập train của dataset @@@
# -- Lưu ý : Dataset tải về nên ở định dạng Ultralytics Yolo detection -> trong file data.yaml đổi biến 
## train: train.txt -> train: images/train
 
class DataAnalyzer:
    def __init__(self, data_yaml_path: str):
        self.data_yaml_path = Path(data_yaml_path)
        with open(self.data_yaml_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.dataset_root = self.data_yaml_path.parent
        self.names = self.cfg.get("names", {})

        # train path (bắt buộc phải có)
        train_cfg = self.cfg.get("train")
        if train_cfg:
            self.train_path = (self.dataset_root / train_cfg.format(path=self.dataset_root)).resolve()
        else:
            self.train_path = None

        # val path (có thể không có khi chưa chia)
        val_cfg = self.cfg.get("val")
        if val_cfg:
            self.val_path = (self.dataset_root / val_cfg.format(path=self.dataset_root)).resolve()
        else:
            self.val_path = None

        # test path (có thể không có khi chưa chia)
        test_cfg = self.cfg.get("test")
        if test_cfg:
            self.test_path = (self.dataset_root / test_cfg.format(path=self.dataset_root)).resolve()
        else:
            self.test_path = None

    def _make_path(self, path_cfg):
        if path_cfg:
            return (self.dataset_root / path_cfg.format(path=self.dataset_root)).resolve()
        return None

    def _count_labels(self, label_dir):
        label_dir = Path(label_dir)
        if not label_dir.exists():
            return Counter()
        counts = Counter()
        for label_file in label_dir.rglob("*.txt"):
            with open(label_file, "r") as f:
                lines = f.readlines()
            classes = [int(line.split()[0]) for line in lines if line.strip()]
            counts.update(classes)
        return counts

    def _count_images(self, image_dir):
        image_dir = Path(image_dir)
        if not image_dir.exists():
            return 0
        return len(list(image_dir.rglob("*.jpg"))) + len(list(image_dir.rglob("*.png")))

    def analyze(self, split="train"):
        splits = []
        if split == "all":
            splits = [("train", self.train_path), ("val", self.val_path), ("test", self.test_path)]
        elif split in ["train", "val", "test"]:
            path = getattr(self, f"{split}_path")
            splits = [(split, path)]
        else:
            raise ValueError(f"Invalid split: {split}. Choose from 'train', 'val', 'test', 'all'.")

        result = {}
        for name, path in splits:
            if path is None:
                print(f"[WARN] Split '{name}' not found, skipping.")
                continue

            label_dir = self.dataset_root / "labels" / name
            if not label_dir.exists():
                label_dir = self.dataset_root / "labels"
            if not label_dir.exists():
                label_dir = path.parent / "labels"
            counts = self._count_labels(label_dir)
            total_images = self._count_images(path)
            total_labels = sum(counts.values())
            per_class = {self.names[c]: counts.get(c, 0) for c in range(len(self.names))}

            result[name] = {
                "dataset": self.dataset_root.name,
                "num_classes": len(self.names),
                "classes": self.names,
                "total_images": total_images,
                "total_labels": total_labels,
                "per_class": per_class
            }

        return result
    
    # ===============================
    # Hàm tính per-class AP
    # ===============================
    def compute_per_class_ap(self, pred_dir, split="val", iou_thresh=0.5):
        """
        Tính AP cho từng class trong tập split (train/val/test).
        pred_dir: thư mục chứa nhãn dự đoán (YOLO format)
        split: chọn "train", "val", hoặc "test"
        """
        pred_dir = Path(pred_dir)
        if not pred_dir.exists():
            raise FileNotFoundError(f"[ERROR] Prediction directory not found: {pred_dir}")

        # Xác định thư mục chứa ground truth
        if split == "train":
            gt_dir = self.train_path / "labels"
        elif split == "val":
            gt_dir = self.val_path / "labels"
        elif split == "test":
            gt_dir = self.test_path / "labels"
        else:
            raise ValueError(f"[ERROR] Unknown split: {split}")

        if not gt_dir.exists():
            raise FileNotFoundError(f"[ERROR] Ground truth directory not found: {gt_dir}")

        # Đọc ground truth
        ground_truths = []
        for label_file in gt_dir.glob("*.txt"):
            image_id = label_file.stem
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    # YOLO format -> convert to (x1, y1, x2, y2)
                    x1, y1 = x - w / 2, y - h / 2
                    x2, y2 = x + w / 2, y + h / 2
                    ground_truths.append({
                        "image_id": image_id,
                        "class_id": cls_id,
                        "bbox": [x1, y1, x2, y2],
                    })

        # Đọc prediction
        detections = []
        for label_file in pred_dir.glob("*.txt"):
            image_id = label_file.stem
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    x, y, w, h, score = map(float, parts[1:])
                    x1, y1 = x - w / 2, y - h / 2
                    x2, y2 = x + w / 2, y + h / 2
                    detections.append({
                        "image_id": image_id,
                        "class_id": cls_id,
                        "bbox": [x1, y1, x2, y2],
                        "score": score,
                    })

        def iou(box1, box2):
            x1, y1, x2, y2 = box1
            x1g, y1g, x2g, y2g = box2
            inter_x1, inter_y1 = max(x1, x1g), max(y1, y1g)
            inter_x2, inter_y2 = min(x2, x2g), min(y2, y2g)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = ((x2 - x1) * (y2 - y1)) + ((x2g - x1g) * (y2g - y1g)) - inter_area
            return inter_area / union_area if union_area > 0 else 0.0

        def compute_ap(rec, prec):
            rec = np.concatenate(([0.0], rec, [1.0]))
            prec = np.concatenate(([0.0], prec, [0.0]))
            for i in range(prec.size - 1, 0, -1):
                prec[i - 1] = np.maximum(prec[i - 1], prec[i])
            idx = np.where(rec[1:] != rec[:-1])[0]
            return np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1])

        # Gom detection và gt theo class
        gt_by_class = defaultdict(list)
        dt_by_class = defaultdict(list)
        for gt in ground_truths:
            gt_by_class[gt["class_id"]].append(gt)
        for dt in detections:
            dt_by_class[dt["class_id"]].append(dt)

        ap_per_class = {}
        for cls_id in sorted(gt_by_class.keys()):
            gts = gt_by_class[cls_id]
            dts = sorted(dt_by_class[cls_id], key=lambda x: -x["score"])

            gt_used = defaultdict(lambda: [])
            for gt in gts:
                gt_used[gt["image_id"]].append(False)

            tp, fp = [], []
            for dt in dts:
                ious = []
                relevant_gts = [g for g in gts if g["image_id"] == dt["image_id"]]
                for g in relevant_gts:
                    ious.append(iou(dt["bbox"], g["bbox"]))
                if not ious:
                    fp.append(1)
                    tp.append(0)
                    continue
                best_iou_idx = np.argmax(ious)
                best_iou = ious[best_iou_idx]
                if best_iou >= iou_thresh:
                    gt_idx = list(gt_by_class[cls_id]).index(relevant_gts[best_iou_idx])
                    if not gt_used[dt["image_id"]][best_iou_idx]:
                        tp.append(1)
                        fp.append(0)
                        gt_used[dt["image_id"]][best_iou_idx] = True
                    else:
                        tp.append(0)
                        fp.append(1)
                else:
                    tp.append(0)
                    fp.append(1)

            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            npos = len(gts)
            rec = tp / npos if npos > 0 else np.zeros_like(tp)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = compute_ap(rec, prec)
            ap_per_class[cls_id] = ap

        # In kết quả
        print("\n===== Per-class AP =====")
        for cls_id, ap in ap_per_class.items():
            cls_name = self.names.get(cls_id, str(cls_id))
            print(f"{cls_name:15s}: {ap:.4f}")

        mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
        print(f"\nMean AP (mAP): {mAP:.4f}\n")

        return ap_per_class
    
class DataProcessor:
    def __init__(self, data_yaml_path: str):
        from core.data_manager import DataAnalyzer
        self.analyzer = DataAnalyzer(data_yaml_path)
        self.dataset_root = self.analyzer.dataset_root
    
    def filter_bboxes_by_area_ratio(
        self,
        min_ratio=0.0,
        max_ratio=1.0,
        classes=None,
        splits=("train", "val", "test"),
        new_dataset_root=None,
        remove_empty_images=True,
    ):
        """
        Lọc bounding box theo tỉ lệ:
            bbox_area / image_area ∈ [min_ratio, max_ratio]

        Args:
            min_ratio: ngưỡng dưới (vd: 1e-4)
            max_ratio: ngưỡng trên (vd: 1e-3)
            classes: list class name hoặc id cần lọc (None = tất cả)
            splits: các split cần xử lý
            new_dataset_root: nếu có → copy dataset sang thư mục mới
            remove_empty_images: xóa ảnh nếu không còn bbox
        """

        import shutil
        import cv2
        import numpy as np
        from pathlib import Path
        from tqdm import tqdm

        # -------------------------------------------------
        # Chuẩn bị dataset
        # -------------------------------------------------
        if new_dataset_root:
            new_dataset_root = Path(new_dataset_root)
            if new_dataset_root.exists():
                raise FileExistsError(f"{new_dataset_root} already exists")
            shutil.copytree(self.dataset_root, new_dataset_root)
            base_dir = new_dataset_root
        else:
            base_dir = self.dataset_root

        # -------------------------------------------------
        # Chuẩn hóa class filter
        # -------------------------------------------------
        if classes is not None:
            class_ids = set()
            name_to_id = {v: k for k, v in self.analyzer.names.items()}
            for c in classes:
                if isinstance(c, str):
                    cid = name_to_id.get(c)
                    if cid is None:
                        raise ValueError(f"Unknown class name: {c}")
                    class_ids.add(cid)
                else:
                    class_ids.add(int(c))
        else:
            class_ids = None  # lọc tất cả

        removed = 0

        # -------------------------------------------------
        # Lọc bbox
        # -------------------------------------------------
        for split in splits:
            label_dir = base_dir / "labels" / split
            img_dir = base_dir / "images" / split
            if not label_dir.exists():
                continue

            for label_file in tqdm(label_dir.glob("*.txt"), desc=f"Filtering {split}"):
                # tìm ảnh
                img_path = None
                for ext in (".jpg", ".png"):
                    p = img_dir / (label_file.stem + ext)
                    if p.exists():
                        img_path = p
                        break
                if img_path is None:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                img_area = h * w

                kept = []

                with open(label_file) as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    cls, x, y, bw, bh = map(float, parts)
                    cls = int(cls)

                    # nếu lọc theo class
                    if class_ids is not None and cls not in class_ids:
                        kept.append(line)
                        continue

                    bbox_area = (bw * w) * (bh * h)
                    area_ratio = bbox_area / img_area

                    # nếu bbox nằm trong khoảng CẦN XOÁ
                    if min_ratio <= area_ratio <= max_ratio:
                        removed += 1
                    else:
                        kept.append(line)

                if len(kept) == 0:
                    label_file.unlink()
                    if remove_empty_images and img_path.exists():
                        img_path.unlink()
                else:
                    with open(label_file, "w") as f:
                        f.writelines(kept)

        print(f"\n[DONE] Removed {removed} bboxes")
        print(f"[INFO] Area ratio kept in [{min_ratio:.1e}, {max_ratio:.1e}]")
        if class_ids is not None:
            print(f"[INFO] Applied to classes: {sorted(class_ids)}")


    def compute_bbox_dimensions(self, subsets):
        """
        Trả về danh sách các bounding box dạng (width_px, height_px, class_name)
        """
        all_dims = []
        

        base_dir = self.dataset_root
        with open(base_dir / "data.yaml", "r") as f:
            data = yaml.safe_load(f)

        for subset in subsets:
            if subset not in data:
                continue

            label_dir = base_dir / "labels" / subset
            image_dir = base_dir / "images" / subset

            if not label_dir.exists():
                continue

            for label_file in label_dir.glob("*.txt"):
                img_file = image_dir / (label_file.stem + ".jpg")
                if not img_file.exists():
                    img_file = image_dir / (label_file.stem + ".png")
                if not img_file.exists():
                    continue

                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                h, w = img.shape[:2]

                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls_id, x_center, y_center, bw, bh = map(float, parts)
                        w_px = round(bw * w, 2)
                        h_px = round(bh * h, 2)
                        area = w_px * h_px
                        img_area = w * h
                        
                        class_name = self.analyzer.names.get(int(cls_id), str(cls_id))
                        all_dims.append((w_px, h_px, area, img_area, class_name))

        return all_dims

    def augment_small_objects(
        self,
        min_px=10,
        max_px=20,
        scale_factor=1.5,
        splits=["train"],
        new_dataset_root=None,
    ):
        """
        Augment (tăng kích thước hoặc crop vùng chứa) các object có kích thước trong khoảng [min_px, max_px].
        Lưu kết quả vào thư mục mới (new_dataset_root).

        Args:
            min_px, max_px: Khoảng kích thước đối tượng theo pixel.
            scale_factor: Mức độ phóng to (ví dụ 1.5 nghĩa là 150%).
            splits: Các tập cần xử lý, mặc định là ["train"].
            new_dataset_root: Đường dẫn dataset mới để lưu kết quả augment.
        """
        if new_dataset_root is None:
            raise ValueError("Bạn cần truyền new_dataset_root để lưu dataset mới.")

        new_dataset_root = Path(new_dataset_root)
        if new_dataset_root.exists():
            raise FileExistsError(f"{new_dataset_root} đã tồn tại. Hãy chọn tên khác.")

        print(f"[INFO] Copying dataset structure to {new_dataset_root}...")
        shutil.copytree(self.dataset_root, new_dataset_root)

        print(f"[INFO] Augmenting small objects ({min_px}px - {max_px}px)...")

        for split in splits:
            old_label_dir = self.dataset_root / "labels" / split
            old_img_dir = self.dataset_root / "images" / split
            new_label_dir = new_dataset_root / "labels" / split
            new_img_dir = new_dataset_root / "images" / split

            if not old_label_dir.exists():
                print(f"[WARN] Split '{split}' không tồn tại, bỏ qua.")
                continue

            for label_file in tqdm(list(old_label_dir.glob("*.txt")), desc=f"Processing {split}"):
                img_path = None
                for ext in [".jpg", ".png"]:
                    candidate = old_img_dir / (label_file.stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
                if img_path is None:
                    continue

                img = cv2.imread(str(img_path))
                h_img, w_img = img.shape[:2]

                with open(label_file, "r") as f:
                    lines = f.readlines()

                new_lines = []
                aug_needed = False

                for line in lines:
                    cls, x, y, w, h = map(float, line.strip().split())
                    w_px = w * w_img
                    h_px = h * h_img

                    if min_px <= min(w_px, h_px) <= max_px:
                        # Object nhỏ → augment
                        aug_needed = True
                        x_c, y_c = int(x * w_img), int(y * h_img)
                        w_new = int(w_px * scale_factor)
                        h_new = int(h_px * scale_factor)

                        # crop vùng quanh object và resize
                        x1 = max(x_c - w_new // 2, 0)
                        y1 = max(y_c - h_new // 2, 0)
                        x2 = min(x1 + w_new, w_img)
                        y2 = min(y1 + h_new, h_img)
                        obj_crop = img[y1:y2, x1:x2]
                        # resize nhưng vẫn phải đảm bảo không vượt quá biên
                        obj_crop = cv2.resize(obj_crop, (x2 - x1, y2 - y1))

                        # dán lại (paste) vào ảnh gốc — nhưng đảm bảo kích thước khớp
                        h_crop, w_crop = obj_crop.shape[:2]
                        h_target, w_target = y2 - y1, x2 - x1
                        h_final = min(h_crop, h_target)
                        w_final = min(w_crop, w_target)

                        img[y1:y1 + h_final, x1:x1 + w_final] = obj_crop[:h_final, :w_final]

                    new_lines.append(line)

                new_img_path = new_img_dir / img_path.name
                new_label_path = new_label_dir / label_file.name

                if aug_needed:
                    cv2.imwrite(str(new_img_path), img)
                else:
                    # Nếu không augment thì copy ảnh cũ
                    shutil.copy2(img_path, new_img_path)

                with open(new_label_path, "w") as f:
                    f.writelines(new_lines)

        print(f"[DONE] Augmented dataset saved at: {new_dataset_root}")
    
    def compute_bbox_areas(self):
        """
        Tính diện tích (pixel²) của tất cả bounding box trong dataset (train/val/test).
        Trả về danh sách diện tích các bbox.
        """ 

        all_areas = []
        subsets = ["train", "val", "test"]

        # đọc lại data.yaml để lấy cấu trúc dữ liệu
        base_dir = self.dataset_root
        with open(base_dir / "data.yaml", "r") as f:
            data = yaml.safe_load(f)

        for subset in subsets:
            if subset not in data:
                continue
                        
            label_dir = base_dir / "labels" / subset
            image_dir = base_dir / "images" / subset

            print(f"Đang lấy labels từ {label_dir}")
            print(f"Đang lấy images từ {image_dir}")  

            if not label_dir.exists():
                print(f"[WARN] Split '{subset}' không tồn tại, bỏ qua.")
                continue

            print(f"[INFO] Processing {subset} labels from {label_dir}...")
            for label_file in tqdm(list(label_dir.glob("*.txt"))):
                img_file = image_dir / (label_file.stem + ".jpg")
                if not img_file.exists():
                    img_file = image_dir / (label_file.stem + ".png")
                if not img_file.exists():
                    continue

                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                h, w = img.shape[:2]

                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        _, x_center, y_center, bw, bh = map(float, parts)
                        bw_px = bw * w
                        bh_px = bh * h
                        area = bw_px * bh_px
                        all_areas.append(area)

        print(f"[INFO] Tổng số bounding box: {len(all_areas)}")
        return all_areas
    
    def filter_class(
        self,
        class_name: str,
        splits=["train", "val", "test"],
        new_dataset_root=None,
    ):
        """
        Lọc các bounding box thuộc class_name, tạo dataset mới.

        Args:
            class_name: Tên class cần lọc (ví dụ "unidentified")
            splits: Các split cần xử lý
            new_dataset_root: Đường dẫn dataset mới lưu kết quả
        """
        if new_dataset_root is None:
            raise ValueError("Bạn cần truyền new_dataset_root để lưu dataset mới.")

        new_dataset_root = Path(new_dataset_root)
        if new_dataset_root.exists():
            raise FileExistsError(f"{new_dataset_root} đã tồn tại. Hãy chọn tên khác.")
        
        # Lấy ID của class cần lọc
        target_ids = [k for k, v in self.analyzer.names.items() if v == class_name]

        print(f"[INFO] Copying dataset structure to {new_dataset_root}...")
        shutil.copytree(self.dataset_root, new_dataset_root)
        
        if not target_ids:
            print(f"[WARN] Không tìm thấy class {class_name} trong dataset.")
            return

        target_id = target_ids[0]

        for split in splits:
            old_label_dir = self.dataset_root / "labels" / split
            old_img_dir = self.dataset_root / "images" / split
            new_label_dir = new_dataset_root / "labels" / split
            new_img_dir = new_dataset_root / "images" / split

            if not old_label_dir.exists():
                print(f"[WARN] Split '{split}' không tồn tại, bỏ qua.")
                continue

            for label_file in old_label_dir.glob("*.txt"):
                img_path = None
                for ext in [".jpg", ".png"]:
                    candidate = old_img_dir / (label_file.stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
                if img_path is None:
                    continue

                # đọc label
                with open(label_file, "r") as f:
                    lines = f.readlines()

                # lọc các line không phải class target
                filtered_lines = [line for line in lines if int(line.strip().split()[0]) != target_id]

                # nếu không còn bbox nào → xóa ảnh
                if len(filtered_lines) == 0:
                    continue  # ảnh sẽ không được copy

                # lưu ảnh và label vào dataset mới
                new_img_path = new_img_dir / img_path.name
                new_label_path = new_label_dir / label_file.name

                shutil.copy2(img_path, new_img_path)
                with open(new_label_path, "w") as f:
                    f.writelines(filtered_lines)

        print(f"[DONE] Dataset mới đã được lưu tại: {new_dataset_root}")
