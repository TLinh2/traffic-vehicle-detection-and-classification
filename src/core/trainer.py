# src/core/trainer.py
from ultralytics import YOLO
from pathlib import Path
import yaml



class Trainer:
    
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.experiment_name = self.cfg["experiment_name"]
        self.project_dir = Path(self.cfg["project_dir"])
        self.exp_dir = self.project_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # paths
        self.data_yaml = self.cfg["data_yaml"]
        self.model_path = self.cfg["model"]

        # init model
        self.model = YOLO(self.model_path)
        self.model.model.info()

    def train(self):
        YOLO_OVERRIDE_KEYS = {
            # Train settings
            "model", "data", "epochs", "time", "patience", "batch", "imgsz",
            "save", "save_period", "cache", "device", "workers",
            "project", "name", "exist_ok", "pretrained", "optimizer",
            "verbose", "seed", "deterministic", "single_cls", "rect",
            "cos_lr", "close_mosaic", "resume", "amp", "fraction",
            "profile", "freeze", "multi_scale", "compile",

            # Segmentation
            "overlap_mask", "mask_ratio",

            # Classification
            "dropout",

            # Val / Test
            "val", "split", "save_json", "conf", "iou", "max_det",
            "half", "dnn", "plots",

            # Predict
            "source", "vid_stride", "stream_buffer", "visualize",
            "augment", "agnostic_nms", "classes", "retina_masks", "embed",

            # Visualize
            "show", "save_frames", "save_txt", "save_conf", "save_crop",
            "show_labels", "show_conf", "show_boxes", "line_width",

            # Export
            "format", "keras", "optimize", "int8", "dynamic", "simplify",
            "opset", "workspace", "nms",

            # Hyperparameters
            "lr0", "lrf", "momentum", "weight_decay",
            "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
            "box", "cls", "dfl", "pose", "kobj", "nbs",
            "hsv_h", "hsv_s", "hsv_v",
            "degrees", "translate", "scale", "shear", "perspective",
            "flipud", "fliplr", "bgr",
            "mosaic", "mixup", "cutmix",
            "copy_paste", "copy_paste_mode",
            "auto_augment", "erasing",

            # Custom
            "cfg",

            # Tracker
            "tracker",
        }
        print(f"[INFO] Training model on {self.data_yaml}")

        train_kwargs = {}

        print("\n[INFO] YOLO parameter overrides:")
        for k, v in self.cfg.items():
            if k in YOLO_OVERRIDE_KEYS:
                train_kwargs[k] = v
                print(f"  ✓ {k} = {v}")
            else:
                print(f"  ⚠ Ignored unknown key: {k}")

        # Bắt buộc phải có
        train_kwargs.update({
            "data": self.data_yaml,
            "project": str(self.project_dir.resolve()),  # experiments
            "name": self.experiment_name,                # v1_yolo11m_2026-02-26
            "exist_ok": True,
        })

        results = self.model.train(**train_kwargs)

        print("[DONE] Training finished")
        return results

    def predict_val(self):
        pred_dir = self.exp_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)

        val_img_dir = self._get_val_images_dir()

        print(f"[INFO] Predicting on val images: {val_img_dir}")

        self.model.predict(
            source=str(val_img_dir),   
            imgsz=self.cfg["imgsz"],
            conf=self.cfg.get("conf", 0.25),
            iou=self.cfg.get("iou", 0.5),
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(pred_dir),
            name="val"
        )
    def run_error_analysis(self):
        from src.error_analysis.run import run

        run(
            gt_dir=self._get_val_label_dir(),
            pred_dir=self.exp_dir / "predictions" / "val" / "labels",
            img_dir=self._get_val_images_dir(),
            out_dir=self.exp_dir / "error_analysis"
        )

    def _get_val_images_dir(self):
        # parse data.yaml
        with open(self.data_yaml, "r") as f:
            data = yaml.safe_load(f)

        val_path = data["val"]

        # nếu val là relative path
        if not Path(val_path).is_absolute():
            val_path = Path(self.data_yaml).parent / val_path

        return Path(val_path)
    
    def _get_val_label_dir(self):
        img_dir = self._get_val_images_dir()
        return img_dir.parent.parent / "labels" / img_dir.name