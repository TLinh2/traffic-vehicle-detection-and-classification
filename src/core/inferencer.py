from ultralytics import YOLO
from pathlib import Path
import os

class Inferencer:
    def __init__(self, model_path, output_dir="outputs/predictions/"):
        self.model = YOLO(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def predict_video(self, video_path):
        video_path = Path(video_path)
        save_dir = self.output_dir / "test_on_real_data"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Running inference on {video_path.name}...")

        results = self.model.predict(
            source=str(video_path),
            save=True,
            project=str(save_dir),   # dùng test_on_real_data làm thư mục lưu
            name="",                 # không tạo thư mục con nữa
            exist_ok=True,           # ghi đè nếu có sẵn
            conf=0.4,
            iou=0.5,
            show=False
        )

        # YOLO sẽ tự lưu video vào `save_dir`
        print(f"[DONE] Output saved to {save_dir}")

