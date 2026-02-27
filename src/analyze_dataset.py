# src/analyze_dataset.py
import argparse
from core.data_manager import DataAnalyzer
from core.visualizer import Visualizer
from pathlib import Path

# @@@ Phân tích tập train của dataset @@@

if __name__ == "__main__":
    """
    Chạy phân tích tập data, với các biến:
    - save : path lưu ảnh class distribution
    - data : path đến data cần phân tích
    - split : tập train/val/test (hoặc all) cần phân tích
    Câu lệnh:
        python src/analyze_dataset.py --data data/(data_cần_phân_tích)/data.yaml --save data/(data_cần_phân_tích) --split "train" (tập cần phân tích)
        Ví dụ : python src/analyze_dataset.py --data data/merged_data/v9_filterBB_all/data.yaml --save data/merged_data/v9_filterBB_all --split "all"
    """


    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--save", type=str, default="outputs/logs/class_distribution.png", help="Path to save chart")
    parser.add_argument("--split", type=str, default="train", help="Choice to analyze")
    args = parser.parse_args()

    # Đường dẫn lưu ảnh biểu đồ dataset
    save_path = Path(args.save)
    if save_path.is_dir() or not save_path.suffix:
        save_path = save_path / "class_distribution.png"

    analyzer = DataAnalyzer(args.data)
    info = analyzer.analyze(args.split)


    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    def plot_class_distribution_splits(
        info,
        splits=("train", "val", "test"),
        save_path=None,
        title="Class Distribution (Train / Val)"
    ):
        # ===== Lấy danh sách class (union) =====
        class_set = set()
        for split in splits:
            if split in info:
                class_set.update(info[split]["per_class"].keys())

        classes = sorted(class_set)
        x = np.arange(len(classes))  # vị trí class
        width = 0.25                 # độ rộng mỗi cột

        plt.figure(figsize=(12, 6))

        # ===== Vẽ từng split =====
        for i, split in enumerate(splits):
            if split not in info:
                continue

            counts = [
                info[split]["per_class"].get(cls, 0)
                for cls in classes
            ]

            bars = plt.bar(
                x + i * width,
                counts,
                width,
                label=split
            )

            # ===== Annotate số =====
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        h,
                        f"{int(h)}",
                        ha="center",
                        va="bottom",
                        fontsize=8
                    )

        plt.title(title)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(x + width, classes, rotation=45)
        plt.legend()
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"[INFO] Saved plot to {save_path}")
        else:
            plt.show()

        plt.close()

    plot_save_path = save_path.parent / "class_distribution_all_splits.png"

    plot_class_distribution_splits(
        info,
        splits=["train", "val"],
        save_path=plot_save_path
    )

