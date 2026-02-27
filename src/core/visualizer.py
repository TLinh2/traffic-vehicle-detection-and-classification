# src/core/visualizer.py
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

class Visualizer:
    @staticmethod
    def plot_class_distribution(per_class_counts, save_path=None, title="Class Distribution"):
        classes = list(per_class_counts.keys())
        counts = list(per_class_counts.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, counts)

        plt.title(title)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        # ===== Annotate số trên mỗi cột =====
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=10
            )

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"[INFO] Saved plot to {save_path}")
        else:
            plt.show()

        plt.close()
        
    @staticmethod
    def plot_bbox_area_distribution(areas, save_path=None, log_scale=True, title="Bounding Box Area Distribution"):
        """
        Vẽ biểu đồ histogram thể hiện phân bố diện tích của bounding boxes.
        Args:
            areas (list): Danh sách diện tích bbox (pixel²)
            save_path (str | Path): Nơi lưu ảnh biểu đồ (nếu có)
            log_scale (bool): Có dùng log-scale cho trục X không
        """
        if not areas:
            print("[WARN] Không có bounding box nào để vẽ.")
            return

        areas = np.array(areas)
        plt.figure(figsize=(20, 10))
        plt.hist(areas, bins=200, color='steelblue', edgecolor='black', alpha=0.5)

        if log_scale:
            plt.xscale("log")

        tick_values = [10, 20, 30, 100, 200, 900, 10000]
        plt.xticks(tick_values, [str(v) for v in tick_values], rotation=30)

        plt.title(title)
        plt.xlabel("Bounding box area (px²)")
        plt.ylabel("Frequency")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()

        mean_area = np.mean(areas)
        median_area = np.median(areas)
        plt.axvline(mean_area, color='red', linestyle='--', linewidth=1.5, label=f"Mean: {mean_area:.1f}")
        plt.axvline(median_area, color='green', linestyle='--', linewidth=1.5, label=f"Median: {median_area:.1f}")
        plt.legend()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"[INFO] Saved bbox area distribution to {save_path}")
        else:
            plt.show()

        plt.close()