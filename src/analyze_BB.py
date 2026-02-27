import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


from core.data_manager import DataProcessor


# ---------------------------
# Plot histogram + KDE (log bins)
# ---------------------------
def plot_hist_kde_area_ratio(
    series: pd.Series,
    title: str,
    out_path: Path,
    bins: int = 10000
):
    if series.empty:
        return

    # ✅ GIỮ TOÀN BỘ DỮ LIỆU HỢP LỆ
    series = series[(series > 0) & (series <= 1)]
    if series.empty:
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 4))

    # ✅ bins phủ toàn miền
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    sns.histplot(
        series,
        bins=bin_edges,
        stat="count",
        kde=False,
        alpha=0.7
    )

    # ✅ zoom cách nhìn, không cắt dữ liệu
    plt.xlim(0, 0.025)

    # ✅ giữ bbox hiếm
    plt.yscale("log")

    plt.xlabel("Bounding Box Area Ratio (bbox_area / image_area)")
    plt.ylabel("Number of bounding boxes (log scale)")
    plt.title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



# python /home/team_thuctap/ltlinh/new_ndanh/src/analyze_BB.py --data /home/team_thuctap/ltlinh/new_ndanh/data/merged_data/v5_filterBB_normalized/data.yaml --out_dir /home/team_thuctap/ltlinh/new_ndanh/data/merged_data/v5_filterBB_normalized/gt_area_analysis

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="gt_area_analysis",
        help="Output directory for plots"
    )
    args = parser.parse_args()

    # ---- Load GT bbox data ----
    dp = DataProcessor(args.data)
    dims = dp.compute_bbox_dimensions()
    # (w_px, h_px, area, img_area, class_name)

    if len(dims) == 0:
        print("[ERROR] No bounding boxes found.")
        return

    df = pd.DataFrame(
        dims,
        columns=["width", "height", "area", "img_area", "class"]
    )
    df["area_ratio"] = df["area"] / df["img_area"]

    out_dir = Path(args.out_dir)

    # =========================
    # Overall (all classes)
    # =========================
    plot_hist_kde_area_ratio(
        df["area_ratio"],
        "GT Bounding Box Area Ratio Distribution (All Classes)",
        out_dir / "overall_area_ratio.png"
    )

    # =========================
    # Per class
    # =========================
    for cls in sorted(df["class"].unique()):
        sub = df[df["class"] == cls]

        # skip very small classes
        if len(sub) < 20:
            continue

        plot_hist_kde_area_ratio(
            sub["area_ratio"],
            f"GT Bounding Box Area Ratio Class: {cls}",
            out_dir / "by_class" / f"{cls}.png"
        )

    print(f"[DONE] Saved GT area ratio plots to {out_dir}")


if __name__ == "__main__":
    main()
