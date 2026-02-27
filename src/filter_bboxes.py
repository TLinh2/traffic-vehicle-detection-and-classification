from core.data_manager import DataProcessor
from pathlib import Path
import argparse


# CLASSES_OF_INTEREST = [
#     "tuktuk",
#     "moto",
#     "pickup truck",
#     "tricycle",
#     "special",
#     "bicycle",
# ]

CLASSES_OF_INTEREST = [
    
]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--old_path",
        type=str,
        required=True,
        help="Old dataset root (contains data.yaml)"
    )
    parser.add_argument(
        "--new_path",
        type=str,
        required=True,
        help="New dataset root"
    )
    parser.add_argument("--min_ratio", type=float, default=0)
    parser.add_argument("--max_ratio", type=float, default=0.001)

    args = parser.parse_args()

    old_yaml_path = Path(args.old_path) / "data.yaml"
    assert old_yaml_path.exists(), f"Missing {old_yaml_path}"

    processor = DataProcessor(
        data_yaml_path=str(old_yaml_path)
    )

    processor.filter_bboxes_by_area_ratio(
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        classes=None,
        new_dataset_root=args.new_path
    )

    print("\n[INFO] Filter finished")
    print("[INFO] Classes:", CLASSES_OF_INTEREST)
    print(f"[INFO] New dataset saved to: {args.new_path}")
