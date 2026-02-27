from core.data_manager import DataProcessor
from pathlib import Path

DATA_YAML = "/home/team_thuctap/ltlinh/new_ndanh/data/merged_data/v4_filterUnidentified/data.yaml"
TARGET_CLASS = "unidentified"
splits = ["train", "val", "test"]

dp = DataProcessor(DATA_YAML)

# tìm tất cả class id tương ứng (cho chắc)
target_ids = [
    k for k, v in dp.analyzer.names.items()
    if v.lower().strip() == TARGET_CLASS.lower()
]

if not target_ids:
    print(f"[ERROR] Không tìm thấy class '{TARGET_CLASS}' trong names")
    exit()

target_ids = set(target_ids)
print(f"[INFO] Target class ids: {target_ids}")

dataset_root = dp.dataset_root

total_instances = 0
found_files = []

for split in splits:
    label_dir = dataset_root / "labels" / split
    if not label_dir.exists():
        continue

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        hit = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
            except:
                continue

            if cls_id in target_ids:
                hit += 1

        if hit > 0:
            total_instances += hit
            found_files.append((split, label_file, hit))

# ===== REPORT =====
print("\n====== FILES STILL CONTAINING Unidentified ======")
for split, lf, cnt in found_files:
    print(f"[{split}] {lf}  --> {cnt} instance(s)")

print(f"\n[SUMMARY] Total remaining Unidentified instances: {total_instances}")
