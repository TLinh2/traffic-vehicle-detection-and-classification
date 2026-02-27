from pathlib import Path
import shutil

src_dir = Path(r"/home/team_thuctap/ltlinh/new_ndanh/data/1000_4th/chunk_001").resolve()      # thư mục đang bị trộn
parent_dir = src_dir.parent.parent       # thư mục cha

# Tạo thư mục đích
images_dir = ".." / src_dir / "images"
labels_dir = ".." / src_dir / "labels"
images_dir.mkdir(exist_ok=True)
labels_dir.mkdir(exist_ok=True)

# Các đuôi ảnh cho YOLO
image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

for file in src_dir.iterdir():
    if file.is_file():
        if file.suffix.lower() in image_exts:
            shutil.move(file, images_dir / file.name)
        elif file.suffix.lower() == ".txt":
            shutil.move(file, labels_dir / file.name)

print("Done: Images → images/, Labels → labels/")
