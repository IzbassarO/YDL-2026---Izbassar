"""Build a YOLO train/val split for the Road Damage dataset (pothole/crack/manhole)."""
import os
import random
import yaml

SRC_IMG = "data/roaddamage/data/images"
SRC_LBL = "data/roaddamage/data/labels-YOLO"
OUT = "data/roaddamage_yolo"
NAMES = {0: "pothole", 1: "crack", 2: "manhole"}
VAL_FRAC = 0.15
random.seed(42)

stems = [os.path.splitext(f)[0] for f in os.listdir(SRC_IMG) if f.lower().endswith(".jpg")]
stems = [s for s in stems if os.path.exists(f"{SRC_LBL}/{s}.txt")]
random.shuffle(stems)
n_val = int(len(stems) * VAL_FRAC)
splits = {"val": stems[:n_val], "train": stems[n_val:]}

for split, items in splits.items():
    for sub in ("images", "labels"):
        os.makedirs(f"{OUT}/{sub}/{split}", exist_ok=True)
    for s in items:
        img_src = os.path.abspath(f"{SRC_IMG}/{s}.jpg")
        lbl_src = os.path.abspath(f"{SRC_LBL}/{s}.txt")
        img_dst = f"{OUT}/images/{split}/{s}.jpg"
        lbl_dst = f"{OUT}/labels/{split}/{s}.txt"
        for src, dst in [(img_src, img_dst), (lbl_src, lbl_dst)]:
            if not os.path.exists(dst):
                os.symlink(src, dst)

with open(f"{OUT}/data.yaml", "w") as f:
    yaml.safe_dump({"path": os.path.abspath(OUT), "train": "images/train",
                    "val": "images/val", "names": NAMES}, f, sort_keys=False)

print(f"train: {len(splits['train'])}  val: {len(splits['val'])}  classes: {list(NAMES.values())}")
print(f"data.yaml -> {OUT}/data.yaml")
