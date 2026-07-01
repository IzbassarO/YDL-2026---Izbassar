"""Run the trained road-damage YOLO on photos -> typed boxes (pothole/crack/manhole).

This is the 'road quality in images' deliverable: each hazard gets a labelled box,
the still-image counterpart of the vehicle detector.
"""
import os
import shutil
import json
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

WEIGHTS = "runs/detect/runs/roaddamage/weights/best.pt"
IMG_DIR = "data/roaddamage/data/images"
LBL_DIR = "data/roaddamage/data/labels-YOLO"
VAL_LIST = "data/roaddamage_yolo/images/val"      # symlinks tell us which are val
shutil.copy(WEIGHTS, "data/road_damage_yolo.pt")

model = YOLO("data/road_damage_yolo.pt")

# pick val images that actually contain potholes/manholes (class 0 or 2) for a clear demo
val_stems = [os.path.splitext(f)[0] for f in os.listdir(VAL_LIST)]
def has_class(stem, cls):
    p = f"{LBL_DIR}/{stem}.txt"
    return os.path.exists(p) and any(l.split() and l.split()[0] == str(cls) for l in open(p))

picks = ([s for s in val_stems if has_class(s, 0)][:4] +
         [s for s in val_stems if has_class(s, 2)][:1] +
         [s for s in val_stems if has_class(s, 1)][:1])
picks = list(dict.fromkeys(picks))[:6]

os.makedirs("artifacts", exist_ok=True)
n = len(picks)
fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.6 * ((n + 1) // 2), 6.4))
axes = axes.ravel()
counts = {}
for i, stem in enumerate(picks):
    res = model(f"{IMG_DIR}/{stem}.jpg", conf=0.25, verbose=False)[0]
    for c in res.boxes.cls.tolist():
        counts[model.names[int(c)]] = counts.get(model.names[int(c)], 0) + 1
    annotated = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"artifacts/road_damage_{i}.jpg", res.plot())
    axes[i].imshow(annotated); axes[i].axis("off")
    axes[i].set_title(f"{len(res.boxes)} hazard(s)", fontsize=10)
for j in range(n, len(axes)):
    axes[j].axis("off")
fig.suptitle("Road-damage detector — typed hazard boxes (pothole / crack / manhole)", fontsize=12)
plt.tight_layout()
plt.savefig("artifacts/road_damage_detections.png", dpi=130, bbox_inches="tight")
json.dump({"weights": WEIGHTS, "classes": model.names, "demo_counts": counts},
          open("artifacts/road_damage_meta.json", "w"), indent=1)
print(f"saved -> artifacts/road_damage_detections.png + {n} stills")
print("demo detections:", counts)
