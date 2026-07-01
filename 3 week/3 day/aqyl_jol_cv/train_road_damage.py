"""Train a YOLOv8 detector for road-surface damage (pothole / crack / manhole).

This is the 'road quality' model: it draws a typed box around each hazard on a photo,
the road-side counterpart to the vehicle detector (same modality). Mirrors Aqyl Jol's
road-hazard perception via detection.
"""
import shutil
import torch
from ultralytics import YOLO

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

model = YOLO("yolov8n.pt")
model.train(
    data="data/roaddamage_yolo/data.yaml",
    epochs=25, imgsz=640, batch=16, device=DEVICE,
    project="runs", name="roaddamage", exist_ok=True,
    patience=8, verbose=True, plots=True,
)
metrics = model.val()
print(f"\nmAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")
for i, name in metrics.names.items():
    if i < len(metrics.box.maps):
        print(f"  {name:9} AP50-95 {metrics.box.maps[i]:.3f}")

shutil.copy("runs/roaddamage/weights/best.pt", "data/road_damage_yolo.pt")
print("saved -> data/road_damage_yolo.pt")
