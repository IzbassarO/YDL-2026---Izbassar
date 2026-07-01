"""Vehicle detection on a traffic video with pretrained YOLOv8 (the Aqyl Jol vehicle layer).

Runs the COCO-pretrained detector on a driving clip, keeps vehicle classes, writes an
annotated MP4, saves a few sample frames, and reports FPS + per-class counts (echoing
Aqyl Jol's 60-125 FPS vehicle-detection story).
"""
import time
import json
import collections
import cv2
import torch
from ultralytics import YOLO

SRC = "data/person-bicycle-car-detection.mp4"
OUT = "artifacts/traffic_detected.mp4"
VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", DEVICE)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(SRC)
w = int(cap.get(3)); h = int(cap.get(4)); fps = cap.get(5) or 12
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counts = collections.Counter()
busiest = []          # keep the 3 frames with the most vehicles for the notebook stills
n_frames = 0
t0 = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = model(frame, classes=list(VEHICLES), conf=0.35, verbose=False, device=DEVICE)[0]
    for c in res.boxes.cls.tolist():
        counts[VEHICLES[int(c)]] += 1
    annotated = res.plot()
    writer.write(annotated)
    cnt = len(res.boxes)
    if cnt > 0:
        if len(busiest) < 3:
            busiest.append((cnt, n_frames, annotated.copy()))
        else:
            lo = min(range(3), key=lambda k: busiest[k][0])
            if cnt > busiest[lo][0]:
                busiest[lo] = (cnt, n_frames, annotated.copy())
    n_frames += 1

for i, (cnt, idx, img) in enumerate(sorted(busiest, reverse=True, key=lambda t: t[0])):
    cv2.imwrite(f"artifacts/traffic_frame_{i}.jpg", img)
saved_frames = len(busiest)

elapsed = time.time() - t0
cap.release(); writer.release()
proc_fps = n_frames / elapsed
json.dump({"frames": n_frames, "seconds": round(elapsed, 1), "fps": round(proc_fps, 1),
           "device": DEVICE, "counts": dict(counts), "source": SRC},
          open("artifacts/yolo_metrics.json", "w"), indent=1)
print(f"processed {n_frames} frames in {elapsed:.1f}s  ->  {proc_fps:.1f} FPS on {DEVICE}")
print("detections per class (summed over frames):", dict(counts))
print(f"saved -> {OUT}  + {saved_frames} sample frames + artifacts/yolo_metrics.json")
