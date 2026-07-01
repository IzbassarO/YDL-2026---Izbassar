"""Detect and TYPE vehicles in ANY video -> annotated MP4 (like 'Traffic in motion').

Usage:
    python detect_video.py <input_video> [output_video]

Examples:
    python detect_video.py ~/Downloads/my_traffic.mp4
    python detect_video.py ~/Downloads/street.mov out.mp4
"""
import sys
import os
import time
import collections
import cv2
import torch
from ultralytics import YOLO

if len(sys.argv) < 2:
    print("Usage: python detect_video.py <input_video> [output_video]")
    sys.exit(1)

SRC = os.path.expanduser(sys.argv[1])
OUT = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else os.path.splitext(SRC)[0] + "_detected.mp4"
VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SRC):
    print("File not found:", SRC); sys.exit(1)

print(f"input : {SRC}\ndevice: {DEVICE}\nloading YOLOv8...")
model = YOLO("yolov8n.pt")                       # auto-downloads if missing

cap = cv2.VideoCapture(SRC)
if not cap.isOpened():
    print("Could not open the video (unsupported codec?)."); sys.exit(1)
w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

counts = collections.Counter()
n, t0 = 0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = model(frame, classes=list(VEHICLES), conf=0.35, verbose=False, device=DEVICE)[0]
    for c in res.boxes.cls.tolist():
        counts[VEHICLES[int(c)]] += 1
    writer.write(res.plot())                     # draws boxes + type + confidence
    n += 1
    if total and n % 50 == 0:
        print(f"  {n}/{total} frames...", flush=True)
cap.release(); writer.release()

print(f"\ndone: {n} frames · {n / (time.time() - t0):.1f} FPS on {DEVICE}")
print("vehicles seen (summed over frames):", dict(counts))
print("saved ->", OUT)
