"""Make a snappy web video: drop empty stretches (keep frames near detections) + speed up."""
import cv2, torch
from ultralytics import YOLO

SRC = "data/person-bicycle-car-detection.mp4"
OUT = "site/assets/traffic_web.mp4"
VEH = [1, 2, 3, 5, 7]
FPS = 18            # source is ~12 fps -> plays ~1.5x faster
WIN = 8             # keep frames within +/- WIN of any detection
dev = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(SRC)
w, h = int(cap.get(3)), int(cap.get(4))
frames, det = [], []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    r = model(fr, classes=VEH, conf=0.35, verbose=False, device=dev)[0]
    frames.append(r.plot()); det.append(len(r.boxes) > 0)
cap.release()

keep = [any(det[max(0, i - WIN):i + WIN + 1]) for i in range(len(frames))]
wr = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"avc1"), FPS, (w, h))
kept = 0
for i in range(len(frames)):
    if keep[i]:
        wr.write(frames[i]); kept += 1
wr.release()
print(f"kept {kept}/{len(frames)} frames (dropped empty stretches), {FPS} fps -> {OUT}")
