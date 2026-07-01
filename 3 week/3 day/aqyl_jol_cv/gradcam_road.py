"""Grad-CAM for the road-condition CNN — 'does my CNN see what I think?'

Overlays the network's attention on sample validation images (correct + confidently
wrong), so we can check it keys on road-surface / sky / weather texture, not artifacts.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cpu"   # grad-cam is happiest on CPU; the dataset is tiny
meta = json.load(open("data/road_meta.json"))
classes = meta["classes"]

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
val_ds = datasets.ImageFolder("data/split/val", tf)

model = resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("data/road_resnet18.pt", map_location=DEVICE)["state_dict"])
model.eval().to(DEVICE)
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

inv = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.255],
                           [1/0.229, 1/0.224, 1/0.255])

# collect a few correct (one per class) + a couple of confident mistakes
picks, seen, wrong = [], set(), []
for i in range(len(val_ds)):
    x, y = val_ds[i]
    with torch.no_grad():
        p = model(x.unsqueeze(0).to(DEVICE)).softmax(1)[0]
    pred = int(p.argmax()); conf = float(p[pred])
    if pred == y and classes[y] not in seen:
        seen.add(classes[y]); picks.append((i, y, pred, conf))
    elif pred != y and conf > 0.6 and len(wrong) < 2:
        wrong.append((i, y, pred, conf))
    if len(seen) == len(classes) and len(wrong) == 2:
        break
picks += wrong

cols = len(picks)
fig, axes = plt.subplots(2, cols, figsize=(2.5 * cols, 5.2))
for c, (idx, y, pred, conf) in enumerate(picks):
    x, _ = val_ds[idx]
    rgb = inv(x).permute(1, 2, 0).clamp(0, 1).numpy()
    grayscale = cam(input_tensor=x.unsqueeze(0), targets=None)[0]
    overlay = show_cam_on_image(rgb.astype(np.float32), grayscale, use_rgb=True)
    ok = pred == y
    axes[0, c].imshow(rgb); axes[0, c].axis("off")
    axes[0, c].set_title(f"true: {classes[y]}", fontsize=10)
    axes[1, c].imshow(overlay); axes[1, c].axis("off")
    axes[1, c].set_title(f"{'✓' if ok else '✗'} {classes[pred]} ({conf:.2f})",
                         color="green" if ok else "red", fontsize=10)
fig.suptitle("Grad-CAM: where the road-condition CNN looks (top: image, bottom: attention)", fontsize=12)
plt.tight_layout()
os.makedirs("artifacts", exist_ok=True)
plt.savefig("artifacts/road_gradcam.png", dpi=130, bbox_inches="tight")
print(f"saved -> artifacts/road_gradcam.png  ({len(picks)} panels: {len(picks)-len(wrong)} correct, {len(wrong)} wrong)")
