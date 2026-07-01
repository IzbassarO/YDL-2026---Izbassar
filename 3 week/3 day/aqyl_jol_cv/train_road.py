"""Road-condition classifier: ResNet18 transfer learning on DAWN (fog/rain/snow/sand).

This is the Day-3 "own architecture on own data": a pretrained CNN (ImageNet) whose
backbone is frozen and re-headed for 4 road-weather classes, then fine-tuned briefly.
Mirrors the road-condition perception model in Aqyl Jol.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data/split"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 12
BATCH = 32
print("device:", DEVICE)

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(), norm,
])
eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

train_ds = datasets.ImageFolder(f"{DATA}/train", train_tf)
val_ds = datasets.ImageFolder(f"{DATA}/val", eval_tf)
classes = train_ds.classes
print("classes:", classes)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

# Pretrained ResNet18, freeze backbone, new 4-class head (feature extraction)
model = resnet18(weights=ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(DEVICE)

opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()


def evaluate():
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in val_dl:
            out = model(x.to(DEVICE))
            preds += out.argmax(1).cpu().tolist()
            gts += y.tolist()
    acc = float(np.mean(np.array(preds) == np.array(gts)))
    return acc, preds, gts


best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    tot, correct, run = 0, 0, 0.0
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = lossf(out, y)
        loss.backward()
        opt.step()
        run += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        tot += x.size(0)
    val_acc, _, _ = evaluate()
    print(f"epoch {epoch:2}  train_loss {run/tot:.3f}  train_acc {correct/tot:.3f}  val_acc {val_acc:.3f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"state_dict": model.state_dict(), "classes": classes}, "data/road_resnet18.pt")

# Final report + confusion matrix on the best snapshot
ckpt = torch.load("data/road_resnet18.pt", map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
acc, preds, gts = evaluate()
print(f"\nBEST val accuracy: {acc:.3f}")
print(classification_report(gts, preds, target_names=classes, digits=3))

cm = confusion_matrix(gts, preds)
fig, ax = plt.subplots(figsize=(5, 4.3))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Road-condition CNN — val confusion (acc {acc:.2f})")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
os.makedirs("artifacts", exist_ok=True)
plt.savefig("artifacts/road_confusion.png", dpi=130)
json.dump({"classes": classes, "val_acc": acc}, open("data/road_meta.json", "w"))
print("saved -> data/road_resnet18.pt, artifacts/road_confusion.png")
